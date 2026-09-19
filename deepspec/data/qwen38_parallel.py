"""GPU-group workers and zero-copy finalization of Qwen3.8 feature caches."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .capture_journal import capture_lock, sync_directory
from .ngram_cache import RECORD
from .target_cache_dataset import INDEX_RECORD_STRUCT, atomic_json_dump


class ProgressReader:
    """Best-effort display only; never use these readings to resume or merge."""
    def __init__(self):
        self.last = {}
        self.last_warning = {}

    def read(self, path):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('Progress state must be an object')
            source, saved = value['next_source'], value['saved']
            if type(source) is not int or type(saved) is not int or not 0 <= saved <= source:
                raise ValueError('Invalid progress counters')
        except (OSError, ValueError, KeyError) as error:
            # A missing state is normal before the child's first checkpoint.
            if not isinstance(error, FileNotFoundError) or path in self.last:
                now = time.monotonic()
                if now - self.last_warning.get(path, float('-inf')) >= 60:
                    print(f'Warning: cannot refresh progress {path}: '
                          f'{type(error).__name__}: {error}; retaining previous display. '
                          'Worker exit status is still monitored.', file=sys.stderr, flush=True)
                    self.last_warning[path] = now
            return self.last.get(path)  # None means unknown, not zero committed.
        self.last[path] = (source, saved)
        return self.last[path]


def partition_ranges(count, workers):
    if workers <= 0 or count < workers:
        raise ValueError('Need at least one source record per instance')
    return [(count * i // workers, count * (i + 1) // workers) for i in range(workers)]


def visible_gpu_groups(size):
    import torch
    value = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = value.split(',') if value is not None else [str(i) for i in range(torch.cuda.device_count())]
    devices = [device.strip() for device in devices if device.strip()]
    if not devices or '-1' in devices or len(set(devices)) != len(devices) or len(devices) % size:
        raise ValueError('Visible GPUs must be distinct and divisible by --gpus-per-instance')
    return [devices[i:i + size] for i in range(0, len(devices), size)]


def link_once(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise RuntimeError(f'Unexpected existing merged shard: {destination}')
    else:
        os.link(source, destination)  # Worker directories are on the same filesystem.


def merge_workers(output, ranges, verify):
    root = Path(output)
    manifests = []
    identities = []
    for rank, (start, end) in enumerate(ranges):
        part = root / '_workers' / f'worker-{rank:03d}'
        manifest = json.loads((part / 'manifest.json').read_text())
        if manifest.get('source_range') != [start, end]:
            raise ValueError(f'Worker {rank}: source partition mismatch')
        if manifest['num_source_samples_examined'] != end - start:
            raise ValueError(f'Worker {rank}: incomplete source partition')
        state = json.loads((part / 'capture_state.json').read_text())
        if state['next_source'] != end - start or state['saved'] != manifest['num_samples']:
            raise ValueError(f'Worker {rank}: incomplete checkpoint')
        identity = {k: v for k, v in state['identity'].items() if k not in {'source_range', 'source_limit'}}
        identities.append(identity)
        verify(str(part), {}, manifest=manifest, allow_empty=True)
        manifests.append(manifest)
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError('Worker model/source/config identities do not match')
    for manifest in manifests[1:]:
        for key in ('target_layer_ids', 'hidden_size', 'target_config', 'chat_template_sha256',
                    'hidden_dtype', 'token_dtype', 'mask_dtype', 'aux_reduction', 'target_final_feature'):
            if manifest.get(key) != manifests[0].get(key):
                raise ValueError(f'Worker metadata mismatch: {key}')
    has_ngram = 'ngram_embedding' in manifests[0].get('extra_features', {})
    if any(('ngram_embedding' in m.get('extra_features', {})) != has_ngram for m in manifests):
        raise ValueError('Workers disagree about ngram capture')
    side_root = root / 'features/ngram_embedding'
    if has_ngram:
        side_root.mkdir(parents=True, exist_ok=True)
    shards, side_shards, total = [], [], 0
    side_handle = (side_root / 'samples.idx.tmp').open('wb') if has_ngram else None
    try:
        with (root / 'samples.idx.tmp').open('wb') as main_index:
            for rank, m in enumerate(manifests):
                part = root / '_workers' / f'worker-{rank:03d}'
                main_base, side_base = len(shards), len(side_shards)
                for shard in m['shards']:
                    name = f'shard-{len(shards):05d}.bin'
                    link_once(part / shard['file_name'], root / name)
                    shards.append(dict(shard_id=len(shards), file_name=name))
                if has_ngram:
                    meta = m['extra_features']['ngram_embedding']
                    original = manifests[0]['extra_features']['ngram_embedding']
                    if {k:v for k,v in meta.items() if k not in {'num_samples','shards'}} != {
                            k:v for k,v in original.items() if k not in {'num_samples','shards'}}:
                        raise ValueError('Worker ngram metadata mismatch')
                    for shard in meta['shards']:
                        name = f'shard-{len(side_shards):05d}.bin'
                        link_once(part / meta['path'] / shard['file_name'], side_root / name)
                        side_shards.append(dict(file_name=name, num_bytes=shard['num_bytes']))
                    with (part / meta['path'] / meta['index_file']).open('rb') as index:
                        for i in range(m['num_samples']):
                            sid, shard, length, offset = RECORD.unpack(index.read(RECORD.size))
                            if sid != i:
                                raise ValueError('Unordered ngram index')
                            side_handle.write(RECORD.pack(total + sid, side_base + shard, length, offset))
                with (part / 'samples.idx').open('rb') as index:
                    for i in range(m['num_samples']):
                        sid, shard, *fields = INDEX_RECORD_STRUCT.unpack(index.read(INDEX_RECORD_STRUCT.size))
                        if sid != i:
                            raise ValueError('Unordered main index')
                        main_index.write(INDEX_RECORD_STRUCT.pack(total + sid, main_base + shard, *fields))
                total += m['num_samples']
            main_index.flush()
            os.fsync(main_index.fileno())
        if side_handle:
            side_handle.flush()
            os.fsync(side_handle.fileno())
    finally:
        if side_handle:
            side_handle.close()
    if not total:
        raise RuntimeError('All instances filtered every sample')
    os.replace(root / 'samples.idx.tmp', root / 'samples.idx')
    if has_ngram:
        os.replace(side_root / 'samples.idx.tmp', side_root / 'samples.idx')
        sync_directory(side_root)
        sync_directory(side_root.parent)
    sync_directory(root)
    merged = copy.deepcopy(manifests[0])
    merged.update(num_samples=total, num_shards=len(shards), shards=shards,
                  source_range=[ranges[0][0], ranges[-1][1]],
                  num_source_samples_examined=sum(end-start for start,end in ranges),
                  num_filtered_samples=sum(m['num_filtered_samples'] for m in manifests),
                  num_instances=len(ranges),
                  worker_source_ranges=[list(r) for r in ranges],
                  worker_device_maps=[m.get('device_map') for m in manifests])
    merged.pop('device_map', None)
    if has_ngram:
        merged['extra_features']['ngram_embedding'].update(num_samples=total, shards=side_shards)
    verify(str(root), {}, manifest=merged)
    merged['verification'] = dict(all_indices=True, worker_caches_verified=True,
                                 payload_verification='see worker manifests')
    atomic_json_dump(merged, str(root / 'manifest.json'))
    sync_directory(root)
    return merged


def run_grouped(args, config, *, verify, snapshot):
    from .jsonl_dataset import JsonLineDataset
    root = Path(args.output_dir).absolute()
    groups = visible_gpu_groups(args.gpus_per_instance)
    sources = snapshot(args.train_data_path)
    data = JsonLineDataset(args.train_data_path)
    try:
        count = min(len(data), args.max_samples) if args.max_samples else len(data)
    finally:
        data.close()
    ranges = partition_ranges(count, len(groups))
    signature = dict(version=1, config=config, source_files=sources,
                     source_limit=count, gpus_per_instance=args.gpus_per_instance,
                     ranges=[list(r) for r in ranges])
    if getattr(args, 'backend', 'transformers') == 'sglang':
        from deepspec.data.qwen38_sglang_capture import backend_identity
        signature['capture_backend'] = backend_identity(args)
        signature['capture_backend']['tp_size'] = args.gpus_per_instance
    with capture_lock(root, resume=args.resume):
        state_path = root / 'group_state.json'
        if args.resume:
            if not state_path.exists() or json.loads(state_path.read_text()) != signature:
                raise ValueError('Grouped resume requires matching group_state.json and partitions')
        else:
            atomic_json_dump(signature, str(state_path))
            sync_directory(root)
        if (root / 'manifest.json').exists():
            verify(str(root), {})
            print(f'Grouped cache already complete: {root}', flush=True)
            return
        jobs, handles = [], []
        try:
            for rank, (devices, (start, end)) in enumerate(zip(groups, ranges)):
                part = root / '_workers' / f'worker-{rank:03d}'
                part.mkdir(parents=True, exist_ok=True)
                log = (root / f'worker-{rank:03d}.log').open('a' if args.resume else 'w')
                handles.append(log)
                options = [s for s in sys.argv[1:] if s != '--resume']
                options += ['--gpus-per-instance', '0', '--worker-start', str(start),
                            '--worker-end', str(end), '--output-dir', str(part)]
                if (part / 'capture_state.json').exists():
                    options += ['--resume']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices))
                job = subprocess.Popen([sys.executable, str(Path(sys.argv[0]).resolve()), *options],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
                jobs.append(job)
                print(f'Instance {rank}: GPUs={devices}, source=[{start},{end}), log={log.name}', flush=True)
            last_progress = None
            progress_reader = ProgressReader()
            while True:
                codes = [job.poll() for job in jobs]
                if any(code is not None and code != 0 for code in codes):
                    raise RuntimeError(f'Capture worker failed (exit codes {codes}); inspect worker logs and rerun --resume')
                if all(code == 0 for code in codes):
                    break
                progress = []
                for rank in range(len(jobs)):
                    state = root / '_workers' / f'worker-{rank:03d}' / 'capture_state.json'
                    progress.append(progress_reader.read(state))
                if progress != last_progress:
                    print(f'Committed (source, saved) per instance: {progress}', flush=True)
                    last_progress = progress
                time.sleep(2)
        finally:
            for job in jobs:
                if job.poll() is None:
                    job.terminate()
            for job in jobs:
                try:
                    job.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    job.kill()
                    job.wait()
            for handle in handles:
                handle.close()
        if snapshot(args.train_data_path) != sources:
            raise RuntimeError('Source changed during grouped capture')
        manifest = merge_workers(root, ranges, verify)
        print(f'Prepared merged cache: samples={manifest["num_samples"]}, output={root}', flush=True)
