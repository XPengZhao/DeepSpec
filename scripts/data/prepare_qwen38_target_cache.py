"""Qwen3.8 full-sequence cache generation with optional independent GPU groups.

Run from the repository root with PYTHONPATH=. (not torchrun).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader, Subset

from deepspec.data import ConversationCollator
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.data.qwen38_capture import capture_features, load_target, verify_ngram_boundaries
from deepspec.data.ngram_cache import NgramCacheReader
from deepspec.data.capture_journal import CaptureJournal, capture_lock, sync_directory
from deepspec.data.target_cache_dataset import (
    CacheDataset, atomic_json_dump, INDEX_RECORD_STRUCT,
    build_target_cache_manifest, write_target_cache_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/data/qwen38_target_cache.json")
    parser.add_argument("--model-path")
    parser.add_argument("--backend", choices=("transformers", "sglang"), default="transformers",
                        help="Transformers layer placement or SGLang true TP across each GPU group")
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.80)
    parser.add_argument("--sglang-linear-backend", choices=("flashinfer", "triton"), default="flashinfer")
    parser.add_argument("--sglang-mamba-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--sglang-ple-offload", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep the TP-sharded PLE embedding table in pinned CPU memory")
    parser.add_argument("--gpus-per-instance", type=int, default=0,
                        help="Group visible GPUs into independent instances; 0 uses all GPUs in one instance")
    parser.add_argument("--worker-start", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-end", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--capture-ngram", action=argparse.BooleanOptionalAction, default=None,
                        help="Save raw 2560-dimensional ngram lookup features (default: config)")
    parser.add_argument("--train-data-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume the last durable source-row checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Checkpoint every N completed capture batches (default: 100)")
    parser.add_argument("--max-samples", type=int, help="Maximum source records to examine (smoke run)")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-memory-gib", type=float, default=120,
                        help="Maximum weight allocation per visible GPU; excludes activation headroom")
    parser.add_argument("--reserve-gib", type=float, default=8,
                        help="Also leave at least this much currently free memory for execution")
    parser.add_argument("--max-shard-gib", type=float, default=64)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    if args.capture_ngram is not None:
        config["capture_ngram"] = args.capture_ngram
    if args.model_path:
        config["target_model_name_or_path"] = args.model_path
    if args.max_length is not None:
        config["max_length"] = args.max_length
    if (args.gpus_per_instance < 0 or args.checkpoint_interval <= 0 or args.local_batch_size <= 0 or args.num_workers < 0 or args.log_interval <= 0
            or args.gpu_memory_gib <= 0 or args.reserve_gib < 0 or args.max_shard_gib <= 0
            or config["max_length"] <= 0 or config["min_loss_tokens"] < 0
            or (args.max_samples is not None and args.max_samples <= 0)):
        parser.error("Invalid batch, length, memory, sample, or worker limits")
    if not 0 < args.sglang_mem_fraction < 1:
        parser.error("sglang-mem-fraction must be between 0 and 1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ:
        parser.error("Run one Python process with multiple visible GPUs; do not use torchrun")
    return args, config


def source_snapshot(paths):
    result = []
    for path in sorted(paths):
        stat = os.stat(path)
        result.append(dict(path=os.path.abspath(path), size=stat.st_size,
                           mtime_ns=stat.st_mtime_ns, inode=stat.st_ino))
    return result


def verify_cache(output_dir, expected, manifest=None, *, allow_empty=False):
    dataset = CacheDataset(output_dir, manifest=manifest)
    ngram_reader = (NgramCacheReader(output_dir, dataset.manifest)
                    if "ngram_embedding" in dataset.manifest.get("extra_features", {}) else None)
    try:
        if not len(dataset) and not allow_empty:
            raise RuntimeError("No valid samples were written")
        # Scan all indices without reading terabytes of payloads. Every field must
        # be contiguous and every sidecar row must have the same sample/length.
        from deepspec.data.ngram_cache import RECORD as NGRAM_RECORD
        from contextlib import ExitStack
        sizes = [Path(dataset.shard_paths[i]).stat().st_size for i in range(len(dataset.shard_paths))]
        ends = [0] * len(sizes)
        ngram_ends = [0] * len(ngram_reader.shards) if ngram_reader else []
        with ExitStack() as stack:
            main_index = stack.enter_context(open(dataset.index_path, 'rb'))
            side_index = stack.enter_context(ngram_reader.index.open('rb')) if ngram_reader else None
            for sample_id in range(len(dataset)):
                stored, shard, length, *offsets = INDEX_RECORD_STRUCT.unpack(main_index.read(INDEX_RECORD_STRUCT.size))
                if stored != sample_id or length <= 0 or shard >= len(sizes):
                    raise RuntimeError(f"Invalid main index at sample {sample_id}")
                cursor = ends[shard]
                field_sizes = [length * 4, length, length,
                               length * dataset.hidden_size * dataset.num_target_layers * 2,
                               length * dataset.hidden_size * 2]
                for offset, size in zip(offsets, field_sizes):
                    if offset != cursor:
                        raise RuntimeError(f"Noncontiguous main fields at sample {sample_id}")
                    cursor += size
                if cursor > sizes[shard]:
                    raise RuntimeError(f"Main shard truncated at sample {sample_id}")
                ends[shard] = cursor
                if side_index:
                    nid, ns, nl, offset = NGRAM_RECORD.unpack(side_index.read(NGRAM_RECORD.size))
                    if (nid != sample_id or nl != length or ns >= len(ngram_ends)
                            or offset != ngram_ends[ns]):
                        raise RuntimeError(f"Ngram/main index mismatch at sample {sample_id}")
                    ngram_ends[ns] += length * ngram_reader.width * 2
        if ends != sizes or any(size == 0 for size in sizes):
            raise RuntimeError("Main shard sizes do not match all indexed samples")
        if ngram_reader and ngram_ends != [p.stat().st_size for p in ngram_reader.shards]:
            raise RuntimeError("Ngram shard sizes do not match all indexed samples")
        for index, fields in expected.items():
            actual = dataset[index]
            if ngram_reader is not None:
                actual["ngram_embedding"] = ngram_reader.read(index, seq_len=actual["input_ids"].numel())
            for key, value in fields.items():
                if key == "attention_mask":
                    continue  # CacheDataset reconstructs padding masks in CacheCollator.
                if not torch.equal(actual[key], value):
                    raise RuntimeError(f"Cache read-back mismatch: sample {index}, {key}")
        print(f"All indices verified; read-back verified: {len(expected)} sample(s); total={len(dataset)}", flush=True)
    finally:
        dataset.close()


def resume_identity(args, config, tokenizer, source_files, count):
    import transformers
    model_path = Path(config['target_model_name_or_path'])
    if not model_path.is_dir():
        raise ValueError('Resumable capture requires a local model directory')
    model_files = sorted(p for p in model_path.rglob('*') if p.is_file()
                         and (p.suffix in {'.safetensors', '.json', '.jinja', '.model', '.txt'}))
    if not any(p.suffix == '.safetensors' for p in model_files):
        raise ValueError('No safetensors weights found in local model directory')
    identity = dict(config=config, source_files=source_files, source_limit=count,
                model_files=source_snapshot(model_files),
                tokenizer_template_sha256=hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
                transformers_version=transformers.__version__, torch_version=str(torch.__version__),
                capture_protocol=1)
    # Leave old HF identities untouched, but never mix TP/HF or TP settings
    # inside the same resumed cache.
    if getattr(args, 'backend', 'transformers') == 'sglang':
        from deepspec.data.qwen38_sglang_capture import backend_identity
        identity['capture_backend'] = backend_identity(args)
    return identity


def run_capture(args, config):
    from transformers import AutoTokenizer
    path = config['target_model_name_or_path']
    layer_ids = config['target_layer_ids']
    if config['chat_template'] != 'qwen38_non_thinking':
        raise ValueError('Expected the non-thinking Qwen3.8 regen dataset')
    if layer_ids != sorted(set(layer_ids)) or not layer_ids:
        raise ValueError('target_layer_ids must be non-empty, sorted and unique')
    output = os.path.abspath(args.output_dir)
    torch.manual_seed(config.get('seed', 42))
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.truncation_side = 'right'
    source_files = source_snapshot(args.train_data_path)
    dataset = JsonLineDataset(args.train_data_path)
    journal = None
    tp_backend = None
    try:
        count = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
        if not count:
            raise ValueError('Empty source dataset')
        source_start = getattr(args, 'worker_start', None)
        source_end = getattr(args, 'worker_end', None)
        is_worker = source_start is not None
        if is_worker:
            if source_end is None or not 0 <= source_start < source_end <= count:
                raise ValueError('Invalid worker source range')
            count = source_end - source_start
        elif source_end is not None:
            raise ValueError('worker-end requires worker-start')
        else:
            source_start = 0
        identity = resume_identity(args, config, tokenizer, source_files, count)
        if is_worker:
            identity['source_range'] = [source_start, source_end]
        journal = CaptureJournal(output, identity=identity, resume=args.resume,
                                 max_shard_bytes=int(args.max_shard_gib * 2**30),
                                 ngram_width=2560 if config.get('capture_ngram', True) else None)
        if journal.complete:
            verify_cache(output, {}, allow_empty=is_worker)
            print(f'Cache already complete: {output}', flush=True)
            return
        start = journal.state['next_source']
        if not 0 <= start <= count:
            raise ValueError('Checkpoint source cursor is outside the dataset')
        print(f'Capture starting at source={start}/{count}, saved={journal.saved}', flush=True)
        expected = {}
        metadata = journal.state['metadata']
        if start < count:
            collator = ConversationCollator(tokenizer, config['chat_template'],
                                            config['max_length'], config['min_loss_tokens'])
            preview = collator([dataset[source_start + start]])
            if preview is not None:
                print('Next sample:', {k: list(v.shape) for k, v in preview.items()},
                      'loss_tokens=', int(preview['loss_mask'].sum()), flush=True)
            if getattr(args, 'backend', 'transformers') == 'sglang':
                import transformers
                from transformers import AutoConfig
                from deepspec.data.qwen38_sglang_capture import SGLangTPCapture
                target_config = AutoConfig.from_pretrained(path)
                if target_config.model_type != 'qwen4_exp' or getattr(target_config, 'quantization_config', None):
                    raise ValueError('Expected an unquantized BF16 qwen4_exp checkpoint')
                transformers_version, device_map = transformers.__version__, None
                model = None
            else:
                model, target_config, device_map, transformers_version = load_target(
                    path, args.gpu_memory_gib, args.reserve_gib)
            text_config = target_config.text_config
            if (text_config.hidden_size != 2560 or text_config.hc_count != 4
                    or any(i < 0 or i >= text_config.num_hidden_layers for i in layer_ids)):
                raise ValueError('Unexpected Qwen3.8 architecture or invalid capture layers')
            if journal.ngram is not None and (text_config.ple_embed_dim != 2560
                                             or list(text_config.ple_layer_ids) != [2]):
                raise ValueError('Expected PLE layer 2 with 2560-dimensional embeddings')
            if 'capture_backend' in identity:
                tp_backend = SGLangTPCapture(args, config, identity['capture_backend'])
            try:
                git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                    cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                git_sha = 'unknown'
            metadata = dict(**config, git_sha=git_sha, transformers_version=transformers_version,
                            source_jsonl_paths=[os.path.abspath(p) for p in sorted(args.train_data_path)],
                            source_file_snapshot=source_files, capture_scope='full_sequence',
                            loss_mask_scope='assistant_response',
                            aux_reduction='post_layer_hc_mean_fp32_to_bfloat16',
                            target_final_feature='last_hidden_state_after_hyper_connection_mixer',
                            hc_count=text_config.hc_count, enable_thinking=False,
                            truncation_side='right', device_map=device_map,
                            chat_template_sha256=identity['tokenizer_template_sha256'],
                            target_config=target_config.to_dict())
            if tp_backend is not None:
                metadata['capture_backend'] = identity['capture_backend']
                metadata['ngram_verification'] = 'packed_vs_standalone; HF comparison required separately'

            loader = DataLoader(Subset(dataset, range(source_start + start, source_start + count)), batch_size=args.local_batch_size,
                                collate_fn=collator, num_workers=args.num_workers,
                                **({'multiprocessing_context': 'spawn'} if args.num_workers else {}))
            started, initial_saved = time.monotonic(), journal.saved
            checked_boundaries = False
            total_tokens, capture_seconds, write_seconds = 0, 0., 0.
            for step, batch in enumerate(loader, 1):
                seen = min(start + step * args.local_batch_size, count)
                if step == 1 or step % args.log_interval == 0:
                    if source_snapshot(args.train_data_path) != source_files:
                        raise RuntimeError('Source JSONL changed during capture')
                if batch is not None:
                    capture_started = time.monotonic()
                    if tp_backend is not None:
                        aux, final, ngram = tp_backend.capture(batch['input_ids'], batch['attention_mask'])
                    else:
                        aux, final, ngram = capture_features(model, batch['input_ids'],
                            batch['attention_mask'], layer_ids, capture_ngram=journal.ngram is not None)
                    capture_seconds += time.monotonic() - capture_started
                    total_tokens += int(batch['attention_mask'].sum())
                    write_started = time.monotonic()
                    for j, length in enumerate(batch['attention_mask'].sum(1).tolist()):
                        fields = {k: v[j, :length].clone() for k, v in batch.items()}
                        fields.update(target_hidden_states=aux[j, :length].clone(),
                                      target_last_hidden_states=final[j, :length].clone())
                        sample_id = journal.saved
                        raw = ngram[j, :length] if ngram is not None else None
                        if raw is not None and not checked_boundaries and tp_backend is None:
                            verify_ngram_boundaries(model, fields['input_ids'], raw)
                            checked_boundaries = True
                        if sample_id == 0:
                            spans, left = [], 0
                            mask, ids = fields['loss_mask'].tolist(), fields['input_ids'].tolist()
                            for stop in range(1, len(mask) + 1):
                                if stop == len(mask) or mask[stop] != mask[left]:
                                    spans.append(dict(start=left, end=stop, supervised=bool(mask[left]),
                                                      text=tokenizer.decode(ids[left:stop], skip_special_tokens=False)))
                                    left = stop
                            atomic_json_dump({'input_ids': ids, 'segments': spans},
                                             os.path.join(output, 'first_sample_preview.json'))
                        journal.write(fields, raw)
                        if raw is not None:
                            fields['ngram_embedding'] = raw.clone()
                        if len(expected) == 2:
                            del expected[max(expected)]
                        expected[sample_id] = fields
                    write_seconds += time.monotonic() - write_started
                if step % args.checkpoint_interval == 0 or seen == count:
                    if source_snapshot(args.train_data_path) != source_files:
                        raise RuntimeError('Source JSONL changed; checkpoint refused')
                    journal.checkpoint(seen, metadata)
                if step == 1 or step % args.log_interval == 0 or seen == count:
                    elapsed = time.monotonic() - started
                    print(f'source={seen}/{count} saved={journal.saved} '
                          f'committed_source={journal.state["next_source"]} elapsed={elapsed:.1f}s '
                          f'samples/s={(journal.saved - initial_saved) / max(elapsed, .001):.3f} '
                          f'tokens/s={total_tokens / max(elapsed, .001):.1f} '
                          f'capture_s={capture_seconds:.1f} write_s={write_seconds:.1f}', flush=True)
        journal.close()
        if source_snapshot(args.train_data_path) != source_files:
            raise RuntimeError('Source JSONL changed; cache will not be published')
        saved = journal.state['saved']
        if not saved and not is_worker:
            raise RuntimeError('All samples filtered; check max_length and response masks')
        shards = [dict(shard_id=i, file_name=name) for i, name in enumerate(journal.state['main_shards'])]
        manifest = build_target_cache_manifest(num_samples=saved, shards=shards,
                    target_layer_ids=layer_ids, hidden_size=2560,
                    extra_fields={**metadata, 'num_source_samples_examined': count,
                                  'num_filtered_samples': count - saved})
        if journal.ngram is not None:
            feature = journal.ngram.metadata()
            text = metadata['target_config']['text_config']
            feature.update(module_path='language_model.layers.1.ple.ple_embedding', decoder_layer_index=1,
                           ngram_orders=list(range(2, text['ngram_size'] + 1)),
                           heads_per_ngram=text['heads_per_ngram'], model_config_source='target_config.text_config')
            manifest['extra_features'] = {'ngram_embedding': feature}
        if is_worker:
            manifest['source_range'] = [source_start, source_end]
        verify_cache(output, expected, manifest=manifest, allow_empty=is_worker)
        manifest['verification'] = dict(all_indices=True, payload_sample_ids=sorted(expected))
        write_target_cache_manifest(output_dir=output, manifest=manifest)
        sync_directory(output)
        print(f'Prepared {saved} full-sequence samples at {output}', flush=True)
    finally:
        if tp_backend is not None:
            tp_backend.close()
        if journal is not None:
            journal.close()
        dataset.close()


def main():
    args, config = parse_args()
    if args.gpus_per_instance:
        from deepspec.data.qwen38_parallel import run_grouped
        run_grouped(args, config, verify=verify_cache, snapshot=source_snapshot)
    else:
        with capture_lock(args.output_dir, resume=args.resume):
            run_capture(args, config)


if __name__ == '__main__':
    main()
