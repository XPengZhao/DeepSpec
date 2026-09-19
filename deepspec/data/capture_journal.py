"""Single-writer transactional checkpoints for Qwen feature collection."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from .ngram_cache import NgramCacheWriter
from .target_cache_dataset import LocalTargetCacheWriter, atomic_json_dump


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def capture_lock(output, *, resume):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode: deleting it would let a second process lock a new inode.
    with (root / '.capture.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another capture process is using this output directory') from error
        if not resume and any(p.name != '.capture.lock' for p in root.iterdir()):
            raise ValueError('Output is not empty; use --resume with the original settings')
        yield


class _MainWriter(LocalTargetCacheWriter):
    def __init__(self, root, limit, state):
        self.rank_dir = str(root)
        self.max_shard_bytes = limit
        self.local_index_path = str(root / 'samples.idx')
        self.index_handle = open(self.local_index_path, 'ab')
        self.local_shard_files = list(state.get('main_shards', []))
        self.current_shard_id = len(self.local_shard_files) - 1
        self.current_shard_handle = None
        self.current_shard_size = 0
        self.num_local_samples = state['saved']
        if self.local_shard_files:
            path = root / self.local_shard_files[-1]
            self.current_shard_size = path.stat().st_size
            self.current_shard_handle = path.open('ab')

    def _open_new_shard(self):
        if self.current_shard_handle is not None:
            self.current_shard_handle.flush()
            os.fsync(self.current_shard_handle.fileno())
            self.current_shard_handle.close()
        self.current_shard_id += 1
        name = f'shard-{self.current_shard_id:05d}.bin'
        self.current_shard_handle = (Path(self.rank_dir) / name).open('xb')
        self.current_shard_size = 0
        self.local_shard_files.append(name)


class _NgramWriter(NgramCacheWriter):
    def __init__(self, root, width, limit, state):
        self.root, self.width, self.limit = root, width, limit
        root.mkdir(parents=True, exist_ok=True)
        self.index = (root / 'samples.idx').open('ab')
        self.shards = [dict(shard) for shard in state.get('ngram_shards', [])]
        self.count = state['saved']
        self.offset = self.shards[-1]['num_bytes'] if self.shards else 0
        self.data = (root / self.shards[-1]['file_name']).open('ab') if self.shards else None


class CaptureJournal:
    """Commit both stores and the *next source row* together, including filtered rows.

    A checkpoint is published only after data and indices are fsynced. Recovery
    discards tails beyond that checkpoint; it never infers progress from output count.
    Caller must hold capture_lock throughout recovery, writing and finalization.
    """
    def __init__(self, root, *, identity, resume, max_shard_bytes, ngram_width=None):
        self.root = Path(root)
        self.path = self.root / 'capture_state.json'
        self.identity = identity
        self.main = self.ngram = None
        self.closed = False
        if resume:
            if not self.path.exists():
                raise ValueError('No capture_state.json: legacy or uninitialized caches cannot resume')
            self.state = json.loads(self.path.read_text())
            if self.state.get('version') != 1 or self.state.get('identity') != identity:
                raise ValueError('Resume settings/source/model/tokenizer differ from the checkpoint')
            if (self.root / 'manifest.json').exists():
                self.complete = True
                return
            self._rollback()
        else:
            self.state = dict(version=1, identity=identity, next_source=0, saved=0,
                              files={}, main_shards=[], ngram_shards=[], metadata={})
        self.complete = False
        self.main = _MainWriter(self.root, max_shard_bytes, self.state)
        try:
            if ngram_width is not None:
                self.ngram = _NgramWriter(self.root / 'features/ngram_embedding',
                                         ngram_width, max_shard_bytes, self.state)
            if not resume:
                self.checkpoint(0, {})
        except BaseException:
            self.close()
            raise

    def _rollback(self):
        # Validate every committed file before making any destructive repair.
        for name, size in self.state['files'].items():
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root.resolve()) or not path.is_file() or path.stat().st_size < size:
                raise RuntimeError(f'Committed cache file missing/truncated: {name}')
        candidates = list(self.root.glob('shard-*.bin')) + [self.root / 'samples.idx']
        side = self.root / 'features/ngram_embedding'
        candidates += list(side.glob('shard-*.bin')) + [side / 'samples.idx']
        for path in candidates:
            if not path.exists():
                continue
            name = str(path.relative_to(self.root))
            if name not in self.state['files']:
                path.unlink()
            elif path.stat().st_size != self.state['files'][name]:
                with path.open('r+b') as handle:
                    handle.truncate(self.state['files'][name])
                    handle.flush()
                    os.fsync(handle.fileno())

    @property
    def saved(self):
        return self.main.num_local_samples

    def write(self, fields, ngram=None):
        if (self.ngram is None) != (ngram is None):
            raise ValueError('Unexpected/missing ngram payload')
        if self.ngram is not None:
            if ngram.shape[0] != fields['input_ids'].numel():
                raise ValueError('Main/ngram sequence length mismatch')
            self.ngram.write(self.saved, ngram)
        self.main.write_sample(sample_id=self.saved, **fields)

    def checkpoint(self, next_source, metadata):
        if next_source < self.state['next_source'] or self.saved > next_source:
            raise ValueError('Invalid source progress')
        if self.ngram is not None and self.ngram.count != self.saved:
            raise RuntimeError('Main/ngram counts differ; checkpoint refused')
        handles = [self.main.current_shard_handle, self.main.index_handle]
        if self.ngram is not None:
            handles += [self.ngram.data, self.ngram.index]
        for handle in handles:
            if handle is not None:
                handle.flush()
                os.fsync(handle.fileno())
        paths = [self.root / 'samples.idx'] + [self.root / n for n in self.main.local_shard_files]
        if self.ngram is not None:
            paths += [self.ngram.root / 'samples.idx'] + [self.ngram.root / s['file_name'] for s in self.ngram.shards]
            sync_directory(self.ngram.root)
            sync_directory(self.ngram.root.parent)
        sync_directory(self.root)
        state = dict(version=1, identity=self.identity, next_source=next_source, saved=self.saved,
                     files={str(p.relative_to(self.root)): p.stat().st_size for p in paths},
                     main_shards=list(self.main.local_shard_files),
                     ngram_shards=[dict(s) for s in self.ngram.shards] if self.ngram else [],
                     metadata=metadata)
        atomic_json_dump(state, str(self.path))
        sync_directory(self.root)
        self.state = state

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.main is not None:
                self.main.close()
        finally:
            if self.ngram is not None:
                self.ngram.close()
