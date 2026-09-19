"""Optional BF16 token-aligned feature sidecar; leaves the v2 target cache intact."""
from pathlib import Path
import os
import struct

import torch

# sample_id, shard_id, sequence_length, byte_offset
RECORD = struct.Struct("<QIIQ")


class NgramCacheWriter:
    def __init__(self, root, *, width, max_shard_bytes):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.width = int(width)
        self.limit = int(max_shard_bytes)
        if self.width <= 0 or self.limit <= 0:
            raise ValueError("Invalid feature width or shard limit")
        self.index = (self.root / 'samples.idx').open('xb')
        self.data = None
        self.shards = []
        self.offset = 0
        self.count = 0

    def _close_shard(self):
        if self.data is not None:
            self.data.flush()
            os.fsync(self.data.fileno())
            self.data.close()
            self.data = None

    def write(self, sample_id, features):
        if sample_id != self.count or features.ndim != 2 or features.shape[1] != self.width:
            raise ValueError("Sidecar sample order or shape mismatch")
        if features.shape[0] <= 0 or not torch.isfinite(features).all():
            raise ValueError("Empty or non-finite ngram embedding")
        tensor = features.detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("ngram embedding overflow on BF16 conversion")
        payload = tensor.view(torch.uint16).numpy().tobytes()
        if self.data is None or (self.offset and self.offset + len(payload) > self.limit):
            self._close_shard()
            name = f'shard-{len(self.shards):05d}.bin'
            self.data = (self.root / name).open('xb')
            self.shards.append({'file_name': name, 'num_bytes': 0})
            self.offset = 0
        self.data.write(payload)
        self.index.write(RECORD.pack(sample_id, len(self.shards) - 1, tensor.shape[0], self.offset))
        self.offset += len(payload)
        self.shards[-1]['num_bytes'] = self.offset
        self.count += 1

    def close(self):
        try:
            self._close_shard()
        finally:
            if not self.index.closed:
                self.index.flush()
                os.fsync(self.index.fileno())
                self.index.close()

    def metadata(self):
        if not self.index.closed:
            raise RuntimeError("Close sidecar before publishing metadata")
        return dict(version=1, path='features/ngram_embedding', dtype='bfloat16',
                    hidden_size=self.width, num_samples=self.count, index_file='samples.idx',
                    index_record_format=RECORD.format, index_record_size=RECORD.size,
                    shards=self.shards, token_alignment='same_position_as_input_ids',
                    feature_stage='raw_ngram_lookup_concat_before_key_value_projection')


class NgramCacheReader:
    """Load with the parent manifest; pass the main sample's sequence length."""
    def __init__(self, cache_dir, manifest):
        metadata = manifest['extra_features']['ngram_embedding']
        if (metadata['version'] != 1 or metadata['dtype'] != 'bfloat16'
                or metadata['index_record_format'] != RECORD.format
                or metadata['index_record_size'] != RECORD.size):
            raise ValueError("Unsupported ngram sidecar format")
        base = Path(cache_dir).resolve()
        self.root = (base / metadata['path']).resolve()
        if not self.root.is_relative_to(base):
            raise ValueError("Invalid sidecar path")
        self.width = int(metadata['hidden_size'])
        self.count = int(metadata['num_samples'])
        if self.width <= 0 or self.count != int(manifest['num_samples']):
            raise ValueError("Sidecar does not match main cache sample count")
        self.index = (self.root / metadata['index_file']).resolve()
        if not self.index.is_relative_to(self.root) or self.index.stat().st_size != self.count * RECORD.size:
            raise ValueError("Invalid sidecar index")
        self.shards = []
        for shard in metadata['shards']:
            path = (self.root / shard['file_name']).resolve()
            if not path.is_relative_to(self.root) or path.stat().st_size != shard['num_bytes']:
                raise ValueError("Invalid sidecar shard")
            self.shards.append(path)

    def read(self, sample_id, *, seq_len):
        if not 0 <= sample_id < self.count:
            raise IndexError(sample_id)
        with self.index.open('rb') as index:
            index.seek(sample_id * RECORD.size)
            record = index.read(RECORD.size)
        stored_id, shard, length, offset = RECORD.unpack(record)
        if stored_id != sample_id or length != seq_len or shard >= len(self.shards):
            raise ValueError("Ngram/main sample alignment mismatch")
        size = length * self.width * 2
        if offset % 2 or offset + size > self.shards[shard].stat().st_size:
            raise ValueError("Ngram feature range outside shard")
        with self.shards[shard].open('rb') as source:
            source.seek(offset)
            payload = source.read(size)
        if len(payload) != size:
            raise ValueError("Truncated ngram shard")
        return torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(length, self.width)
