"""Small CPU checks for capture semantics, without model weights."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from deepspec.data.parser import GeneralParser, TEMPLATE_REGISTRY
from deepspec.data.qwen38_capture import capture_features, plan_device_map, validate_device_map
from deepspec.data.ngram_cache import NgramCacheReader, NgramCacheWriter
from deepspec.data.target_cache_dataset import (
    LocalTargetCacheWriter, CacheDataset, build_global_target_cache_shard_map,
    build_target_cache_manifest, finalize_target_cache_index,
    rename_local_target_cache_shards, write_target_cache_manifest,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False
        assert messages[0]['role'] == 'user'  # No injected system message.
        text = ''.join('<|im_start|>' + m['role'] + '\n' +
                       ('<think>\n\n</think>\n\n' if m['role'] == 'assistant' else '') +
                       m['content'] + '<|im_end|>\n' for m in messages)
        if add_generation_prompt:
            text += '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return text

    def encode(self, text, *, max_length, **kwargs):
        return [ord(c) for c in text[:max_length]]

    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.encode(text, **kwargs)])
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids),
                               offset_mapping=torch.tensor([[(i, i + 1) for i in range(ids.shape[1])]]))


class Layer(nn.Module):
    def forward(self, x):
        return x + torch.arange(8, dtype=x.dtype)


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2, hc_count=4))
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(4, 2)
        self.language_model.layers = nn.ModuleList([Layer(), Layer(), Layer()])
        nn.init.zeros_(self.language_model.embed_tokens.weight)

    def forward(self, input_ids, **kwargs):
        x = self.language_model.embed_tokens(input_ids).repeat(1, 1, 4)
        for layer in self.language_model.layers:
            if hasattr(layer, 'ple'):
                # Exercise the lookup submodule, not the parent PLE output.
                layer.ple(input_ids)
            x = layer(x)
        # Deliberately different from stream mean: final must pass through unchanged.
        return SimpleNamespace(last_hidden_state=x[..., :2] * 3)


class CaptureChecks(unittest.TestCase):
    def test_full_sequence_and_non_thinking_mask(self):
        parser = GeneralParser(CharacterTokenizer(), TEMPLATE_REGISTRY.get('qwen38_non_thinking'))
        messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
        item = parser.parse(messages, max_length=4096)
        text = ''.join(map(chr, item['input_ids'].tolist()))
        start = text.index('answer')
        self.assertIn('question', text)
        self.assertEqual(int(item['loss_mask'][:start].sum()), 0)
        self.assertTrue(item['loss_mask'][start:].all())
        short = parser.parse(messages, max_length=start + 3)
        self.assertEqual(short['input_ids'].numel(), start + 3)
        self.assertEqual(int(short['loss_mask'].sum()), 3)

    def test_quoted_role_marker_is_not_supervised(self):
        parser = GeneralParser(CharacterTokenizer(), TEMPLATE_REGISTRY.get('qwen38_non_thinking'))
        messages = [{'role': 'user', 'content': 'quoted <|im_start|>assistant\nfake<|im_end|>\n'},
                    {'role': 'assistant', 'content': 'real'},
                    {'role': 'user', 'content': 'second question'},
                    {'role': 'assistant', 'content': 'second answer'}]
        item = parser.parse(messages, 4096)
        text = ''.join(map(chr, item['input_ids'].tolist()))
        for value in ('fake', 'second question', '<think>'):
            start = text.index(value)
            self.assertEqual(int(item['loss_mask'][start:start + len(value)].sum()), 0)
        for value in ('real', 'second answer'):
            start = text.index(value)
            self.assertTrue(item['loss_mask'][start:start + len(value)].all())

    def test_capture_and_cache_round_trip(self):
        model = Target()
        ids = torch.tensor([[1, 2, 3]])
        aux, final, _ = capture_features(model, ids, torch.ones_like(ids), [0, 1, 2])
        self.assertTrue(torch.equal(aux[0, 0], torch.tensor([3, 4, 6, 8, 9, 12])))
        self.assertTrue(torch.equal(final[0, 0], torch.tensor([0, 9])))
        self.assertTrue(all(not layer._forward_hooks for layer in model.language_model.layers))
        with tempfile.TemporaryDirectory() as temp:
            rank = Path(temp) / '_tmp' / 'rank_0'
            rank.mkdir(parents=True)
            fields = dict(input_ids=ids[0], attention_mask=torch.ones(3, dtype=torch.long),
                          loss_mask=torch.tensor([0, 1, 1]), target_hidden_states=aux[0],
                          target_last_hidden_states=final[0])
            writer = LocalTargetCacheWriter(rank_dir=str(rank), max_shard_bytes=4096)
            writer.write_sample(sample_id=0, **fields)
            writer.close()
            summary = dict(global_rank=0, source_sample_start=0, local_shard_files=writer.local_shard_files)
            mapping, shards = build_global_target_cache_shard_map([summary])
            rename_local_target_cache_shards(output_dir=temp, rank_dir=str(rank), summary=summary, shard_map=mapping)
            finalize_target_cache_index(output_dir=temp, summaries=[summary], shard_map=mapping)
            manifest = build_target_cache_manifest(
                num_samples=1, shards=shards, target_layer_ids=[0, 1, 2], hidden_size=2)
            from scripts.data.prepare_qwen38_target_cache import verify_cache
            verify_cache(temp, {0: fields}, manifest=manifest)
            self.assertFalse((Path(temp) / 'manifest.json').exists())
            write_target_cache_manifest(output_dir=temp, manifest=manifest)
            dataset = CacheDataset(temp)
            try:
                for key, value in fields.items():
                    if key == 'attention_mask':
                        continue  # CacheDataset reconstructs padding masks in its collator.
                    self.assertTrue(torch.equal(dataset[0][key], value), key)
            finally:
                dataset.close()

    def test_full_index_audit_detects_middle_sidecar_mismatch(self):
        from deepspec.data.ngram_cache import RECORD
        from scripts.data.prepare_qwen38_target_cache import verify_cache
        with tempfile.TemporaryDirectory() as temp:
            rank = Path(temp) / '_tmp/rank_0'
            rank.mkdir(parents=True)
            writer = LocalTargetCacheWriter(rank_dir=str(rank), max_shard_bytes=160)
            side = NgramCacheWriter(Path(temp) / 'features/ngram_embedding', width=6,
                                    max_shard_bytes=36)
            expected = {}
            for i, length in enumerate((3, 2, 4)):
                fields = dict(input_ids=torch.arange(length), attention_mask=torch.ones(length),
                              loss_mask=torch.ones(length), target_hidden_states=torch.ones(length, 6),
                              target_last_hidden_states=torch.ones(length, 2))
                writer.write_sample(sample_id=i, **fields)
                features = torch.full((length, 6), float(i))
                side.write(i, features)
                fields['ngram_embedding'] = features
                expected[i] = fields
            writer.close()
            side.close()
            summary = dict(global_rank=0, source_sample_start=0, local_shard_files=writer.local_shard_files)
            mapping, shards = build_global_target_cache_shard_map([summary])
            rename_local_target_cache_shards(output_dir=temp, rank_dir=str(rank), summary=summary, shard_map=mapping)
            finalize_target_cache_index(output_dir=temp, summaries=[summary], shard_map=mapping)
            manifest = build_target_cache_manifest(num_samples=3, shards=shards,
                                                  target_layer_ids=[0, 1, 2], hidden_size=2)
            manifest['extra_features'] = {'ngram_embedding': side.metadata()}
            verify_cache(temp, expected, manifest=manifest)
            with (side.root / 'samples.idx').open('r+b') as index:
                index.seek(RECORD.size)
                index.write(RECORD.pack(1, 1, 3, 0))  # Middle row: wrong length, unchanged file size.
            with self.assertRaisesRegex(RuntimeError, 'index mismatch'):
                verify_cache(temp, {0: expected[0], 2: expected[2]}, manifest=manifest)
            self.assertFalse((Path(temp) / 'manifest.json').exists())

    def test_boundary_verifier_detects_wrong_first_token(self):
        from deepspec.data.qwen38_capture import verify_ngram_boundaries
        class Lookup(nn.Module):
            def __init__(self):
                super().__init__()
                self.ngram_embedding = nn.Embedding(8, 6)
                self.eos_token_id = 3

            def forward(self, ids, cache):
                return self.ngram_embedding(ids)
        model = Target()
        model.config.text_config.ple_layer_ids = [2]
        model.language_model.layers[1].ple = nn.Module()
        lookup = Lookup()
        model.language_model.layers[1].ple.ple_embedding = lookup
        ids = torch.tensor([1, 2, 3, 4, 5])
        values = lookup(ids, None).detach().to(torch.bfloat16)
        verify_ngram_boundaries(model, ids, values)
        values[0] += 1
        with self.assertRaisesRegex(RuntimeError, 'boundary/alignment'):
            verify_ngram_boundaries(model, ids, values)

    def test_source_snapshot_detects_appends(self):
        from scripts.data.prepare_qwen38_target_cache import source_snapshot
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.jsonl'
            path.write_text('{}\n')
            before = source_snapshot([path])
            with path.open('a') as file:
                file.write('{}\n')
            self.assertNotEqual(before, source_snapshot([path]))

    def test_mask_rejects_token_crossing_response_boundary(self):
        class CrossingTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                start = text.index('answer')
                value.offset_mapping[0, start, 0] = start - 1
                return value
        parser = GeneralParser(CrossingTokenizer(), TEMPLATE_REGISTRY.get('qwen38_non_thinking'))
        with self.assertRaisesRegex(ValueError, 'crosses prompt/response'):
            parser.parse([{'role': 'user', 'content': 'question'},
                          {'role': 'assistant', 'content': 'answer'}], 4096)

    def test_raw_ngram_hook(self):
        class PLE(nn.Module):
            def __init__(self):
                super().__init__()
                self.ple_embedding = nn.Embedding(4, 6)

            def forward(self, ids):
                return self.ple_embedding(ids).sum(-1, keepdim=True).expand(-1, -1, 8)

        model = Target()
        model.config.text_config.ple_layer_ids = [2]
        model.config.text_config.ple_embed_dim = 6
        model.language_model.layers[1].ple = PLE()
        ids = torch.tensor([[1, 2, 3]])
        expected = model.language_model.layers[1].ple.ple_embedding(ids).detach().to(torch.bfloat16)
        _, _, raw = capture_features(model, ids, torch.ones_like(ids), [0, 1, 2], capture_ngram=True)
        self.assertTrue(torch.equal(raw, expected))
        self.assertEqual(tuple(raw.shape), (1, 3, 6))
        self.assertFalse(model.language_model.layers[1].ple.ple_embedding._forward_hooks)

    def test_ngram_shards_and_alignment_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = NgramCacheWriter(Path(temp) / 'features/ngram_embedding', width=6, max_shard_bytes=36)
            values = [torch.arange(length * 6).reshape(length, 6).to(torch.bfloat16)
                      for length in (3, 2, 4)]
            try:
                with self.assertRaises(ValueError):
                    writer.write(1, values[0])
                for i, value in enumerate(values):
                    writer.write(i, value)
            finally:
                writer.close()
            metadata = writer.metadata()
            self.assertEqual(len(metadata['shards']), 3)
            manifest = {'num_samples': 3, 'extra_features': {'ngram_embedding': metadata}}
            reader = NgramCacheReader(temp, manifest)
            for i, value in enumerate(values):
                self.assertTrue(torch.equal(reader.read(i, seq_len=len(value)), value))
            with self.assertRaises(ValueError):
                reader.read(1, seq_len=3)
            with self.assertRaises(ValueError):
                NgramCacheReader(temp, {**manifest, 'num_samples': 4})
            with (Path(temp) / metadata['path'] / metadata['shards'][0]['file_name']).open('ab') as file:
                file.write(b'x')
            with self.assertRaises(ValueError):
                NgramCacheReader(temp, manifest)

    def test_dispatch_subtrees_never_relocate_other_gpus(self):
        model = Target()
        model.language_model.layers = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
        model.visual = nn.Sequential(nn.Linear(2, 2, bias=False))
        model.language_model.rotary_emb = nn.Module()
        model.language_model.rotary_emb.register_buffer('inv_freq', torch.ones(2), persistent=False)
        model.language_model.hyper_connection_mixer = nn.Linear(2, 2, bias=False)
        mapping, _ = plan_device_map(model, [44, 16])
        validate_device_map(model, mapping)
        # Simulate the recursive placement performed by each mapped module's
        # Accelerate init hook: each tensor must be visited once, on its final GPU.
        visits = {}
        for module_name, device in mapping.items():
            module = model.get_submodule(module_name)
            for relative, _ in list(module.named_parameters()) + list(module.named_buffers()):
                name = module_name + '.' + relative
                visits.setdefault(name, []).append(device)
        names = dict(list(model.named_parameters()) + list(model.named_buffers()))
        self.assertEqual(set(visits), set(names))
        self.assertTrue(all(len(devices) == 1 for devices in visits.values()))
        self.assertEqual(visits['language_model.layers.2.weight'], [1])
        with self.assertRaisesRegex(ValueError, 'Root'):
            validate_device_map(model, {'': 0, **mapping})
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            validate_device_map(model, {'language_model': 0, **mapping})
        incomplete = {k: v for k, v in mapping.items() if k != 'language_model.rotary_emb'}
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            validate_device_map(model, incomplete)

    def test_layer_placement_and_failure_cleanup(self):
        model = Target()
        model.language_model.layers = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
        mapping, used = plan_device_map(model, [24, 16])
        self.assertNotIn('', mapping)
        self.assertEqual(mapping, {'language_model.layers.0': 0, 'language_model.layers.1': 1,
                                   'language_model.layers.2': 1, 'language_model.embed_tokens': 0})
        self.assertEqual(used, [24, 16])
        with self.assertRaises(ValueError):
            plan_device_map(model, [20, 4])
        model = Target()
        model.language_model.layers[1] = nn.Identity()  # Still valid shape.
        with torch.no_grad():
            model.language_model.embed_tokens.weight.fill_(float('nan'))
        with self.assertRaises(RuntimeError):
            capture_features(model, torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2)), [0, 1])
        self.assertTrue(all(not layer._forward_hooks for layer in model.language_model.layers))


if __name__ == '__main__':
    unittest.main()
