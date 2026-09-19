"""Crash/recovery tests without downloading weights or requiring a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from deepspec.data.capture_journal import CaptureJournal, capture_lock
from deepspec.data.target_cache_dataset import build_target_cache_manifest, CacheDataset
from scripts.data.prepare_qwen38_target_cache import verify_cache


def sample(value=1, length=3):
    return dict(input_ids=torch.full((length,), value), attention_mask=torch.ones(length),
                loss_mask=torch.ones(length), target_hidden_states=torch.full((length, 6), float(value)),
                target_last_hidden_states=torch.full((length, 2), float(value)))


def journal(root, resume=False, ngram=True, identity=None):
    return CaptureJournal(root, identity=identity or {'source': 'fixed'}, resume=resume,
                          max_shard_bytes=90, ngram_width=6 if ngram else None)


def manifest(j):
    result = build_target_cache_manifest(num_samples=j.saved,
        shards=[dict(shard_id=i, file_name=n) for i, n in enumerate(j.main.local_shard_files)],
        target_layer_ids=[0, 1, 2], hidden_size=2)
    if j.ngram:
        result['extra_features'] = {'ngram_embedding': j.ngram.metadata()}
    return result


class ResumeTests(unittest.TestCase):
    def test_rollback_partial_pair_and_filtered_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            j = journal(temp)
            j.write(sample(1), torch.ones(3, 6))
            j.checkpoint(3, {'note': 'two source rows were filtered'})
            j.write(sample(2), torch.full((3, 6), 2.))  # Uncommitted main shard and sidecar tail.
            j.ngram.write(2, torch.full((3, 6), 99.))  # Crash between the two writers.
            with self.assertRaisesRegex(RuntimeError, 'counts differ'):
                j.checkpoint(5, {})
            j.close()
            resumed = journal(temp, resume=True)
            self.assertEqual(resumed.state['next_source'], 3)
            self.assertEqual(resumed.saved, 1)
            resumed.write(sample(3), torch.full((3, 6), 3.))
            resumed.checkpoint(4, {})
            resumed.close()
            m = manifest(resumed)
            verify_cache(temp, {}, manifest=m)
            data = CacheDataset(temp, manifest=m)
            try:
                self.assertEqual([data[i]['input_ids'][0].item() for i in range(len(data))], [1, 3])
            finally:
                data.close()

    def test_killed_process_and_zero_sample_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            program = '''
import os, sys, torch
from deepspec.data.capture_journal import CaptureJournal
j=CaptureJournal(sys.argv[1], identity={'source':'fixed'}, resume=False,
                 max_shard_bytes=90, ngram_width=6)
j.checkpoint(7, {})  # All seven source records filtered.
j.ngram.write(0, torch.ones(3,6))
j.ngram.data.flush()
j.ngram.index.flush()
os._exit(19)  # No finally/close handlers.
'''
            result = subprocess.run([sys.executable, '-c', program, temp], env=os.environ.copy())
            self.assertEqual(result.returncode, 19)
            j = journal(temp, resume=True)
            self.assertEqual(j.saved, 0)
            self.assertEqual(j.state['next_source'], 7)
            j.write(sample(4), torch.full((3, 6), 4.))
            j.checkpoint(8, {})
            j.close()
            verify_cache(temp, {}, manifest=manifest(j))

    def test_changed_identity_and_truncated_committed_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            j = journal(temp)
            j.write(sample(), torch.ones(3, 6))
            j.checkpoint(1, {})
            j.close()
            state_before = (Path(temp) / 'capture_state.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'differ'):
                journal(temp, resume=True, identity={'source': 'changed'})
            self.assertEqual((Path(temp) / 'capture_state.json').read_bytes(), state_before)
            shard = Path(temp) / 'shard-00000.bin'
            with shard.open('r+b') as f:
                f.truncate(1)
            with self.assertRaisesRegex(RuntimeError, 'missing/truncated'):
                journal(temp, resume=True)
            self.assertEqual(shard.stat().st_size, 1)

    def test_aux_only_resume_and_finalized_idempotence(self):
        with tempfile.TemporaryDirectory() as temp:
            j = journal(temp, ngram=False)
            j.write(sample())
            j.checkpoint(1, {'ready': True})
            j.close()
            j = journal(temp, resume=True, ngram=False)
            self.assertEqual(j.state['metadata'], {'ready': True})
            j.write(sample(2))
            j.checkpoint(2, {'ready': True})
            j.close()
            m = manifest(j)
            verify_cache(temp, {}, manifest=m)
            (Path(temp) / 'manifest.json').write_text(json.dumps(m))
            completed = journal(temp, resume=True, ngram=False)
            self.assertTrue(completed.complete)
            completed.close()

    def test_exclusive_lock_and_legacy_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            with capture_lock(temp, resume=False):
                with self.assertRaisesRegex(RuntimeError, 'Another capture'):
                    with capture_lock(temp, resume=True):
                        pass
                with self.assertRaisesRegex(ValueError, 'No capture_state'):
                    journal(temp, resume=True)

    def test_checkpoint_publish_failure_replays_uncommitted_sample(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            j = journal(temp)
            j.write(sample(), torch.ones(3, 6))
            with patch('deepspec.data.capture_journal.atomic_json_dump', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    j.checkpoint(1, {})
            j.close()
            j = journal(temp, resume=True)
            self.assertEqual(j.saved, 0)
            self.assertEqual(j.state['next_source'], 0)
            j.close()

    def test_entrypoint_resume_and_finalize_without_reloading_model(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.data.prepare_qwen38_target_cache import run_capture
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            model_dir = base / 'model'
            model_dir.mkdir()
            (model_dir / 'weights.safetensors').write_bytes(b'test')
            source = base / 'source.jsonl'
            source.write_text('{}\n{}\n{}\n')
            output = base / 'cache'
            output.mkdir()
            class Dataset:
                def __init__(self, paths): pass
                def __len__(self): return 3
                def __getitem__(self, i): return i
                def close(self): pass
            class Tokenizer:
                def get_chat_template(self): return 'test-template'
                def decode(self, ids, **kwargs): return str(ids)
            tokenizer = Tokenizer()
            fake_transformers = SimpleNamespace(__version__='5.17.0', AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda path: tokenizer))
            text_config = SimpleNamespace(hidden_size=2560, hc_count=4, num_hidden_layers=48,
                ple_embed_dim=2560, ple_layer_ids=[2], ngram_size=3, heads_per_ngram=8)
            target_config = SimpleNamespace(text_config=text_config,
                to_dict=lambda: {'text_config': vars(text_config)})
            def collate(rows):
                if rows == [1]: return None
                return dict(input_ids=torch.tensor([[rows[0] + 1] * 3]),
                            attention_mask=torch.ones(1, 3, dtype=torch.long),
                            loss_mask=torch.ones(1, 3, dtype=torch.long))
            def capture(model, ids, mask, layers, **kwargs):
                return (torch.ones(1, 3, 7680, dtype=torch.bfloat16),
                        torch.ones(1, 3, 2560, dtype=torch.bfloat16),
                        torch.ones(1, 3, 2560, dtype=torch.bfloat16))
            args = SimpleNamespace(output_dir=str(output), train_data_path=[str(source)], max_samples=None,
                resume=False, max_shard_gib=.001, local_batch_size=1, num_workers=0,
                gpu_memory_gib=120, reserve_gib=8, log_interval=1, checkpoint_interval=1)
            config = dict(target_model_name_or_path=str(model_dir), target_layer_ids=[45,46,47],
                          chat_template='qwen38_non_thinking', max_length=4096, min_loss_tokens=1,
                          capture_ngram=True)
            prefix = 'scripts.data.prepare_qwen38_target_cache.'
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {'transformers': fake_transformers}))
                stack.enter_context(patch(prefix+'JsonLineDataset', Dataset))
                stack.enter_context(patch(prefix+'ConversationCollator', return_value=collate))
                load = stack.enter_context(patch(prefix+'load_target', return_value=(object(), target_config, {}, '5.17.0')))
                stack.enter_context(patch(prefix+'verify_ngram_boundaries'))
                calls = 0
                def interrupt(*a, **kw):
                    nonlocal calls
                    calls += 1
                    if calls == 2: raise RuntimeError('simulated capture crash')
                    return capture(*a, **kw)
                with patch(prefix+'capture_features', side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'simulated capture crash'):
                        run_capture(args, config)
                state = json.loads((output / 'capture_state.json').read_text())
                self.assertEqual((state['next_source'], state['saved']), (2, 1))
                args.resume = True
                with patch(prefix+'capture_features', side_effect=capture), patch(prefix+'verify_cache', side_effect=RuntimeError('finalization crash')):
                    with self.assertRaisesRegex(RuntimeError, 'finalization crash'):
                        run_capture(args, config)
                self.assertFalse((output / 'manifest.json').exists())
                load.reset_mock()
                run_capture(args, config)
                load.assert_not_called()
                final = json.loads((output / 'manifest.json').read_text())
                self.assertEqual(final['num_samples'], 2)
                self.assertEqual(final['num_filtered_samples'], 1)
                run_capture(args, config)  # Completed --resume is idempotent.
                load.assert_not_called()
                # Exercise the actual entry point with a nonzero global source offset.
                for begin, end, saved in [(2, 3, 1), (1, 2, 0)]:
                    worker_output = base / f'worker-{begin}'
                    worker_output.mkdir()
                    args.output_dir = str(worker_output)
                    args.resume = False
                    args.worker_start, args.worker_end = begin, end
                    with patch(prefix+'capture_features', side_effect=capture):
                        run_capture(args, config)
                    worker_manifest = json.loads((worker_output/'manifest.json').read_text())
                    self.assertEqual(worker_manifest['source_range'], [begin, end])
                    self.assertEqual(worker_manifest['num_samples'], saved)
                    if saved:
                        from deepspec.data.target_cache_dataset import CacheDataset
                        worker_data = CacheDataset(str(worker_output))
                        try:
                            self.assertEqual(worker_data[0]['input_ids'][0].item(), 3)
                        finally:
                            worker_data.close()
                    args.resume = True
                    load.reset_mock()
                    run_capture(args, config)
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
