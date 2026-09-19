"""CPU feature-contract tests; GPU TP/kernel parity needs the smoke comparison."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from deepspec.data.qwen38_sglang_capture import (
    FeatureCollector, SGLangTPCapture, backend_identity, unpad_rows, pad_features,
)


class Layer(nn.Module):
    def forward(self, x):
        return x + 1, None


class CaptureTests(unittest.TestCase):
    def test_post_layer_mean_final_mixer_and_prefetched_ngram(self):
        layers = nn.ModuleList([Layer(), Layer()])
        layers[1].ple = nn.Module()
        layers[1].ple.key_proj = nn.Identity()
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        collector = FeatureCollector(model, [0, 1], True, hidden=2, streams=4)
        x = torch.arange(24).reshape(3, 8).float()
        raw = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        a, _ = layers[0](x)
        # Simulate prefetch: no call to an embedding module, just the projection.
        layers[1].ple.key_proj(raw)
        b, _ = layers[1](a)
        final = b[:, :2] * 7  # Deliberately differs from HC stream mean.
        collector.save_final(final)
        raw.zero_()  # Hook must own its snapshot.
        aux, last, ngram = collector.finish(3)
        torch.testing.assert_close(aux[:, :2], a.reshape(3, 4, 2).mean(1).bfloat16())
        torch.testing.assert_close(last, final.bfloat16())
        self.assertEqual(ngram[0].tolist(), [1., 2.])
        with self.assertRaises(RuntimeError):
            collector.finish(4)
        collector.reset()
        with self.assertRaises(RuntimeError):
            collector.finish(3)
        collector.close()
        self.assertFalse(layers[0]._forward_hooks)
        self.assertFalse(layers[1].ple.key_proj._forward_pre_hooks)

    def test_nonfinite_and_wrong_final_rejected(self):
        layers = nn.ModuleList([Layer()])
        c = FeatureCollector(SimpleNamespace(model=SimpleNamespace(layers=layers)), [0], False, hidden=2)
        with self.assertRaises(RuntimeError):
            c.save_final(torch.zeros(2, 8))
        layers[0](torch.zeros(2, 8))
        c.save_final(torch.full((2, 2), float('nan')))
        with self.assertRaises(RuntimeError):
            c.finish(2)
        c.close()

    def test_packed_requests_exclude_padding_and_preserve_positions(self):
        ids = torch.tensor([[7, 8, 0], [9, 0, 0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.assertEqual(unpad_rows(ids, mask), [[7, 8], [9]])
        packed = torch.tensor([[70., 80.], [81., 82.], [90., 91.]])
        padded = pad_features(packed, [2, 1], 3)
        self.assertEqual(padded[1].tolist(), [[90., 91.], [0., 0.], [0., 0.]])
        for bad in (torch.tensor([[1, 0, 1], [1, 0, 0]]), torch.zeros_like(mask)):
            with self.assertRaises(ValueError):
                unpad_rows(ids, bad)

    def test_packed_ngram_boundary_check_detects_history_leak(self):
        engine = SGLangTPCapture.__new__(SGLangTPCapture)
        engine.options = {'ngram': True}
        engine.checked_boundaries = False
        def run(rows):
            values = torch.tensor([t for row in rows for t in row], dtype=torch.bfloat16)[:, None]
            raw = values.clone()
            if len(rows) > 1:
                raw[len(rows[0])] += 1  # Simulate history leaking into second req.
            return values, values, raw
        engine._run = run
        with self.assertRaisesRegex(RuntimeError, 'standalone'):
            engine.capture(torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2, dtype=torch.long))
        self.assertFalse(engine.checked_boundaries)

    def test_wrong_sglang_version_fails_before_loading(self):
        with patch.dict('sys.modules', {'sglang': SimpleNamespace(__version__='0.5.18')}):
            with self.assertRaisesRegex(RuntimeError, '1aeeb25e8'):
                backend_identity(None)

    def test_comparison_uses_aligned_tokens_and_reports_feature_errors(self):
        import json
        import tempfile
        from pathlib import Path
        from deepspec.data.capture_journal import CaptureJournal
        from deepspec.data.target_cache_dataset import build_target_cache_manifest
        from scripts.data.compare_qwen38_caches import compare
        with tempfile.TemporaryDirectory() as temp:
            paths = [str(Path(temp) / x) for x in ('hf', 'tp', 'wrong_ids')]
            for k, path in enumerate(paths):
                Path(path).mkdir()
                j = CaptureJournal(path, identity={'test': True}, resume=False,
                                   max_shard_bytes=100000, ngram_width=2)
                fields = dict(input_ids=torch.tensor([1, 2 + (k == 2)]),
                    attention_mask=torch.ones(2), loss_mask=torch.tensor([0, 1]),
                    target_hidden_states=torch.ones(2, 6),
                    target_last_hidden_states=torch.full((2, 2), 1. + .125 * k))
                j.write(fields, torch.ones(2, 2))
                j.checkpoint(1, {})
                j.close()
                m = build_target_cache_manifest(num_samples=1,
                    shards=[dict(shard_id=i, file_name=n) for i, n in enumerate(j.state['main_shards'])],
                    target_layer_ids=[0, 1, 2], hidden_size=2)
                m['extra_features'] = {'ngram_embedding': j.ngram.metadata()}
                (Path(path) / 'manifest.json').write_text(json.dumps(m))
            result = compare(paths[0], paths[1], 32)['features']
            self.assertEqual(result['ngram_embedding']['different_fraction'], 0.)
            self.assertEqual(result['target_last_hidden_states']['rmse'], .125)
            with self.assertRaisesRegex(ValueError, 'input_ids mismatch'):
                compare(paths[0], paths[2], 32)

    def test_backend_identity_keeps_precision_and_tp_configuration(self):
        args = SimpleNamespace(sglang_mem_fraction=.8, sglang_linear_backend='flashinfer',
            sglang_mamba_dtype='bfloat16', sglang_ple_offload=True)
        with patch.dict('sys.modules', {'sglang': SimpleNamespace(__version__='0.5.6.post3.dev10555+g1aeeb25e8')}), patch('torch.cuda.device_count', return_value=4):
            original = backend_identity(args)
            args.sglang_mamba_dtype = 'float32'
            changed = backend_identity(args)
        self.assertEqual(original['tp_size'], 4)
        self.assertNotEqual(original, changed)


    def test_normal_shutdown_keeps_peers_alive_for_collectives(self):
        from unittest.mock import Mock
        events = []
        child = Mock(exitcode=None)
        child.is_alive.side_effect = lambda: child.exitcode is None
        def join(timeout):
            events.append('join')
            child.exitcode = 0
        child.join.side_effect = join
        connection = Mock()
        connection.send.side_effect = lambda msg: events.append(('send', msg))
        engine = SGLangTPCapture.__new__(SGLangTPCapture)
        engine.processes, engine.connections = [child], [connection]
        engine._healthy, engine._closed = True, False
        engine.runner, engine.collector = object(), None
        engine._wait = lambda status, **kw: events.append(status)
        with patch('deepspec.data.qwen38_sglang_capture._shutdown_rank', side_effect=lambda: events.append('collectives')):
            engine.close()
            engine.close()  # Idempotent, no second collective.
        self.assertEqual(events[:4], [('send', None), 'collectives', 'closed', 'join'])
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        connection.close.assert_called_once()

    def test_failed_forward_shutdown_does_not_enter_collectives(self):
        from unittest.mock import Mock
        child = Mock(exitcode=None)
        child.is_alive.side_effect = lambda: child.exitcode is None
        child.terminate.side_effect = lambda: setattr(child, 'exitcode', -15)
        engine = SGLangTPCapture.__new__(SGLangTPCapture)
        engine.processes, engine.connections = [child], [Mock()]
        engine._healthy, engine._closed = False, False
        engine.runner, engine.collector = object(), None
        with patch('deepspec.data.qwen38_sglang_capture._shutdown_rank') as shutdown:
            engine.close()
            shutdown.assert_not_called()
        child.terminate.assert_called_once()
        engine.connections[0].send.assert_not_called()

    def test_ipc_communicator_closes_before_process_groups(self):
        from unittest.mock import Mock
        from deepspec.data.qwen38_sglang_capture import _shutdown_rank
        events = []
        comm = Mock()
        comm.close.side_effect = lambda: events.append('ipc_close')
        group = SimpleNamespace(ca_comm=comm)
        # Several SGLang group names can refer to the same communicator.
        state = SimpleNamespace(_TP=group, _ATTN_TP=group,
            destroy_model_parallel=lambda: events.append('destroy_model'),
            destroy_distributed_environment=lambda: events.append('destroy_world'))
        with patch.dict('sys.modules', {'sglang.srt.distributed': SimpleNamespace(parallel_state=state)}), patch('torch.cuda.synchronize', side_effect=lambda: events.append('sync')):
            _shutdown_rank()
        self.assertEqual(events, ['sync', 'ipc_close', 'destroy_model', 'destroy_world'])
        comm.close.assert_called_once()
        self.assertIsNone(group.ca_comm)


    def test_acknowledged_shutdown_exit_problems_do_not_fail_capture(self):
        from unittest.mock import Mock
        for exit_status in (None, 1):
            with self.subTest(exit_status=exit_status):
                child = Mock(pid=123, exitcode=None)
                child.is_alive.side_effect = lambda: child.exitcode is None
                def join(timeout):
                    if timeout == 60:
                        child.exitcode = exit_status
                child.join.side_effect = join
                child.terminate.side_effect = lambda: setattr(child, 'exitcode', -15)
                engine = SGLangTPCapture.__new__(SGLangTPCapture)
                engine.processes, engine.connections = [child], [Mock()]
                engine._healthy, engine._closed = True, False
                engine.runner, engine.collector = object(), None
                engine._wait = Mock()
                with patch('deepspec.data.qwen38_sglang_capture._shutdown_rank'), patch('builtins.print') as warning:
                    engine.close()
                engine._wait.assert_called_once_with('closed', timeout=180)
                warning.assert_called_once()
                self.assertFalse(child.is_alive())

    def test_missing_shutdown_acknowledgement_still_fails(self):
        from unittest.mock import Mock
        child = Mock(pid=123, exitcode=None)
        child.is_alive.side_effect = lambda: child.exitcode is None
        child.terminate.side_effect = lambda: setattr(child, 'exitcode', -15)
        engine = SGLangTPCapture.__new__(SGLangTPCapture)
        engine.processes, engine.connections = [child], [Mock()]
        engine._healthy, engine._closed = True, False
        engine.runner, engine.collector = object(), None
        engine._wait = Mock(side_effect=RuntimeError('missing shutdown ack'))
        with patch('deepspec.data.qwen38_sglang_capture._shutdown_rank'):
            with self.assertRaisesRegex(RuntimeError, 'missing shutdown ack'):
                engine.close()
        child.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
