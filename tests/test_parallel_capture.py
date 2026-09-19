import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from deepspec.data.capture_journal import CaptureJournal
from deepspec.data.qwen38_parallel import partition_ranges, visible_gpu_groups, merge_workers, run_grouped
from deepspec.data.target_cache_dataset import build_target_cache_manifest, atomic_json_dump, CacheDataset
from deepspec.data.ngram_cache import NgramCacheReader
from scripts.data.prepare_qwen38_target_cache import verify_cache


def make_worker(root, rank, start, end, values, ngram=True):
    part = Path(root) / '_workers' / f'worker-{rank:03d}'
    part.mkdir(parents=True, exist_ok=True)
    j = CaptureJournal(part, identity={'source':'fixed', 'source_range':[start,end], 'source_limit':end-start},
                       resume=False, max_shard_bytes=100, ngram_width=6 if ngram else None)
    for value, length in values:
        fields = dict(input_ids=torch.full((length,), value), attention_mask=torch.ones(length),
                      loss_mask=torch.ones(length), target_hidden_states=torch.full((length, 6), float(value)),
                      target_last_hidden_states=torch.full((length, 2), float(value)))
        j.write(fields, torch.full((length, 6), float(value)) if ngram else None)
    j.checkpoint(end-start, {})
    j.close()
    m = build_target_cache_manifest(num_samples=j.saved, target_layer_ids=[0,1,2], hidden_size=2,
        shards=[dict(shard_id=i,file_name=n) for i,n in enumerate(j.main.local_shard_files)],
        extra_fields={'source_range':[start,end], 'num_source_samples_examined':end-start,
                      'num_filtered_samples':end-start-j.saved})
    if ngram:
        m['extra_features']={'ngram_embedding':j.ngram.metadata()}
    atomic_json_dump(m, str(part / 'manifest.json'))
    return part


class ParallelCaptureTests(unittest.TestCase):
    def test_partition_and_gpu_groups(self):
        self.assertEqual(partition_ranges(5,2), [(0,2),(2,5)])
        for count in range(2,20):
            ranges=partition_ranges(count,2)
            self.assertEqual([i for a,b in ranges for i in range(a,b)], list(range(count)))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'2,3,4,5,6,7,8,9'}):
            self.assertEqual(visible_gpu_groups(4), [['2','3','4','5'],['6','7','8','9']])
            with self.assertRaises(ValueError): visible_gpu_groups(3)

    def test_merge_reindexes_both_streams_and_reuses_payloads(self):
        with tempfile.TemporaryDirectory() as root:
            part0=make_worker(root,0,0,3,[(1,3),(2,2)])
            part1=make_worker(root,1,3,6,[(3,4),(4,3)])
            m=merge_workers(root,[(0,3),(3,6)],verify_cache)
            self.assertEqual(m['num_samples'],4)
            self.assertEqual(m['num_filtered_samples'],2)
            data=CacheDataset(root)
            side=NgramCacheReader(root,m)
            try:
                for i,length in enumerate([3,2,4,3]):
                    self.assertTrue((data[i]['input_ids']==i+1).all())
                    self.assertTrue((side.read(i,seq_len=length)==i+1).all())
            finally: data.close()
            self.assertTrue(os.path.samefile(Path(root)/'shard-00000.bin',part0/'shard-00000.bin'))
            before=(Path(root)/'samples.idx').read_bytes()
            merge_workers(root,[(0,3),(3,6)],verify_cache)
            self.assertEqual(before,(Path(root)/'samples.idx').read_bytes())

    def test_empty_worker_and_aux_only(self):
        for ngram in [False,True]:
            with tempfile.TemporaryDirectory() as root:
                make_worker(root,0,0,2,[],ngram)
                make_worker(root,1,2,4,[(7,3)],ngram)
                m=merge_workers(root,[(0,2),(2,4)],verify_cache)
                self.assertEqual((m['num_samples'],m['num_filtered_samples']),(1,3))

    def test_merge_interruption_and_mismatched_partition(self):
        with tempfile.TemporaryDirectory() as root:
            make_worker(root,0,0,2,[(1,3)])
            make_worker(root,1,2,4,[(2,3)])
            from deepspec.data.qwen38_parallel import link_once
            calls=0
            def fail_once(a,b):
                nonlocal calls
                calls+=1
                if calls==3: raise OSError('simulated merge interruption')
                link_once(a,b)
            with patch('deepspec.data.qwen38_parallel.link_once',side_effect=fail_once):
                with self.assertRaises(OSError): merge_workers(root,[(0,2),(2,4)],verify_cache)
            self.assertFalse((Path(root)/'manifest.json').exists())
            merge_workers(root,[(0,2),(2,4)],verify_cache)
            with self.assertRaisesRegex(ValueError,'partition mismatch'):
                merge_workers(root,[(0,1),(1,4)],verify_cache)

    def test_group_launcher_assigns_gpu_sets_and_resumes_completed(self):
        from types import SimpleNamespace
        import sys
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'output'
            args=SimpleNamespace(output_dir=str(root),gpus_per_instance=4,
                                 train_data_path=['dummy'],max_samples=None,resume=False)
            class Dataset:
                def __init__(self,*a): pass
                def __len__(self): return 4
                def close(self): pass
            calls=[]
            class Process:
                def __init__(self,command,**kw):
                    calls.append((command,kw['env']['CUDA_VISIBLE_DEVICES']))
                    start=int(command[command.index('--worker-start')+1])
                    end=int(command[command.index('--worker-end')+1])
                    rank=start//2
                    if '--resume' not in command:
                        make_worker(root,rank,start,end,[(rank+1,3)])
                def poll(self): return 0
                def wait(self,**kw): return 0
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}), \
                 patch('deepspec.data.jsonl_dataset.JsonLineDataset',Dataset), \
                 patch('deepspec.data.qwen38_parallel.subprocess.Popen',Process), \
                 patch.object(sys,'argv',['scripts/data/prepare_qwen38_target_cache.py','--output-dir',str(root)]):
                run_grouped(args,{},verify=verify_cache,snapshot=lambda p:[])
                self.assertEqual([gpu for _,gpu in calls],['0,1,2,3','4,5,6,7'])
                (root/'manifest.json').unlink()  # interrupted final publication: children already done
                args.resume=True
                run_grouped(args,{},verify=verify_cache,snapshot=lambda p:[])
                self.assertTrue(all('--resume' in command for command,_ in calls[2:]))
                calls.clear()
                run_grouped(args,{},verify=verify_cache,snapshot=lambda p:[])
                self.assertEqual(calls,[])


class ProgressReaderTests(unittest.TestCase):
    def test_failed_progress_read_retains_previous_value_and_recovers(self):
        from deepspec.data.qwen38_parallel import ProgressReader
        reader = ProgressReader()
        path = Path('/unused/capture_state.json')
        with patch.object(Path, 'read_text', side_effect=[
                '{"next_source":160,"saved":157}', OSError(5, 'I/O error'),
                '{bad', '{"next_source":320,"saved":315}']), \
             patch('builtins.print') as warning:
            self.assertEqual(reader.read(path), (160,157))
            self.assertEqual(reader.read(path), (160,157))
            self.assertEqual(reader.read(path), (160,157))
            self.assertEqual(reader.read(path), (320,315))
            self.assertEqual(warning.call_count, 1)

    def test_missing_state_is_unknown_and_interrupt_still_propagates(self):
        from deepspec.data.qwen38_parallel import ProgressReader
        reader = ProgressReader()
        path = Path('/unused/capture_state.json')
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            self.assertIsNone(reader.read(path))
        with patch.object(Path, 'read_text', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                reader.read(path)


if __name__=='__main__': unittest.main()
