"""Offline full-prefill TP capture for SGLang commit 1aeeb25e8.

Rank 0 runs in the cache writer process: large CPU features never go through
JSON, multiprocessing queues or /dev/shm. Other TP ranks receive token IDs only.
This adapter intentionally fails on unsupported SGLang versions.
"""
from array import array
import multiprocessing as mp
import os
import signal
import socket
import sys
import time
import traceback

import torch

PINNED_COMMIT = '1aeeb25e8'


def backend_identity(args):
    import sglang
    version = sglang.__version__
    if PINNED_COMMIT not in version:
        raise RuntimeError(f'TP capture supports SGLang commit {PINNED_COMMIT}; installed: {version}')
    size = torch.cuda.device_count()
    if size < 1:
        raise RuntimeError('TP capture requires visible CUDA GPUs')
    return dict(backend='sglang', sglang_version=version, tp_size=size,
                adapter_protocol=1, mem_fraction_static=args.sglang_mem_fraction,
                linear_attn_backend=args.sglang_linear_backend,
                mamba_ssm_dtype=args.sglang_mamba_dtype,
                ple_offload_embedding=args.sglang_ple_offload)


def unpad_rows(input_ids, attention_mask):
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError('Expected matching [batch, sequence] IDs and attention mask')
    lengths = attention_mask.sum(1).tolist()
    expected = torch.arange(input_ids.shape[1])[None, :] < torch.tensor(lengths)[:, None]
    if not torch.equal(attention_mask.cpu(), expected.to(attention_mask.dtype)) or not all(lengths):
        raise ValueError('TP capture requires nonempty, right-padded sequences')
    return [row[:n].tolist() for row, n in zip(input_ids.cpu(), lengths)]


def pad_features(packed, lengths, width):
    if packed.ndim != 2 or packed.shape[0] != sum(lengths):
        raise RuntimeError(f'Wrong packed feature shape: {packed.shape}, lengths={lengths}')
    result = torch.zeros((len(lengths), width, packed.shape[-1]), dtype=torch.bfloat16)
    offset = 0
    for i, n in enumerate(lengths):
        result[i, :n].copy_(packed[offset:offset+n])
        offset += n
    return result


class FeatureCollector:
    """Observe complete decoder residuals and the raw PLE projection input."""
    def __init__(self, model, layer_ids, capture_ngram, hidden=2560, streams=4):
        self.hidden, self.streams = hidden, streams
        self.layer_ids = list(layer_ids)
        self.capture_ngram = capture_ngram
        self.handles = []
        self.reset()
        for i in layer_ids:
            self.handles.append(model.model.layers[i].register_forward_hook(self._layer_hook(i)))
        if capture_ngram:
            ple = model.model.layers[1].ple
            if ple is None:
                raise ValueError('Missing PLE at decoder layer 1')
            # Both CPU-prefetched and direct embeddings converge here. Hooking
            # ple_embedding.forward alone silently misses the prefetch path.
            self.handles.append(ple.key_proj.register_forward_pre_hook(self._ngram_hook))

    def reset(self):
        self.aux, self.final, self.ngram = {}, None, None

    def _layer_hook(self, i):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if i in self.aux or value.ndim != 2 or value.shape[-1] != self.hidden * self.streams:
                raise RuntimeError(f'Unexpected post-layer HC feature at layer {i}: {value.shape}')
            self.aux[i] = value.detach().unflatten(-1, (self.streams, self.hidden)).float().mean(-2).to(torch.bfloat16)
        return hook

    def _ngram_hook(self, module, inputs):
        value = inputs[0]
        if self.ngram is not None or value.ndim != 2 or value.shape[-1] != self.hidden:
            raise RuntimeError(f'Unexpected raw ngram feature: {value.shape}')
        self.ngram = value.detach().clone().to(torch.bfloat16)

    def save_final(self, value):
        if self.final is not None or value.ndim != 2 or value.shape[-1] != self.hidden:
            raise RuntimeError(f'Expected post-mixer feature [tokens, {self.hidden}], got {value.shape}')
        self.final = value.detach().clone().to(torch.bfloat16)

    def finish(self, tokens):
        if set(self.aux) != set(self.layer_ids) or self.final is None or (self.capture_ngram and self.ngram is None):
            raise RuntimeError('One or more requested capture hooks did not execute')
        values = [torch.cat([self.aux[i] for i in self.layer_ids], -1), self.final, self.ngram]
        for value in values:
            if value is not None and (value.shape[0] != tokens or not torch.isfinite(value).all().item()):
                raise RuntimeError('Feature token count mismatch or nonfinite values')
        return tuple(value.cpu() if value is not None else None for value in values)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.reset()


def _make_runner(rank, size, port, options):
    import sglang
    if PINNED_COMMIT not in sglang.__version__:
        raise RuntimeError('SGLang version changed since capture identity was checked')
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.runtime_context import publish
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.manual_seed(options['seed'])
    server = ServerArgs(
        model_path=options['model_path'], dtype='bfloat16', tp_size=size, pp_size=1,
        ep_size=1, dp_size=1, random_seed=options['seed'], skip_tokenizer_init=True,
        context_length=options['max_length'], startup_weight_load_mode='serial',
        mem_fraction_static=options['mem_fraction_static'],
        max_running_requests=options['batch_size'],
        max_total_tokens=options['batch_size'] * (options['max_length'] + 256),
        max_prefill_tokens=options['batch_size'] * options['max_length'],
        chunked_prefill_size=-1, disable_radix_cache=True,
        disable_overlap_schedule=True, disable_cuda_graph=True,
        linear_attn_prefill_backend=options['linear_attn_backend'],
        linear_attn_decode_backend=options['linear_attn_backend'],
        mamba_ssm_dtype=options['mamba_ssm_dtype'],
        ple_offload_embedding=options['ple_offload_embedding'],
        dist_timeout=180,
    )
    publish(server, role='scheduler')
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    initialize_bf16_gemm_config()
    ps = ParallelState.trivial(tp_rank=rank, tp_size=size, attn_tp_rank=rank,
                               attn_tp_size=size, gpu_id=rank)
    model_config = ModelConfig.from_server_args(server)
    runner = ModelRunner(model_config=model_config, mem_fraction_static=server.mem_fraction_static,
                         gpu_id=rank, ps=ps, nccl_port=port, server_args=server)
    if type(runner.model).__name__ != 'Qwen4ExpForConditionalGeneration':
        raise ValueError(f'Unexpected SGLang model class: {type(runner.model).__name__}')
    runner.account_preloaded_weights(runner.preloaded_weights_bytes)
    runner.alloc_memory_pool()
    if runner.max_running_requests < options['batch_size'] or runner.max_total_num_tokens < options['batch_size'] * (options['max_length'] + runner.page_size):
        raise RuntimeError('SGLang KV/state pools are too small for this batch size; reduce --local-batch-size')
    runner.init_attention_backends()
    runner.init_cuda_graphs(capture_decode_cuda_graph=False)
    collector = FeatureCollector(runner.model, options['layers'], options['ngram']) if rank == 0 else None

    class CaptureOnly(torch.nn.Module):
        def forward(self, input_ids, hidden_states, lm_head, forward_batch, *args, **kwargs):
            # Already the learned mixer output. The outer Qwen4 forward replaces
            # output.hidden_states with HC; retain our own post-mixer copy.
            if collector is not None:
                collector.save_final(hidden_states)
            return LogitsProcessorOutput(next_token_logits=None)

    runner.model.logits_processor = CaptureOnly()
    return runner, collector


@torch.inference_mode()
def _forward(runner, collector, rows):
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, CaptureHiddenMode
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    pool, allocator = runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    pool.clear()
    allocator.clear()
    cache = RadixCache(CacheInitParams(disable=True, req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator, page_size=runner.page_size))
    reqs = []
    for i, row in enumerate(rows):
        sampling = SamplingParams(temperature=0, max_new_tokens=1)
        sampling.normalize(tokenizer=None)
        req = Req(rid=f'capture-{i}', origin_input_text='', origin_input_ids=array('q', row),
                  sampling_params=sampling, vocab_size=runner.model_config.vocab_size)
        req.init_next_round_input(cache)
        req.set_extend_range(0, len(row))
        reqs.append(req)
    batch = ScheduleBatch.init_new(reqs=reqs, req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator, tree_cache=cache, model_config=runner.model_config,
        enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
    batch.prepare_for_extend()
    # Synchronous pure-prefill subset of resolve_forward_inputs; no relay or
    # sampling payload is needed. Keep packed sequences and per-request offsets.
    batch.input_ids = batch.prefill_input_ids_cpu.to(runner.device, non_blocking=True)
    batch.prefill_input_ids_cpu = None
    forward = ForwardBatch.init_new(batch, runner, capture_hidden_mode=CaptureHiddenMode.NULL,
                                    return_hidden_states_before_norm=False)
    if collector is not None:
        collector.reset()
    runner.forward(forward)
    if collector is not None:
        try:
            return collector.finish(sum(map(len, rows)))
        finally:
            collector.reset()
    return None


def _shutdown_rank():
    """All healthy ranks call this before any peer or process group exits."""
    from sglang.srt.distributed import parallel_state

    torch.cuda.synchronize()
    # In the pinned SGLang version GroupCoordinator.destroy drops cpu_group
    # before ca_comm. CustomAllReduceV2.close itself uses a CPU-group barrier,
    # so close the IPC communicators explicitly while their groups are alive.
    seen = set()
    for name in ('_TP', '_PP', '_DCP', '_MOE_EP', '_MOE_TP', '_MOE_DP',
                 '_ATTN_CP', '_ATTN_TP', '_PDMUX_PREFILL_TP_GROUP', '_WORLD'):
        group = getattr(parallel_state, name, None)
        comm = getattr(group, 'ca_comm', None)
        if comm is not None:
            if id(comm) not in seen:
                seen.add(id(comm))
                comm.close()
            group.ca_comm = None
    parallel_state.destroy_model_parallel()
    parallel_state.destroy_distributed_environment()


def _rank_worker(connection, rank, size, port, options, parent_pid):
    # Grouped capture terminates its worker on sibling failure. Also terminate
    # this worker's TP children, rather than leaving NCCL/GPU allocations alive.
    if sys.platform == 'linux':
        import ctypes
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
            raise RuntimeError('Cannot install TP parent-death signal')
        if os.getppid() != parent_pid:
            return
    try:
        runner, collector = _make_runner(rank, size, port, options)
        connection.send(('ready', None))
        while True:
            rows = connection.recv()
            if rows is None:
                _shutdown_rank()
                connection.send(('closed', None))
                break
            _forward(runner, collector, rows)
            connection.send(('done', None))
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class SGLangTPCapture:
    def __init__(self, args, config, identity):
        self.processes, self.connections = [], []
        self._healthy, self._closed = False, False
        self.runner = self.collector = None
        self.checked_boundaries = False
        self.size = identity['tp_size']
        self.options = dict(identity, model_path=config['target_model_name_or_path'],
            max_length=config['max_length'], batch_size=args.local_batch_size,
            layers=config['target_layer_ids'], ngram=config.get('capture_ngram', True),
            seed=config.get('seed', 42))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        context = mp.get_context('spawn')
        try:
            for rank in range(1, self.size):
                parent, child = context.Pipe()
                process = context.Process(target=_rank_worker,
                    args=(child, rank, self.size, port, self.options, os.getpid()))
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            self.runner, self.collector = _make_runner(0, self.size, port, self.options)
            self._wait('ready')
            self._healthy = True
            print(f'SGLang TP capture ready: TP={self.size}, EP=1, packed full prefill; no LM-head computation', flush=True)
        except BaseException:
            self.close()
            raise

    def _wait(self, status, timeout=1800):
        from multiprocessing.connection import wait
        pending = list(self.connections)
        deadline = time.monotonic() + timeout
        while pending:
            for connection in wait(pending, timeout=0.25):
                try:
                    tag, detail = connection.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError('TP worker exited unexpectedly') from error
                if tag != status:
                    raise RuntimeError(f'TP worker failed: {detail or tag}')
                pending.remove(connection)
            if time.monotonic() > deadline:
                raise TimeoutError(f'Timed out waiting for TP workers: {status}')
            for process, connection in zip(self.processes, self.connections):
                if connection in pending and process.exitcode is not None:
                    raise RuntimeError(f'TP rank died: pid={process.pid}, exit={process.exitcode}')

    def _run(self, rows):
        self._healthy = False
        for connection in self.connections:
            connection.send(rows)
        values = _forward(self.runner, self.collector, rows)
        self._wait('done')
        self._healthy = True
        return values

    def capture(self, input_ids, attention_mask):
        rows = unpad_rows(input_ids, attention_mask)
        values = self._run(rows)
        if self.options['ngram'] and not self.checked_boundaries:
            # Check request-boundary isolation against separate prefill. In
            # particular, request 2 must not inherit request 1's ngram history.
            offset = 0
            for row in rows[:2]:
                alone = self._run([row])[2]
                if not torch.equal(values[2][offset:offset+len(row)], alone):
                    raise RuntimeError('Packed vs standalone raw ngram mismatch')
                offset += len(row)
            self.checked_boundaries = True
            print('Raw ngram packed/standalone request boundaries verified; compare with HF cache before full run', flush=True)
        lengths = list(map(len, rows))
        return tuple(pad_features(v, lengths, input_ids.shape[1]) if v is not None else None for v in values)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._healthy and all(p.is_alive() for p in self.processes):
                # Keep every peer alive until IPC barriers and group destruction
                # have completed on all ranks. No terminate() on normal exit.
                for connection in self.connections:
                    connection.send(None)
                _shutdown_rank()
                self._wait('closed', timeout=180)
                for rank, process in enumerate(self.processes, 1):
                    process.join(timeout=60)
                    if process.is_alive() or process.exitcode != 0:
                        # Every rank already acknowledged completed collective
                        # cleanup above. Interpreter/allocator teardown cannot
                        # invalidate completed captures and must not abort a
                        # sibling capture instance. Reap stragglers in finally.
                        status = ('still alive after 60s' if process.is_alive()
                                  else f'exit code {process.exitcode}')
                        print(f'Warning: TP rank {rank} pid={process.pid}: {status} '
                              'after shutdown acknowledgement; capture completed, '
                              'remaining process cleanup will be forced.',
                              file=sys.stderr, flush=True)
        finally:
            # A failed forward/startup cannot safely enter collective cleanup.
            # Reap remaining workers; let process exit release rank 0 resources.
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            for connection in self.connections:
                connection.close()
            if self.collector is not None:
                self.collector.close()
            self.collector = self.runner = None
