# Qwen3.8-Flash-Next target cache

This entry point captures text-only, non-thinking regen conversations using the
BF16 checkpoint and Transformers 5.17.0+, with an optional SGLang TP backend.
It writes the existing DeepSpec v2
binary cache; it does not implement Qwen3.8 draft training or serving.

## Feature definition

- Input: the complete rendered conversation, right-truncated to 4096 tokens by
  default. Prompt, system messages, and earlier turns count toward this limit.
- Aux layers: zero-based `[45, 46, 47]`, the last three complete decoder outputs.
  Reshape `[B, L, 10240]` to `[B, L, 4, 2560]`, average the four residual streams
  in FP32, cast to BF16, then concatenate the three layers: `[B, L, 7680]`.
- Final feature: the model's `last_hidden_state`, `[B, L, 2560]`, after its learned
  `hyper_connection_mixer`. Do not apply an extra norm or average to this output.
- Raw n-gram lookup output is also saved by default: `[B, L, 2560]` from
  `language_model.layers[1].ple.ple_embedding`. This concatenates 8 bigram and
  8 trigram hash-head embeddings (160 dimensions each), **before** key/value
  projection, hidden-state gating, and convolution. It is not the 10240-dimensional
  output of the entire PLE module.
- Every retained sequence position is stored, including prompt positions.
  `loss_mask` marks assistant content and the end-of-turn suffix, excluding the
  template's empty thinking prefix. It does not crop the saved sequence.
- Rendering uses the model tokenizer's template with `enable_thinking=False` and
  does not inject DeepSpec's generic Qwen system prompt. Explicit source system
  messages are preserved. Use text-only system/user/assistant regen records.
- Samples with fewer than 14 supervised tokens after truncation are filtered.
  Thus full-sequence training can reuse retained records, but the cache is not an
  unfiltered copy of every source record.

The manifest records reduction semantics, tokenizer template hash, model config,
Transformers version, selected layers, source count, and actual device map.
A future draft inference implementation must use the same aux reduction and
layer indices. Existing Qwen3 draft code is not automatically a Qwen3.8 trainer.

## Environment

Install in the environment used for capture (leave an active regen process alone):

```bash
uv pip install 'transformers==5.17.0' accelerate
uv pip check
```

Do not reinstall the repository's older Transformers pin from `requirements.txt`
after this step. CUDA PyTorch must already be installed. The short run checks
whether the installed model kernels work on your server.

## Short run

Stop any serving processes occupying these four GPUs first. From the DeepSpec
repository root:

```bash
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=.:${PYTHONPATH:-} \
python scripts/data/prepare_qwen38_target_cache.py \
  --config config/data/qwen38_target_cache.json \
  --train-data-path train_datasets/qwen38_flash_next/perfectblend_train_regen.jsonl \
  --output-dir /public/workspace/dspark/cache/deepspec/qwen38_flash_next_smoke \
  --max-samples 8 \
  --max-length 512 \
  --local-batch-size 1 \
  2>&1 | tee logs/qwen38_cache_smoke.log
```

`--max-samples` limits source records examined, not the number surviving filtering.
If the first eight records have long prompts and no response within 512 tokens,
use 4096. Successful completion prints `All indices verified; read-back verified` and a final sample
count. The reader checks the first and last saved records against the captured
CPU tensors, including IDs, loss masks, aux, final states, and raw n-gram embeddings.
Every index row is checked for contiguous byte ranges, exact shard sizes, and
matching main/sidecar sample IDs and sequence lengths before publishing the manifest.
This is a structural audit plus sampled payload verification, not a full payload checksum.
`first_sample_preview.json` contains token IDs and decoded supervised/unsupervised
segments from the first retained record; review this before starting a full run.

Then use a **new output directory** to run 8 records at `--max-length 4096` before
launching the full job. Inspect a saved mask with the tokenizer to confirm prompt
and response boundaries. Real-model forward and memory use require server
validation; CPU checks cannot establish their correctness or throughput.

## Full run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=.:${PYTHONPATH:-} \
python scripts/data/prepare_qwen38_target_cache.py \
  --config config/data/qwen38_target_cache.json \
  --train-data-path train_datasets/qwen38_flash_next/perfectblend_train_regen.jsonl \
  --output-dir /public/workspace/dspark/cache/deepspec/qwen38_flash_next_target_cache \
  --local-batch-size 1 \
  2>&1 | tee logs/qwen38_cache_full.log
```

By default one Python process uses all visible GPUs. **Do not use torchrun.** This is
sequential layer placement, not tensor parallel execution: it prioritizes a
simple working capture path, not SGLang-like throughput. Each complete decoder
layer, including its PLE table if present, must fit on one GPU. The planner counts
PLE weights and fails if it cannot place them; it does not silently offload to
CPU or disk. Other modules, including the final mixer and unused vision encoder,
are placed on logical GPU 0.

`--gpu-memory-gib 120` caps weights per GPU; `--reserve-gib 8` additionally reserves
currently free memory for activations. A short forward can still reveal a need
for more headroom. Use `--model-path` to override the local checkpoint directory.
The default checkpoint is `/public/llm_models/Qwen/Qwen3.8-Flash-Next`.

For a new job the output directory must be empty. This version supports `--resume`
using `capture_state.json`; old caches generated without this file cannot resume.
Use a completed, stable regen JSONL as input; do not capture from a growing file.

## Resume

On interruption, rerun the same command and add `--resume`:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=.:${PYTHONPATH:-} \
python scripts/data/prepare_qwen38_target_cache.py \
  --config config/data/qwen38_target_cache.json \
  --train-data-path train_datasets/qwen38_flash_next/perfectblend_train_regen.jsonl \
  --output-dir /public/workspace/dspark/cache/deepspec/qwen38_flash_next_target_cache \
  --local-batch-size 1 \
  --checkpoint-interval 100 \
  --resume \
  2>&1 | tee -a logs/qwen38_cache_full.log
```

- Both the first run and resumed runs save a durable checkpoint every 100 completed
  capture batches by default. Override with `--checkpoint-interval`; it is independent
  of `--log-interval`. At most one checkpoint interval is recomputed after a crash.
- Progress is the next **source record**, including filtered records, not the number
  of saved samples. Multi-file order follows the sorted input paths.
- Data and both indices are flushed and fsynced before atomically publishing the
  checkpoint. Recovery truncates uncommitted tails and removes uncommitted shards.
  Both feature streams restart at the same sample ID. Committed missing/truncated
  files cause an error rather than being silently regenerated.
- Recovery requires the same source snapshot, capture config, model file metadata,
  tokenizer template, library versions, and source limit (`--max-samples`). It does
  not append new source data or extend a completed smoke cache into a full cache.
  GPU placement, logging/checkpoint intervals and batch size may be changed.
- A process lock prevents concurrent writers. The final manifest is still published
  only after validation. If interrupted during finalization, `--resume` retries
  validation without loading the model. For an already completed cache it verifies
  the indices and exits without model loading.
- Writes are synchronous in this capture entry point so both stores share a clear
  commit boundary; disk/fsync overhead should be measured on the server.
- Keep `capture_state.json` and `.capture.lock` in place. No resume metadata is
  required by existing training readers; the finished v2 cache layout is unchanged.

Optional CPU checks (no model download):

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*capture*.py'
```

## Optional n-gram sidecar

`capture_ngram` defaults to `true` in the provided config. Existing commands capture
it automatically. Use `--no-capture-ngram` for an aux-only run in a new directory.

```text
cache/
  manifest.json
  samples.idx                       # original v2 index, unchanged
  shard-00000.bin                    # original aux/final/token/mask data
  features/ngram_embedding/
    samples.idx                     # <QIIQ: sample ID, shard ID, length, offset
    shard-00000.bin                  # contiguous [length, 2560] BF16 values
```

The manifest's `extra_features.ngram_embedding` describes the independent version-1
sidecar: width, dtype, ordered shard names/sizes, index layout, lookup hook path,
ngram orders, and token alignment. Sidecar shards rotate independently of the main
shards; **join by sample ID, never by shard filename or byte offset**. Both writers
receive exactly the same retained samples, positions, and lengths. Manifest
publication requires successful completion of both writers; an interrupted run must be resumed before
being consumed as a completed dataset. This does not backfill earlier aux-only caches.

Existing DeepSpec and SpecForge v2 readers ignore the extra metadata and continue
reading the original features. Enhanced training must explicitly load the sidecar:

```python
from deepspec.data.target_cache_dataset import CacheDataset
from deepspec.data.ngram_cache import NgramCacheReader

cache = CacheDataset('/path/to/cache')
ngram_reader = NgramCacheReader(cache.cache_dir, cache.manifest)
try:
    sample_id = 0
    sample = cache[sample_id]
    ngram = ngram_reader.read(sample_id, seq_len=sample['input_ids'].numel())
    # ngram.shape == [sequence_length, 2560], dtype == torch.bfloat16
finally:
    cache.close()
```

The capture entry point automatically compares the first and last sidecar records
to the captured values, in addition to checking main-cache features. CPU checks
cover raw-hook selection, shard rotation, and rejection of length/count mismatch
and truncated/modified shards. Real-model collection still needs a server smoke run.

This feature adds 5120 bytes per retained token (~25% over the four existing 2560-D
BF16 feature vectors). At 4096 tokens this is 20 MiB per sample. It includes the
current token: feature at position t can condition predictions after t; it must not
supply the true token t to a prediction of t. No token shift is performed in storage.
A draft rollout that recomputes features must use generated predecessor IDs, not
look up ground-truth future rows from this cache.

## Audit safeguards

- Qwen3.8 supervision spans come from structured message prefixes and tokenizer
  offsets; quoted role markers inside user text cannot create supervised spans.
  Prefix changes or tokens crossing a response boundary stop capture for inspection.
- Source file size, modification time, and inode are recorded and checked during
  capture. A changing regen input aborts without publishing a completed manifest.
- Feature shapes and finiteness are checked, including after BF16 conversion.
- On the first saved sample, raw n-gram features at the first two positions and
  after up to two EOS boundaries are compared with fresh short lookups using the
  real embedding module. No extra backbone pass is needed.
- The final manifest is published only after all index checks and first/last
  payload comparisons succeed. Interrupted jobs cannot masquerade as complete caches.

The local tests use small CPU models and a synthetic tokenizer. They do not validate
real Qwen weights, tokenizer offsets, multi-GPU kernels, or future draft consumption.
A real-model 4096-token smoke run remains required before full collection.

## Multi-GPU dispatch

The device map must contain disjoint module subtrees: individual decoder layers,
embedding, rotary module, final mixer, and vision encoder. Do not add a root `""`
entry or a `language_model` parent entry alongside per-layer entries. Accelerate
initializes those mapped parent hooks with recursive placement, which can move
other GPUs' parameters onto GPU 0 before the child hooks are installed and OOM.
The loader validates coverage and prints actual tensor bytes and CUDA allocations
after dispatch. The PLE table still resides on GPU in this implementation.

## Multiple instances, automatic partition and merge

Use the original complete JSONL and a new output directory. No manual splitting
or separate worker commands are needed. Eight visible GPUs with
`--gpus-per-instance 4` start two independent model replicas, each spread across
four GPUs by layer placement (not tensor parallelism).

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=.:${PYTHONPATH:-} \
python scripts/data/prepare_qwen38_target_cache.py \
  --config config/data/qwen38_target_cache.json \
  --train-data-path train_datasets/qwen38_flash_next/perfectblend_train_regen.jsonl \
  --output-dir /public/workspace/dspark/cache/deepspec/qwen38_flash_next_2instances \
  --gpus-per-instance 4 \
  --capture-ngram \
  --max-length 4096 \
  --gpu-memory-gib 110 \
  --local-batch-size 1 \
  --checkpoint-interval 100 \
  2>&1 | tee logs/qwen38_cache_2instances.log
```

For the first grouped smoke test, add `--max-samples 64` and use a separate smoke
output directory. This limits the total input across all workers to 64 records
(32 each for two workers), not 64 per worker. `--local-batch-size` is per instance,
not per GPU. The GPU count must be divisible by `--gpus-per-instance`; omission
or zero preserves the original single-instance behavior.

Each worker receives a disjoint contiguous source range and maintains its own
`capture_state.json` under `_workers/worker-NNN`. Parent output reports committed
progress; detailed loading/forward logs go to `worker-000.log`, `worker-001.log`
inside the output directory. Tail these files to inspect kernel warnings or errors.

Resume the same grouped command with `--resume` and `tee -a`. Keep the source,
capture config, source limit and group topology unchanged. Completed workers
validate and exit without loading weights; incomplete workers continue from their
own checkpoint. A worker failure stops the other workers and leaves checkpoints
for recovery. Changing a previous single-instance job into a grouped job in the
same directory is not supported.

Once all workers complete, the parent validates their caches, assigns global
sample/shard IDs to both main and n-gram indices, and publishes a single root
`manifest.json`. The source ranges cover the original source order; filtering
preserves relative order. An empty worker cache is allowed if its entire range
was filtered, but an entirely empty combined dataset is rejected.

Binary shards are hard-linked into the final layout: no second copy of the hidden
states is created. Main and n-gram shard IDs are remapped independently. Merger
interruption is restartable with the same `--resume` command. Keep `_workers`,
`group_state.json`, and the worker logs for recovery/provenance; training should
read the root cache directory. Copy only the finished root cache files and feature
sidecar when exporting, or preserve hard links when copying the entire directory.

CPU tests cover source partitioning, physical GPU grouping, nonzero source offsets,
empty worker output, merged main/ngram alignment, hard-link reuse, interrupted
merging and grouped resume. Real eight-GPU throughput and memory usage still
require a server smoke run after installing the optimized kernels.

## SGLang TP capture (server commit `1aeeb25e8`)

Use `--backend sglang` for actual tensor parallelism across all GPUs visible to
one capture instance. `--gpus-per-instance 4` on eight visible GPUs starts two
independent TP4 instances and merges their caches. The default backend remains
`transformers`, which places whole layers across GPUs and is the reference path.

This adapter targets `0.5.6.post3.dev10555+g1aeeb25e8` and rejects other commits:
SGLang's internal ModelRunner, request-state, and runtime-context APIs are not
stable. It does not modify the installed SGLang package.

Implementation:

- BF16, TP with EP=1; FlashInfer linear-attention kernels by default.
- Packed full-sequence prefill, without padding computation, prefix reuse,
  generation, or LM-head logits. CUDA graphs and overlapping scheduling are
  disabled so all selected per-token hooks execute on every batch.
- Rank 0 collects post-layer HC means and the post-mixer final feature. SGLang's
  generic returned hidden states are **not** used: Qwen4 replaces them with HC.
- Raw ngram is captured at the input of `ple.key_proj`, after either direct
  lookup or PLE prefetch, before projection/gating/convolution. PLE offload to
  pinned host memory defaults to enabled, like SGLang's BF16 serving path.
- Only token IDs cross the TP control pipes. Rank 0 writes CPU tensors directly
  through the existing transactional main-cache/ngram-sidecar writer.
- Each batch gets fresh requests and reset KV/Mamba/ngram pools. The first batch
  checks raw ngram against standalone prefill for its first two valid samples.
  This checks request isolation; it does not replace the HF numerical comparison.

### Smoke run and numerical comparison

From the DeepSpec repository on the server:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=.:${PYTHONPATH:-} \
python scripts/data/prepare_qwen38_target_cache.py \
  --backend sglang \
  --config config/data/qwen38_target_cache.json \
  --train-data-path /public/workspace/dspark/DeepSpec/train_datasets/qwen38_flash_next/perfectblend_train_regen.jsonl \
  --output-dir /public/workspace/dspark/cache/deepspec/qwen38_flash_next_tp_smoke \
  --capture-ngram --max-samples 32 --max-length 4096 \
  --local-batch-size 4 --checkpoint-interval 2 --log-interval 1 \
  2>&1 | tee logs/qwen38_cache_tp_smoke.log

PYTHONPATH=. python scripts/data/compare_qwen38_caches.py \
  /public/workspace/dspark/cache/deepspec/qwen38_flash_next_smoke \
  /public/workspace/dspark/cache/deepspec/qwen38_flash_next_tp_smoke
```

Use identical source records, tokenizer, truncation, and layer selection for the
HF reference. The comparison requires exact input IDs and masks and reports
max absolute error, RMSE, relative L2, cosine similarity and unequal-element
fraction for each feature. Raw ngram should match exactly for the same BF16 table
and hash semantics; investigate any discrepancy. Aux/final are not expected to
be bit-identical across different attention/MoE/HC kernels; inspect their errors
before approving a full capture. No universal acceptance tolerance is assumed.
If needed, compare `--sglang-mamba-dtype float32` to isolate the SSM cache dtype.

These changes have CPU contract/resume tests, but require a GPU smoke run to
validate the installed kernels and model weights. Do not interpret a successful
read-back check as HF/TP numerical equivalence.

### Full capture: two TP4 instances

After validating the smoke cache:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=.:${PYTHONPATH:-} \
python scripts/data/prepare_qwen38_target_cache.py \
  --backend sglang --gpus-per-instance 4 \
  --config config/data/qwen38_target_cache.json \
  --train-data-path /public/workspace/dspark/DeepSpec/train_datasets/qwen38_flash_next/perfectblend_train_regen.jsonl \
  --output-dir /public/workspace/dspark/cache/deepspec/qwen38_flash_next_tp \
  --capture-ngram --max-length 4096 --local-batch-size 8 \
  --checkpoint-interval 10 --log-interval 10 \
  2>&1 | tee logs/qwen38_cache_tp.log
```

Add `--resume` after interruption, using the same source partition and backend
settings. HF caches cannot be resumed into TP caches. Use a new output directory
when switching backend. The grouped worker logs are `worker-000.log` and
`worker-001.log` under the output directory.

Tune microbatch 4 → 8 → 16 using representative long samples and steady-state
`tokens/s`; first-batch compilation and boundary checks distort short-run rates.
`capture_s` includes prefill, feature transfer, and the first-batch check;
`write_s` includes tensor preparation/writes but excludes durable checkpoint
fsync. End-to-end `tokens/s` includes all capture-loop costs. TP is not a promise
of saturation: MoE/PLE access, CPU transfer and disk writes may become limiting.

`--gpu-memory-gib` and `--reserve-gib` only affect the Transformers backend.
For TP use `--sglang-mem-fraction` (default 0.80). KV capacity is capped for the
configured batch and sequence length. To keep the PLE table on GPUs instead of
CPU, use `--no-sglang-ple-offload` and recheck memory/throughput in a new cache.
