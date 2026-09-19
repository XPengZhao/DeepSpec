"""Qwen3.8 text feature capture; imports Transformers only when loading weights."""

import torch


def plan_device_map(model, budgets):
    """Pack complete decoder layers in order, accounting for the large PLE table.

    Other modules (including the final mixer and optional vision encoder) live
    on GPU 0. Budgets are weight budgets in bytes, with activation headroom
    already removed by the caller. No CPU/disk offloading is implicit.
    """
    prefix = "language_model.layers."
    layer_sizes = [0] * len(model.language_model.layers)
    root_size = 0
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        size = tensor.numel() * (2 if tensor.is_floating_point() else tensor.element_size())
        if name.startswith(prefix):
            layer_sizes[int(name[len(prefix):].split(".", 1)[0])] += size
        else:
            root_size += size
    used = [0] * len(budgets)
    used[0] = root_size
    if root_size > budgets[0]:
        raise ValueError("Non-decoder weights exceed GPU 0's weight budget")
    placement = {}
    device = 0
    for layer_id, size in enumerate(layer_sizes):
        while device < len(budgets) and used[device] + size > budgets[device]:
            device += 1
        if device == len(budgets):
            raise ValueError(
                f"Cannot place complete layer {layer_id} ({size / 2**30:.1f} GiB). "
                "Free more GPU memory, increase the weight budget, or expose more GPUs. "
                "The PLE layer must fit on one GPU; this loader does not tensor-shard it."
            )
        placement[f"{prefix}{layer_id}"] = device
        used[device] += size
    # Never combine a parent map entry with descendants. Accelerate's
    # AlignDevicesHook(place_submodules=True) recursively moves that parent's
    # weights before installing child hooks, temporarily collapsing the model.
    layer_paths = set(placement)
    def assign_other_modules(module, name):
        if name in layer_paths:
            return
        contains_layers = not name or any(p.startswith(name + ".") for p in layer_paths)
        if not contains_layers:
            placement[name] = 0
            return
        if list(module.parameters(recurse=False)) or list(module.buffers(recurse=False)):
            raise ValueError(f"Cannot safely map direct tensors on split container {name!r}")
        for child_name, child in module.named_children():
            assign_other_modules(child, f"{name}.{child_name}" if name else child_name)
    assign_other_modules(model, "")
    validate_device_map(model, placement)
    return placement, used


def validate_device_map(model, placement):
    """Require disjoint module subtrees and exactly one owner for every tensor."""
    if "" in placement:
        raise ValueError("Root device-map entry can recursively move all weights onto one GPU")
    names = set(dict(model.named_modules()))
    for name in placement:
        if name not in names:
            raise ValueError(f"Unknown mapped module: {name}")
        if any(other != name and name.startswith(other + ".") for other in placement):
            raise ValueError(f"Overlapping device-map entries at {name}")
    for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
        owners = [key for key in placement if name.startswith(key + ".")]
        if len(owners) != 1:
            raise ValueError(f"Expected exactly one device-map owner for {name}, got {owners}")


def load_target(model_path, gpu_memory_gib, reserve_gib):
    from accelerate import init_empty_weights
    from packaging.version import Version
    import transformers
    from transformers import AutoConfig, AutoModel

    if Version(transformers.__version__) < Version("5.17.0"):
        raise RuntimeError("Qwen3.8 capture requires Transformers >= 5.17.0")
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type != "qwen4_exp" or getattr(config, "quantization_config", None):
        raise ValueError("Expected the unquantized BF16 qwen4_exp checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUs are required for Qwen3.8 weight loading")
    budgets = []
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        budget = min(int(gpu_memory_gib * 2**30), free - int(reserve_gib * 2**30))
        if budget <= 0:
            raise RuntimeError(f"GPU {i} has insufficient free memory")
        budgets.append(budget)
    with init_empty_weights(include_buffers=True):
        skeleton = AutoModel.from_config(config, dtype=torch.bfloat16)
    placement, used = plan_device_map(skeleton, budgets)
    del skeleton
    print("Device map:", placement, flush=True)
    print("Estimated BF16 weights per GPU (GiB):", [round(x / 2**30, 2) for x in used], flush=True)
    model, info = AutoModel.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16,
        device_map=placement, attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors")):
        raise RuntimeError(f"Incomplete checkpoint load: {info}")
    unexpected = [k for k in info.get("unexpected_keys", [])
                  if not k.startswith(("lm_head.", "mtp.", "model.mtp."))]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:20]}")
    validate_device_map(model, placement)
    misplaced = []
    actual_bytes = [0] * torch.cuda.device_count()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        owner = next(key for key in placement if name.startswith(key + "."))
        expected_device = torch.device("cuda", placement[owner])
        if tensor.device != expected_device:
            misplaced.append(f"{name}: {tensor.device}, expected {expected_device}")
        elif tensor.device.type == "cuda":
            actual_bytes[tensor.device.index] += tensor.numel() * tensor.element_size()
    if misplaced:
        raise RuntimeError(f"Incorrect tensor placement after dispatch: {misplaced[:10]}")
    print("Actual parameter/buffer bytes per GPU (GiB):",
          [round(x / 2**30, 2) for x in actual_bytes], flush=True)
    print("CUDA allocated per GPU (GiB):",
          [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(len(actual_bytes))], flush=True)
    model.requires_grad_(False).eval()
    return model, config, placement, transformers.__version__


def capture_features(model, input_ids, attention_mask, layer_ids, *, capture_ngram=False):
    """Return CPU BF16 (aux, final, optional raw ngram), at every input position."""
    backbone = model.language_model
    config = model.config.text_config
    hidden, streams = int(config.hidden_size), int(config.hc_count)
    if list(layer_ids) != sorted(set(layer_ids)) or not layer_ids:
        raise ValueError("Capture layers must be sorted and distinct")
    if any(i < 0 or i >= len(backbone.layers) for i in layer_ids):
        raise ValueError("Capture layer out of range")
    captures, handles = {}, []
    ngram = None

    def save_ngram(_module, _args, output):
        nonlocal ngram
        expected = (*input_ids.shape, int(config.ple_embed_dim))
        if ngram is not None or tuple(output.shape) != expected or not torch.isfinite(output).all():
            raise RuntimeError("Invalid or repeated raw ngram embedding capture")
        ngram = output.detach().to(device="cpu", dtype=torch.bfloat16)
        if not torch.isfinite(ngram).all():
            raise RuntimeError("Raw ngram features overflow BF16")

    def hook(layer_id):
        def save(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            expected = (*input_ids.shape, streams * hidden)
            if tuple(value.shape) != expected:
                raise RuntimeError(f"Layer {layer_id}: expected {expected}, got {tuple(value.shape)}")
            # Accumulate in FP32, then store BF16. This is an aux convention,
            # deliberately distinct from the learned final hyper-connection mixer.
            value = value.detach().unflatten(-1, (streams, hidden)).float().mean(-2)
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite auxiliary features at layer {layer_id}")
            stored = value.to(device="cpu", dtype=torch.bfloat16)
            if not torch.isfinite(stored).all():
                raise RuntimeError(f"Auxiliary features overflow BF16 at layer {layer_id}")
            captures[layer_id] = stored
        return save

    try:
        if capture_ngram:
            if len(config.ple_layer_ids) != 1:
                raise ValueError("Ngram capture currently requires exactly one PLE layer")
            # ple_layer_ids are one-based; decoder layer indices are zero-based.
            ple = backbone.layers[int(config.ple_layer_ids[0]) - 1].ple
            handles.append(ple.ple_embedding.register_forward_hook(save_ngram))
        for i in layer_ids:
            handles.append(backbone.layers[i].register_forward_hook(hook(i)))
        device = backbone.embed_tokens.weight.device
        with torch.inference_mode():
            result = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                           use_cache=False, output_hidden_states=False, return_dict=True)
            final = result.last_hidden_state
            if tuple(final.shape) != (*input_ids.shape, hidden) or not torch.isfinite(final).all():
                raise RuntimeError("Invalid final target hidden states")
            final = final.detach().to(device="cpu", dtype=torch.bfloat16)
            if not torch.isfinite(final).all():
                raise RuntimeError("Final target features overflow BF16")
        if set(captures) != set(layer_ids):
            raise RuntimeError("Not all selected layers executed")
        if capture_ngram and ngram is None:
            raise RuntimeError("The raw ngram embedding hook did not execute")
        return torch.cat([captures[i] for i in layer_ids], dim=-1), final, ngram
    finally:
        for handle in handles:
            handle.remove()


def verify_ngram_boundaries(model, input_ids, saved_ngram):
    """Cheap first-sample check using the real lookup, without another backbone pass.

    Initial positions and positions following EOS must equal a fresh short lookup;
    this also detects accidentally capturing a hidden-conditioned PLE output.
    """
    config = model.config.text_config
    lookup = model.language_model.layers[int(config.ple_layer_ids[0]) - 1].ple.ple_embedding
    device = lookup.ngram_embedding.weight.device
    ids = input_ids.reshape(-1)
    starts = [0]
    eos_positions = (ids == lookup.eos_token_id).nonzero().flatten().tolist()
    starts.extend(i + 1 for i in eos_positions[:2] if i + 1 < len(ids))
    with torch.inference_mode():
        for start in starts:
            stop = min(start + 2, len(ids))
            short = ids[start:stop].unsqueeze(0).to(device)
            actual = lookup(short, None)[0].to(device="cpu", dtype=torch.bfloat16)
            if not torch.equal(actual, saved_ngram[start:stop]):
                raise RuntimeError(f"Raw ngram boundary/alignment mismatch at token {start}")
    print("Raw ngram first-token, second-token and available EOS boundaries verified", flush=True)
