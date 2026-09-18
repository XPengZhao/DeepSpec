import argparse
import hashlib
import fcntl
from collections import Counter, defaultdict
from pathlib import Path
import signal
from threading import Event
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate JSONL conversations through OpenAI-compatible sglang servers."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-address", nargs="+", required=True)
    parser.add_argument("--input-file-path", required=True)
    parser.add_argument("--output-file-path", required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--is-reasoning-model", action="store_true")
    parser.add_argument("--is-gpt-oss", action="store_true")

    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", action="store_true")
    thinking_group.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if not 0.0 <= args.temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")
    if args.top_p is not None and not 0.0 <= args.top_p <= 1.0:
        raise ValueError("top-p must be between 0.0 and 1.0")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("top-k must be greater than 0")
    if args.min_p is not None and not 0.0 <= args.min_p <= 1.0:
        raise ValueError("min-p must be between 0.0 and 1.0")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be greater than 0")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be greater than 0")


def get_random_reasoning_effort():
    return random.choices(["low", "medium", "high"], weights=[4, 4, 2], k=1)[0]


def compute_context_length(conversations):
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    length += len(part["text"].split())
    return length


def build_query_kwargs(args, messages, max_tokens=None):
    query_kwargs = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty

    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if args.enable_thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
    if args.disable_thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if extra_body:
        query_kwargs["extra_body"] = extra_body

    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


def error_sample(sample, message):
    sample["status"] = "error"
    sample["error"] = message
    return sample


def call_sglang(args, server_address, sample, max_tokens=None):
    conversations = sample.get("conversations")
    if not conversations:
        return error_sample(sample, "Missing conversations")
    if conversations[0].get("role") == "assistant":
        return error_sample(sample, "Data starts with an assistant message")

    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")
    regenerated = []

    for message in conversations:
        role = message.get("role")
        if role == "system":
            regenerated.append(message)
            continue
        if role == "assistant":
            continue
        if role != "user":
            return error_sample(sample, f"Invalid message role: {role}")

        regenerated.append(message)
        try:
            response = client.chat.completions.create(
                **build_query_kwargs(args, regenerated, max_tokens=max_tokens)
            )
        except Exception as exc:
            return error_sample(sample, str(exc))

        response_message = {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
        if args.is_reasoning_model:
            response_message["thinking"] = response.choices[0].message.reasoning_content
        regenerated.append(response_message)

    sample["conversations"] = regenerated
    sample["status"] = "success"
    return sample


RESUME_KEY = "_regen"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def prompt_fingerprint(sample):
    # Old outputs preserve source metadata and system/user turns, but replace assistants.
    value = {k: v for k, v in sample.items()
             if k not in ("conversations", "status", "error", RESUME_KEY)}
    value["conversations"] = [m for m in sample.get("conversations", [])
                              if m.get("role") != "assistant"]
    return fingerprint(value)


def source_rows(path):
    occurrences = Counter()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            sample = json.loads(line)
            if not isinstance(sample, dict) or RESUME_KEY in sample:
                raise ValueError(f"Invalid source or reserved {RESUME_KEY} field at line {line_number}")
            digest = fingerprint(sample)
            occurrence = occurrences[digest]
            occurrences[digest] += 1
            key = f"{digest}:{occurrence}"
            yield key, line_number, sample


def read_output_rows(path):
    """Never truncate existing results automatically, including a partial last line."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                sample = json.loads(line)
                if not isinstance(sample, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}; preserve/repair this line before resuming") from exc
            yield sample


def find_completed_samples(input_path, output_path):
    """Match successful rows by source identity, never by completion order or line count."""
    expected = {}
    legacy_candidates = defaultdict(list)
    for key, _, sample in source_rows(input_path):
        prompt = prompt_fingerprint(sample)
        expected[key] = prompt
        legacy_candidates[prompt].append(key)
    completed = set()
    legacy_counts = Counter()
    for sample in read_output_rows(output_path):
        if sample.get("status") != "success":
            continue
        prompt = prompt_fingerprint(sample)
        meta = sample.get(RESUME_KEY)
        if meta is None:
            legacy_counts[prompt] += 1
            continue
        if not isinstance(meta, dict) or meta.get("version") != 1:
            raise ValueError("Unsupported regen resume metadata")
        key = meta.get("source_key")
        if key not in expected or expected[key] != prompt:
            raise ValueError("Output does not match the current input dataset; use the original input")
        if key in completed:
            raise ValueError(f"Duplicate successful source key in output: {key}")
        completed.add(key)
    for prompt, count in legacy_counts.items():
        candidates = legacy_candidates.get(prompt, [])
        if not candidates:
            raise ValueError("Legacy output cannot be matched to this input dataset")
        if len(candidates) != 1 or count != 1 or candidates[0] in completed:
            raise ValueError("Ambiguous/duplicate legacy output: preserved prompts and metadata do not identify a unique source row; reconcile before resuming")
        completed.add(candidates[0])
    return completed, len(expected)


def ensure_append_boundary(path):
    # A valid final JSON object may have no newline. Keep it, but delimit the next row.
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")


def validate_server(args, server_address, probe):
    start_time = time.perf_counter()
    try:
        result = call_sglang(args, server_address, dict(probe), max_tokens=1)
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    elapsed = time.perf_counter() - start_time
    return server_address, result, elapsed


def validate_servers(args):
    probe = {"conversations": [{"role": "user", "content": "Hello"}]}
    server_results = {}
    server_count = len(args.server_address)

    print(f"Validating {server_count} sglang servers in parallel...", flush=True)
    with ThreadPoolExecutor(max_workers=server_count) as executor:
        future_to_server = {
            executor.submit(
                validate_server,
                args,
                server_address,
                probe,
            ): server_address
            for server_address in args.server_address
        }

        for completed_count, future in enumerate(as_completed(future_to_server), 1):
            server_address = future_to_server[future]
            try:
                _, result, elapsed = future.result()
            except Exception as exc:
                result = {"status": "error", "error": str(exc)}
                elapsed = 0.0

            if result.get("status") == "success":
                server_results[server_address] = True
                print(
                    f"[validate {completed_count}/{server_count}] "
                    f"ok server {server_address} elapsed={elapsed:.2f}s",
                    flush=True,
                )
            else:
                server_results[server_address] = False
                print(
                    f"[validate {completed_count}/{server_count}] "
                    f"skip server {server_address} elapsed={elapsed:.2f}s: "
                    f"{result.get('error')}",
                    flush=True,
                )

    valid_servers = [
        server_address
        for server_address in args.server_address
        if server_results[server_address]
    ]
    invalid_servers = [
        server_address
        for server_address in args.server_address
        if not server_results[server_address]
    ]

    print(f"Available servers ({len(valid_servers)}/{server_count}): {valid_servers}")
    print(
        f"Unavailable servers ({len(invalid_servers)}/{server_count}): "
        f"{invalid_servers}"
    )
    if not valid_servers:
        raise RuntimeError("No available sglang server")
    return valid_servers


def write_finished_result(
    future,
    output_handle,
    error_handle,
    stats,
):
    sample = future.result()
    if sample["status"] == "error":
        error_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        error_handle.flush()
        stats["errors"] += 1
        return

    context_length = compute_context_length(sample.get("conversations", []))
    stats["context_sum"] += context_length
    stats["context_min"] = (
        context_length
        if stats["context_min"] is None
        else min(stats["context_min"], context_length)
    )
    stats["context_max"] = max(stats["context_max"], context_length)
    stats["success"] += 1
    output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    output_handle.flush()


def print_config(args):
    print("Configuration:")
    print(f"  model: {args.model}")
    print(f"  servers: {args.server_address}")
    print(f"  input: {args.input_file_path}")
    print(f"  output: {args.output_file_path}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  top_k: {args.top_k}")
    print(f"  min_p: {args.min_p}")
    print(f"  resume: {args.resume}")


def run(args, stop_requested=None):
    stop_requested = stop_requested if stop_requested is not None else Event()
    print_config(args)
    error_path = str(Path(args.output_file_path).with_suffix("")) + "_error.jsonl"
    completed, total_lines = find_completed_samples(
        args.input_file_path, args.output_file_path if args.resume else os.devnull
    )
    if args.resume:
        # Failed attempts are history, not completed samples; retry them automatically.
        for _ in read_output_rows(error_path):
            pass
        print(f"Resume: {len(completed)} successful source rows; "
              f"{total_lines - len(completed)} pending (including errors)")
    if len(completed) == total_lines:
        print(f"All {total_lines} samples succeeded.")
        return

    valid_servers = validate_servers(args)
    print(f"Using servers: {valid_servers}")

    file_mode = "a" if args.resume else "w"
    if args.resume:
        ensure_append_boundary(args.output_file_path)
        ensure_append_boundary(error_path)
    stats = {
        "success": 0,
        "errors": 0,
        "context_sum": 0,
        "context_min": None,
        "context_max": 0,
    }
    queues = {server_address: [] for server_address in valid_servers}
    next_server_index = 0
    submitted_count = 0

    with (
        open(args.output_file_path, file_mode, encoding="utf-8") as output_handle,
        open(error_path, file_mode, encoding="utf-8") as error_handle,
        ThreadPoolExecutor(max_workers=args.concurrency * len(valid_servers)) as executor,
    ):
        pending = total_lines - len(completed)
        progress_total = pending if args.num_samples is None else min(pending, args.num_samples)
        progress = tqdm(total=progress_total, desc="Completed")

        def finish(future):
            write_finished_result(future, output_handle, error_handle, stats)
            progress.update(1)

        try:
            for key, line_number, sample in source_rows(args.input_file_path):
                if stop_requested.is_set():
                    break
                if key in completed:
                    continue
                if args.num_samples is not None and submitted_count >= args.num_samples:
                    break
                server_address = valid_servers[next_server_index]
                next_server_index = (next_server_index + 1) % len(valid_servers)
                while len(queues[server_address]) >= args.concurrency and not stop_requested.is_set():
                    for future in list(queues[server_address]):
                        if future.done():
                            finish(future)
                            queues[server_address].remove(future)
                    if len(queues[server_address]) >= args.concurrency:
                        time.sleep(0.05)
                if stop_requested.is_set():
                    break
                sample[RESUME_KEY] = {"version": 1, "source_key": key, "source_line": line_number}
                future = executor.submit(call_sglang, args, server_address, sample)
                queues[server_address].append(future)
                submitted_count += 1
        except KeyboardInterrupt:
            print("Stopping submission; waiting for in-flight samples and saving their results...", flush=True)
        finally:
            # Ctrl+C/SIGTERM drains submitted work before closing files.
            for server_address in valid_servers:
                for future in queues[server_address]:
                    finish(future)
            progress.close()

    print("Stopped; submitted work saved." if stop_requested.is_set() else "Processing completed.")
    print(f"  success: {stats['success']}")
    print(f"  errors: {stats['errors']}")
    if stats["success"] > 0:
        avg_context = stats["context_sum"] / stats["success"]
        print(f"  context_min: {stats['context_min']}")
        print(f"  context_max: {stats['context_max']}")
        print(f"  context_avg: {avg_context:.2f}")


def main():
    args = parse_args()
    validate_args(args)
    if args.num_samples is not None and args.num_samples < 0:
        raise ValueError("num-samples must be nonnegative")
    output = Path(args.output_file_path).resolve()
    source = Path(args.input_file_path).resolve()
    error = output.with_suffix("").with_name(output.stem + "_error.jsonl")
    if source in (output, error):
        raise ValueError("Input and output/error paths must be different")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Prevent two regen jobs from appending to the same output simultaneously.
    with open(str(output) + ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another regen job owns {output}") from None
        stop_requested = Event()

        def request_stop(*_):
            if not stop_requested.is_set():
                print("Stop requested; draining in-flight samples before exit...", flush=True)
            stop_requested.set()

        previous = {sig: signal.signal(sig, request_stop)
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            run(args, stop_requested)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
