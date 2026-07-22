#!/usr/bin/env python
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from prismatic.models.action_heads import ActionHeadRecurrent, RecurrentConfigInternal
from prismatic.utils.rdvla_profiler import RDVLAProfiler, rdvla_range


TIMING_FIELDS = {
    "recurrent_loop_total_ms": "RDVLA/action_head/recurrent_loop_total",
    "recurrent_one_iteration_ms": "RDVLA/action_head/recurrent_one_iteration",
    "get_output_each_iter_ms": "RDVLA/action_head/get_output_each_iter",
    "final_get_output_ms": "RDVLA/action_head/final_get_output",
}


def parse_batch_sizes(value: str) -> List[int]:
    batch_sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not batch_sizes:
        raise argparse.ArgumentTypeError("At least one batch size is required")
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise argparse.ArgumentTypeError("Batch sizes must be positive")
    return batch_sizes


def find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern} in {path}, found {len(matches)}")
    return matches[0]


def load_state_dict(path: Path):
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")

    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_action_head(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> ActionHeadRecurrent:
    if not checkpoint.is_dir():
        raise ValueError(f"Expected local checkpoint directory, got: {checkpoint}")

    config_path = find_one(checkpoint, "action_head_config--*.json")
    with config_path.open("r", encoding="utf-8") as f:
        saved_cfg = json.load(f)

    saved_cfg.pop("_type", None)
    for key in ("prelude_vlm_layers", "recurrent_vlm_layers", "coda_vlm_layers"):
        if key in saved_cfg:
            saved_cfg[key] = tuple(saved_cfg[key])

    head_cfg = RecurrentConfigInternal(**saved_cfg)
    action_head = ActionHeadRecurrent(hidden_dim=head_cfg.hidden_dim, cfg=head_cfg)

    checkpoint_path = find_one(checkpoint, "action_head--*checkpoint*.pt")
    action_head.load_state_dict(load_state_dict(checkpoint_path))
    action_head = action_head.to(device=device, dtype=dtype)
    action_head.eval()
    return action_head


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")
    return device


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def max_vlm_layer_index(model) -> int:
    layers = []
    layers.extend(getattr(model, "prelude_vlm_layers", []))
    layers.extend(getattr(model, "recurrent_vlm_layers", []))
    layers.extend(getattr(model, "coda_vlm_layers", []))
    return max(layers) if layers else 0


def synthesize_inputs(
    model,
    batch_size: int,
    num_task_tokens: int,
    action_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
):
    cfg = model.cfg
    num_vlm_layers = max_vlm_layer_index(model) + 1
    hidden_dim = cfg.hidden_dim
    h_a = torch.randn(batch_size, num_vlm_layers, action_tokens, hidden_dim, device=device, dtype=dtype)
    h_t = torch.randn(batch_size, num_vlm_layers, num_task_tokens, hidden_dim, device=device, dtype=dtype)
    proprio_features = torch.randn(batch_size, 1, hidden_dim, device=device, dtype=dtype)
    return h_a, h_t, proprio_features


def tensor_mb(*tensors) -> float:
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return total_bytes / (1024.0 ** 2)


def run_recurrent_core_once(model, h_a, h_t, proprio_features, num_iter: int, include_get_output_each_iter: bool):
    B = h_a.size(0)
    device, dtype = h_a.device, h_a.dtype
    shape_info = {}

    with torch.inference_mode():
        with rdvla_range("RDVLA/action_head/action_queries"):
            x = model.action_queries.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)

        with rdvla_range("RDVLA/action_head/prelude_total"):
            if model.prelude_vlm_layers:
                for i, layer in enumerate(model.prelude):
                    with rdvla_range(f"RDVLA/action_head/prelude_layer_{i}"):
                        x = layer(
                            x,
                            h_a[:, model.prelude_vlm_layers[i]],
                            h_t[:, model.prelude_vlm_layers[i]],
                            proprio_features,
                        )
        prelude_out = x

        state = model.init_state(B, device, dtype)
        shape_info.update({
            "h_a": list(h_a.shape),
            "h_t": list(h_t.shape),
            "proprio_features": list(proprio_features.shape),
            "prelude_out": list(prelude_out.shape),
            "state": list(state.shape),
        })

        with rdvla_range("RDVLA/action_head/recurrent_loop_total"):
            for _ in range(num_iter):
                state = model._run_one_iteration(state, prelude_out, h_a, h_t, proprio_features)
                if include_get_output_each_iter:
                    with rdvla_range("RDVLA/action_head/get_output_each_iter"):
                        model._get_output(state, h_a, h_t, proprio_features)

        with rdvla_range("RDVLA/action_head/final_get_output"):
            final_output = model._get_output(state, h_a, h_t, proprio_features)

    return final_output, shape_info


def timed_core_run(model, h_a, h_t, proprio_features, num_iter: int, include_get_output_each_iter: bool):
    RDVLAProfiler.set_timing_enabled(True)
    RDVLAProfiler.start_timing_record()

    if h_a.device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output, shape_info = run_recurrent_core_once(
            model, h_a, h_t, proprio_features, num_iter, include_get_output_each_iter
        )
        end_event.record()
        timing_record = RDVLAProfiler.finish_timing_record()
        total_ms = float(start_event.elapsed_time(end_event))
    else:
        start = time.perf_counter()
        output, shape_info = run_recurrent_core_once(
            model, h_a, h_t, proprio_features, num_iter, include_get_output_each_iter
        )
        timing_record = RDVLAProfiler.finish_timing_record()
        total_ms = (time.perf_counter() - start) * 1000.0

    RDVLAProfiler.set_timing_enabled(False)
    timings = timing_record.get("timings_ms", {}) if timing_record else {}
    row = {"total_ms": total_ms}
    for field, range_name in TIMING_FIELDS.items():
        row[field] = float(timings.get(range_name, 0.0))
    row["output_shape"] = list(output.shape)
    row["shape_info"] = shape_info
    return row


def mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def summarize_runs(runs: List[Dict[str, float]]) -> Dict[str, float]:
    fields = ["total_ms", *TIMING_FIELDS.keys(), "cuda_peak_allocated_mb", "input_tensor_mb"]
    summary = {}
    for field in fields:
        values = [run[field] for run in runs if field in run and run[field] is not None]
        summary[field] = mean(values)
    summary["shape_info"] = runs[-1].get("shape_info") if runs else {}
    summary["output_shape"] = runs[-1].get("output_shape") if runs else None
    return summary


def benchmark_batch_size(
    model,
    batch_size: int,
    num_iter: int,
    num_task_tokens: int,
    action_tokens: int,
    warmup: int,
    repeat: int,
    device: torch.device,
    dtype: torch.dtype,
    include_get_output_each_iter: bool,
):
    h_a, h_t, proprio_features = synthesize_inputs(
        model, batch_size, num_task_tokens, action_tokens, device, dtype
    )
    input_tensor_mb = tensor_mb(h_a, h_t, proprio_features)

    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(warmup):
        timed_core_run(model, h_a, h_t, proprio_features, num_iter, include_get_output_each_iter)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    runs = []
    for _ in range(repeat):
        row = timed_core_run(model, h_a, h_t, proprio_features, num_iter, include_get_output_each_iter)
        row["input_tensor_mb"] = input_tensor_mb
        if device.type == "cuda":
            row["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        else:
            row["cuda_peak_allocated_mb"] = None
        runs.append(row)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return summarize_runs(runs)


def print_shape_info(results):
    print("Representative synthetic shapes:")
    first = next(iter(results.values()))
    for name, shape in first.get("shape_info", {}).items():
        print(f"- {name}: {shape}")
    print(f"- output: {first.get('output_shape')}")
    print()


def print_results(results):
    base = results.get(1)
    base_per_lane = base["total_ms"] if base else None

    print("| B | total ms | per-lane ms | speedup vs B=1 | recurrent loop ms | one-iter sum ms | get-output-each ms | final-output ms | input MB | CUDA peak MB |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for batch_size, summary in results.items():
        total_ms = summary["total_ms"]
        per_lane_ms = total_ms / batch_size
        speedup = (base_per_lane / per_lane_ms) if base_per_lane else None
        cuda_peak = summary.get("cuda_peak_allocated_mb")
        print(
            f"| {batch_size} | {total_ms:.3f} | {per_lane_ms:.3f} | "
            f"{(speedup if speedup is not None else float('nan')):.3f}x | "
            f"{summary['recurrent_loop_total_ms']:.3f} | "
            f"{summary['recurrent_one_iteration_ms']:.3f} | "
            f"{summary['get_output_each_iter_ms']:.3f} | "
            f"{summary['final_get_output_ms']:.3f} | "
            f"{summary['input_tensor_mb']:.1f} | "
            f"{(cuda_peak if cuda_peak is not None else float('nan')):.1f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark RD-VLA recurrent action-head batching.")
    parser.add_argument("--pretrained_checkpoint", required=True, type=Path)
    parser.add_argument("--batch_sizes", type=parse_batch_sizes, default=parse_batch_sizes("1,2,4"))
    parser.add_argument("--num_iter", type=int, default=None)
    parser.add_argument("--num_task_tokens", type=int, default=512)
    parser.add_argument("--action_tokens", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip_get_output_each_iter", action="store_true")
    parser.add_argument("--json_output", type=Path, default=None)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    action_head = load_action_head(args.pretrained_checkpoint, device, dtype)
    model = action_head.model

    num_iter = args.num_iter if args.num_iter is not None else model.cfg.mean_recurrence
    action_tokens = args.action_tokens if args.action_tokens is not None else model.cfg.action_chunk_len * model.cfg.action_dim
    include_get_output_each_iter = not args.skip_get_output_each_iter

    RDVLAProfiler.set_enabled(False)
    RDVLAProfiler.set_timing_cuda_sync(False)

    print(f"Checkpoint: {args.pretrained_checkpoint}")
    print(f"Device: {device}, dtype: {dtype}, num_iter: {num_iter}, repeat: {args.repeat}, warmup: {args.warmup}")
    print(f"VLM layer slots: {max_vlm_layer_index(model) + 1}, action_tokens: {action_tokens}, task_tokens: {args.num_task_tokens}")
    print()

    results = {}
    for batch_size in args.batch_sizes:
        results[batch_size] = benchmark_batch_size(
            model=model,
            batch_size=batch_size,
            num_iter=num_iter,
            num_task_tokens=args.num_task_tokens,
            action_tokens=action_tokens,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
            dtype=dtype,
            include_get_output_each_iter=include_get_output_each_iter,
        )

    print_shape_info(results)
    print_results(results)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "checkpoint": str(args.pretrained_checkpoint),
            "device": str(device),
            "dtype": str(dtype),
            "num_iter": num_iter,
            "num_task_tokens": args.num_task_tokens,
            "action_tokens": action_tokens,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "results": {str(batch_size): summary for batch_size, summary in results.items()},
        }
        args.json_output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"\nSaved JSON results to {args.json_output}")


if __name__ == "__main__":
    main()
