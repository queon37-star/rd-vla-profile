#!/usr/bin/env python
import argparse
import json
import statistics
from pathlib import Path


COMPONENTS = [
    ("total", "total_ms"),
    ("prepare_images", "prepare_images_ms"),
    ("processor_primary", "processor_primary_ms"),
    ("vla_predict_action", "vla_predict_action_ms"),
    ("vision_backbone", "vision_backbone_ms"),
    ("projector", "projector_ms"),
    ("language_model_forward", "language_model_forward_ms"),
    ("extract_hidden_states", "extract_hidden_states_ms"),
    ("action_head_total", "action_head_total_ms"),
    ("init_state", "init_state_ms"),
    ("prelude", "prelude_ms"),
    ("recurrent_loop", "recurrent_loop_ms"),
    ("get_output_each_iter", "get_output_each_iter_ms"),
    ("final_get_output", "final_get_output_ms"),
]


def load_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def format_ms(value):
    return "-" if value is None else f"{value:.3f}"


def format_pct(value):
    return "-" if value is None else f"{value:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Analyze RD-VLA timing summary JSONL/JSON files.")
    parser.add_argument("summary_path", type=Path)
    args = parser.parse_args()

    records = load_records(args.summary_path)
    if not records:
        print(f"No timing records found in {args.summary_path}")
        return

    total_values = [as_float(record.get("total_ms")) for record in records]
    total_values = [value for value in total_values if value is not None]
    total_mean = statistics.mean(total_values) if total_values else None

    print(f"Records: {len(records)}")
    print()
    print("| Component | Count | Mean ms | Median ms | P95 ms | Mean / Total |")
    print("|---|---:|---:|---:|---:|---:|")

    for component, field in COMPONENTS:
        values = [as_float(record.get(field)) for record in records]
        values = [value for value in values if value is not None]
        if values:
            mean_value = statistics.mean(values)
            median_value = statistics.median(values)
            p95_value = percentile(values, 95)
            ratio = (mean_value / total_mean * 100.0) if total_mean else None
        else:
            mean_value = median_value = p95_value = ratio = None

        print(
            f"| {component} | {len(values)} | {format_ms(mean_value)} | "
            f"{format_ms(median_value)} | {format_ms(p95_value)} | {format_pct(ratio)} |"
        )


if __name__ == "__main__":
    main()
