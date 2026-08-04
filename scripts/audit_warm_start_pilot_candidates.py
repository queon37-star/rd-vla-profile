#!/usr/bin/env python3
"""Audit existing LIBERO result JSONs for paired warm-start pilot eligibility.

Diagnostic only: legacy files are never upgraded into paired evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence


PAIR_FIELDS = ("paired_trial_id", "initial_state_id", "episode_seed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def task_name_map(manifest_path: Path) -> dict[str, int]:
    manifest = load_json(manifest_path)
    if manifest is None or manifest.get("task_suite_name") != "libero_spatial":
        raise ValueError("A valid LIBERO Spatial manifest is required")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("Manifest has no tasks mapping")
    mapping = {}
    for task_id in range(10):
        entry = tasks.get(str(task_id))
        if not isinstance(entry, dict) or not isinstance(entry.get("task_name"), str):
            raise ValueError(f"Manifest is missing task {task_id}")
        mapping[entry["task_name"]] = task_id
    return mapping


def config_bool(text: str, field: str) -> bool | None:
    match = re.search(rf"\b{re.escape(field)}\s*=\s*(True|False)\b", text)
    return None if match is None else match.group(1) == "True"


def config_text(text: str, field: str) -> str | None:
    for pattern in (
        rf"\b{re.escape(field)}\s*=\s*'([^']*)'",
        rf'\b{re.escape(field)}\s*=\s*"([^"]*)"',
        rf"\b{re.escape(field)}\s*=\s*([^,\)\s]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def audit_file(path: Path, names: dict[str, int]) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None or not isinstance(payload.get("tasks"), dict):
        return None

    config = str(payload.get("config", ""))
    use_warm = config_bool(config, "use_warm_start")
    warm_source = config_text(config, "warm_start_source")
    if use_warm is False:
        arm = "cold_candidate"
    elif use_warm is True and warm_source == "midpoint":
        arm = "midpoint_warm_candidate"
    elif use_warm is True:
        arm = "other_warm_source"
    else:
        arm = "unknown"

    task_ids = set()
    episodes = complete = incomplete = duplicates = successes = 0
    identities = set()
    phases = set()
    reasons = []

    for task_name, records in payload["tasks"].items():
        task_id = names.get(task_name)
        if task_id is None:
            reasons.append(f"unknown_task:{task_name}")
            continue
        task_ids.add(task_id)
        if not isinstance(records, list):
            reasons.append(f"task_{task_id}_records_not_list")
            continue
        for record in records:
            episodes += 1
            if not isinstance(record, dict):
                incomplete += 1
                continue
            successes += int(record.get("success") is True)
            if record.get("evaluation_protocol_phase") is not None:
                phases.add(str(record["evaluation_protocol_phase"]))
            values = [record.get(field) for field in PAIR_FIELDS]
            valid = all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            if not valid:
                incomplete += 1
                continue
            complete += 1
            identity = (task_id, *values)
            duplicates += int(identity in identities)
            identities.add(identity)

    protocol = payload.get("evaluation_protocol")
    root_protocol = isinstance(protocol, dict)
    paired_rng = protocol.get("paired_rng") if root_protocol else None
    if episodes == 0:
        pairability = "no_episode_records"
    elif incomplete:
        pairability = "legacy_or_incomplete_unpaired"
        reasons.append("missing_integer_pair_identity")
    elif duplicates:
        pairability = "invalid_duplicate_pair_identity"
        reasons.append("duplicate_pair_identity")
    elif not root_protocol or paired_rng is not True:
        pairability = "identity_present_but_protocol_unverified"
        reasons.append("missing_root_paired_protocol_metadata")
    else:
        pairability = "paired_source_candidate"

    if len(task_ids) != 10:
        reasons.append(f"partial_task_coverage:{len(task_ids)}/10")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "task_ids": sorted(task_ids),
        "episode_count": episodes,
        "success_count": successes,
        "complete_identity_count": complete,
        "incomplete_identity_count": incomplete,
        "duplicate_identity_count": duplicates,
        "protocol_phases": sorted(phases),
        "root_protocol_present": root_protocol,
        "root_paired_rng": paired_rng,
        "likely_arm": arm,
        "pairability": pairability,
        "inferred_config": {
            "use_warm_start": use_warm,
            "warm_start_source": warm_source,
            "use_cached_final_output": config_bool(config, "use_cached_final_output"),
            "use_latent_precheck": config_bool(config, "use_latent_precheck"),
            "recurrence_strategy": config_text(config, "recurrence_strategy"),
        },
        "reasons": sorted(set(reasons)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-state-manifest",
        type=Path,
        default=Path("experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"),
    )
    parser.add_argument(
        "--path-regex",
        default=r"warm|midpoint|adaptive_base|adaptive_cold|phase3|phase5|stage0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    names = task_name_map(args.initial_state_manifest)
    pattern = re.compile(args.path_regex, re.IGNORECASE)
    files = []
    seen = set()
    for root in args.root:
        candidates = [root] if root.is_file() else root.rglob("*.json") if root.is_dir() else []
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen or not pattern.search(str(path)):
                continue
            seen.add(resolved)
            result = audit_file(path, names)
            if result is not None:
                files.append(result)

    files.sort(key=lambda item: item["path"])
    counts = {}
    for item in files:
        counts[item["pairability"]] = counts.get(item["pairability"], 0) + 1
    paired = [item for item in files if item["pairability"] == "paired_source_candidate"]
    cold = [item for item in paired if item["likely_arm"] == "cold_candidate"]
    warm = [item for item in paired if item["likely_arm"] == "midpoint_warm_candidate"]
    payload = {
        "schema_version": 1,
        "role": "diagnostic_only_no_evidence_upgrade",
        "pairability_counts": counts,
        "paired_candidate_summary": {
            "cold_candidate_files": len(cold),
            "midpoint_warm_candidate_files": len(warm),
            "cold_task_union": sorted({task for item in cold for task in item["task_ids"]}),
            "midpoint_warm_task_union": sorted({task for item in warm for task in item["task_ids"]}),
        },
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"Result JSONs audited: {len(files)}")
    print(f"Pairability counts: {counts}")
    print(
        "Paired candidates: "
        f"cold_files={len(cold)}, midpoint_warm_files={len(warm)}"
    )
    print(
        "Task coverage: "
        f"cold={payload['paired_candidate_summary']['cold_task_union']}, "
        f"midpoint_warm={payload['paired_candidate_summary']['midpoint_warm_task_union']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
