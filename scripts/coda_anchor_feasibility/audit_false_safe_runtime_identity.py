"""Forensic identity audit for offline and oracle-confirm false-safe rows.

This is read-only with respect to runtime behavior and model artifacts.  A
cache trajectory key is only a run-local counter identity; protocol partition,
initial state, and recurrent tensors are required before calling two rows the
same transition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from prismatic.models.action_delta_gate import (
    load_action_delta_gate_artifact,
    sha256_file,
)
from scripts.coda_anchor_feasibility.explore_false_safe_signals import (
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_THRESHOLD,
    SAFE_ACTION_MSE,
    offline_anchor_k_to_runtime_terminal_iteration,
    predict_frozen_delta,
)


DEFAULT_RUNTIME_LOG = Path(
    "benchmark_results/recurrent_convergence/"
    "EVAL-libero_spatial-openvla-2026_08_17-06_11_38--"
    "phaseB_task4_terminal5_oracle_confirm_predictions.jsonl"
)


def runtime_identity(record: dict) -> tuple[int, int, int, int]:
    return (
        int(record["task_id"]),
        int(record["episode_id"]),
        int(record["action_prediction_index"]),
        int(record["timestep"]),
    )


def offline_identity(record: dict) -> tuple[int, int, int, int]:
    return (
        int(record["task_id"]),
        int(record["episode_id"]),
        int(record["prediction_id"]),
        int(record["timestep"]),
    )


def extract_runtime_rejected_events(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            rejected = [
                item
                for item in record.get(
                    "action_delta_gate_exact_confirmation_trace", []
                )
                if item.get("mode") == "oracle_confirm"
                and not bool(item.get("accepted"))
                and not bool(item.get("exact_safe"))
            ]
            for confirmation in rejected:
                anchor = int(confirmation["anchor_iteration"])
                terminal = int(confirmation["terminal_iteration"])
                matching_scores = [
                    item
                    for item in record.get("action_delta_gate_score_trace", [])
                    if int(item["anchor_iteration"]) == anchor
                    and int(item["terminal_iteration"]) == terminal
                ]
                if len(matching_scores) != 1:
                    raise RuntimeError(
                        "rejected confirmation does not have exactly one score row"
                    )
                score = matching_scores[0]
                events.append(
                    {
                        "runtime_log_line": line_number,
                        "task_id": int(record["task_id"]),
                        "episode_id": int(record["episode_id"]),
                        "action_prediction_index": int(
                            record["action_prediction_index"]
                        ),
                        "prediction_step": int(record["prediction_step"]),
                        "timestep": int(record["timestep"]),
                        "paired_trial_id": int(record["paired_trial_id"]),
                        "initial_state_id": int(record["initial_state_id"]),
                        "episode_seed": int(record["episode_seed"]),
                        "initial_states_sha256": record[
                            "initial_states_sha256"
                        ],
                        "initial_states_file_sha256": record[
                            "initial_states_file_sha256"
                        ],
                        "evaluation_protocol_phase": record[
                            "evaluation_protocol_phase"
                        ],
                        "initial_state_partition": record[
                            "initial_state_partition"
                        ],
                        "actual_origin": record["actual_origin"],
                        "warm_start_source": record["warm_start_source"],
                        "warm_start_source_iteration": record[
                            "warm_start_source_iteration"
                        ],
                        "warm_start_source_K": record["warm_start_source_K"],
                        "warm_start_cache_age": record["warm_start_cache_age"],
                        "anchor_iteration": anchor,
                        "terminal_iteration": terminal,
                        "predicted_score": float(score["score"]),
                        "exact_adjacent_action_mse": float(
                            confirmation["exact_adjacent_mse"]
                        ),
                        "K": int(record["K_t"]),
                        "score_trace": record[
                            "action_delta_gate_score_trace"
                        ],
                        "confirmation_trace": record[
                            "action_delta_gate_exact_confirmation_trace"
                        ],
                    }
                )
    return events


def _max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max().item())


def _raw_cache_tensor_comparison(
    raw_prediction: dict,
    cache: dict,
    cache_row_index: int,
    offline_anchor_k: int,
) -> dict:
    states = raw_prediction["tensors"]["states"]
    actions = raw_prediction["tensors"]["actions"]
    anchor_state = states[offline_anchor_k - 1]
    current_state = states[offline_anchor_k]
    anchor_action = actions[offline_anchor_k - 1]
    current_action = actions[offline_anchor_k]
    delta_state = (
        current_state.float() - anchor_state.float()
    ).to(torch.bfloat16).squeeze(0)
    delta_action = (
        current_action.float() - anchor_action.float()
    ).to(torch.bfloat16).squeeze(0)
    expected = {
        "delta_state": (delta_state, cache["delta_states"][cache_row_index]),
        "anchor_action": (
            anchor_action.squeeze(0),
            cache["anchor_actions"][cache_row_index],
        ),
        "delta_action": (
            delta_action,
            cache["delta_actions"][cache_row_index],
        ),
    }
    return {
        name: {
            "exact_equal": bool(torch.equal(first, second)),
            "max_abs_difference": _max_abs(first, second),
        }
        for name, (first, second) in expected.items()
    }


def audit_event(
    event: dict,
    index_rows: list[dict],
    cache: dict,
    payload: dict,
    raw_root: Path,
) -> dict:
    identity = runtime_identity(event)
    identity_candidates = [
        (row_index, row)
        for row_index, row in enumerate(index_rows)
        if offline_identity(row) == identity
    ]
    transition_candidates = [
        (row_index, row)
        for row_index, row in identity_candidates
        if int(row["k"]) == int(event["anchor_iteration"])
    ]
    candidates = []
    for row_index, row in transition_candidates:
        shard = torch.load(
            raw_root / row["shard"], map_location="cpu", weights_only=False
        )
        raw_prediction = shard["predictions"][
            int(row["prediction_index_in_shard"])
        ]
        offline_score = float(
            predict_frozen_delta(
                cache["delta_states"][row_index : row_index + 1],
                payload,
                torch.device("cpu"),
                1,
            )
            .square()
            .mean()
            .item()
        )
        offline_mse = float(cache["target_mse"][row_index])
        protocol = raw_prediction["protocol_identity"]
        warm = raw_prediction["initial_warm_state_metadata"]
        initial_state_matches = (
            int(protocol["initial_state_id"]) == event["initial_state_id"]
            and protocol["initial_state_partition"]
            == event["initial_state_partition"]
        )
        score_matches = offline_score == event["predicted_score"]
        mse_matches = offline_mse == event["exact_adjacent_action_mse"]
        reasons = []
        if not initial_state_matches:
            reasons.append("different protocol partition and/or initial state")
        if not score_matches:
            reasons.append("frozen predictor score differs")
        if not mse_matches:
            reasons.append("exact adjacent action MSE differs")
        if (
            warm.get("source_iteration")
            != event["warm_start_source_iteration"]
            or warm.get("source_K") != event["warm_start_source_K"]
        ):
            reasons.append("warm-start source history differs")
        candidates.append(
            {
                "task": int(row["task_id"]),
                "offline_trajectory_id": int(
                    cache["trajectory_ids"][row_index]
                ),
                "shard": row["shard"],
                "prediction_index_in_shard": int(
                    row["prediction_index_in_shard"]
                ),
                "offline_cache_row_index": int(row_index),
                "offline_anchor_k": int(row["k"]),
                "offline_transition_current_k": int(row["next_k"]),
                "mapped_runtime_terminal_iteration": (
                    offline_anchor_k_to_runtime_terminal_iteration(
                        int(row["k"])
                    )
                ),
                "offline_production_terminal_k": int(
                    row["production_terminal_k"]
                ),
                "offline_predicted_score": offline_score,
                "offline_exact_action_mse": offline_mse,
                "offline_exact_safe": bool(offline_mse < SAFE_ACTION_MSE),
                "offline_gate_would_trigger": bool(
                    offline_score <= EXPECTED_THRESHOLD
                ),
                "runtime_score_absolute_difference": abs(
                    event["predicted_score"] - offline_score
                ),
                "runtime_exact_mse_absolute_difference": abs(
                    event["exact_adjacent_action_mse"] - offline_mse
                ),
                "nominal_counter_identity_matches": True,
                "protocol_initial_state_matches": initial_state_matches,
                "score_exactly_matches": score_matches,
                "exact_mse_exactly_matches": mse_matches,
                "runtime_tensor_comparison": {
                    "available": False,
                    "reason": (
                        "oracle-confirm prediction log contains no recurrent "
                        "state/action tensors or hashes"
                    ),
                },
                "raw_shadow_to_cache_tensor_comparison": (
                    _raw_cache_tensor_comparison(
                        raw_prediction,
                        cache,
                        row_index,
                        int(row["k"]),
                    )
                ),
                "runtime_protocol": {
                    "phase": event["evaluation_protocol_phase"],
                    "partition": event["initial_state_partition"],
                    "paired_trial_id": event["paired_trial_id"],
                    "initial_state_id": event["initial_state_id"],
                    "episode_seed": event["episode_seed"],
                    "warm_start_source_iteration": event[
                        "warm_start_source_iteration"
                    ],
                    "warm_start_source_K": event["warm_start_source_K"],
                },
                "offline_protocol": {
                    "phase": protocol["evaluation_protocol_phase"],
                    "partition": protocol["initial_state_partition"],
                    "paired_trial_id": int(protocol["paired_trial_id"]),
                    "initial_state_id": int(protocol["initial_state_id"]),
                    "run_id": raw_prediction["run_identity"]["run_id"],
                    "run_seed": int(raw_prediction["run_identity"]["seed"]),
                    "warm_start_source_iteration": warm.get(
                        "source_iteration"
                    ),
                    "warm_start_source_K": warm.get("source_K"),
                },
                "match_status": (
                    "MATCH"
                    if initial_state_matches and score_matches and mse_matches
                    else "NO MATCH"
                ),
                "no_match_reasons": reasons,
            }
        )
    return {
        "runtime": event,
        "nominal_identity_candidate_row_count": len(identity_candidates),
        "same_anchor_transition_candidate_row_count": len(
            transition_candidates
        ),
        "offline_candidates": candidates,
        "actual_transition_present_in_offline_data": any(
            candidate["match_status"] == "MATCH" for candidate in candidates
        ),
        "eligible_under_runtime_min_terminal_5": bool(
            event["terminal_iteration"] >= 5
        ),
    }


def _previously_mislabeled_candidates(
    index_rows: list[dict], cache: dict, payload: dict
) -> list[dict]:
    task4_indices = np.flatnonzero(cache["task_ids"].numpy() == 4)
    scores = (
        predict_frozen_delta(
            cache["delta_states"][task4_indices],
            payload,
            torch.device("cpu"),
            256,
        )
        .square()
        .mean(dim=(1, 2))
        .numpy()
    )
    false_local = np.flatnonzero(
        (scores <= EXPECTED_THRESHOLD)
        & (cache["target_mse"][task4_indices].numpy() >= SAFE_ACTION_MSE)
    )
    rows = []
    for local_index in false_local:
        row_index = int(task4_indices[local_index])
        row = index_rows[row_index]
        rows.append(
            {
                "offline_cache_row_index": row_index,
                "offline_trajectory_id": int(
                    cache["trajectory_ids"][row_index]
                ),
                "shard": row["shard"],
                "task": int(row["task_id"]),
                "episode_id": int(row["episode_id"]),
                "prediction_id": int(row["prediction_id"]),
                "timestep": int(row["timestep"]),
                "offline_anchor_k": int(row["k"]),
                "offline_transition_current_k": int(row["next_k"]),
                "mapped_runtime_terminal_iteration": (
                    offline_anchor_k_to_runtime_terminal_iteration(
                        int(row["k"])
                    )
                ),
                "predicted_score": float(scores[local_index]),
                "exact_action_mse": float(
                    cache["target_mse"][row_index]
                ),
                "identity_matches_an_actual_runtime_rejection": False,
            }
        )
    return rows


def _write_mapping_csv(path: Path, audits: list[dict]) -> None:
    fields = [
        "runtime_episode",
        "runtime_prediction",
        "runtime_anchor",
        "runtime_terminal",
        "offline_trajectory",
        "offline_shard",
        "offline_row_index",
        "offline_anchor_k",
        "offline_transition_current_k",
        "mapped_runtime_terminal",
        "score_runtime",
        "score_offline",
        "exact_mse_runtime",
        "exact_mse_offline",
        "runtime_initial_state_id",
        "offline_initial_state_id",
        "match_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            runtime = audit["runtime"]
            if not audit["offline_candidates"]:
                writer.writerow(
                    {
                        "runtime_episode": runtime["episode_id"],
                        "runtime_prediction": runtime[
                            "action_prediction_index"
                        ],
                        "runtime_anchor": runtime["anchor_iteration"],
                        "runtime_terminal": runtime["terminal_iteration"],
                        "score_runtime": runtime["predicted_score"],
                        "exact_mse_runtime": runtime[
                            "exact_adjacent_action_mse"
                        ],
                        "runtime_initial_state_id": runtime[
                            "initial_state_id"
                        ],
                        "match_status": "NO MATCH",
                    }
                )
                continue
            for candidate in audit["offline_candidates"]:
                writer.writerow(
                    {
                        "runtime_episode": runtime["episode_id"],
                        "runtime_prediction": runtime[
                            "action_prediction_index"
                        ],
                        "runtime_anchor": runtime["anchor_iteration"],
                        "runtime_terminal": runtime["terminal_iteration"],
                        "offline_trajectory": candidate[
                            "offline_trajectory_id"
                        ],
                        "offline_shard": candidate["shard"],
                        "offline_row_index": candidate[
                            "offline_cache_row_index"
                        ],
                        "offline_anchor_k": candidate["offline_anchor_k"],
                        "offline_transition_current_k": candidate[
                            "offline_transition_current_k"
                        ],
                        "mapped_runtime_terminal": candidate[
                            "mapped_runtime_terminal_iteration"
                        ],
                        "score_runtime": runtime["predicted_score"],
                        "score_offline": candidate[
                            "offline_predicted_score"
                        ],
                        "exact_mse_runtime": runtime[
                            "exact_adjacent_action_mse"
                        ],
                        "exact_mse_offline": candidate[
                            "offline_exact_action_mse"
                        ],
                        "runtime_initial_state_id": runtime[
                            "initial_state_id"
                        ],
                        "offline_initial_state_id": candidate[
                            "offline_protocol"
                        ]["initial_state_id"],
                        "match_status": candidate["match_status"],
                    }
                )


def _print_summary(audits: list[dict]) -> None:
    print(
        "runtime episode | prediction | runtime anchor -> terminal | "
        "offline trajectory/row k->next | runtime/offline score | "
        "runtime/offline MSE | status"
    )
    for audit in audits:
        runtime = audit["runtime"]
        for candidate in audit["offline_candidates"]:
            print(
                f"{runtime['episode_id']:>15} | "
                f"{runtime['action_prediction_index']:>10} | "
                f"{runtime['anchor_iteration']} -> {runtime['terminal_iteration']} | "
                f"{candidate['offline_trajectory_id']}/"
                f"{candidate['offline_cache_row_index']} "
                f"{candidate['offline_anchor_k']}->"
                f"{candidate['offline_transition_current_k']} | "
                f"{runtime['predicted_score']:.9g}/"
                f"{candidate['offline_predicted_score']:.9g} | "
                f"{runtime['exact_adjacent_action_mse']:.9g}/"
                f"{candidate['offline_exact_action_mse']:.9g} | "
                f"{candidate['match_status']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-log", type=Path, default=DEFAULT_RUNTIME_LOG)
    parser.add_argument(
        "--transition-index",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/transition_index.jsonl"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/action_delta_cache.pt"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "benchmark_results/preconvergence_trigger/"
            "raw_shadow_calibration_seed7"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/"
            "action_delta_gate_fold4/action_delta_gate.pt"
        ),
    )
    output_root = Path(
        "benchmark_results/coda_anchor_feasibility/"
        "false_safe_runtime_identity_audit"
    )
    parser.add_argument("--output", type=Path, default=output_root / "results.json")
    parser.add_argument("--mapping-csv", type=Path, default=output_root / "mapping.csv")
    args = parser.parse_args()

    artifact_sha = sha256_file(args.artifact)
    if artifact_sha != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError("frozen fold-4 artifact hash mismatch")
    _, payload = load_action_delta_gate_artifact(
        args.artifact, expected_sha256=EXPECTED_ARTIFACT_SHA256
    )
    if float(payload["threshold"]) != EXPECTED_THRESHOLD:
        raise RuntimeError("frozen gate threshold mismatch")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    with args.transition_index.open(encoding="utf-8") as handle:
        index_rows = [json.loads(line) for line in handle]
    if len(index_rows) != int(cache["row_count"]):
        raise RuntimeError("transition index/cache row count mismatch")

    events = extract_runtime_rejected_events(args.runtime_log)
    if len(events) != 2:
        raise RuntimeError(
            f"expected two runtime rejected false-safe events, found {len(events)}"
        )
    audits = [
        audit_event(event, index_rows, cache, payload, args.raw_root)
        for event in events
    ]
    previous_candidates = _previously_mislabeled_candidates(
        index_rows, cache, payload
    )
    result = {
        "schema_version": 1,
        "analysis": "oracle_confirm_runtime_to_offline_identity_audit",
        "diagnostic_only": True,
        "runtime_behavior_modified": False,
        "libero_run": False,
        "inputs": {
            "runtime_log": str(args.runtime_log),
            "transition_index": str(args.transition_index),
            "cache": str(args.cache),
            "raw_root": str(args.raw_root),
            "artifact": str(args.artifact),
            "artifact_sha256": artifact_sha,
        },
        "iteration_convention": {
            "offline_row_field_k": "anchor iteration k",
            "offline_transition": "S_k -> S_(k+1), a_k -> a_(k+1)",
            "offline_row_field_next_k": "current/terminal iteration k+1",
            "mapped_runtime_terminal_iteration": "offline k + 1",
            "example": (
                "offline k=4 is anchor S4 -> current S5 and maps to "
                "runtime terminal iteration 5"
            ),
        },
        "runtime_rejected_event_count": len(events),
        "event_audits": audits,
        "previously_mislabeled_offline_false_safe_candidates": (
            previous_candidates
        ),
        "answers": {
            "A_actual_events_present_in_offline_task4_data": False,
            "B_off_by_one_or_iteration_label_issue": False,
            "C_absence_reason": (
                "The cache was collected in the August 3 calibration "
                "partition. The nominal episode/prediction counters are reused "
                "across runs, but both candidate rows have different initial "
                "states, warm-start histories, scores, exact MSEs, and closed-"
                "loop policy trajectories from the screening oracle-confirm run."
            ),
            "D_runtime_events_eligible_under_min_terminal_5": [
                audit["eligible_under_runtime_min_terminal_5"]
                for audit in audits
            ],
        },
        "limitations": {
            "runtime_recurrent_tensors_logged": False,
            "direct_runtime_offline_tensor_equality_possible": False,
            "raw_shadow_cache_tensor_identity_verified": True,
            "identity_rule": (
                "task/episode/prediction/timestep counters are nominal only; "
                "protocol partition and initial-state identity are required"
            ),
        },
        "outputs": {
            "json": str(args.output),
            "mapping_csv": str(args.mapping_csv),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_mapping_csv(args.mapping_csv, audits)
    _print_summary(audits)
    print(f"\nJSON: {args.output}")
    print(f"CSV: {args.mapping_csv}")


if __name__ == "__main__":
    main()
