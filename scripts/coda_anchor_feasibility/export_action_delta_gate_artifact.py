"""Train, export, and replay-validate the Phase-B fold-4 Action-Delta Gate."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from evaluate_linear_action_predictor import predict_rows, replay_metrics, train_model
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_ARTIFACT_TYPE,
    ACTION_DELTA_GATE_CALIBRATION_METHOD,
    ACTION_DELTA_GATE_MODEL_TYPE,
    ACTION_DELTA_GATE_SCHEMA_VERSION,
    evaluate_action_delta_gate,
    load_action_delta_gate_artifact,
    prepare_action_delta_gate,
    sha256_file,
)


OUTER_FOLD = 4
HELD_OUT_TASK_IDS = [4, 5]
BASE_SEED = 7
TRAINING_SEED = BASE_SEED + 1000 + OUTER_FOLD
EPOCHS = 60
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
AUDIT_THRESHOLD = 0.000732466738008497
EXPECTED_REPLAY = {
    "trajectory_count": 616,
    "activated": 379,
    "correct_safe_stop": 374,
    "false_early_stop": 5,
    "no_skip": 237,
}


def _source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _runtime_scores(cache, row_indices, gate) -> np.ndarray:
    scores = []
    for global_index in np.asarray(row_indices, dtype=np.int64):
        # The cache already contains the offline BF16-quantized delta. Using a
        # zero anchor reconstructs a pair whose runtime subtraction and BF16
        # re-quantization produce exactly that cached delta.
        current = cache["delta_states"][int(global_index)].unsqueeze(0).to(gate.x_mean.device)
        anchor = torch.zeros_like(current)
        score, _ = evaluate_action_delta_gate(gate, anchor, current)
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def _payload(model, stats, threshold, cache_path, calibration_path, train_rows):
    return {
        "schema_version": ACTION_DELTA_GATE_SCHEMA_VERSION,
        "artifact_type": ACTION_DELTA_GATE_ARTIFACT_TYPE,
        "model_type": ACTION_DELTA_GATE_MODEL_TYPE,
        "hidden_dim": 896,
        "action_dim": 7,
        "action_chunk_len": 8,
        "held_out_task_ids": HELD_OUT_TASK_IDS,
        "outer_fold": OUTER_FOLD,
        "threshold": float(threshold),
        "x_mean": stats["x_mean"].detach().cpu().float().contiguous(),
        "x_std": stats["x_std"].detach().cpu().float().contiguous(),
        "y_mean": stats["y_mean"].detach().cpu().float().contiguous(),
        "y_std": stats["y_std"].detach().cpu().float().contiguous(),
        "linear_weight": model.linear.weight.detach().cpu().float().contiguous(),
        "linear_bias": model.linear.bias.detach().cpu().float().contiguous(),
        "delta_quantization_dtype": "bfloat16",
        "training_seed": TRAINING_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "calibration_method": ACTION_DELTA_GATE_CALIBRATION_METHOD,
        "training_row_count": int(len(train_rows)),
        "provenance": {
            "source_repository_commit": _source_commit(),
            "source_cache_sha256": sha256_file(cache_path),
            "source_calibration_results_sha256": sha256_file(calibration_path),
            "source_trace_set_identity": "action_delta_cache.pt:fold!=4",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("benchmark_results/coda_anchor_feasibility/action_delta_cache.pt"),
    )
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/"
            "multitask_risk_calibration_results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if not args.cache.is_file() or not args.calibration_results.is_file():
        raise FileNotFoundError("real cache and calibration results are required; synthetic artifacts are not exported")

    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    with args.calibration_results.open(encoding="utf-8") as stream:
        calibration = json.load(stream)

    threshold = float(
        calibration["outer_folds"][str(OUTER_FOLD)]["methods"]
        [ACTION_DELTA_GATE_CALIBRATION_METHOD]["threshold"]
    )
    if not math.isclose(threshold, AUDIT_THRESHOLD, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(
            "fold-4 calibration threshold differs from the audited value: "
            f"actual={threshold}, audited={AUDIT_THRESHOLD}"
        )

    folds = cache["folds"].numpy()
    task_ids = cache["task_ids"].numpy()
    train_idx = np.where(folds != OUTER_FOLD)[0]
    test_idx = np.where(folds == OUTER_FOLD)[0]
    held_out = sorted(np.unique(task_ids[test_idx]).tolist())
    if held_out != HELD_OUT_TASK_IDS:
        raise RuntimeError(f"fold-4 held-out tasks differ: {held_out}")

    model, stats = train_model(
        cache=cache,
        train_indices=train_idx,
        model_name="delta_only",
        device=args.device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=TRAINING_SEED,
    )
    test_indices, offline_pred_delta = predict_rows(
        cache=cache,
        row_indices=test_idx,
        model=model,
        stats=stats,
        model_name="delta_only",
        device=args.device,
        batch_size=BATCH_SIZE,
    )
    offline_scores = offline_pred_delta.float().square().mean(dim=(1, 2)).numpy()
    payload = _payload(model, stats, threshold, args.cache, args.calibration_results, train_idx)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="action_delta_gate_fold4_", dir=args.output_dir.parent))
    artifact_path = temporary_root / "action_delta_gate.pt"
    torch.save(payload, artifact_path)
    artifact_sha = sha256_file(artifact_path)
    manifest = {
        "schema_version": ACTION_DELTA_GATE_SCHEMA_VERSION,
        "artifact_type": ACTION_DELTA_GATE_ARTIFACT_TYPE,
        "artifact_file": artifact_path.name,
        "artifact_sha256": artifact_sha,
        "model_type": ACTION_DELTA_GATE_MODEL_TYPE,
        "outer_fold": OUTER_FOLD,
        "held_out_task_ids": HELD_OUT_TASK_IDS,
        "calibration_method": ACTION_DELTA_GATE_CALIBRATION_METHOD,
        "threshold": threshold,
    }
    (temporary_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    try:
        _, loaded = load_action_delta_gate_artifact(temporary_root, expected_sha256=artifact_sha)
        gate = prepare_action_delta_gate(loaded, device=args.device, task_id=4)
        runtime_scores = _runtime_scores(cache, test_indices, gate)
        max_score_difference = float(np.max(np.abs(runtime_scores - offline_scores)))
        if not np.allclose(runtime_scores, offline_scores, rtol=1e-6, atol=1e-10):
            raise RuntimeError(
                "runtime/offline score parity failed: "
                f"max_abs_difference={max_score_difference}"
            )

        replay = replay_metrics(
            runtime_scores,
            cache["target_safe"][test_indices].numpy(),
            cache["trajectory_ids"][test_indices].numpy(),
            cache["ks"][test_indices].numpy(),
            threshold,
        )
        observed = {key: replay[key] for key in EXPECTED_REPLAY}
        if observed != EXPECTED_REPLAY:
            raise RuntimeError(
                f"fold-4 replay mismatch: expected={EXPECTED_REPLAY}, observed={observed}"
            )

        manifest["validation"] = {
            "runtime_offline_max_score_difference": max_score_difference,
            "fold4_sequential_replay": observed,
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact directory: {args.output_dir}")
        temporary_root.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(json.dumps({
        "artifact_path": str(args.output_dir / artifact_path.name),
        "artifact_sha256": artifact_sha,
        "threshold": threshold,
        "runtime_offline_max_score_difference": max_score_difference,
        "fold4_sequential_replay": observed,
    }, indent=2))


if __name__ == "__main__":
    main()
