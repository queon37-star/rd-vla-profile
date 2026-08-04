# Scalar calibration depth audit

## Purpose

This stage audits whether the formal scalar calibration population contains enough deep-recurrence examples and whether the existing task-OOF scalar policy fires too early on those examples.

It follows the fixed-depth Coda microprofile, which established that repeated output-decode removal has measurable value at identical recurrent depth. This audit does not retrain the scalar model and does not change the runtime state machine.

## Latency terminology

All latency discussion in the current optimization study uses the **post-VLM action-policy scope** unless explicitly stated otherwise. The VLM backbone is excluded because both warm-start and Coda scheduling operate after VLM hidden states are produced. See `docs/latency_reporting_scope.md`.

## Formal inputs

- Raw schema-2 calibration trajectories:
  `benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7/manifest.json`
- Finalized rollout step log with episode success labels:
  `benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7/steps.jsonl`
- Existing task-OOF scalar runtime policy:
  `benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_kfirst_v1`
- Frozen audit protocol:
  `experiments/robot/libero/manifests/scalar_calibration_depth_audit_v1.json`

The formal contract expects 2,398 predictions: 2,298 `ACTUAL_WARM` and 100 cold predictions across task IDs 0 through 9.

## Reported questions

The report answers the following questions for the actual-warm population:

1. How many predictions have authoritative `K_action=3`, `K_action=4`, and `K_action=5`?
2. How many have `K_action>=6` and `K_action>=8`?
3. Are deep predictions concentrated in a small number of tasks or episodes?
4. How do the depth distributions differ between successful and failed episodes?
5. Among predictions with `K_action>=6`, how often does the existing task-OOF scalar policy already trigger at `k=3`?

The last group is the primary hard-negative definition:

```text
actual_origin = ACTUAL_WARM
K_action >= 6
scalar_score(k=3) >= task-specific OOF threshold
```

These are states for which the current policy requests an early stop near K=4 even though authoritative Action-MSE requires substantially deeper recurrence.

## Run

```bash
cd /home/siwon/RD-VLA_test/rd-vla

git fetch origin
git switch experiment/scalar-calibration-depth-audit
git pull --ff-only

python scripts/audit_scalar_calibration_depth.py \
  --output benchmark_results/preconvergence_trigger/seed7/calibration_depth_audit/report.json
```

The command is offline and does not require LIBERO rollout or GPU inference. It loads roughly 1 GiB of raw latent/action tensors, so the process may temporarily use several GiB of host memory.

Expected validation prefix:

```text
Formal run: True
Predictions: 2398 (warm=2298, cold=100)
```

The remaining terminal lines report the warm `K>=6` and `K>=8` coverage and the number of k=3 scalar hard negatives. The full JSON also contains exact histograms, per-task shares, per-episode concentration, success/failure summaries, the top 50 hard negatives, and one compact row per prediction.

## Decision after the audit

- If `K>=6` examples are rare or heavily concentrated, keep the same seven features and runtime contract, then test data balancing or additional non-final development collection first.
- If deep examples are sufficiently represented but k=3 hard negatives remain frequent, the evidence points toward a feature/target limitation rather than simple data scarcity.
- Do not use the previously observed 300-pair final partition for threshold tuning, feature selection, weighting selection, or hard-example mining.
