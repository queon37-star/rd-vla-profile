# Scalar k=3 separability audit

## Motivation

The formal calibration-depth audit found:

- `ACTUAL_WARM` predictions: 2,298
- `K_action >= 6`: 869 predictions (37.815%)
- `K_action >= 8`: 320 predictions (13.925%)
- k=3 scalar hard negatives among `K_action >= 6`: 732 / 869

Thus deep-recurrence examples are not globally rare, while the deployed task-OOF scalar threshold fires at k=3 for about 84.2% of the severe-deep group. This rules out simple global data scarcity as a sufficient explanation, but it does not yet distinguish among:

1. an overly aggressive deployed threshold;
2. weak separability of the seven k=3 scalar features;
3. a target that is not aligned with task-success preservation.

This audit addresses the first two possibilities without retraining.

## Definitions

At k=3, scalar `confirm_next` produces terminal depth K=4. The descriptive cohorts are:

- safe early gate: `K_action <= 4`
- unsafe early gate: `K_action >= 5`
- severe unsafe gate: `K_action >= 6`
- very severe unsafe gate: `K_action >= 8`

A higher scalar score means a stronger request to stop early. The deployed decision uses each task-specific OOF threshold, equivalently `k3_score_margin >= 0`.

## Analyses

The script reports:

- deployed false-trigger rates for K>=5, K>=6, and K>=8;
- score and task-threshold margin distributions by depth cohort;
- threshold-free ROC-AUC and average precision for safe versus unsafe/deep examples;
- per-task safe-versus-severe ROC-AUC;
- success/failure episode summaries at the deployed operating point;
- descriptive operating points obtained by adding one uniform offset to all task-specific thresholds.

The offset sweep reports the maximum safe-gate recall observed while restricting the severe false-trigger rate to 1%, 5%, 10%, or 20%. These are diagnostic points only. They are not new runtime thresholds and must not be selected for deployment from this seen calibration population.

## Latency scope

Any latency discussion remains limited to the post-VLM action-policy path. The VLM backbone is excluded.

## Run

```bash
cd /home/siwon/RD-VLA_test/rd-vla

git fetch origin
git switch experiment/scalar-k3-separability-audit
git pull --ff-only

python scripts/audit_scalar_k3_separability.py \
  --output benchmark_results/preconvergence_trigger/seed7/k3_separability_audit/report.json
```

Expected prefix:

```text
Formal run: True
Actual-warm predictions: 2298
Deployed k=3 severe false-trigger rate: 84.235%
Safe-vs-severe k=3 score ROC-AUC: ...
```

## Interpretation

- High ROC-AUC with useful safe recall at low severe false-trigger rates indicates that the current task thresholds or their selection objective are the immediate problem.
- Low or inconsistent ROC-AUC, especially across tasks, indicates that the current seven-feature summary cannot reliably distinguish safe K=4 exits from predictions that require deeper recurrence.
- Neither result proves task-success alignment. A score that reproduces Action-MSE depth can still fail to preserve closed-loop success.

Do not retrain, alter features, or modify the runtime state machine until this audit is reviewed.
