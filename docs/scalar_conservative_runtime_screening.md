# Conservative scalar runtime screening

## Decision from the full-trajectory replay

The formal ACTUAL_WARM replay contained 2,298 predictions. Its five threshold conditions produced:

| Condition | Early terminal | Exact K | Mean delta K | p95 delta K | Planning net saving |
|---|---:|---:|---:|---:|---:|
| deployed offset 0 | 57.920% | 19.452% | -1.0627 | 2.0 | 7.0124 ms |
| k=3 severe-FPR cap 1% | 8.921% | 26.066% | +0.9735 | 3.0 | 2.1003 ms |
| k=3 severe-FPR cap 5% | 20.627% | 32.855% | +0.4125 | 2.0 | 3.4535 ms |
| k=3 severe-FPR cap 10% | 28.808% | 33.203% | +0.1110 | 2.0 | 4.1810 ms |
| k=3 severe-FPR cap 20% | 38.903% | 30.983% | -0.2550 | 2.0 | 5.0639 ms |

The 5% and 10% points advance to runtime screening:

- **offset05** is the safety-leaning candidate. It reduces early terminals substantially while retaining a positive mechanism-only saving projection.
- **offset10** is the balanced candidate. It has mean delta K closest to zero, the highest exact-K rate, and a larger planning saving than offset05.

The 1% point is not advanced because it shifts recurrence almost one iteration later on average and has the smallest planning saving. The 20% point remains too aggressive because its average K is already earlier than Action-MSE and 38.903% of predictions terminate early.

This selection is descriptive and uses the seen calibration population. It cannot be reported as a new task-OOF threshold result.

## Runtime artifact strategy

No model weight is retrained. `scripts/build_scalar_offset_policy.py` copies the existing hash-verified task-OOF scalar artifact and adds one uniform threshold offset to every task policy. The resulting artifact:

- preserves all OOF model parameters and normalizers;
- records the original and effective task thresholds;
- records the source k=3 report and operating point;
- is explicitly marked screening-only;
- is re-opened through the production scalar-policy loader after saving.

The runtime receives an ordinary validated scalar artifact, so no inference code, checkpoint file, or state-machine implementation is changed for this smoke.

## Build the two candidates

```bash
cd /home/siwon/RD-VLA_test/rd-vla

git fetch origin
git switch experiment/scalar-conservative-runtime-screening
git pull --ff-only

python scripts/build_scalar_offset_policy.py \
  --severe-fpr-limit 0.05 \
  --output-dir benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_offset05

python scripts/build_scalar_offset_policy.py \
  --severe-fpr-limit 0.10 \
  --output-dir benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_offset10
```

Expected output for each command includes:

```text
Built scalar offset runtime artifact
Severe-FPR operating point: severe_FPR<=...
Uniform threshold offset: ...
Artifact SHA256: ...
Manifest: .../manifest.json
```

The builder refuses to overwrite a non-empty output directory.

## Paired smoke

The frozen smoke covers tasks 0 and 5, three paired episodes per task, and three arms:

1. corrected adjacent Action-MSE baseline;
2. scalar confirm-next with offset05;
3. scalar confirm-next with offset10.

Task 5 is included because it produced the largest success loss in the previous 300-pair scalar confirm-next evaluation. Task 0 supplies a second task with previously observed discordance. Smoke uses the first three calibration-partition states through the existing `smoke` protocol and remains excluded from fitting and performance claims.

Inspect the six commands without running them:

```bash
python scripts/run_scalar_conservative_smoke.py \
  --output-root benchmark_results/scalar_conservative_runtime_smoke/seed7 \
  --dry-run
```

Run the smoke:

```bash
python scripts/run_scalar_conservative_smoke.py \
  --output-root benchmark_results/scalar_conservative_runtime_smoke/seed7
```

The launcher verifies both candidate manifests, enables `NUMBA_DISABLE_JIT=1`, uses read-only checkpoint configuration, runs the six task/arm processes sequentially, and writes:

```text
benchmark_results/scalar_conservative_runtime_smoke/seed7/
  run_plan.json
  execution_report.json
  task0/action_mse/{result.json,steps.jsonl}
  task0/offset05/{result.json,steps.jsonl}
  task0/offset10/{result.json,steps.jsonl}
  task5/action_mse/{result.json,steps.jsonl}
  task5/offset05/{result.json,steps.jsonl}
  task5/offset10/{result.json,steps.jsonl}
```

Expected final terminal form:

```text
Smoke completed
Task 0: action_mse=.../3, offset05=.../3, offset10=.../3
Task 5: action_mse=.../3, offset05=.../3, offset10=.../3
Wrote: .../execution_report.json
```

## Progression rule

The smoke is a runtime and gross-regression check only. Do not choose a final candidate from six episodes.

Proceed to the full 10-task by 10-episode screening partition only when:

- all six processes complete;
- candidate artifacts and hashes match the run plan;
- paired initial-state IDs and episode seeds match across arms;
- scalar ACTUAL_WARM predictions execute exactly one terminal output decode;
- COLD first predictions use the corrected Action-MSE fallback;
- no non-finite abort or runtime-contract violation occurs;
- neither conservative candidate shows an obvious task-0 or task-5 regression relative to Action-MSE.

The screening partition may select between offset05 and offset10. The already consumed final partition must not be reused to tune offsets or select the candidate.

## Latency scope

All latency discussion is limited to the **post-VLM action-policy path**. The VLM backbone is excluded. The replay values above are mechanism-only planning projections with scalar gate overhead set to zero, not measured runtime latency.
