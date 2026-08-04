# Fixed-Depth Coda Microprofile

This experiment isolates the latency value of removing repeated action decodes
while holding recurrent depth fixed. It is the first follow-up required by
`docs/scalar_direct_confirm_replay_handoff_20260804.md`.

## Question

For the same captured action-head workload, selected initial latent state, and
fixed recurrent depth `K`, how much synchronized action-head latency is removed
when the output path is executed only at the terminal state?

This is not a scalar-policy evaluation and does not support a success-rate
claim.

## Compared conditions

The formal protocol compares the following paired conditions:

| Condition | Recurrent iterations | Output decodes |
|---|---:|---:|
| `legacy_fixed_k4` | 4 | 4 |
| `terminal_only_k4` | 4 | 1 |
| `legacy_fixed_k6` | 6 | 6 |
| `terminal_only_k6` | 6 | 1 |
| `legacy_fixed_k8` | 8 | 8 |
| `terminal_only_k8` | 8 | 1 |

`terminal_only_k5` is also measured. Its paired difference from
`terminal_only_k4` estimates the latency of one additional recurrent iteration
when both arms execute one terminal decode.

The legacy fixed path includes the existing per-iteration action-output metric
logging. Therefore, the paired difference represents removal of repeated
`_get_output` execution and the associated fixed-path output-comparison work,
not only the transformer Coda layers in isolation. The terminal-only path also
retains its current fail-closed finite checks. These implementation details must
be preserved when interpreting the result.

## Correctness preflight

Before timing, the preflight replays every paired `K=4`, `K=6`, and `K=8`
condition and requires exact tensor equality between the terminal action from
the legacy fixed path and the terminal-only path. It also validates recurrent
depth, output-decode count, warm/cold origin handling, and terminal decode
execution.

The correctness check is intentionally separate from the timed runner so that
output comparison and diagnostic synchronization cannot contaminate the primary
latency measurement.

Development preflight:

```bash
python scripts/run_fixed_coda_preflight.py \
  --max-workloads 2 \
  --output benchmark_results/fixed_coda_microprofile/dev/preflight.json
```

Formal preflight:

```bash
python scripts/run_fixed_coda_preflight.py \
  --output benchmark_results/fixed_coda_microprofile/seed7/preflight.json
```

Do not start the formal timing run unless the formal preflight reports zero
output-equivalence failures.

## Timed boundary

The primary measurement is CPU wall-clock time around
`ActionHeadRecurrent.predict_action`, with `torch.cuda.synchronize()` immediately
before and after the call. It includes:

- proprio projection;
- action/task hidden-state split;
- Prelude;
- exactly `K` recurrent iterations;
- the condition-specific output-decode schedule;
- current runtime validation and synchronization overhead.

It excludes workload deserialization, validation, CPU-to-GPU copies, and the
captured proprio-feature equivalence check.

The runner reuses the 200 frozen action-head workloads captured during formal
seed-7 calibration. For cold workloads, it executes the production random
initialization for its cost but substitutes the saved selected initial state,
matching the previous GPU replay protocol. The primary scope is the 100
`ACTUAL_WARM` workloads; the 100 `COLD` workloads are supplementary.

## Aggregation

Each workload/condition is measured five times in a deterministic balanced
complete-block order. The median across repeats is computed within each
workload. The report then records descriptive mean, median, and p95 values and
paired terminal-only versus legacy comparisons at each `K`.

No post-hoc pass/fail latency threshold is used. The raw synchronized
measurements and descriptive summaries are retained in the JSON report.

## Development timing run

```bash
cd /home/siwon/RD-VLA_test/rd-vla

git switch experiment/fixed-coda-microprofile
git pull --ff-only

python scripts/run_fixed_coda_microprofile.py \
  --max-workloads 2 \
  --measurement-repeats 3 \
  --warmup-rounds 1 \
  --output benchmark_results/fixed_coda_microprofile/dev/report.json
```

A development subset is never marked as a formal run.

## Formal timing run

```bash
python scripts/run_fixed_coda_microprofile.py \
  --output benchmark_results/fixed_coda_microprofile/seed7/report.json
```

The runner refuses to overwrite an existing report unless `--overwrite` is
provided. The report records the code commit, protocol hash, initial-state
manifest hash, checkpoint component hashes, CUDA environment, workload counts,
raw measurements, schedule validation, and aggregated results.

## Required interpretation

Use the result to answer only the following question:

> At identical recurrent depth, does removing repeated output decodes produce a
> meaningful reduction in synchronized action-head latency on the captured
> workload distribution?

If the reduction is small, do not invest further work in scalar retraining
before reconsidering whether repeated decode removal can be a main systems
contribution. If the reduction is meaningful, proceed to the calibration-depth
distribution audit described in the handoff document.
