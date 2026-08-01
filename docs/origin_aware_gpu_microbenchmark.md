# Origin-Aware GPU Schedule Microbenchmark

This stage replays the frozen OOF Top-6 configurations and the clean midpoint
warm-only baseline on the 200 action-head workloads captured during formal
seed-7 calibration. It is baseline-conditioned candidate pruning, not a
closed-loop or VLM end-to-end experiment.

## Timed boundary

Every workload is deserialized, validated, and copied to `cuda:0` before the
timer starts. The primary measurement is CPU wall-clock time around
`ActionHeadRecurrent.predict_action`, with a CUDA synchronization immediately
before and after the call. This includes the proprio projector, prelude,
recurrence, latent gate and its host synchronization, Coda/backfill calls,
adjacent action-MSE synchronization, and cached-final return path. Tensor I/O,
VLM inference, environment work, and validation are excluded.

Captured cold states are required to reproduce the exact baseline-conditioned
trajectory. For a cold workload the runner executes the production random
initialization for its cost, discards its value, and then uses the saved initial
state in every arm. Cold results are supplementary. The primary promotion
scope is the 100 `ACTUAL_WARM` workloads, one paired workload per task and
episode.

Each workload/condition is measured five times in deterministic balanced
complete-block orders. The within-workload median is averaged by episode, then
task, and finally across the fixed ten tasks with equal task weight. The
one-sided 95% simultaneous lower bound uses paired episode resampling within
each task and the maximum centered downward error across the frozen six
candidates.

A candidate may enter online screening only if all replayed schedules match
the offline expectation and its primary simultaneous lower bound is at least
5%. At most two candidates are promoted. A subset run always disables
promotion.

## Development subset

```bash
python scripts/run_origin_aware_gpu_microbenchmark.py \
  --max-workloads 2 \
  --measurement-repeats 3 \
  --output benchmark_results/origin_aware_gpu_microbenchmark/dev/report.json
```

## Formal run

```bash
python scripts/run_origin_aware_gpu_microbenchmark.py \
  --output benchmark_results/origin_aware_gpu_microbenchmark/seed7/report.json
```

The runner refuses to overwrite an existing report and records the code
commit, protocol and shortlist hashes, source OOF/calibration hashes,
checkpoint hashes, GPU environment, raw synchronized measurements, schedule
mismatches, and promotion decision.
