# Scalar conservative-offset full-trajectory replay

## Why this stage exists

The k=3 separability audit found that the existing seven scalar features are informative at the first eligible gate:

- safe-vs-severe ROC-AUC: `0.934630`
- deployed severe false-trigger rate: `84.235%`
- at severe FPR <= 5%, descriptive safe recall: `74.340%`
- at severe FPR <= 10%, descriptive safe recall: `81.732%`

This means the immediate problem is not simple global data scarcity or complete feature collapse. The deployed threshold is too aggressive at k=3. However, increasing one global margin offset can also delay gates at later iterations, so the k=3 table alone is not enough to propose a runtime change.

This stage replays the entire recurrent trajectory for each descriptive offset.

## Central question

Can a more conservative scalar threshold move terminal K closer to the authoritative Action-MSE K while retaining the one-terminal-Coda advantage?

This question matters because the fixed-depth microprofile showed that terminal-only output decoding remains faster even when recurrent K is held constant. Therefore a useful policy does not need to reduce K aggressively. It may preserve K more closely and obtain efficiency mainly by removing repeated intermediate output decoding.

## Replay contract

Primary population:

```text
ACTUAL_WARM = 2,298 predictions
LIBERO Spatial tasks 0..9
```

For every prediction and every `k=3..32`, the runner recomputes the existing task-OOF scalar score from the stored raw latent trajectory. For each candidate margin offset:

```text
effective threshold = task-specific deployed threshold + margin offset
first score >= effective threshold -> gate k
confirm-next terminal K = min(k+1, max_iter)
no gate -> terminal K = max_iter
Coda calls = 1
```

The baseline is the authoritative first Action-MSE hit `K_action`, with one Coda/output decode per recurrent iteration.

The report includes:

- early, exact, and late terminal-K rates;
- mean, median, p95, and extreme delta-K counts;
- early-terminal rates inside `K_action>=6` and `K_action>=8`;
- no-gate/max-iteration rates;
- total Coda-call reduction;
- per-task and success/failure descriptive summaries;
- one record per prediction and candidate.

## Candidate offsets

The runner reads the exact descriptive offsets from:

```text
benchmark_results/preconvergence_trigger/seed7/k3_separability_audit/report.json
```

It evaluates:

- deployed offset `0`;
- k=3 severe-FPR caps of 1%, 5%, 10%, and 20%.

These offsets were obtained after inspecting the same calibration population. They are diagnostic candidates, not deployable thresholds.

## Latency scope and planning estimate

All latency language uses the post-VLM action-policy scope. The VLM backbone is excluded.

The report contains an optimistic mechanism-only delta projection using:

```text
recurrent iteration = 2.4125 ms
removed repeated output path = 1.0715 ms
gate evaluation = 0 ms assumption
```

The first two values summarize the formal fixed-depth microprofile. The gate value is deliberately an explicit zero-overhead assumption. Therefore the projection is useful for ranking feasibility, but it is not a measured runtime latency result.

## Run

```bash
cd /home/siwon/RD-VLA_test/rd-vla

git fetch origin
git switch experiment/scalar-conservative-offset-replay
git pull --ff-only

python scripts/replay_scalar_conservative_offsets.py \
  --output benchmark_results/preconvergence_trigger/seed7/conservative_offset_replay/report.json
```

The command is offline. It loads roughly 1 GiB of raw latent/action data and scores about 71,000 recurrent states on CPU, so it may take several minutes and use several GiB of host memory.

Expected terminal structure:

```text
Scored trajectories: .../2298
Formal input validation: True
Descriptive replay only: True
Actual-warm predictions: 2298
deployed_offset_0: early=..., exact=..., mean_delta_K=..., p95_delta_K=..., projected_net=... ms
k3_severe_fpr_cap_01pct: ...
...
Wrote: benchmark_results/preconvergence_trigger/seed7/conservative_offset_replay/report.json
```

## Interpretation

A promising descriptive offset should satisfy all three directions:

1. sharply reduce early terminal-K events, especially for `K_action>=6`;
2. avoid replacing early stopping with widespread max-iteration overshoot;
3. retain a positive post-VLM mechanism-only saving after the extra recurrence cost.

No candidate from this replay may be promoted directly. A candidate that looks useful must be regenerated through a leakage-safe train-only or nested task-level threshold-selection protocol, then evaluated on fresh episodes or another untouched partition. The previously inspected 300-pair final evaluation cannot be reused as a confirmatory test.
