# Formal Origin-Aware OOF Selection

This stage consumes only the validated seed-7 calibration artifacts. It ranks
scheduler families for an exact GPU schedule microbenchmark; it does not
estimate closed-loop success or authorize online screening.

## Frozen family pool

```text
(fixed {0.075, 0.08, 0.10, 0.12, 0.15}
 union train-fold actual-warm {Q20, Q40, Q60, Q80})
x max_skip {1, 2, 3}
x confirmation {next_iter, backfill_pair}
```

This produces 54 families. Quantiles use only finite `ACTUAL_WARM` transitions
that are gate-eligible, have `k >= 3`, and are earlier than `max_iter`. A
quantile is fit on the four training folds and replayed only on the held-out
fold. All safety metrics and latency ranks are formed from the concatenated OOF
validation predictions, with episode then task macro aggregation.

Safety constraints are applied before latency ranking:

- adjacent convergence capture at least 99.5%;
- mean delta-K at most 0.25;
- episode/task-macro p95 delta-K at most 1;
- no increase in max-iteration rate;
- no non-finite/retry or invalid action-space convergence event.

The persistent-tail metric is diagnostic. It asks whether a convergence event
remains below threshold through the full shadow tail, which is stricter than
RD-VLA's original first-adjacent-pair stopping rule; it is not silently treated
as the method's authoritative stopping criterion.

## Cost sensitivity and interpretation

The primary model retains the frozen 3.56 ms recurrence and 1.83 ms full-decode
planning anchors. Its 0.166328 ms action-comparison value is the median of 4,221
synchronized steady comparison measurements from the recorded three-task
cached-baseline profile. Candidate-only latent-gate and finite-check costs are
set to zero in the primary model, deliberately favoring the candidate.

Additional scenarios test zero control overhead, an unusually high comparison
cost favorable to the candidate, and low/moderate nonzero candidate overhead.
These are selection-conditioned sensitivity estimates, not confidence
intervals or deployment measurements. The selected six numerical configs must
therefore be run through the saved-workload GPU schedule microbenchmark. Even a
linear estimate above 5% cannot directly promote a candidate to online
screening.

## Command

```bash
python scripts/run_formal_origin_aware_oof.py \
  --run-root benchmark_results/origin_aware_calibration/20260801_ca1b7d3_seed7_10x10 \
  --output benchmark_results/origin_aware_oof/20260801_seed7/report.json
```

The command revalidates all 100 episodes and 200 workload shards before loading
the 2,398 shadow predictions. It refuses to overwrite a report unless
`--overwrite` is explicitly supplied and records SHA-256 hashes for every trace
and manifest.
