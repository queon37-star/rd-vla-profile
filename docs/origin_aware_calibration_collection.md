# Origin-Aware Coda Calibration Collection Contract

This document freezes the formal seed-7 calibration collection used to prune
origin-aware Coda scheduler candidates. It is a data-collection contract, not
an efficiency result: workload serialization and full-depth shadow tracing are
deliberately inside the measured action call and therefore invalidate the run's
latency as deployment evidence.

## Frozen inputs

- Suite: `libero_spatial`, task IDs `0..9`.
- Checkpoint: `outputs/12_24-24_24_Spatial_40k` unless an explicitly recorded
  replacement is supplied.
- Initial-state manifest:
  `experiments/robot/libero/manifests/libero_spatial_official_50_v1.json`.
- Partition: the ten `calibration` state IDs for each task; smoke states are not
  additional samples and smoke results are never used for fitting.
- Base seed: `7`; the paired episode seed is derived from phase, task, initial
  state, and paired-trial ID by the frozen evaluation protocol.
- Production policy: clean midpoint warm-start, adjacent-action MSE stopping,
  cached-final output, latent pre-check `off`, and legacy non-finite behavior.
- Shadow policy: continue every production prediction to full recurrence depth
  in a separate trace buffer without changing the returned action, actual K,
  terminal state, next midpoint cache, or stop reason.

## Artifacts

Every prediction records a complete scalar shadow trace. The first two
predictions of every episode additionally save one action-head workload shard:

- prediction 0: `COLD`;
- prediction 1: `ACTUAL_WARM`.

Each shard contains the action hidden states, raw proprio input, projected
proprio features, incoming warm cache when present, and the selected initial
latent state. Tensors must be finite and contiguous at the production
action-head boundary. They are copied to CPU, saved atomically, and referenced
from the step log by an identity tuple and SHA-256 digest. Incoming caches are
stored inline, so sampled records have no producer-reference dependency.

The formal validator rejects missing or extra shards, path escapes, digest or
identity mismatches, non-finite or non-contiguous tensors, incomplete shadow
traces, numerical retries, unexpected origins, incorrect partitions or seeds,
and any populated legacy latent-precheck trace in clean `off` mode.

## Task-level OOF folds

The outcome-independent five-fold manifest is
`experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json`:

| Fold | Validation tasks |
| --- | --- |
| 0 | 0, 9 |
| 1 | 1, 8 |
| 2 | 2, 7 |
| 3 | 3, 6 |
| 4 | 4, 5 |

All candidate-family ranking uses validation records pooled out of fold. Only
after that ranking may a quantile threshold be refit on the complete calibration
set. Refit configurations are deduplicated by their full-precision numerical
configuration before schedule microbenchmarking.

## Commands and gates

Run the collector from an activated `rdvla_env` environment:

```bash
RUN_TAG=<immutable-run-tag> ./run_origin_aware_calibration_10x10.sh
```

Validate an existing complete run independently:

```bash
python scripts/validate_origin_aware_calibration.py \
  --run-root benchmark_results/origin_aware_calibration/<run-tag> \
  --manifest experiments/robot/libero/manifests/libero_spatial_official_50_v1.json
```

The final report must have `valid=true`, `complete_10_task_gate=true`, 100
episodes, 200 workload shards (100 cold and 100 actual-warm), and all ten task
reports. Formal OOF replay must not start before this gate passes.
