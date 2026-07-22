# RD-VLA Warm-Start Integration Plan

## Goal

Integrate temporal warm-start into the paper baseline that already contains recurrent convergence logging, Coda profiling, cached-final-output reuse, latent/action delta logging, latent pre-check, PyTorch profiler ranges, and detailed timing summaries.

Candidate-batch recurrent inference is intentionally excluded from this branch and remains only on `feature/rdvla-pytorch-profiler` after commit `fd423c87e7c286476ed7fd5c4d315ac8b90b5566`.

## Integration Baseline

- Target repository: `queon37-star/rd-vla-profile`
- Integration branch: `paper/warm-start-integration`
- Base commit: `cab870eac0468cc15e0daa140a90d87f8b749b19`
- Included research path: convergence logging, Coda reduction, latent pre-check, and detailed timing instrumentation
- Excluded experiment: candidate-batch recurrent inference
- Source warm-start implementation: `choijustin-east/choi_east`, especially `warm-start-S1` and related state-selection branches

The base commit is selected because it contains the complete Coda-optimization and profiling path needed for the paper while stopping immediately before the experimental candidate-batch commit.

## Design Principles

1. Warm-start must be optional and disabled by default.
2. Cold-start and warm-start must run from the same code branch for fair A/B comparison.
3. The cache must reset at every episode boundary.
4. Cached tensors must remain detached and on the inference device.
5. Existing recurrence, Coda, latent-precheck, and timing metrics must remain available.
6. Candidate-batch code must not be introduced into this branch.
7. Avoid fragile return-value proliferation. Return inference metadata as one structured dictionary rather than repeatedly extending positional tuples.
8. The first implementation should reproduce S1 warm-start before adding more aggressive variants.

## Proposed Configuration

Add the following evaluation options:

```python
use_warm_start: bool = False
warm_start_source: str = "s1"  # initially only s1
warm_start_blend_alpha: float = 1.0
warm_start_reset_interval: int = 0  # 0 means episode-boundary reset only
```

Optional safeguards can be added after the basic implementation is validated:

```python
warm_start_max_proprio_delta: float | None = None
warm_start_max_action_delta: float | None = None
```

## Files to Modify

### 1. `prismatic/models/action_heads.py`

- Accept `warm_start_state=None` in the recurrent action-head path.
- Initialize recurrent state from the cache when enabled; otherwise call `init_state()`.
- Save the first recurrent update state for S1 warm-start.
- Use `detach().clone()` for the returned cache state.
- Preserve `last_recurrence_debug` and all Coda/latent-precheck/timing fields.
- Add warm-start metadata without placing tensors in JSON-serializable debug records.
- Do not add candidate-batch parameters or per-lane computations.

Initial state-selection semantics:

- `s1`: first recurrent update state.

Only after S1 is validated may later ablations compare midpoint, three-fifths, or final recurrent states.

### 2. `prismatic/extern/hf/modeling_prismatic.py`

- Thread `warm_start_state` into the regression action-head call.
- Preserve existing normalized-action and recurrence outputs.
- Preserve Coda and timing instrumentation.
- Prefer a structured inference metadata object over new positional tuple variants.

### 3. `outputs/12_24-24_24_Spatial_40k/modeling_prismatic.py`

- Keep this checkpoint-side source synchronized only if runtime inspection confirms that it is imported during evaluation.
- Avoid diverging behavior between the canonical and checkpoint copies.

### 4. `experiments/robot/openvla_utils.py`

- Accept the current cache state.
- Pass it to `predict_action()`.
- Read both recurrence debug data and the next cache state.
- Return one `inference_metadata` dictionary, for example:

```python
{
    "recurrence_debug": recurrence_debug,
    "next_warm_start_state": next_warm_start_state,
}
```

### 5. `experiments/robot/robot_utils.py`

- Forward the current warm-start cache if this layer is part of the active call path.
- Preserve the structured inference metadata.

### 6. `experiments/robot/libero/run_libero_eval.py`

- Initialize `warm_start_state = None` at episode start.
- Pass the cache only when `use_warm_start=True`.
- Replace it after each action-chunk prediction.
- Reset it at every episode boundary and on explicit safety conditions.
- Update warm-start only at policy-query boundaries, not every environment timestep.
- Extend prediction logs with warm-start fields.
- Preserve the existing Coda, latent-precheck, convergence, and timing summaries.

## Logging Extensions

Add the following fields to each prediction JSONL record:

```text
warm_start_enabled
warm_start_used
warm_start_source
warm_start_source_iteration
warm_start_cache_age
warm_start_reset
warm_start_reset_reason
initial_state_origin          # random or cached
```

Keep the existing fields, including:

```text
recurrent_iteration_count
max_recurrent_iteration
adaptive_stop
iteration_mse
final_mse
latency_ms
success
latent_precheck_skip_count
latent_precheck_call_count
latent_precheck_skip_ratio
get_output_call_count
coda_ms_total
run_one_iteration_ms_total
```

Add run-level comparisons for:

- mean, median, p90, and p95 recurrent iterations;
- mean, median, p90, and p95 action-inference latency;
- Coda call count and Coda time;
- latent-precheck skip rate;
- max-iteration hit rate;
- adaptive-stop rate;
- success rate;
- warm-start usage and reset rates.

## Implementation Phases

### Phase 0 — Freeze the Paper Baseline

- Confirm the branch starts from `cab870eac0468cc15e0daa140a90d87f8b749b19` plus this plan commit.
- Confirm candidate-batch symbols and benchmark scripts are absent.
- Record exact commands for cold-start Adaptive `1e-4`, `5e-4`, and `1e-3`.
- Preserve fixed K=8 and K=12 as reference baselines.
- Run syntax and import checks before warm-start changes.

### Phase 1 — Add Disabled Warm-Start Plumbing

- Add configuration and call-chain parameters.
- Keep `use_warm_start=False`.
- Verify that outputs, K statistics, Coda statistics, and success behavior are unchanged.

Acceptance criteria:

- cold-start smoke test runs without return-value errors;
- existing JSONL records remain valid;
- fixed and adaptive recurrence both execute;
- Coda and timing summaries are still generated;
- no candidate-batch configuration appears in the branch.

### Phase 2 — Implement S1 Cache

- Cache the first recurrent update state.
- Reuse it at the next policy query in the same episode.
- Reset cache at episode start.

Acceptance criteria:

- first query reports `warm_start_used=false`;
- later queries report `warm_start_used=true`;
- no cache crosses episode boundaries;
- tensor shape, dtype, device, and batch size are validated;
- cached state is detached and does not retain a computation graph.

### Phase 3 — Smoke and Regression Tests

Run one LIBERO-Spatial task with one episode for:

1. cold-start adaptive;
2. warm-start S1 adaptive;
3. cold-start fixed K=12;
4. warm-start enabled with fixed K=12 as a control.

Check:

- no NaNs;
- no tuple-unpacking errors;
- action chunk length remains valid;
- cache resets correctly;
- logging files contain all required fields;
- disabled warm-start matches the pre-change cold path;
- Coda and latent-precheck options still work.

### Phase 4 — Paper Baseline Experiment

Use the same checkpoint, seeds, initial states, task ordering, action horizon, Coda settings, and number of trials.

Initial comparison matrix:

| Method | Warm start | State | Threshold / K |
|---|---:|---|---|
| Fixed | No | Random | K=8 |
| Fixed | No | Random | K=12 |
| Adaptive | No | Random | 1e-4 |
| Adaptive | No | Random | 5e-4 |
| Adaptive | No | Random | 1e-3 |
| Adaptive | Yes | S1 | 1e-4 |
| Adaptive | Yes | S1 | 5e-4 |
| Adaptive | Yes | S1 | 1e-3 |

Primary paper metrics:

- LIBERO success rate;
- mean and p95 recurrent depth;
- mean and p95 steady-state inference latency;
- recurrent-iteration reduction relative to the matching cold baseline;
- Coda call/time reduction under matched settings;
- failure and max-iteration-hit analysis.

### Phase 5 — Warm-State Ablation

Only after S1 is validated, consider:

- S1;
- midpoint state;
- three-fifths state;
- final state.

This phase is optional and must not delay the first paper-ready S1 result.

### Phase 6 — Safety and Reset Ablation

Evaluate optional policies only after the direct S1 result is established:

- periodic cache reset;
- reset when proprioception changes sharply;
- random/cached state blending;
- fallback to cold-start after max-iteration hits.

## Experimental Fairness Rules

- Use one code branch for both warm and cold runs.
- Change only warm-start-related flags between matched experiments.
- Use identical checkpoint, task order, seed, initial states, action horizon, thresholds, and Coda settings.
- Exclude startup/warm-up calls from final latency statistics.
- Report both success rate and compute reduction; do not claim optimization from K reduction alone.
- Keep candidate-batch disabled and absent from the paper branch.
- Treat 30 episodes as preliminary. Use more trials for the final paper table when compute permits.

## Immediate Next Actions

1. Verify the rebased branch contains `cab870...` and this plan only beyond that point.
2. Confirm candidate-batch files and flags are absent.
3. Run syntax/import checks on the paper baseline.
4. Implement disabled warm-start plumbing.
5. Run cold-start regression smoke tests.
6. Implement and validate S1 cache reuse.
7. Expand to matched LIBERO-Spatial cold/warm experiments.
