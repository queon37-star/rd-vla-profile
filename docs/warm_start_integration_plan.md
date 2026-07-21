# RD-VLA Warm-Start Integration Plan

## Goal

Integrate temporal warm-start into the existing RD-VLA convergence-profiling codebase while preserving fixed/adaptive recurrence baselines, step-level logging, and reproducible LIBERO evaluation for paper experiments.

## Integration Baseline

- Target repository: `queon37-star/rd-vla-profile`
- Integration branch: `paper/warm-start-integration`
- Base commit: `7682a379ed8b499b42809b99cff310646240dad0`
- Source implementation: `choijustin-east/choi_east`, especially `warm-start-S1` and related state-selection branches.

The base commit is selected because it contains the latest recurrent convergence logging on top of the preserved profiling state.

## Design Principles

1. Warm-start must be optional and disabled by default.
2. Cold-start and warm-start must run from the same code branch for fair A/B comparison.
3. The cache must reset at every episode boundary.
4. Cached tensors must remain detached and on the inference device.
5. Existing recurrence metrics and JSONL schemas must remain available.
6. Avoid fragile return-value proliferation. Return inference metadata as one structured dictionary rather than repeatedly extending positional tuples.
7. The first implementation should reproduce S1 warm-start before adding more aggressive variants.

## Proposed Configuration

Add the following evaluation options:

```python
use_warm_start: bool = False
warm_start_source: str = "s1"  # s1, mid, three_fifths, sk
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
- Save recurrent states required by `warm_start_source`.
- Use `detach().clone()` for the returned cache state.
- Preserve `last_recurrence_debug`.
- Add warm-start metadata without placing tensors in JSON-serializable debug records.

Proposed state-selection semantics:

- `s1`: first recurrent update state.
- `mid`: state near `K/2`.
- `three_fifths`: state near `3K/5`.
- `sk`: final recurrent state.

Start with `s1` only. Add the other variants after cold/warm equivalence tests pass.

### 2. `prismatic/extern/hf/modeling_prismatic.py`

- Thread `warm_start_state` into the regression action-head call.
- Preserve existing normalized-action and recurrence outputs.
- Prefer a structured inference metadata object over new positional tuple variants.

### 3. `experiments/robot/openvla_utils.py`

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

### 4. `experiments/robot/robot_utils.py`

- Forward the current warm-start cache.
- Preserve the structured inference metadata.

### 5. `experiments/robot/libero/run_libero_eval.py`

- Initialize `warm_start_state = None` at episode start.
- Pass the cache only when `use_warm_start=True`.
- Replace it after each action-chunk prediction.
- Reset it at every episode boundary and on explicit safety conditions.
- Ensure warm-start is updated at policy-query boundaries, not every environment timestep.
- Extend prediction logs with warm-start fields.

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

Keep the existing fields:

```text
recurrent_iteration_count
max_recurrent_iteration
adaptive_stop
iteration_mse
final_mse
latency_ms
success
```

Add run-level comparisons for:

- mean, median, p90, and p95 recurrent iterations;
- mean, median, p90, and p95 action-inference latency;
- max-iteration hit rate;
- adaptive-stop rate;
- success rate;
- warm-start usage and reset rates.

## Implementation Phases

### Phase 0 — Freeze Baselines

- Confirm the branch starts from the latest convergence-logging commit.
- Record exact commands for cold-start Adaptive `1e-4`, `5e-4`, and `1e-3`.
- Preserve fixed K=8 and K=12 as reference baselines.

### Phase 1 — Add Disabled Warm-Start Plumbing

- Add configuration and call-chain parameters.
- Keep `use_warm_start=False`.
- Verify that outputs, K statistics, and success behavior are unchanged from the current branch.

Acceptance criteria:

- cold-start smoke test runs without return-value errors;
- existing JSONL records remain valid;
- fixed and adaptive recurrence both execute.

### Phase 2 — Implement S1 Cache

- Cache the first recurrent update state.
- Reuse it at the next policy query in the same episode.
- Reset cache at episode start.

Acceptance criteria:

- first query reports `warm_start_used=false`;
- later queries report `warm_start_used=true`;
- no cache crosses episode boundaries;
- tensor shape, dtype, and device are validated.

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
- logging files contain all required fields.

### Phase 4 — Paper Baseline Experiment

Use the same checkpoint, seeds, initial states, task ordering, and number of trials.

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
- mean and p95 inference latency;
- recurrent-iteration reduction relative to the matching cold baseline;
- failure and max-iteration-hit analysis.

### Phase 5 — Warm-State Ablation

Only after S1 is validated, compare:

- S1;
- midpoint state;
- three-fifths state;
- final SK state.

This isolates the trade-off between retaining reusable trajectory information and carrying stale, over-converged information from the previous observation.

### Phase 6 — Safety and Reset Ablation

Evaluate optional reset or blending policies:

- periodic cache reset;
- reset when proprioception changes sharply;
- random/cached state blending;
- fallback to cold-start after max-iteration hits.

These are secondary experiments and should not block the first paper-ready S1 result.

## Experimental Fairness Rules

- Use one code branch for both warm and cold runs.
- Change only warm-start-related flags between matched experiments.
- Use identical checkpoint, task order, seed, initial states, action horizon, and thresholds.
- Separate cold-start startup latency from steady-state latency.
- Report both success rate and compute reduction; do not claim optimization from K reduction alone.
- Treat 30 episodes as preliminary. Use more trials for the final paper table when compute permits.

## Immediate Next Actions

1. Implement configuration and disabled call-chain plumbing.
2. Integrate only S1 warm-start.
3. Add warm-start JSONL fields.
4. Run cold-start regression smoke tests.
5. Run matched cold/warm S1 smoke tests.
6. Expand to the LIBERO-Spatial benchmark after validation.
