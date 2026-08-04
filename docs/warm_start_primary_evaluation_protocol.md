# Warm-Start Primary Evaluation Protocol

Status: **planning freeze v1; final run is not authorized**  
Branch: `experiment/warm-start-primary-evaluation`  
Base: `experiment/scalar-conservative-runtime-screening`

## 1. Purpose

This document freezes the research question, paired comparison, success analysis,
latency wording, and data-reuse constraints for the midpoint warm-start study
before any new confirmatory rollout is started.

The primary question is:

> While preserving RD-VLA's adaptive adjacent action-MSE stopping rule, does
> midpoint latent-state reuse preserve closed-loop task success and reduce
> recurrent computation and measured policy-query cost relative to
> cold initialization at every prediction?

This protocol does **not** evaluate a fixed-depth policy. Fixed-depth runs remain
mechanism-isolation controls only. It also does not revive latent-only or scalar
stopping as a candidate method.

## 2. Compared arms

Only the initial latent-state source differs between the two arms.

### 2.1 Cold-initialized adaptive baseline

```text
use_recurrent = true
recurrence_strategy = adjacent_action_mse
recurrence_kl_thresh = 0.001
recurrence_max_iter = 32
use_warm_start = false
use_cached_final_output = true
```

Every prediction starts from the checkpoint's original random truncated-normal
state. The adaptive stopping criterion remains consecutive action-output MSE.

### 2.2 Midpoint warm-start adaptive policy

```text
use_recurrent = true
recurrence_strategy = adjacent_action_mse
recurrence_kl_thresh = 0.001
recurrence_max_iter = 32
use_warm_start = true
warm_start_source = midpoint
warm_start_min_iter = 2
validate_warm_start_finite = true
use_cached_final_output = true
```

The first prediction of every episode has no prior latent and therefore uses the
same cold initialization as the baseline. Later predictions may reuse the
midpoint state selected from the preceding prediction. The midpoint source is
the implementation-defined state returned by `select_warm_start_candidate`; it
must not be redefined after the evaluation begins.

### 2.3 Frozen common configuration

```text
checkpoint = outputs/12_24-24_24_Spatial_40k
sync_checkpoint_source_config = false
task_suite_name = libero_spatial
task_ids = 0..9
num_exec_actions = 5
reset_rng_each_episode = true
recurrence_strategy = adjacent_action_mse
recurrence_kl_thresh = 0.001
recurrence_max_iter = 32
use_cached_final_output = true
use_latent_precheck = false
latent_precheck_mode = off
latent_precheck_trace_level = off
shadow_full_depth = false
collect_preconvergence_raw_shadow = false
adaptive_exec = false
dynamic_exec = false
use_linear_decay_horizon = false
scalar policy = disabled
```

The checkpoint digest, source commit, environment versions, initial-state
manifest digest, and final paired-seed namespace must be recorded before the
formal run.

## 3. Runtime-contract preflight

A development preflight must finish before any final data are collected. It
must fail closed unless all of the following hold.

1. The checkpoint snapshot is identical before and after the run.
2. The two arms receive identical task IDs, initial-state IDs, paired-trial IDs,
   and episode seeds.
3. Arm identity is absent from the paired episode-seed derivation.
4. The first prediction is cold in both arms.
5. In the warm arm, later eligible predictions report a provided and used
   midpoint state unless a documented finite-validation reset occurs.
6. In the baseline arm, every prediction reports random initialization and no
   cached state use.
7. Latent pre-check, scalar stopping, shadow recurrence, adaptive action
   execution, and other inactive mechanisms remain disabled.
8. Adjacent action-MSE uses one output decode per recurrent iteration and
   returns the cached terminal output without an extra duplicate decode.
9. `Coda/get_output call count = terminal K` for every normal prediction.
10. No non-finite state, numerical retry, protocol violation, process failure,
    or missing episode metadata is accepted.

A finite-validation reset in the warm arm is not silently counted as successful
warm reuse. Its frequency and reason must be reported.

## 4. Data-reuse declaration

The existing official 50 LIBERO Spatial initial states per task were already
partitioned into calibration, screening, and final subsets for the scalar
stopping study. Those partitions have been consumed. Consequently, a future
warm-start result must not be described as using an "untouched initial-state
partition."

The currently admissible design is a **fresh, pre-registered paired RNG
replicate over the frozen official initial states**, subject to all of the
following constraints:

- policy and analysis are frozen before outcome inspection;
- a new warm-start-specific paired-seed namespace is used;
- each selected official initial state appears at most once per arm in the
  primary confirmatory analysis;
- both arms receive the same initial state and episode seed;
- no scalar-study result is used to choose individual warm-start test states;
- the reuse of official states is disclosed explicitly.

This design provides at most 50 unique pairs per task. If the power analysis
requires more than 50 pairs per task, the study must obtain new initial states
or pre-register a clustered repeated-state analysis. Repeating the same 50
states as if they were independent pairs is prohibited.

## 5. Primary success estimand and analysis

Let

```text
p01 = P(cold baseline fails, warm-start succeeds)
p10 = P(cold baseline succeeds, warm-start fails)
```

The primary estimand is the pooled paired success difference across the ten
fixed LIBERO Spatial tasks:

```text
Delta = success(warm-start) - success(cold baseline) = p01 - p10
```

Every task receives the same number of paired trials. The task set is treated as
the fixed benchmark population rather than a random sample of tasks.

The planned primary analysis is:

1. compute the pooled paired success difference;
2. resample paired trials independently within each task while keeping all ten
   tasks in every bootstrap replicate;
3. form a one-sided 95% percentile-bootstrap lower bound;
4. declare success preservation only when the lower bound is strictly greater
   than the pre-registered negative non-inferiority margin.

The final margin has not yet been selected. Candidate planning margins are
`-5`, `-3`, and `-2` percentage points. The margin must be justified and frozen
before the final paired-seed namespace and trial plan are generated.

Secondary success reporting includes:

- `both success`, `cold-only success`, `warm-only success`, and `both failure`;
- exact McNemar testing as a descriptive paired diagnostic;
- task-level success differences and intervals as descriptive results;
- a pre-registered catastrophic-regression guardrail, if adopted.

The protocol does not claim formal non-inferiority separately for every task
unless a new task-level multiplicity and sample-size design is written first.

## 6. Power planning

Use:

```bash
python scripts/simulate_warm_start_power.py \
  --output benchmark_results/warm_start_power/planning_v1.json
```

The planner simulates the same fixed-task, within-task paired bootstrap intended
for the primary analysis. It requires directional discordance, not only total
discordance.

The default planning grid varies:

```text
pairs per task: 10, 20, 30, 40, 50
total discordance p01+p10: 3%, 5%, 8%, 10%
true paired difference p01-p10: 0, -1%, -2%
non-inferiority margin: 5%, 3%, 2%
```

A warm-start-specific development pilot or an existing methodologically valid
paired warm-start dataset should be used to construct task-level `p01/p10`
profiles. Pilot outcomes are for planning only and are excluded from the final
claim. The selected sample size must report Monte Carlo uncertainty and should
be checked at both 80% target power and 90% sensitivity power.

The current official-state limit is built into the planner: a request above 50
pairs per task is rejected because it would require a repeated-state clustered
design not represented by the script.

## 7. Efficiency outcomes

Success preservation is the primary gate. Efficiency is interpreted only after
that gate and must be reported at both prediction and episode levels.

Required outcomes are:

- recurrent iterations per prediction;
- recurrent calls per episode;
- output/Coda calls per prediction and per episode;
- policy predictions per episode;
- synchronized policy-query latency per prediction;
- summed inference time per episode;
- environment steps per episode;
- success-stratified descriptive summaries, clearly labeled secondary.

Episode inference time is the sum of the observed prediction latencies. The
analysis may describe a trade-off between lower per-prediction latency and more
predictions per episode, but it must not claim a causal additive decomposition
into success, episode length, and re-query effects.

## 8. Latency scopes

Two distinct timer scopes already exist and must not be merged.

### 8.1 Online rollout metric

The current `run_libero_eval.py` `latency_ms` synchronizes CUDA before and after
`get_action`. That call includes image collection/preparation, processor work,
VLM prediction, the action policy/action head, and action post-processing.
Accordingly, this metric should be described as:

```text
synchronized online policy-query latency
```

It is not the existing post-VLM action-head microprofile.

### 8.2 Fixed-Coda mechanism microprofile

The fixed-Coda microprofile synchronizes around the action-head call after VLM
features are available. It is a post-VLM action-head mechanism measurement and
must remain separate from online rollout latency.

Before the final warm-start evaluation, choose and freeze one of the following:

1. report the existing synchronized online policy-query latency for the paired
   rollout and keep the action-head microprofile as a separate mechanism study;
2. add and validate a dedicated synchronized action-head timer to the online
   runner, then report both scopes separately.

No unmeasured VLM-inclusive percentage may be inferred by adding averages from
unmatched timing experiments.

## 9. Explicit exclusions

The following are outside the primary method and may not be introduced after
outcome inspection:

- fixed recurrence depth as a deployment candidate;
- task-specific warm-start rules;
- proprioception or visual staleness gates;
- noise-interpolation coefficients;
- latent-only, scalar, or learned-probe stopping;
- latent pre-check scheduling;
- adaptive action-execution horizon changes;
- threshold, midpoint, or minimum-iteration retuning based on final outcomes.

The scalar/latent stopping work remains a separate post-hoc negative-result
analysis. It does not determine this warm-start policy.

## 10. Final-run authorization checklist

The final paired rollout remains unauthorized until all fields below are frozen
and recorded in a machine-readable manifest:

- source commit;
- checkpoint snapshot digest;
- environment/runtime identity;
- initial-state allocation and disclosure of prior use;
- fresh paired-seed namespace;
- chosen non-inferiority margin;
- target power and task-level `p01/p10` planning provenance;
- selected pairs per task;
- primary bootstrap implementation and aggregation contract;
- catastrophic-regression guardrail, or an explicit decision not to use one;
- selected online latency boundary;
- successful runtime-contract preflight.

No final result should be interpreted before this checklist is complete.
