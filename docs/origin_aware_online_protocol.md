# Origin-aware online evaluation protocol

The LIBERO runner keeps `evaluation_protocol_phase=legacy` as its default. The
frozen protocol is enabled only when one of `smoke`, `calibration`, `screening`,
or `final` is selected.

## Frozen initial states

The committed manifest is:

```text
experiments/robot/libero/manifests/libero_spatial_official_50_v1.json
```

For every LIBERO Spatial task it records:

- the SHA-256 of the exact serialized LIBERO initial-state file;
- the SHA-256 of the ordered tensors loaded from that file;
- a deterministic hash-ranked `10/10/30` calibration, screening, and final split.

The runner validates both hashes before creating the environment. A missing or
mismatched source file, state collection, task, or partition aborts the run.

To regenerate the manifest intentionally:

```bash
python scripts/build_libero_initial_state_manifest.py \
  --task-suite-name libero_spatial \
  --output experiments/robot/libero/manifests/libero_spatial_official_50_v1.json \
  --overwrite
```

If the repository-local LIBERO editable install is unavailable, prepend
`$PWD/LIBERO` to `PYTHONPATH` for this command. Regeneration changes the frozen
manifest SHA-256 and therefore requires updating its golden unit test.

## Phase contract

| Phase | State partition | Trials per task | Statistical use |
| --- | --- | ---: | --- |
| `smoke` | first three calibration IDs | 3 | excluded from fitting |
| `calibration` | calibration | 10 | OOF candidate fitting |
| `screening` | screening | 10 | online candidate screening |
| `final` | final | 30 | final paired comparison |

The trial count, `initial_states_path=DEFAULT`, LIBERO Spatial suite, manifest
path, and `reset_rng_each_episode=True` are enforced for every non-legacy phase.

Example smoke invocation arguments:

```bash
--task_suite_name libero_spatial \
--evaluation_protocol_phase smoke \
--initial_state_manifest_path experiments/robot/libero/manifests/libero_spatial_official_50_v1.json \
--initial_states_path DEFAULT \
--num_trials_per_task 3 \
--reset_rng_each_episode True
```

## Paired RNG and reset order

The episode seed is a deterministic hash of base seed, phase, task suite, task
ID, initial-state ID, and paired-trial ID. Condition/arm identity is not part of
the seed. Each paired arm therefore executes the following order independently:

1. restore Python, NumPy, PyTorch CPU, and CUDA RNGs;
2. call the environment seed API when available;
3. call `env.reset()`;
4. apply the manifest-selected fixed initial state;
5. initialize the action queue, warm cache, and episode counters;
6. start rollout.

Step logs and result JSON include the phase, paired-trial ID, initial-state ID,
episode seed, manifest SHA-256, source-file SHA-256, and state-content SHA-256.
The offline calibration parser rejects smoke, screening, and final records so
that smoke observations cannot be fitted accidentally.
