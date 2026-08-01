#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-outputs/12_24-24_24_Spatial_40k}"
MANIFEST="${MANIFEST:-experiments/robot/libero/manifests/libero_spatial_official_50_v1.json}"
GPU_ID="${GPU_ID:-0}"
BASE_SEED="${BASE_SEED:-7}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-benchmark_results/origin_aware_smoke/${RUN_TAG}}"

if [[ ! -f "${CHECKPOINT}/model.safetensors" ]]; then
    echo "Missing checkpoint weights: ${CHECKPOINT}/model.safetensors" >&2
    exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing initial-state manifest: ${MANIFEST}" >&2
    exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
    echo "Refusing to overwrite existing smoke run: ${RUN_ROOT}" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}"
if [[ -d "${REPO_ROOT}/LIBERO" ]]; then
    export PYTHONPATH="${REPO_ROOT}/LIBERO${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "=================================================="
echo "Origin-aware calibration smoke gate"
echo "Checkpoint : ${CHECKPOINT}"
echo "Manifest   : ${MANIFEST}"
echo "GPU        : ${GPU_ID}"
echo "Base seed  : ${BASE_SEED}"
echo "Tasks      : 0 1 2 3 4 5 6 7 8 9"
echo "Episodes   : 3 per task (calibration partition only)"
echo "Output     : ${RUN_ROOT}"
echo "=================================================="

for task_id in 0 1 2 3 4 5 6 7 8 9; do
    task_dir="${RUN_ROOT}/task${task_id}"
    mkdir -p "${task_dir}/local_logs"
    echo "[$(date --iso-8601=seconds)] starting task ${task_id}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${CHECKPOINT}" \
      --task_suite_name libero_spatial \
      --task_id "${task_id}" \
      --evaluation_protocol_phase smoke \
      --initial_state_manifest_path "${MANIFEST}" \
      --initial_states_path DEFAULT \
      --num_trials_per_task 3 \
      --seed "${BASE_SEED}" \
      --reset_rng_each_episode True \
      --episode_seed_stride 1 \
      --use_recurrent True \
      --recurrence_strategy adjacent_action_mse \
      --recurrence_kl_thresh 0.001 \
      --recurrence_max_iter 32 \
      --use_warm_start True \
      --warm_start_source midpoint \
      --warm_start_min_iter 2 \
      --validate_warm_start_finite True \
      --use_cached_final_output True \
      --use_latent_precheck False \
      --latent_precheck_mode "'off'" \
      --latent_precheck_trace_level "'off'" \
      --latent_precheck_min_iter 2 \
      --nonfinite_policy legacy \
      --shadow_full_depth True \
      --profile_coda_cost False \
      --profile_pytorch False \
      --profile_timing_summary False \
      --num_exec_actions 5 \
      --adaptive_exec False \
      --dynamic_exec False \
      --use_linear_decay_horizon False \
      --use_wandb False \
      --run_id_note "origin_aware_smoke_task${task_id}" \
      --save_version origin_aware_smoke \
      --local_log_dir "${task_dir}/local_logs" \
      --json_log_file "${task_dir}/result.json" \
      --step_log_file "${task_dir}/steps.jsonl" \
      --recurrent_convergence_log_file "${task_dir}/steps.jsonl" \
      --recurrent_convergence_summary_file "${task_dir}/summary.json" \
      > "${task_dir}/console.log" 2>&1

    "${PYTHON_BIN}" scripts/validate_origin_aware_smoke.py \
      --run-root "${RUN_ROOT}" \
      --manifest "${MANIFEST}" \
      --base-seed "${BASE_SEED}" \
      --task-ids "${task_id}" \
      --output "${task_dir}/validation.json" \
      > "${task_dir}/validation.log"
    echo "[$(date --iso-8601=seconds)] task ${task_id} passed"
done

"${PYTHON_BIN}" scripts/validate_origin_aware_smoke.py \
  --run-root "${RUN_ROOT}" \
  --manifest "${MANIFEST}" \
  --base-seed "${BASE_SEED}" \
  --output "${RUN_ROOT}/validation.json" \
  > "${RUN_ROOT}/validation.log"

echo "=================================================="
echo "Smoke gate passed: ${RUN_ROOT}"
echo "Validation report: ${RUN_ROOT}/validation.json"
echo "=================================================="
