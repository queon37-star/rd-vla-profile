#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="outputs/12_24-24_24_Spatial_40k"
ROOT_DIR="benchmark_results/coda_warm_2x2"
EVAL_DIR="${ROOT_DIR}/task0_ep10"
PROFILE_DIR="${ROOT_DIR}/task0_profile_ep3"

mkdir -p "${EVAL_DIR}" "${PROFILE_DIR}"

COMMON_ARGS=(
  --pretrained_checkpoint "${CHECKPOINT}"
  --task_suite_name libero_spatial
  --task_id 0
  --seed 7
  --reset_rng_each_episode True
  --episode_seed_stride 1
  --use_recurrent True
  --recurrence_strategy kl_divergence
  --recurrence_max_iter 32
  --recurrence_kl_thresh 0.001
  --validate_warm_start_finite False
  --profile_pytorch False
  --profile_timing_summary False
  --latent_precheck_thresh 0.2
  --latent_precheck_min_iter 2
  --latent_precheck_force_interval 0
)

run_case() {
  local trials="$1"
  local profile_coda="$2"
  local result_dir="$3"
  local name="$4"
  local use_warm="$5"
  local warm_min="$6"
  local use_cached="$7"
  local use_precheck="$8"

  echo
  echo "=================================================="
  echo "Running: ${name}"
  echo "trials=${trials}"
  echo "profile_coda=${profile_coda}"
  echo "warm=${use_warm}"
  echo "cached_final=${use_cached}"
  echo "latent_precheck=${use_precheck}"
  echo "=================================================="

  python experiments/robot/libero/run_libero_eval.py \
    "${COMMON_ARGS[@]}" \
    --num_trials_per_task "${trials}" \
    --profile_coda_cost "${profile_coda}" \
    --use_warm_start "${use_warm}" \
    --warm_start_source s1 \
    --warm_start_min_iter "${warm_min}" \
    --use_cached_final_output "${use_cached}" \
    --use_latent_precheck "${use_precheck}" \
    --run_id_note "${name}" \
    --json_log_file "${result_dir}/${name}.json"
}

# ==================================================
# Phase A: 10-episode functional and latency test
# ==================================================

run_case \
  10 False "${EVAL_DIR}" \
  "phase4_baseline_adaptive_task0_ep10_seed7" \
  False 2 False False

run_case \
  10 False "${EVAL_DIR}" \
  "phase4_coda_only_task0_ep10_seed7" \
  False 2 True True

run_case \
  10 False "${EVAL_DIR}" \
  "phase4_warm_only_task0_ep10_seed7" \
  True 4 False False

run_case \
  10 False "${EVAL_DIR}" \
  "phase4_coda_warm_task0_ep10_seed7" \
  True 4 True True

# ==================================================
# Phase B: 3-episode Coda component profiling
# ==================================================

run_case \
  3 True "${PROFILE_DIR}" \
  "phase4_profile_baseline_task0_ep3_seed7" \
  False 2 False False

run_case \
  3 True "${PROFILE_DIR}" \
  "phase4_profile_coda_only_task0_ep3_seed7" \
  False 2 True True

run_case \
  3 True "${PROFILE_DIR}" \
  "phase4_profile_warm_only_task0_ep3_seed7" \
  True 4 False False

run_case \
  3 True "${PROFILE_DIR}" \
  "phase4_profile_coda_warm_task0_ep3_seed7" \
  True 4 True True

echo
echo "=================================================="
echo "Phase 4 Coda + Warm-start 2x2 completed"
echo "Functional results: ${EVAL_DIR}"
echo "Profile results:    ${PROFILE_DIR}"
echo "=================================================="
