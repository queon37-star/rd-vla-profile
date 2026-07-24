#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="outputs/12_24-24_24_Spatial_40k"
RESULT_DIR="benchmark_results/warm_start_validation/phase3_task0_ep10"

mkdir -p "${RESULT_DIR}"

COMMON_ARGS=(
  --pretrained_checkpoint "${CHECKPOINT}"
  --task_suite_name libero_spatial
  --task_id 0
  --num_trials_per_task 10
  --seed 7
  --reset_rng_each_episode True
  --episode_seed_stride 1
  --use_recurrent True
  --recurrence_strategy kl_divergence
  --recurrence_max_iter 32
  --recurrence_kl_thresh 0.001
  --validate_warm_start_finite False
  --profile_coda_cost False
  --use_cached_final_output False
  --use_latent_precheck False
)

echo "=================================================="
echo "[1/3] Cold adaptive baseline"
echo "=================================================="

python experiments/robot/libero/run_libero_eval.py \
  "${COMMON_ARGS[@]}" \
  --use_warm_start False \
  --warm_start_source s1 \
  --warm_start_min_iter 2 \
  --run_id_note phase3_cold_adaptive_task0_ep10_seed7 \
  --json_log_file \
    "${RESULT_DIR}/cold_adaptive_task0_ep10_seed7.json"

echo "=================================================="
echo "[2/3] Warm S1 min_iter=2"
echo "=================================================="

python experiments/robot/libero/run_libero_eval.py \
  "${COMMON_ARGS[@]}" \
  --use_warm_start True \
  --warm_start_source s1 \
  --warm_start_min_iter 2 \
  --run_id_note phase3_warm_s1_min2_task0_ep10_seed7 \
  --json_log_file \
    "${RESULT_DIR}/warm_s1_min2_task0_ep10_seed7.json"

echo "=================================================="
echo "[3/3] Warm S1 min_iter=4"
echo "=================================================="

python experiments/robot/libero/run_libero_eval.py \
  "${COMMON_ARGS[@]}" \
  --use_warm_start True \
  --warm_start_source s1 \
  --warm_start_min_iter 4 \
  --run_id_note phase3_warm_s1_min4_task0_ep10_seed7 \
  --json_log_file \
    "${RESULT_DIR}/warm_s1_min4_task0_ep10_seed7.json"

echo "=================================================="
echo "Phase 3 Task 0 ep10 completed"
echo "Results: ${RESULT_DIR}"
echo "=================================================="
