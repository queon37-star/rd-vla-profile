#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT="outputs/12_24-24_24_Spatial_40k"
GPU_ID="${GPU_ID:-0}"

MEASURE_MODE="${MEASURE_MODE:-component}"
TRIALS="${TRIALS:-1}"
TASKS="${TASKS:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${MEASURE_MODE}" == "component" ]]; then
    PROFILE_CODA=True
elif [[ "${MEASURE_MODE}" == "latency" ]]; then
    PROFILE_CODA=False
else
    echo "Unsupported MEASURE_MODE=${MEASURE_MODE}"
    exit 1
fi

RUN_ROOT="benchmark_results/paper_stage0_v2/${MEASURE_MODE}_${RUN_TAG}"
mkdir -p "${RUN_ROOT}/console_logs"

echo "================================================"
echo "Measurement mode : ${MEASURE_MODE}"
echo "Trials per task  : ${TRIALS}"
echo "Tasks            : ${TASKS}"
echo "Profile Coda     : ${PROFILE_CODA}"
echo "Output root      : ${RUN_ROOT}"
echo "================================================"

run_method() {
    local method="$1"
    local warm_start="$2"
    local latent_precheck="$3"
    local cached_final="$4"

    mkdir -p "${RUN_ROOT}/${method}"

    for task_id in ${TASKS}; do
        echo "================================================"
        echo "Method=${method}, task=${task_id}"
        echo "Warm=${warm_start}"
        echo "Latent precheck=${latent_precheck}"
        echo "Cached final=${cached_final}"
        echo "================================================"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python experiments/robot/libero/run_libero_eval.py \
          --pretrained_checkpoint "${CHECKPOINT}" \
          --task_suite_name libero_spatial \
          --task_id "${task_id}" \
          --num_trials_per_task "${TRIALS}" \
          --seed 7 \
          --reset_rng_each_episode True \
          --episode_seed_stride 1 \
          --use_recurrent True \
          --recurrence_strategy kl_divergence \
          --recurrence_kl_thresh 0.001 \
          --recurrence_max_iter 32 \
          --use_warm_start "${warm_start}" \
          --warm_start_source midpoint \
          --warm_start_min_iter 2 \
          --validate_warm_start_finite False \
          --use_latent_precheck "${latent_precheck}" \
          --latent_precheck_thresh 0.2 \
          --latent_precheck_min_iter 2 \
          --latent_precheck_force_interval 0 \
          --use_cached_final_output "${cached_final}" \
          --profile_coda_cost "${PROFILE_CODA}" \
          --profile_pytorch False \
          --profile_timing_summary False \
          --profile_timing_cuda_sync False \
          --num_exec_actions 5 \
          --adaptive_exec False \
          --dynamic_exec False \
          --use_linear_decay_horizon False \
          --json_log_file "${RUN_ROOT}/${method}/task${task_id}.json" \
          --step_log_file "${RUN_ROOT}/${method}/task${task_id}_steps.jsonl" \
          > "${RUN_ROOT}/console_logs/${method}_task${task_id}.log" 2>&1

        echo "Finished ${method}, task ${task_id}"
    done
}

# 1. Original adaptive inference
run_method "adaptive_base" False False True

# 2. Warm-start only
run_method "warm_only" True False True

# 3. Coda optimization only
run_method "coda_only" False True True

# 4. Warm-start + Coda optimization
run_method "full" True True True

echo "================================================"
echo "All runs finished."
echo "Results: ${RUN_ROOT}"
echo "================================================"
