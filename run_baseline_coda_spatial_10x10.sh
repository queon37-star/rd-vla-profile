#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT="outputs/12_24-24_24_Spatial_40k"
GPU_ID="${GPU_ID:-0}"

# component: Coda component profiling enabled
# latency:   end-to-end latency measurement (Coda profiling disabled)
MEASURE_MODE="${MEASURE_MODE:-latency}"
TRIALS="${TRIALS:-10}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
LATENT_PRECHECK_THRESH="${LATENT_PRECHECK_THRESH:-0.2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PROFILE_TIMING_SUMMARY="${PROFILE_TIMING_SUMMARY:-False}"
PROFILE_TIMING_CUDA_SYNC="${PROFILE_TIMING_CUDA_SYNC:-False}"
PROFILE_TIMING_STEPS="${PROFILE_TIMING_STEPS:-100000}"

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
echo "Measurement mode          : ${MEASURE_MODE}"
echo "Trials per task           : ${TRIALS}"
echo "Tasks                     : ${TASKS}"
echo "Latent pre-check threshold: ${LATENT_PRECHECK_THRESH}"
echo "Profile Coda              : ${PROFILE_CODA}"
echo "Output root               : ${RUN_ROOT}"
echo "Timing summary            : ${PROFILE_TIMING_SUMMARY}"
echo "Timing CUDA sync          : ${PROFILE_TIMING_CUDA_SYNC}"
echo "Timing steps              : ${PROFILE_TIMING_STEPS}"
echo "================================================"

run_method() {
    local method="$1"
    local latent_precheck="$2"
    local cached_final="$3"

    mkdir -p "${RUN_ROOT}/${method}"

    for task_id in ${TASKS}; do
        echo "================================================"
        echo "Method=${method}, task=${task_id}"
        echo "Warm=False"
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
          --use_warm_start False \
          --warm_start_source midpoint \
          --warm_start_min_iter 2 \
          --validate_warm_start_finite False \
          --use_latent_precheck "${latent_precheck}" \
          --latent_precheck_thresh "${LATENT_PRECHECK_THRESH}" \
          --latent_precheck_min_iter 2 \
          --latent_precheck_force_interval 0 \
          --use_cached_final_output "${cached_final}" \
          --profile_coda_cost "${PROFILE_CODA}" \
          --profile_pytorch False \
          --profile_timing_summary "${PROFILE_TIMING_SUMMARY}" \
          --profile_timing_summary_path "${RUN_ROOT}/${method}/task${task_id}_timing.jsonl" \
          --profile_timing_steps "${PROFILE_TIMING_STEPS}" \
          --profile_timing_cuda_sync "${PROFILE_TIMING_CUDA_SYNC}" \
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

# 1. Baseline
#    No warm start and no latent pre-check.
#    Cached final output is enabled as a common implementation correction.
run_method "adaptive_base" False True

# 2. Coda optimization only
#    Latent pre-check is enabled with threshold 0.2.
#    Cached final output remains enabled as a common condition.
run_method "coda_only" True True

echo "================================================"
echo "All runs finished."
echo "Results: ${RUN_ROOT}"
echo "================================================"
