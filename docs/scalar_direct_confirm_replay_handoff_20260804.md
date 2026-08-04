# Scalar Direct/Confirm Replay 인수인계

작성일: 2026-08-04  
대상 브랜치: `experiment/scalar-direct-confirm-replay`  
구현 기준 커밋: `184a902a13d3b9de4ccdb3d6e1cf95e017e64939`  
저장소: `queon37-star/rd-vla-profile`

---

## 1. 문서 목적

이 문서는 RD-VLA의 반복 Coda 호출을 latent dynamics 기반 stopping으로 줄이기 위해 진행한 scalar direct/confirm-next 실험의 현재 상태를 인수인계한다.

현재까지의 핵심 결론은 다음과 같다.

> Scalar confirm-next는 반복 Coda 호출을 크게 줄였지만, 최종 300-pair 평가에서 Action-MSE 대비 성공률을 보존하지 못했다. 또한 관측된 prediction latency 감소에는 Coda 감소뿐 아니라 recurrence depth 감소가 함께 섞여 있으므로, 다음 작업은 동일 K 조건에서 Coda 제거 효과를 분리 측정하는 것이다.

이 문서를 읽은 뒤에는 이전 대화 기록을 다시 확인하지 않고도 다음 실험을 시작할 수 있어야 한다.

---

## 2. 연구 질문

### 2.1 원래 질문

RD-VLA의 기본 adaptive stopping은 각 recurrent iteration마다 Coda를 실행하여 consecutive action output의 MSE를 계산한다.

연구 질문은 다음과 같다.

> 매 iteration마다 action을 decode하지 않고 latent dynamics만으로 convergence를 판단하여, task success를 유지하면서 Coda 연산과 inference latency를 줄일 수 있는가?

### 2.2 최종 비교 정책

최종 평가는 다음 두 정책을 비교했다.

1. **Action-MSE baseline**
   - corrected/default adjacent action-MSE stopping
   - 각 recurrent iteration마다 Coda 실행
   - threshold: `0.001`
   - loop 안의 마지막 action output을 그대로 반환하여 duplicate final Coda를 실행하지 않음

2. **Scalar confirm-next**
   - COLD prediction에서는 Action-MSE fallback 사용
   - ACTUAL_WARM prediction에서는 latent scalar score만 계산
   - 처음 threshold를 만족한 iteration을 gate로 기록
   - 정확히 한 번 더 recurrence를 수행
   - terminal latent state에서 Coda를 한 번만 실행

Scalar direct는 screening 단계에서 비교했지만 최종 primary 후보에서는 제외했다. Fixed-depth는 secondary ablation 후보이며 primary 비교가 아니다.

---

## 3. 저장소와 주요 커밋

현재 작업 브랜치:

```text
experiment/scalar-direct-confirm-replay
```

주요 구현 커밋:

| Commit | 내용 |
|---|---|
| `5e21dc2f185f0d6e7ecdf078fa1f040fe312d119` | Task-OOF scalar recurrence stopping 구현 |
| `240c2f128ba992c38b03e05c4992d5be234af3a3` | Read-only checkpoint evaluation mode 추가 |
| `2f6de805e86344de3245a4a16efc179bf8d8e714` | Terminal-only fixed recurrence control 추가 |
| `184a902a13d3b9de4ccdb3d6e1cf95e017e64939` | Scalar COLD fallback에서 terminal action output 재사용 |

테스트 상태:

```text
407 passed, 3 warnings
```

최종 평가 전후 repository worktree와 checkpoint mirror는 clean이었다.

---

## 4. Runtime 정책 계약

### 4.1 Action-MSE baseline

각 prediction에서:

```text
Recurrent core -> Coda -> action
Recurrent core -> Coda -> action
...
Adjacent action MSE < 0.001이면 종료
```

정상 계약:

```text
Coda call count = K
get_output call count = K
returned_cached_final_output = true
```

마지막 loop output을 재사용하므로 별도의 duplicate final Coda는 없다. 이 수정은 correctness/default behavior이며 연구 기여로 주장하지 않는다.

### 4.2 Scalar confirm-next

#### COLD prediction

Warm state가 없으므로 corrected Action-MSE fallback을 사용한다.

정상 계약:

```text
scalar_policy_applied = false
scalar_policy_score_call_count = 0
scalar_policy_gate_iteration = null
Coda call count = K
get_output call count = K
returned_cached_final_output = true
```

#### ACTUAL_WARM prediction

```text
latent feature 계산
-> scalar threshold 검사
-> 첫 trigger를 gate k로 기록
-> recurrence 한 번 추가
-> terminal K = gate + 1
-> Coda 한 번 실행
```

정상 계약:

```text
scalar_policy_applied = true
scalar_policy_execution_mode = confirm_next
terminal K = gate K + 1
Coda call count = 1
get_output call count = 1
```

Non-finite state에서는 fail-closed 동작을 유지한다.

---

## 5. Scalar policy와 calibration artifact

### 5.1 Scalar features

현재 scalar model은 다음 7개 feature를 사용한다.

```text
iteration_k
delta_rms
previous_delta_rms
relative_delta_rms
delta_ratio
delta_cosine
second_difference_rms
```

### 5.2 Calibration raw data

Formal raw root:

```text
benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7
```

규모:

```text
10 tasks x 10 episodes
109 shards
2398 predictions
2298 warm predictions
100 cold predictions
1880 feature-applicable predictions
518 history-unavailable predictions
```

Dataset SHA:

```text
e0fe73e606167939cf559f45802ea74d880ab3442738870cf65a8c27f2829ad0
```

Boundary OOF root:

```text
benchmark_results/preconvergence_trigger/seed7/boundary_latent_oof_v1
```

Boundary dataset SHA:

```text
defd0b...
```

Boundary bundle SHA:

```text
98c0ae...
```

### 5.3 Runtime scalar artifact

Artifact path:

```text
benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_kfirst_v1/scalar_policy.pt
```

Artifact SHA256:

```text
e1d981f91fd8be729a73ea94b6bd56c30232ac2a3c18f85f7367b80e81fb5fef
```

Task-level OOF fold map:

```text
{0:0, 1:1, 2:2, 3:3, 4:4, 5:4, 6:3, 7:2, 8:1, 9:0}
```

Task별 runtime threshold:

| Task | Fold | Threshold |
|---:|---:|---:|
| 0 | 0 | 0.10759558528661728 |
| 1 | 1 | 0.10642359405755997 |
| 2 | 2 | 0.05942179262638092 |
| 3 | 3 | 0.10363844782114029 |
| 4 | 4 | 0.08715094625949860 |
| 5 | 4 | 0.08715094625949860 |
| 6 | 3 | 0.10363844782114029 |
| 7 | 2 | 0.05942179262638092 |
| 8 | 1 | 0.10642359405755997 |
| 9 | 0 | 0.10759558528661728 |

OOF scalar 성능:

```text
AUC = 0.7882
task macro AUC = 0.7983
minimum task AUC = 0.7533
```

이 수치는 action-space convergence boundary 예측 성능이며 task success 보존을 직접 의미하지 않는다.

---

## 6. Checkpoint와 평가 프로토콜

Checkpoint:

```text
outputs/12_24-24_24_Spatial_40k
```

Checkpoint 파일 수:

```text
341
```

Config SHA256:

```text
56421ebf621a0a70dbe0e19bf3578d16e35ce76e354864cc88dbe2836ae1d935
```

평가 시 반드시 다음을 유지한다.

```text
sync_checkpoint_source_config=False
```

Checkpoint snapshot digest:

```text
e1b2dd3b3ce55511cb8efc665fc06d3ce11689cd200035a64b7f04d3731620a6
```

Initial-state manifest:

```text
experiments/robot/libero/manifests/libero_spatial_official_50_v1.json
```

Manifest SHA256:

```text
0e3c6609b719d6b0a05f79efd769dff67141b52d00b42d9e0bea904ecf493144
```

환경 특이사항:

```text
NUMBA_DISABLE_JIT=1
```

LIBERO 실행에서 Numba cache locator 문제를 피하기 위해 사용했다.

---

## 7. Smoke와 screening 결과

### 7.1 Task 0 3-way smoke

Root:

```text
benchmark_results/scalar_latent_3way_paired_smoke/20260804_072115_head2f6de80_task0_seed7
```

결과:

```text
Action-MSE: 3/3 success
Scalar direct: 3/3 success
Scalar confirm-next: 3/3 success
```

Smoke 결과는 기능 확인용이며 성능 결론에 사용하지 않는다.

### 7.2 10 tasks x 3 episodes screening

Root:

```text
benchmark_results/scalar_latent_3way_screening/20260804_073711_head184a902_spatial10_seed7
```

규모:

```text
10 tasks x 3 episodes x 3 policies
90 total episodes
30 GPU processes
0 process failures
0 runtime contract violations
```

성공:

```text
Action-MSE: 28/30
Scalar direct: 25/30
Scalar confirm-next: 28/30
```

Screening에서 confirm-next를 final 후보로 선택했지만 이 결과는 screening일 뿐이며 formal success-preservation 결론에 사용하지 않는다.

Screening 주요 효율 수치:

| Metric | Action-MSE | Direct | Confirm-next |
|---|---:|---:|---:|
| Predictions | 713 | 881 | 768 |
| Mean K | 5.271 | 3.356 | 4.214 |
| Total Coda | 3758 | 1025 | 912 |
| Coda/prediction | 5.271 | 1.163 | 1.188 |
| Prediction latency mean | 135.1 ms | 123.5 ms | 126.5 ms |
| Episode inference mean | 3211.4 ms | 3627.4 ms | 3238.9 ms |

---

## 8. 최종 평가

### 8.1 Final protocol

정책:

```text
Action-MSE baseline vs Scalar confirm-next
```

규모:

```text
LIBERO Spatial
10 tasks
30 paired episodes per task
300 paired trials
600 policy episodes total
```

Primary outcome:

```text
paired episode success difference = confirm-next - Action-MSE
```

사전 고정 practical preservation 기준:

```text
one-sided 95% lower bound > -5 percentage points
```

Final trial plan SHA:

```text
c99d5eccc5256ad3bbb32af32b8d49f1560015ce640db7fb853abc84984ffa19
```

Final root:

```text
benchmark_results/scalar_latent_final_evaluation/20260804_094502_head184a902_spatial10_seed7
```

생성된 집계 파일:

```text
aggregate_report.json
aggregate_report.md
episode_level.csv
discordant_pairs.csv
```

### 8.2 Execution validity

```text
20/20 processes completed
600/600 episode metadata
14868 step metadata records
0 process failures
0 protocol violations
0 aggregation contract violations
checkpoint before/after exact match
repository worktree clean
```

### 8.3 Overall result

| Policy | Success | Rate | Predictions/Ep | Mean K | Warm K | Recurrences/Ep | Coda/Pred | Coda/Ep | Prediction latency | Episode latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Action-MSE | 273/300 | 91.00% | 23.883 | 5.154 | 5.123 | 123.100 | 5.154 | 123.100 | 119.87 ms | 2862.98 ms |
| Confirm-next | 262/300 | 87.33% | 25.677 | 4.238 | 4.172 | 108.813 | 1.190 | 30.550 | 113.10 ms | 2904.13 ms |

### 8.4 Paired success result

```text
Success difference: -3.67 percentage points
Task-stratified paired bootstrap 95% CI: [-6.00, -1.67] pp
One-sided 95% lower bound: -5.67 pp
Predeclared margin: -5.00 pp
Preservation criterion met: false
Exact McNemar two-sided p = 0.003418
```

Paired outcomes:

| Outcome | Count |
|---|---:|
| Both success | 261 |
| Action-MSE only | 12 |
| Confirm-next only | 1 |
| Both failure | 26 |

현재 confirm-next는 Action-MSE 대비 task success를 보존했다고 주장할 수 없다.

### 8.5 Task-level result

| Task | Action-MSE | Confirm-next | Difference |
|---:|---:|---:|---:|
| 0 | 29/30 | 27/30 | -6.67 pp |
| 1 | 30/30 | 28/30 | -6.67 pp |
| 2 | 30/30 | 30/30 | 0.00 pp |
| 3 | 30/30 | 30/30 | 0.00 pp |
| 4 | 27/30 | 26/30 | -3.33 pp |
| 5 | 10/30 | 5/30 | -16.67 pp |
| 6 | 30/30 | 30/30 | 0.00 pp |
| 7 | 30/30 | 30/30 | 0.00 pp |
| 8 | 27/30 | 26/30 | -3.33 pp |
| 9 | 30/30 | 30/30 | 0.00 pp |

Task 5가 가장 큰 손실을 만들었지만 Task 0, 1, 4, 8에서도 confirm-next가 낮았다. 따라서 문제를 Task 5 하나로만 설명할 수 없다.

---

## 9. Confirm-next 동작 진단

### 9.1 K=4 집중

Confirm-next warm prediction 수:

```text
7403
```

Gate histogram:

```text
{3: 6482, 4: 624, 5: 251, 6: 41, 7: 4, 8: 1}
```

Terminal K histogram:

```text
{4: 6482, 5: 624, 6: 251, 7: 41, 8: 4, 9: 1}
```

따라서 warm prediction의 약 87.6%가 terminal K=4에서 종료됐다.

Action-MSE warm K:

```text
mean = 5.123
median = 5
p95 = 9
```

Confirm-next warm K:

```text
mean = 4.172
median = 4
p95 = 5
```

현재 confirm-next는 state-dependent depth allocation을 충분히 유지하지 못하고 사실상 K=4 부근에 집중됐다.

### 9.2 현재 가능한 원인 가설

아래는 확정 결론이 아니라 후속 검증이 필요한 가설이다.

#### 가설 A: 어려운 calibration 사례 부족

Calibration은 task당 10 episodes였다. 깊은 K가 필요한 다음 상태가 충분히 포함되지 않았을 수 있다.

```text
grasp 직전 정밀 정렬
접촉 이후 자세 복구
실패 직전 불안정한 trajectory
희귀한 어려운 initial state
이전 action error가 누적된 state
```

이 경우 runtime state machine을 바꾸지 않고 어려운 development data를 추가한 뒤 같은 7-feature scalar model을 재학습할 수 있다.

#### 가설 B: Feature 또는 학습 target 한계

현재 scalar model의 target은 action-space convergence boundary `K_first`이다. 이는 다음 질문과 다르다.

```text
이 latent state가 안정적인가?
```

대

```text
이 깊이에서 생성한 action이 task success를 보존할 만큼 올바른가?
```

데이터를 늘려도 두 대상의 차이 때문에 한계가 남을 수 있다.

현재 단계에서는 복잡한 fallback이나 별도 verification head를 먼저 추가하지 않는다. 먼저 Coda 최적화 자체의 가치와 calibration 난이도 분포를 분리해서 확인한다.

---

## 10. Latency 해석에서 반드시 주의할 점

Final 결과에서 confirm-next prediction latency는 다음과 같이 감소했다.

```text
119.87 ms -> 113.10 ms
5.65% reduction
```

하지만 동시에 두 변수가 바뀌었다.

```text
Warm K: 5.123 -> 4.172, 18.57% 감소
Coda/prediction: 5.154 -> 1.190, 76.92% 감소
```

따라서 5.65% latency 감소를 Coda 제거만의 효과라고 해석할 수 없다.

정확한 해석:

> Confirm-next는 recurrence depth와 반복 Coda 호출을 동시에 줄였고, 이 두 효과와 scalar overhead를 모두 포함한 결과 prediction latency가 5.65% 감소했다.

현재 결과로는 아래를 분리할 수 없다.

```text
recurrent iteration 감소 효과
Coda 호출 감소 효과
scalar scoring overhead
```

또한 confirm-next는 predictions/episode를 증가시켰다.

```text
23.883 -> 25.677
7.51% 증가
```

그 결과 episode latency는 오히려 증가했다.

```text
2862.98 ms -> 2904.13 ms
1.44% 증가
```

따라서 현재 방법을 episode-level latency 개선으로 주장하면 안 된다.

---

## 11. 다음 섹션의 첫 번째 작업

### 11.1 목표

동일한 recurrent depth K에서 반복 Coda 호출만 제거했을 때 latency가 얼마나 감소하는지 측정한다.

이 실험은 scalar policy를 평가하는 실험이 아니다. Coda 제거 자체의 시스템 가치만 분리하는 microprofile이다.

### 11.2 가장 단순한 비교

이미 구현된 terminal-only fixed control을 사용한다.

| Policy | Recurrent iterations | Coda calls | 목적 |
|---|---:|---:|---|
| Legacy fixed K4 | 4 | 4 | 기준 |
| Terminal-only fixed K4 | 4 | 1 | Coda 3회 제거 효과 |

추가로 가능하면 K6과 K8에서도 같은 비교를 한다.

| K | Legacy fixed | Terminal-only fixed |
|---:|---:|---:|
| 4 | 4 Coda | 1 Coda |
| 6 | 6 Coda | 1 Coda |
| 8 | 8 Coda | 1 Coda |

Recurrence 한 번의 추가 비용을 분리하려면 다음 비교도 유용하다.

```text
Terminal-only K4 vs Terminal-only K5
```

두 정책 모두 Coda 1회이므로 차이는 주로 recurrent core 한 번의 비용이다.

### 11.3 측정 원칙

- 동일 observation/workload를 사용한다.
- 동일 initial latent와 RNG 조건을 유지한다.
- 동일 K끼리 비교한다.
- CUDA asynchronous timing의 영향을 피하기 위해 synchronization을 명시한다.
- 첫 실행 warm-up을 결과에서 제외한다.
- 충분한 반복 측정 후 mean, median, p95를 기록한다.
- 가능하면 action head 전체 latency와 아래 component를 함께 기록한다.

```text
recurrent core cumulative time
_get_output cumulative time
pure Coda time
output projection and wrapper time
```

- 이 microprofile은 full rollout success 평가가 필요하지 않다.
- checkpoint는 read-only로 유지한다.

### 11.4 이 실험으로 답할 질문

```text
동일 K에서 Coda를 K회에서 1회로 줄이면 action-head latency가 실제로 얼마나 감소하는가?
```

결과에 따른 판단:

1. 동일 K에서도 의미 있는 latency 감소가 확인됨
   - Coda 제거 연구를 계속할 가치가 있다.
   - 다음으로 calibration의 깊은 K 분포를 감사한다.

2. 동일 K에서 감소가 매우 작음
   - 반복 Coda 제거를 주요 시스템 기여로 삼기 어렵다.
   - scalar 재학습에 추가 시간을 투입할 필요를 재검토한다.

현재는 의미 있는 감소의 threshold를 사후적으로 정하지 않는다. raw component timing과 end-to-end action-head timing을 함께 보고 판단한다.

---

## 12. Coda microprofile 이후 작업

### 12.1 Calibration 난이도 분포 감사

현재 calibration 데이터에서 다음을 task별로 계산한다.

```text
K=3 count
K=4 count
K=5 count
K>=6 count
K>=8 count
```

추가로 확인할 것:

```text
k=3 scalar score는 높지만 authoritative Action-MSE K가 6 이상인 hard negative 수
깊은 K 사례의 task 편중
깊은 K 사례의 episode 편중
success/failure episode별 K 분포
```

가장 중요한 비율:

```text
K>=6 warm predictions / all warm predictions
```

### 12.2 깊은 K 데이터가 부족한 경우

Runtime 구조와 7개 feature를 유지한다.

변경 후보:

```text
새로운 non-final development episodes 수집
K>=6 examples oversampling
깊은 K일수록 높은 sample weight
false-early examples에 더 큰 penalty
task-level sample balance
```

한 번에 여러 변경을 적용하지 않는다. 가장 먼저 동일 모델과 동일 runtime에서 data balancing만 바꿔 원인을 분리한다.

### 12.3 깊은 K 데이터가 충분한 경우

데이터 부족보다는 현재 feature 또는 `K_first` target의 한계 가능성이 커진다.

이 경우 복잡한 방식으로 바로 확장하지 말고 다음을 먼저 결론으로 검토한다.

> 현재 latent-dynamics summary만으로는 Action-MSE stopping을 안정적으로 완전 대체하기 어렵다.

---

## 13. 향후 평가 규칙

### 13.1 Final partition 재사용 금지

현재 final partition은 이미 결과를 확인했다. 다음 용도로 재사용하면 안 된다.

```text
threshold tuning
feature selection
model selection
loss weighting 선택
hard-example mining 기준 보정
```

현재 final 결과는 동결한다.

### 13.2 재학습 후 개발 평가

새 scalar model을 만든다면 새로운 non-final development data에서 먼저 평가한다.

권장 최소 평가:

```text
10 tasks
10 paired episodes per task
Action-MSE vs retrained confirm-next
100 paired trials
200 policy rollouts
```

이 단계에서 success degradation이 지속되면 더 큰 holdout 평가로 넘어가지 않는다.

### 13.3 최종 성공 주장 기준

새로운 holdout 평가를 수행하게 되더라도 사전에 다음을 고정한다.

```text
primary outcome
paired confidence method
success-preservation margin
latency aggregation method
infrastructure retry policy
```

성공률과 episode-level cost를 함께 본다. Prediction-level latency만으로 전체 시스템 개선을 주장하지 않는다.

---

## 14. 현재 주장 가능한 내용과 금지할 내용

### 14.1 현재 주장 가능

- Scalar confirm-next runtime은 계약 위반 없이 정상 동작했다.
- Warm prediction에서 Coda 호출을 1회로 줄였다.
- Final 평가에서 Coda/episode는 75.18% 감소했다.
- Warm mean K는 18.57% 감소했다.
- Prediction latency mean은 5.65% 감소했다.
- Confirm-next는 Action-MSE 대비 성공률 보존 기준을 충족하지 못했다.
- Episode latency는 1.44% 증가했다.
- Confirm-next exits는 K=4에 크게 집중됐다.

### 14.2 현재 주장 금지

- Coda 제거만으로 prediction latency가 5.65% 감소했다.
- Confirm-next가 task success를 보존했다.
- Confirm-next가 episode latency를 개선했다.
- Final 결과가 단순히 Task 5 하나 때문에 실패했다.
- Calibration data 부족이 원인으로 확정됐다.
- Latent convergence가 action correctness를 보장한다.
- Duplicate final Coda 제거를 독립 연구 기여로 주장한다.

---

## 15. 핵심 경로 모음

### Calibration

```text
benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7
benchmark_results/preconvergence_trigger/seed7/boundary_latent_oof_v1
benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_kfirst_v1
```

### Screening

```text
benchmark_results/scalar_latent_3way_screening/20260804_073711_head184a902_spatial10_seed7
```

### Final preflight

```text
benchmark_results/scalar_latent_final_preflight/20260804_093530_head184a902_spatial10_seed7
```

### Final evaluation

```text
benchmark_results/scalar_latent_final_evaluation/20260804_094502_head184a902_spatial10_seed7
```

### Final aggregate files

```text
benchmark_results/scalar_latent_final_evaluation/20260804_094502_head184a902_spatial10_seed7/aggregate_report.json
benchmark_results/scalar_latent_final_evaluation/20260804_094502_head184a902_spatial10_seed7/aggregate_report.md
benchmark_results/scalar_latent_final_evaluation/20260804_094502_head184a902_spatial10_seed7/episode_level.csv
benchmark_results/scalar_latent_final_evaluation/20260804_094502_head184a902_spatial10_seed7/discordant_pairs.csv
```

### Local analysis archive

```text
/tmp/scalar_confirm_final_analysis.tar.gz
```

이 `/tmp` 파일은 재부팅이나 환경 정리 시 사라질 수 있으므로 필요하면 별도 위치에 복사한다.

---

## 16. 다음 섹션 시작 체크리스트

다음 섹션에서 가장 먼저 다음을 확인한다.

```bash
cd /home/siwon/RD-VLA_test/rd-vla

git switch experiment/scalar-direct-confirm-replay
git pull --ff-only

git branch --show-current
git log -1 --oneline
git status --short
```

기대 상태:

```text
branch = experiment/scalar-direct-confirm-replay
worktree clean
```

그 다음 이 문서를 읽고 첫 작업을 수행한다.

```text
동일 K에서 Legacy fixed recurrence와 terminal-only fixed recurrence의 latency 비교
```

이 실험이 끝나기 전에는 scalar model을 재학습하거나 새로운 fallback/state machine을 구현하지 않는다.

---

## 17. 한 문장 요약

> Confirm-next는 반복 Coda를 크게 줄였지만 K=4에 집중되며 성공률을 보존하지 못했고, 현재 관측된 latency 이득에는 K 감소가 섞여 있으므로 다음 단계는 동일 K microprofile로 Coda 제거 자체의 latency 가치를 먼저 분리하는 것이다.
