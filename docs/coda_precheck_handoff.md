# Coda Pre-check 연구 인수인계

이 문서는 RD-VLA의 adaptive recurrence에서 **Coda action decode 호출을 줄이기 위해 수행한 연구·구현·검증 과정**을 인수인계하기 위한 요약이다. 코드 사용법만 설명하는 문서가 아니라, 왜 각 시도를 했고 어떤 근거로 채택 또는 기각했는지를 함께 기록한다.

## 1. 현재 최종 결정

| 항목 | 최종 상태 | 이유 |
|---|---|---|
| Midpoint warm-start | **채택** | 이전 prediction의 유효한 midpoint recurrent state를 재사용해 recurrence depth를 줄이며, 별도의 Coda scheduler 없이 clean path를 유지한다. |
| Cached final output | **채택** | adaptive loop에서 마지막으로 계산된 action output을 다시 Coda로 계산하지 않고 재사용한다. |
| Origin-aware latent Coda pre-check | **기각** | offline replay에서는 Coda 호출을 줄였지만, RTX 4070 Ti 동기화 측정에서 모든 후보가 clean warm-only보다 20.3–24.3% 느렸다. |
| Learned scalar convergence probe | **기각** | final-only Coda 정책에서 Coda 호출은 약 80.6% 줄었지만, 수렴 판정이 평균 3.1–3.7 iterations 늦었고 false convergence가 발생했다. Probe 비용을 0으로 둔 추정에서도 12.3–19.3% 느렸다. |
| Online integration / screening | **진행하지 않음** | 두 Coda 절감 방식 모두 promotion gate를 통과하지 못했다. |

**현재 유지해야 하는 최종 정책은 `clean midpoint warm-only + cached final output`이다.**

> 이 프로젝트에서 midpoint warm-start는 이전 prediction의 midpoint recurrent state(SK1)를 의미한다. 초기 recurrent state인 S1을 warm-start 후보로 해석하지 않는다.

---

## 2. 문제 정의

기존 adaptive recurrence는 adjacent action convergence를 확인하기 위해 각 recurrent iteration에서 Coda를 실행한다.

```text
S_k 생성
  -> Coda(S_k)로 action a_k 생성
  -> MSE(a_k, a_{k-1}) < 0.001인지 확인
  -> 수렴 시 종료
```

이 방식은 action-space 수렴을 직접 확인하므로 신뢰할 수 있지만, 중간 iteration마다 Coda가 호출된다. 연구 목표는 다음과 같았다.

> Coda action을 매 iteration 생성하지 않고 latent 변화만으로 수렴 가능성을 판단하여 Coda 호출과 action-head latency를 줄일 수 있는가?

핵심 제약은 다음과 같다.

- action convergence의 기존 기준 `action_mse < 0.001`을 임의로 완화하지 않는다.
- numerical non-finite 또는 불확실한 상태에서는 fail closed한다.
- offline 호출 감소가 아니라 **실제 GPU latency 감소**를 최종 채택 기준으로 본다.
- midpoint warm-start가 이미 적용된 actual-warm 경로를 primary scope로 본다.

---

## 3. 브랜치와 연구 순서

```text
paper/warm-start-state-selection
        |
        |  midpoint warm-start 기준선
        v
codex/origin-aware-coda-scheduler
        |
        |  latent pre-check, shadow trace, OOF, GPU microbenchmark
        |  최종 결과 커밋: 7c5ce2184c2e06f062bda616f607fc9b1c93762b
        v
codex/learned-convergence-probe
           scalar learned probe와 corrected final-only Coda 평가
```

주요 origin-aware 진행 커밋은 다음과 같다.

| 커밋 | 의미 |
|---|---|
| `e6ba328` | Origin-aware Coda scheduler 추가 |
| `129b6c7` | Production과 비간섭하는 full-depth shadow trace 추가 |
| `aaec39b` | Offline replay 및 후보 선택 코드 추가 |
| `907b1ef` | Paired LIBERO 평가 protocol 고정 |
| `bfd59d7` | Fail-closed smoke gate 추가 |
| `ca1b7d3` | Formal calibration collection 추가 |
| `d3e9dee` | Task-level 5-fold OOF selection formalization |
| `64381e9` | OOF shortlist와 provenance 기록 |
| `7c5ce21` | Formal GPU schedule 결과와 기각 결정 기록 |

---

## 4. 기준선: clean midpoint warm-only

최종 비교 기준은 다음 특성을 갖는다.

- 유효한 이전 midpoint state가 있으면 actual-warm으로 시작한다.
- 그렇지 않으면 cold primary로 시작한다.
- 기존 adjacent action MSE stopping을 사용한다.
- adaptive loop에서 이미 계산된 마지막 Coda output을 재사용한다.
- 별도의 latent gate, confirmation scheduler, learned probe를 실행하지 않는다.

Origin-aware 및 learned-probe 결과는 모두 이 기준선에 조건부로 비교된 결과다. 원 논문의 전체 end-to-end VLM latency나 closed-loop success를 직접 대체하는 결과가 아니다.

---

## 5. 시도 1: Origin-aware latent Coda pre-check

### 5.1 설계 의도

단순 latent threshold를 모든 prediction에 동일하게 적용하지 않고, 초기 state의 출처를 구분했다.

- `ACTUAL_WARM`: 이전 prediction에서 전달된 유효한 midpoint state
- `COLD_PRIMARY`: 새로 초기화된 state
- `COLD_RETRY`: numerical fallback 후 한 번만 허용되는 cold retry

Actual-warm 경로에서는 latent 변화량이 작을 때 Coda를 건너뛰고, 일정 간격 또는 confirmation 조건에서 action-space convergence를 다시 확인하도록 scheduler를 설계했다. Non-finite, max-iteration, forced initial decode, confirmation pending, maximum skip 등의 우선순위를 명시적으로 관리했다.

### 5.2 Shadow trace와 비간섭

정책 후보를 production output에 영향을 주지 않고 검증하기 위해 full-depth shadow trace를 추가했다.

- Production terminal K, action output, cached final output, next midpoint와 RNG 상태를 먼저 확정한다.
- 이후 별도 namespace에서 max depth까지 shadow tail을 계산한다.
- Shadow 결과는 production action, cache, retry, profiling에 영향을 주지 않는다.
- Production 구간의 action MSE는 실제 BF16 control-flow `iteration_mse`를 authoritative source로 사용한다.
- Baseline 종료 이후에만 FP32 shadow diagnostic action MSE를 사용한다.

### 5.3 Calibration과 OOF

Formal calibration은 LIBERO Spatial 10 tasks × 10 episodes로 수행되었다.

- 100 episodes
- 2,398 predictions
- Actual-warm 2,298 / cold 100
- 200 captured action-head workload shards
- task-level 5-fold OOF
- prediction 및 episode overlap 0

후보는 latent threshold, `max_skip`, confirmation policy 조합으로 구성했고, 다음 안전 gate를 latency보다 먼저 적용했다.

- task-macro convergence capture ≥ 99.5%
- false convergence 0
- mean ΔK ≤ 0.25
- task-macro p95 ΔK ≤ 1
- max-iteration rate 증가 없음
- candidate-only retry-required 없음
- 모든 task에서 finite/evaluable

OOF에서는 일부 후보가 Coda decode를 약 10.2% 줄일 수 있었지만, planning anchor 기준 예상 개선은 promotion 목표에 미달했다.

### 5.4 Formal GPU microbenchmark

`docs/origin_aware_gpu_microbenchmark.md`와
`experiments/robot/libero/manifests/origin_aware_gpu_microbenchmark_seed7_result_v1.json`을 기준으로 확인한다.

측정 조건:

- GPU: RTX 4070 Ti
- Workloads: 200
- Primary actual-warm: 100
- Conditions: clean baseline + 6 frozen candidates
- Repeats: 5
- 총 synchronized calls: 7,000
- Offline schedule mismatch: 0

결과:

| 조건 | Primary latency |
|---|---:|
| Clean midpoint warm-only | 15.5225 ms |
| Origin-aware candidates | 18.6789–19.2978 ms |
| 상대 변화 | 20.3–24.3% 느림 |

모든 simultaneous lower bound가 promotion 기준보다 낮아 online screening을 중단했다.

### 5.5 기각 이유

Coda 호출 수 감소 자체는 발생했다. 그러나 현재 구현은 iteration마다 다음 비용을 유발했다.

- latent predicate의 GPU→CPU scalar synchronization
- 반복적인 `.item()` 기반 Python control flow
- `isfinite(...).all()` 등의 numerical checks
- confirmation/backfill 관리

즉, **모델 호출 수 감소가 실제 시스템 latency 감소를 보장하지 않았다.** 현재 구조에서는 줄어든 Coda 비용보다 host synchronization과 control overhead가 더 컸다.

---

## 6. 시도 2: Learned scalar convergence probe

### 6.1 연구 질문

Origin-aware pre-check는 일반 latent MSE가 action convergence와 충분히 정렬되지 않아 Coda confirmation을 필요로 했다. 다음 질문을 검증했다.

> 기존 scalar latent dynamics를 이용해 `action_mse < 0.001`을 예측하는 매우 작은 모델을 학습하면, 중간 Coda 없이 안전한 종료 시점을 찾을 수 있는가?

Online action head에는 통합하지 않고 existing calibration trace만 사용하는 offline feasibility study로 제한했다.

### 6.2 Dataset과 label

`docs/learned_convergence_probe_feasibility.md`를 기준으로 확인한다.

- Predictions: 2,398
- Finite transitions: 74,338
- Label: `action_mse < 0.001`
- Production 구간: native control-flow `iteration_mse`
- Shadow tail: FP32 diagnostic `shadow_trace.action_mse`

18개 입력 feature:

- 현재 `latent_mse`, `latent_l2`
- iteration index 및 normalized iteration index
- 이전 1·2 iteration의 latent MSE/L2
- 1·2-step slope와 ratio
- 2-step history availability
- warm/cold origin

Raw latent tensor와 cosine scalar는 frozen trace에 없어 사용하지 않았다.

### 6.3 학습한 모델

| 모델 | 구조 | 파라미터 |
|---|---|---:|
| Latent-MSE threshold | 단일 scalar threshold baseline | 1 |
| Logistic regression | 18 features + bias | 19 |
| Class-weighted logistic | 동일 구조, class balancing | 19 |
| Tiny MLP | 18 → 16 → 1, ReLU + sigmoid | 321 |

Tiny MLP 학습 설정:

- batch size ≤ 1,024
- 1,600 fixed steps
- Adam 형태 update
- learning rate 0.003
- weight regularization 1e-4
- deterministic seed

Task-level 5-fold OOF를 유지했고, 각 fold의 normalization, weights, threshold는 네 개 train folds에서만 fit했다. Threshold는 train-only score 후보 중 capture 99.5% 조건을 만족하면서 false convergence를 먼저 최소화하도록 선택했다.

### 6.4 V1 evaluator 오류와 의미

최초 evaluator는 원래 제안한 mechanism이 아닌 `legacy_cached_action_precheck`를 계산했다.

```text
probe negative -> 해당 iteration에서 Coda 실행
probe positive -> 현재 Coda 생략, 이전 cached action 반환
```

이 정책에서 나타난 negative decode reduction은 **final-only Coda 아이디어의 결과가 아니다.** V1 manifest는 재현용 diagnostic으로 보존하되, 최종 판단에는 사용하지 않는다.

### 6.5 Corrected primary policy

V2의 `final_only_coda`가 원래 의도한 정책이다.

```text
S_k 생성
  -> probe가 수렴 여부 판단
  -> positive면 recurrence 종료
  -> terminal latent S_k에 Coda를 정확히 1회 실행
```

Positive가 없으면 max iteration의 latent에 Coda를 한 번 실행한다.

Committed result:

`experiments/robot/libero/manifests/learned_convergence_probe_seed7_final_only_coda_result_v2.json`

### 6.6 Corrected 결과

Actual-warm 2,298 predictions 기준:

| 모델 | False conv. | Capture | Mean ΔK | p95 ΔK | Coda 감소 | Zero-overhead latency 변화 |
|---|---:|---:|---:|---:|---:|---:|
| Latent-MSE | 1 | 99.629% | 3.689 | 8.085 | 80.572% | +19.293% |
| Logistic | 2 | 99.915% | 3.376 | 7.500 | 80.572% | +15.332% |
| Weighted logistic | 2 | 99.915% | 3.341 | 7.560 | 80.572% | +14.892% |
| Tiny MLP | 4 | 99.842% | 3.134 | 7.765 | 80.572% | +12.266% |

Coda 호출은 전체 기준 `11,839 -> 2,298`회로 감소했다. 그러나 frozen planning anchor인 recurrent 3.56 ms/call, Coda 1.83 ms/call을 적용하면, 추가 recurrence 비용이 절약한 Coda 비용을 초과한다. 표의 latency 변화는 probe와 synchronization 비용을 0으로 둔 mechanism-only 추정이다. 실제 probe/control 비용을 포함하면 더 유리해질 수 없다.

### 6.7 기각 이유

- 모든 모델에서 false convergence가 발생해 strict safety gate 실패
- 평균 종료가 baseline보다 3.1–3.7 iterations 늦음
- task-macro p95 ΔK가 7.5–8.1로 허용치 1을 크게 초과
- probe 비용을 0으로 두어도 12.3–19.3% 느림

최종 결론:

```text
safety_failed_efficiency_also_unfavorable
online_integration_worth_investigating = false
```

### 6.8 Learned probe 결과의 해석 한계

이 결과는 현재 **scalar-only probe 설계**를 기각하기에 충분하지만, 모든 learned latent probe가 불가능하다는 증거는 아니다.

- 학습 objective는 transition-level binary classification이며 first-hit stopping sequence loss가 아니다.
- 수렴 이후의 많은 positive transitions가 학습 행 수를 지배한다.
- raw latent 방향 정보는 사용하지 않았다.
- production BF16 MSE와 shadow-tail FP32 diagnostic MSE가 한 sequence에 공존한다.
- Tiny MLP에 대한 광범위한 hyperparameter search나 다중 seed 탐색은 하지 않았다.
- closed-loop rollout 또는 deployment latency를 측정하지 않았다.

이 한계를 보완하는 작업은 현재 브랜치의 단순 연장이 아니라 별도 연구 범위로 다루는 것이 적절하다.

---

## 7. 결과를 해석할 때 주의할 점

### 주장 가능한 것

- Origin-aware latent pre-check는 offline에서 Coda 호출을 줄일 수 있었다.
- 현재 host-synchronized 구현에서는 실제 action-head latency가 증가했다.
- Corrected final-only learned probe는 Coda를 약 80.6% 줄였다.
- Scalar probe는 안전한 first-hit stopping을 제때 찾지 못했고 additional recurrence가 절감 효과를 상쇄했다.
- 최종 채택 대상은 clean midpoint warm-only이다.

### 주장하면 안 되는 것

- Origin-aware 또는 learned probe가 closed-loop success를 유지했다고 주장하지 않는다.
- Learned probe V1의 negative decode reduction을 final-only mechanism의 결과로 사용하지 않는다.
- Planning-anchor latency estimate를 현재 commit의 실제 deployment latency로 표현하지 않는다.
- Learned probe 실패를 raw-latent 또는 sequence-aware probe 전체의 불가능성으로 일반화하지 않는다.
- Rejected scheduler를 production/default path로 병합하지 않는다.

---

## 8. 주요 파일

### 문서

- `docs/origin_aware_gpu_microbenchmark.md`
- `docs/learned_convergence_probe_feasibility.md`
- `docs/coda_precheck_handoff.md` — 현재 문서

### Learned probe 코드

- `scripts/build_learned_convergence_dataset.py`
- `scripts/train_learned_convergence_probe.py`
- `scripts/evaluate_learned_convergence_probe.py`
- `scripts/learned_convergence_probe_lib.py`

### Learned probe tests

- `tests/test_learned_convergence_probe.py`
- `tests/test_learned_convergence_policies.py`

### Committed manifests

- `experiments/robot/libero/manifests/origin_aware_gpu_microbenchmark_seed7_result_v1.json`
- `experiments/robot/libero/manifests/learned_convergence_probe_seed7_result_v1.json`
- `experiments/robot/libero/manifests/learned_convergence_probe_seed7_model_v1.json`
- `experiments/robot/libero/manifests/learned_convergence_probe_seed7_final_only_coda_result_v2.json`

---

## 9. GitHub에 포함되지 않은 local artifacts

`.gitignore`에 의해 `benchmark_results/`, `*.pt`, logs, checkpoint outputs는 일반적으로 커밋되지 않는다. 따라서 repository clone만으로 formal calibration과 학습을 그대로 다시 실행할 수는 없다.

대표 local paths:

```text
benchmark_results/origin_aware_calibration/20260801_ca1b7d3_seed7_10x10
benchmark_results/origin_aware_gpu_microbenchmark/20260801_seed7_b9523c8/report.json
benchmark_results/learned_convergence_probe/20260801_seed7/
```

필요한 사람에게는 다음을 별도로 전달해야 한다.

- formal calibration traces
- 200 workload shards
- 29 MB derived learned-probe dataset
- full OOF/evaluation reports

Committed compact manifest에는 provenance SHA-256가 저장돼 있으므로 전달받은 artifact가 같은 입력인지 검증할 수 있다.

---

## 10. 재현 순서

Raw artifacts가 있는 환경에서 learned probe를 재현하는 기본 명령은 다음과 같다.

```bash
python scripts/build_learned_convergence_dataset.py \
  --run-root benchmark_results/origin_aware_calibration/20260801_ca1b7d3_seed7_10x10 \
  --output-dir benchmark_results/learned_convergence_probe/20260801_seed7/dataset \
  --base-seed 7

python scripts/train_learned_convergence_probe.py \
  --dataset-dir benchmark_results/learned_convergence_probe/20260801_seed7/dataset \
  --output benchmark_results/learned_convergence_probe/20260801_seed7/training_bundle.json \
  --seed 7

python scripts/evaluate_learned_convergence_probe.py \
  --dataset-dir benchmark_results/learned_convergence_probe/20260801_seed7/dataset \
  --training-bundle benchmark_results/learned_convergence_probe/20260801_seed7/training_bundle.json \
  --output benchmark_results/learned_convergence_probe/20260801_seed7/report.json \
  --compact-manifest experiments/robot/libero/manifests/learned_convergence_probe_seed7_result_v1.json \
  --model-artifact experiments/robot/libero/manifests/learned_convergence_probe_seed7_model_v1.json
```

V2 corrected policy 결과는 frozen V1 model artifact와 dataset을 사용해 생성한다. 기존 V1 manifest와 model artifact를 덮어쓰지 말고 SHA-256가 보존되는지 확인한다.

최근 로컬 검증 결과는 `141 passed`로 보고되었다. 이 수치는 repository CI 기록이 아니라 작업 환경에서 수행된 local test 결과다.

---

## 11. 후속 작업 우선순위

### 현재 논문화 범위

1. Clean midpoint warm-start를 최종 채택 방법으로 기술한다.
2. Origin-aware pre-check는 decode 감소와 GPU latency 악화의 systems ablation으로 정리한다.
3. Learned scalar probe는 Coda 감소에도 additional recurrence가 더 비싸다는 negative feasibility result로 정리한다.
4. 두 기각 결과에 대해 closed-loop 성능 주장을 하지 않는다.

### 연구를 다시 시작할 경우

다음은 동일 branch에서 threshold를 더 튜닝하기보다 새 연구 질문으로 시작해야 한다.

- first-hit stopping을 직접 최적화하는 sequence-aware objective
- prediction별 균등 가중치와 first-convergence transition weighting
- raw latent 또는 action-aware projection을 사용하는 probe
- per-iteration CPU synchronization이 없는 GPU-resident control flow
- theoretical headroom 확인 후에만 synchronized microbenchmark 수행

현재 결과에서는 threshold sweep, confirmation state 추가, 작은 MLP의 단순 확대를 우선시하지 않는다.

---

## 12. 새 팀원을 위한 체크리스트

- [ ] `codex/learned-convergence-probe` 브랜치를 checkout한다.
- [ ] 이 문서와 두 feasibility 문서를 먼저 읽는다.
- [ ] 최종 채택 정책이 clean midpoint warm-only임을 확인한다.
- [ ] V1 legacy evaluator와 V2 final-only policy를 구분한다.
- [ ] Raw artifact가 필요한 작업인지 확인한다.
- [ ] Committed manifest의 SHA-256와 local artifact를 대조한다.
- [ ] Rejected scheduler를 online inference에 병합하지 않는다.
- [ ] 새 실험은 기존 결론을 덮어쓰지 않고 새 manifest/version으로 기록한다.
