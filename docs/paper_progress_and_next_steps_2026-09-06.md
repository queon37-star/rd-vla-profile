# RD-VLA 최적화 논문 진행 현황 및 향후 작업

> 기준일: 2026-09-06  
> 대상 브랜치: `experiment/paper-libero-spatial-50x10`  
> 목적: 현재 논문에 실제로 사용할 수 있는 **최종 방법 정의, 실험 프로토콜, 검증된 수치, 해석 범위, 남은 작성/분석 작업**을 한 곳에 고정한다.

---

## 0. 이 문서의 역할

이 저장소에는 warm-start, latent pre-check, origin-aware scheduler, learned convergence probe 등 여러 개발 단계의 문서가 남아 있다. 특히 `docs/coda_precheck_handoff.md`는 과거 origin-aware latent pre-check 및 learned probe를 평가하고 당시 기준으로 기각했던 연구 이력을 기록한다. 그 문서는 역사적 기록으로 유지하되, **현재 논문에 사용할 최종 방법과 결과의 authoritative summary는 본 문서**로 본다.

현재 논문의 핵심은 두 가지 후단(post-VLM) 최적화다.

1. **Midpoint Warm-start**: 이전 prediction의 유효한 midpoint recurrent state를 다음 prediction의 초기 state로 재사용하여 recurrent depth `K`를 줄인다.
2. **LDCE (Latent-guided Deferred Coda Evaluation)**: stopping criterion 자체를 바꾸지 않고, latent로부터 action 변화가 아직 클 것으로 예상되는 iteration에서 Coda 평가를 지연하여 불필요한 Coda evaluation을 줄인다.

논문 표현에서 중요한 구분은 다음과 같다.

> **LDCE는 stopping criterion을 근사하지 않는다. stopping criterion의 평가 시점을 근사적으로 선별한다.**

최종 stopping decision은 기존과 동일한 adjacent action MSE 기준을 사용한다.

---

# 1. 현재 고정된 논문 실험 정의

## 1.1 4개 method

현재 공식 4-arm 정의는 아래와 같다.

| Method | Warm-start | LDCE | `apply_to_cold` | 의미 |
|---|---:|---:|---:|---|
| **Baseline** | False | False | - | 원래 adaptive RD-VLA |
| **LDCE** | False | True | **True** | cold path에서도 LDCE 활성화 |
| **Warm-start** | True | False | - | midpoint warm-start만 사용 |
| **Combined** | True | True | **True** | midpoint warm-start + LDCE |

코드 수준의 고정 정의:

```text
BASELINE: warm=False, ldce=False
LDCE:     warm=False, ldce=True,  apply_to_cold=True
WARM:     warm=True,  ldce=False
COMBINED: warm=True,  ldce=True,  apply_to_cold=True
```

Combined의 prediction origin은 다음처럼 해석한다.

- episode의 첫 prediction: `COLD + LDCE`
- 이후 유효한 warm state가 있는 prediction: `ACTUAL_WARM + LDCE`

즉 현재 Combined는 과거 일부 실험에서 사용했던 `apply_to_cold=False` 정의와 다르다. **현재 논문 결과에는 반드시 `apply_to_cold=True` 정의만 사용한다.**

## 1.2 공통 inference 조건

최종 실험의 공통 조건은 다음과 같다.

- Benchmark: **LIBERO Spatial**
- Tasks: 10 tasks
- Checkpoint: `outputs/12_24-24_24_Spatial_40k`
- Recurrent stopping strategy: adjacent action MSE
- Exact stopping threshold: `0.001`
- Max recurrent depth: `K_max = 32`
- Minimum terminal iteration: 2
- `num_exec_actions = 5`
- Cached final output: ON
- Legacy latent pre-check: OFF
- PyTorch profiler: OFF (공식 latency run)
- Video: OFF (공식 latency run)
- Workload replay: OFF (live LIBERO component validation)

LDCE runtime predictor는 predicted action-delta score를 사용하며, runtime high-side threshold는 `0.0015`이다. 단, 이 threshold는 **Coda 평가 여부를 정하는 scheduler threshold**이지 원래 stopping threshold `0.001`을 대체하지 않는다.

---

# 2. 최종 closed-loop formal evaluation: 2,000 episodes

## 2.1 프로토콜

Formal protocol:

```text
libero-spatial-final-onepass-50x10-4arm-v2
```

구성:

- 10 tasks
- task당 50 episodes / method
- 500 episodes / method
- 4 methods
- **총 2,000 measured episodes**
- official initial states / paired seeds 사용
- resume 없이 단일 formal acquisition

당시 formal run 기준 provenance:

- code commit: `ea1e2701930f3a8cc6249821fd400e2f7b3ddcd4`
- manifest SHA256: `0e3c6609b719d6b0a05f79efd769dff67141b52d00b42d9e0bea904ecf493144`
- LDCE artifact SHA256: `b4f9e938c72108f164b9d997a86eb4f5d9ea15b146754b9af83f9735e1ecfcf8`

관련 runner:

```text
scripts/run_spatial_paper_final_50x10.py
```

## 2.2 Formal 결과

| Method | Success | Avg. K | Coda eval./pred. | Coda elimination |
|---|---:|---:|---:|---:|
| **Baseline** | **460/500 = 92.0%** | **7.311** | **7.311** | - |
| **LDCE** | **459/500 = 91.8%** | **7.325** | **2.694** | **63.22% of own potential** |
| **Warm-start** | **454/500 = 90.8%** | **5.151** | **5.151** | - |
| **Combined** | **454/500 = 90.8%** | **5.154** | **2.772** | **46.23% of own potential** |

추가 해석:

- Warm-start의 Baseline 대비 recurrent depth 감소:

```text
7.311 -> 5.151  ~= 29.5% reduction
```

- LDCE는 Baseline과 거의 동일한 `K`를 유지하면서 Coda 호출을 크게 줄인다.
- Combined는 Warm과 거의 동일한 `K`를 유지하면서 Coda evaluation을 추가로 줄인다.
- Combined의 Baseline 대비 **Coda eval./pred. 감소율**은 약 `62.09%`이다.
- `46.23%`는 Combined 자체에서 가능한 Coda opportunity 중 제거된 비율이고, `62.09%`는 Baseline 대비 실제 Coda/pred 감소율이다. 두 수치를 혼동하지 않는다.

## 2.3 Paired success 분석

공식 paired outcome 해석:

### Baseline vs LDCE

- 499/500 episodes에서 success/failure label 동일
- Baseline-success -> LDCE-fail: 1
- net success change: `-0.2 pp`
- McNemar exact `p = 1`

LDCE는 공식 formal run에서 Baseline과 거의 동일한 closed-loop success를 보였다.

### Baseline vs Warm-start

- success -> fail: 17
- fail -> success: 11
- net: `-6 / 500 = -1.2 pp`
- McNemar exact `p ~= 0.345`

이 결과만으로 **Warm-start success preservation 또는 non-inferiority를 주장하지 않는다.** 단순히 관측 success와 paired flip 결과를 보고한다.

### Warm-start vs Combined

- both success: 454
- both fail: 46
- success/failure flip: **0**
- label identity: **500/500**

따라서 LDCE를 Warm-start에 추가했을 때 formal run의 success/failure label은 모든 paired episodes에서 동일했다.

단, prediction count는 완전히 같지 않으므로 **trajectory/action identity를 주장하지 않는다.**

## 2.4 Task별 success

| Task | Baseline | LDCE | Warm | Combined |
|---:|---:|---:|---:|---:|
| 0 | 100% | 100% | 96% | 96% |
| 1 | 98% | 96% | 94% | 94% |
| 2 | 100% | 100% | 100% | 100% |
| 3 | 100% | 100% | 100% | 100% |
| 4 | 88% | 88% | 84% | 84% |
| 5 | 50% | 50% | 44% | 44% |
| 6 | 100% | 100% | 100% | 100% |
| 7 | 98% | 98% | 96% | 96% |
| 8 | 90% | 90% | 94% | 94% |
| 9 | 96% | 96% | 100% | 100% |

---

# 3. 2,000-episode formal run의 latency는 headline으로 사용하지 않음

## 3.1 당시 측정 범위

Formal run에는 두 종류의 latency가 기록되었다.

1. **Action-head CUDA-event elapsed**
2. `get_action` 주변의 **policy-query wall-clock**

그러나 장시간 method-sequential 실행 때문에 method와 시간대가 완전히 confounded되었다.

대략적인 실행 순서:

```text
Baseline  -> LDCE -> Warm -> Combined
03:16       05:57    08:40   11:27 -> 14:12
```

같은 logical Action-head workload가 시간에 따라 약 25~30% 이상 변하는 사례가 확인되었다. CUDA Event도 GPU clock, power, thermal/runtime state 변화까지 제거하지 못했다.

따라서 formal 2,000-run의 latency는 다음 원칙으로 처리한다.

> 장시간 sequential closed-loop run은 **success, K, Coda accounting**의 공식 근거로 사용하고, latency는 관측치로만 취급한다. method 간 causal speedup headline에는 사용하지 않는다.

논문용 권장 표현:

> The long sequential closed-loop run revealed substantial temporal platform drift in both policy-query wall-clock and CUDA-event Action-head timing. We therefore use this run primarily to establish closed-loop success and algorithmic compute reductions, and use a separate interleaved component benchmark for latency comparisons.

---

# 4. 최종 Action-head latency benchmark: 150 episodes / method

## 4.1 측정 범위

공식 latency headline은 별도 live LIBERO component benchmark에서 측정한다.

Timer boundary:

```text
torch.cuda.synchronize()
start = perf_counter()
action_head.predict_action(...)
torch.cuda.synchronize()
stop
```

즉 **synchronized wall-clock Action-head latency**이다.

포함:

- Prelude
- recurrent core
- warm-start state handling
- LDCE predictor/scheduler
- Coda evaluation
- exact stopping check
- cached output / Python wrapper overhead inside `predict_action`

제외:

- VLM forward
- image preprocessing
- action unnormalization outside `predict_action`
- environment stepping

따라서 이 수치는 **E2E latency가 아니다.**

저장소의 `docs/latency_reporting_scope.md` 원칙에 따라 논문에서는 `Action-head latency` 또는 정확한 boundary를 명시한 `post-VLM action-head latency`라고 표현한다.

## 4.2 첫 50 episodes / method

Runner:

```text
scripts/profile_spatial_paper_action_head_10x5_interleaved.py
```

주요 commit:

```text
a982cf487d14e780be0fa32d26ca177d81e9b407
```

구성:

- 10 tasks
- task당 5 official states
- 50 episodes / method
- 200 measured episodes total
- task x method short block interleaving
- task마다 method order rotating

초기 50-episode 결과:

| Method | Mean AH latency | vs Baseline | Avg. K |
|---|---:|---:|---:|
| Baseline | 26.755 ms | - | 7.165 |
| Warm | 19.820 ms | -25.9% | 5.172 |
| LDCE | 22.551 ms | -15.7% | 7.172 |
| Combined | 17.785 ms | -33.5% | 5.180 |

Task-level 분석에서도 Warm, LDCE, Combined가 모두 10/10 tasks에서 Baseline보다 낮은 Action-head latency를 보였고, order-position residual은 이전 2,000-run의 큰 drift에 비해 매우 작았다.

## 4.3 추가 100 episodes / method

Runner:

```text
scripts/profile_spatial_paper_action_head_additional_10x10_interleaved.py
```

주요 commit:

```text
7ce3c3a1f53da797a2b9f724cce94c029a951062
```

기존 5개 state와 겹치지 않는 10개 state를 task마다 추가했다.

추가 결과:

| Method | Mean AH latency | vs Baseline | Avg. K |
|---|---:|---:|---:|
| Baseline | 27.489 ms | - | 7.375 |
| Warm | 20.002 ms | -27.2% | 5.192 |
| LDCE | 23.100 ms | -16.0% | 7.367 |
| Combined | 17.888 ms | -34.9% | 5.194 |

첫 50ep와 별도 100ep subset 사이에서 relative speedup이 잘 재현되었다.

## 4.4 Raw prediction sample pooling: 최종 150 episodes / method

최종 수치는 두 run의 평균값을 단순 평균한 것이 아니라 **raw prediction timing samples를 합쳐 다시 계산한 pooled 결과**를 사용한다.

최종 corpus:

- 15 official states / task
- 10 tasks
- **150 episodes / method**
- **600 measured episodes total**

| Method | Episodes | Predictions | Avg. K | Mean latency | Median | P95 | vs Baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | 150 | 3,437 | **7.307** | **27.253 ms** | 25.643 | 40.441 | - |
| **Warm-start** | 150 | 3,535 | **5.186** | **19.943 ms** | 18.822 | 34.013 | **-26.8%** |
| **LDCE** | 150 | 3,416 | **7.304** | **22.922 ms** | 21.712 | 34.088 | **-15.9%** |
| **Combined** | 150 | 3,533 | **5.190** | **17.855 ms** | 16.998 | 29.165 | **-34.5%** |

논문용 headline:

```text
Baseline : 27.25 ms
Warm     : 19.94 ms  (-26.8%)
LDCE     : 22.92 ms  (-15.9%)
Combined : 17.86 ms  (-34.5%)
```

중요한 decomposition:

- Baseline `K = 7.307`
- LDCE `K = 7.304`

즉 LDCE speedup은 K 감소로 설명되지 않는다.

- Warm `K = 5.186`
- Combined `K = 5.190`

즉 Warm과 Combined는 거의 동일한 recurrent depth를 갖고, Combined의 추가 speedup은 LDCE의 Coda reduction 효과와 일관된다.

Warm -> Combined:

```text
19.943 ms -> 17.855 ms
additional reduction ~= 10.5%
```

Task-macro 결과도 prediction-weighted 결과와 같은 결론을 보였다.

---

# 5. Coda evaluation vs LDCE Predictor microprofile

## 5.1 목적

LDCE는 작은 predictor를 추가로 실행하는 대신 비싼 Coda evaluation을 일부 생략한다. 따라서 다음 질문을 component level에서 직접 측정했다.

> Coda evaluation 1회가 실제 production LDCE predictor 1회보다 얼마나 비싼가?

전용 runner:

```text
scripts/profile_coda_vs_ldce_predictor.py
```

추가 commit:

```text
a637ebd78737604c90fa09601ab2ef69a7b97002
```

Protocol:

```text
libero-spatial-coda-vs-ldce-predictor-microprofile-v1
```

구성:

- LDCE-only
- 10 tasks x task당 1 official state
- 10 measured episodes
- 261 formal prediction records
- 실제 timing samples는 component call 단위로 수집

이 run은 내부 synchronization을 의도적으로 삽입하므로 **Action-head, policy-query, E2E latency를 report하지 않는다.**

## 5.2 결과

| Component | Calls | Mean | Median | P95 |
|---|---:|---:|---:|---:|
| **LDCE predictor** | **1,485** | **0.217 ms** | 0.189 ms | 0.411 ms |
| **Coda evaluation (`_get_output`)** | **718** | **1.231 ms** | 1.173 ms | 1.610 ms |
| Coda internal sub-block | 718 | 1.076 ms | 1.032 ms | 1.407 ms |

Primary ratio:

```text
Coda evaluation / Predictor = 5.67x (mean)
Median ratio                = 6.22x
Mean absolute difference    = 1.014 ms / call
```

Coda evaluation은 LDCE가 실제로 회피하는 전체 `_get_output` boundary를 의미한다.

```text
_get_output
= Coda processing
+ output norm/projection
```

따라서 논문 main text의 component comparison은 `1.231 ms vs 0.217 ms = 5.67x`를 사용한다. `1.076 ms`의 pure Coda sub-block 수치는 필요 시 부가 분석으로만 사용한다.

권장 논문 표현:

> The LDCE predictor requires 0.217 ms per evaluation on average, whereas a full Coda evaluation takes 1.231 ms, making Coda approximately 5.67x more expensive. This confirms that the predictor is substantially cheaper than the computation it is designed to avoid.

단순 component-level break-even 관점에서는 predictor cost가 Coda evaluation cost의 약 17.6%다.

```text
0.217 / 1.231 ~= 0.176
```

따라서 다른 모든 비용을 무시한 단순 모델에서는 Coda opportunity의 약 17.6% 이상을 제거하면 predictor 비용을 상쇄한다. 이 값은 **설명용 component break-even**이며 전체 Action-head speedup을 직접 예측하는 공식은 아니다.

---

# 6. 과거 Changed-path 결과와 현재 component microprofile의 관계

과거 120-episode 실험에서는 아래와 같은 결과가 있었다.

| Method | Success | Mean K | Coda/pred. | Changed-path/pred. | AH-core/pred. |
|---|---:|---:|---:|---:|---:|
| Baseline | 83.3% | 7.728 | 7.728 | 11.856 ms | 33.898 ms |
| Warm-only | 81.7% | 5.045 | 5.045 | 9.251 ms | 26.594 ms |
| Deferred-only | 83.3% | 7.738 | 2.820 | 5.857 ms | 28.135 ms |
| Warm + Deferred | 81.7% | 5.045 | 2.951 | 6.357 ms | 24.422 ms |

Deferred-only의 Changed-path는 Baseline 대비 약 `50.6%` 감소했다. 이 값과 현재 `Coda / Predictor = 5.67x`는 서로 모순되지 않는다.

이유:

- `5.67x`는 **per-call component ratio**다.
- Changed-path는 prediction 하나 동안 누적된 전체 changed path 비용이다.
- LDCE/Deferred는 Coda 한 번을 Predictor 한 번으로 1:1 교체하는 구조가 아니다.
- 한 prediction에서 predictor는 여러 번 실행되고, 일부 Coda는 여전히 실행되며, exact check / backfill / scheduler overhead도 존재한다.

현재 microprofile의 call frequency:

```text
Predictor calls/pred ~= 1485 / 261 = 5.69
Coda calls/pred      ~=  718 / 261 = 2.75
```

따라서 per-call gap이 약 5.7x여도 aggregate path speedup은 그보다 작게 나타나는 것이 정상이다.

또한 이 과거 표의 일부 Combined semantics는 현재 최종 `apply_to_cold=True`와 다를 수 있으므로, **과거 Changed-path 표는 historical/mechanistic context로만 사용하고 현재 main result table의 공식 수치로 섞지 않는다.**

---

# 7. 논문에서 사용할 공식 evidence map

현재 결과를 한 표에 섞을 때는 각 metric의 provenance를 명확히 분리해야 한다.

| Claim / Metric | 공식 근거 | 사용 여부 |
|---|---|---|
| Closed-loop success | 2,000-episode formal run | **Main** |
| Avg. K reduction | 2,000-episode formal run | **Main** |
| Coda eval./pred. | 2,000-episode formal run | **Main** |
| Coda elimination | 2,000-episode formal run | **Main** |
| Action-head latency | 150 episodes/method pooled interleaved benchmark | **Main** |
| Predictor vs Coda per-call cost | 10-episode component microprofile, 1,485/718 calls | **Mechanism / Ablation** |
| 2,000-run raw AH CUDA-event latency | sequential drift confounded | **Do not use as causal headline** |
| 2,000-run policy-query latency | sequential drift confounded | **Do not use as causal headline** |
| Environment/robot E2E latency | not measured in current final validation | **Do not claim** |
| Historical Changed-path / AH-core | old protocol / historical semantics | **Context only** |

---

# 8. 현재 main result table 초안

논문 main table은 closed-loop formal 결과와 별도 latency validation을 같은 row에 배치하되, caption에서 source/protocol이 다름을 명확히 적는다.

| Method | Success ↑ | Avg. K ↓ | Coda eval./pred. ↓ | Coda elimination ↑ | Action-head latency ↓ | AH reduction |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline** | **92.0%** | **7.311** | **7.311** | - | **27.25 ms** | - |
| **Warm-start** | 90.8% | **5.151** | 5.151 | - | **19.94 ms** | **26.8%** |
| **LDCE** | **91.8%** | 7.325 | **2.694** | **63.22%** | **22.92 ms** | **15.9%** |
| **Combined** | 90.8% | 5.154 | **2.772** | **46.23% own-potential** | **17.86 ms** | **34.5%** |

Caption에서 반드시 명시할 내용:

- Success/K/Coda: `500 episodes/method`, closed-loop formal evaluation
- Action-head latency: `150 episodes/method`, separate interleaved component benchmark
- latency excludes VLM and environment stepping
- Combined uses `apply_to_cold=True`

`Coda elimination` 열은 LDCE와 Combined의 denominator 정의가 오해되지 않도록 caption 또는 footnote에서 `own potential Coda opportunities`임을 명시한다. 필요하면 main table에서는 `Coda eval./pred.`만 두고 elimination 비율은 별도 figure로 보내는 것도 고려한다.

---

# 9. 현재 논문의 핵심 결과 서술

현재 수치가 지지하는 가장 강한 서술은 다음과 같다.

## 9.1 Warm-start

- Baseline formal `K = 7.311`
- Warm formal `K = 5.151`
- recurrent depth 약 `29.5%` 감소
- 별도 latency benchmark에서 Action-head latency `27.25 -> 19.94 ms`, 약 `26.8%` 감소

해석:

> Warm-start는 이전 prediction의 midpoint recurrent state를 재사용해 adaptive action generation이 반복해야 하는 recurrent refinement depth를 줄인다.

## 9.2 LDCE

- Baseline formal `K = 7.311`
- LDCE formal `K = 7.325` (사실상 동일)
- Coda/pred `7.311 -> 2.694`
- own-potential Coda evaluation의 `63.22%` 제거
- Action-head latency `27.25 -> 22.92 ms`, `15.9%` 감소
- Coda evaluation 1회 `1.231 ms`
- LDCE predictor 1회 `0.217 ms`
- per-call Coda가 predictor보다 `5.67x` 비쌈

해석:

> LDCE는 recurrent depth를 줄이는 방법이 아니라, 동일한 exact action-space stopping rule을 유지하면서 그 평가 시점을 선택적으로 줄이는 방법이다.

## 9.3 Combined

- Warm `K = 5.151`
- Combined `K = 5.154` (사실상 동일)
- Combined Coda/pred `2.772`
- Action-head latency `17.86 ms`
- Baseline 대비 `34.5%` 감소
- Warm 대비 추가 약 `10.5%` 감소
- Warm vs Combined paired success/failure label: `500/500` 동일

해석:

> Warm-start가 recurrent refinement 횟수를 줄이고, LDCE가 남아 있는 refinement 과정에서 Coda evaluation을 줄이므로 두 최적화는 서로 다른 비용 축을 줄인다. 단, Warm이 먼저 K를 줄이면 LDCE가 제거할 수 있는 Coda opportunity도 줄기 때문에 효과는 완전 additive하지 않다.

---

# 10. 논문에서 피해야 할 주장

다음 표현은 현재 evidence보다 강하므로 사용하지 않는다.

## 10.1 E2E latency

현재 공식 latency benchmark는 `action_head.predict_action`만 측정한다.

따라서 다음 표현은 사용하지 않는다.

```text
end-to-end VLA latency reduced by 34.5%
robot end-to-end latency reduced by 34.5%
policy-query latency reduced by 34.5%
```

대신:

```text
Action-head latency reduced by 34.5%
post-VLM action-head latency reduced by 34.5%
```

## 10.2 Warm-start success preservation / non-inferiority

Warm은 Baseline 대비 관측 success가 `92.0% -> 90.8%`이고 formal paired non-inferiority를 입증한 실험이 아니다.

따라서:

```text
Warm-start preserves success with no degradation
Warm-start is non-inferior to baseline
```

이라고 쓰지 않는다.

가능한 표현:

> Warm-start achieved 90.8% success compared with 92.0% for Baseline in the 500-episode formal evaluation, while reducing average recurrent depth by 29.5%.

## 10.3 LDCE trajectory identity

Warm과 Combined는 formal run에서 success/failure label 500/500 동일했지만 prediction count가 완전히 같지 않다. 따라서 동일 trajectory / identical actions를 주장하지 않는다.

## 10.4 Coda elimination denominator 혼동

- LDCE `63.22%`: LDCE own-potential Coda opportunities eliminated
- Combined `46.23%`: Combined own-potential Coda opportunities eliminated
- Combined vs Baseline Coda/pred reduction: 약 `62.09%`

서로 다른 denominator다.

---

# 11. 논문 framing 및 구조

지도교수 피드백에 따라 논문 framing은 RD-VLA 하나에만 묶지 않고 **Adaptive Action Generation / test-time adaptive compute** 관점으로 일반화한다.

권장 Introduction 흐름:

1. 단순한 action과 복잡한 action에 같은 계산량을 쓰는 것은 비효율적이다.
2. recurrent/iterative action generator는 difficulty에 따라 test-time compute를 조절할 수 있지만, 실제 runtime에는 두 종류의 낭비가 남는다.
3. 첫째, 연속 prediction 사이에 유사한 recurrent computation을 다시 시작한다.
4. 둘째, recurrent state가 아직 충분히 변하고 있어 stopping이 불가능한 iteration에서도 비싼 action decode/Coda를 반복 평가한다.
5. **Warm-start**로 반복 recurrent refinement를 줄인다.
6. **LDCE**로 불필요한 intermediate Coda evaluation을 지연한다.
7. 두 방법을 결합해 recurrent depth와 decode frequency라는 서로 다른 compute axis를 동시에 줄인다.
8. 대표 결과: Combined Action-head latency `34.5%` 감소, formal closed-loop success `90.8%`.

Introduction에서 RD-VLA는 대표적인 recurrent-depth VLA baseline으로 후반에 소개한다.

---

# 12. Figure / Table 계획

## 12.1 Main overview figure

보여줄 흐름:

```text
VLM features
   |
Prelude
   |
Initial recurrent state
   |<------ previous midpoint state (Warm-start)
Recurrent Core
   |
LDCE predictor ---- high predicted action change ---> defer Coda
   |
Coda / action decode
   |
exact adjacent-action MSE stopping
```

그림에서 반드시 분명하게 구분:

- Warm-start는 **initialization / recurrent-depth reduction**
- LDCE는 **Coda evaluation scheduling / decode-frequency reduction**
- exact stopping criterion은 변경되지 않음

## 12.2 Warm-start figure

Baseline vs Warm의 recurrent depth 분포 또는 task별 Avg. K를 보여준다.

핵심 annotation:

```text
Avg K: 7.311 -> 5.151 (-29.5%)
```

## 12.3 LDCE figure

가능한 구성:

- Coda evaluations / prediction
- Baseline: 7.311
- LDCE: 2.694
- Combined: 2.772

또는 Coda elimination percentage를 별도 bar로 제시한다.

## 12.4 Component-cost figure

Coda vs Predictor:

```text
LDCE predictor: 0.217 ms
Coda eval.:     1.231 ms
ratio:          5.67x
```

이 figure는 “왜 작은 predictor를 추가하는 것이 계산상 합리적인가”를 직관적으로 설명하는 데 사용한다.

## 12.5 Main results table

Section 8의 table을 기반으로 작성한다.

---

# 13. 앞으로 해야 할 일

아래 순서대로 진행하면 된다. 현재 핵심 성능 측정은 충분히 완료되었으므로 **새로운 대규모 rollout을 추가하는 것이 우선순위가 아니다.**

## Priority A. 결과 artifact 정리 및 수치 freeze

- [ ] 2,000-episode formal archive를 논문 결과의 authoritative artifact로 지정
- [ ] 첫 50ep latency archive와 추가 100ep latency archive를 함께 보관
- [ ] pooled 150ep 결과를 재생성 가능한 summary script 또는 JSON으로 저장
- [ ] `component_report.json`을 Coda-vs-Predictor 공식 artifact로 보관
- [ ] 각 table/figure에서 사용한 source artifact, commit, protocol을 별도 provenance 표에 기록
- [ ] main numbers를 이후 코드 변경과 분리하기 위해 `paper_results_frozen/` 또는 equivalent immutable result directory를 만드는 방안 검토

## Priority B. Main Results Table 확정

- [ ] Success / Avg. K / Coda-per-pred / Action-head latency를 한 표에 정리
- [ ] metric마다 서로 다른 evaluation protocol이 사용되었음을 caption에 명시
- [ ] Coda elimination denominator 설명
- [ ] Warm success를 과도하게 해석하지 않는 문구 확인
- [ ] Combined `apply_to_cold=True` 표기 및 provenance 확인

## Priority C. Figure 제작

- [ ] Overall method overview
- [ ] Warm-start recurrent-depth reduction figure
- [ ] LDCE Coda evaluation reduction figure
- [ ] Coda vs Predictor `5.67x` component-cost figure
- [ ] 필요 시 per-task Action-head latency reduction figure

Figure는 단순 장식이 아니라 본문에서 직접 읽어야 한다. 본문 문장에 “Fig. X의 red bar / shaded region / task trend”처럼 구체적으로 연결한다.

## Priority D. Experimental Results section 작성

권장 subsection:

```text
5.1 Experimental Setup
5.2 Closed-loop Performance and Compute Reduction
5.3 Action-head Latency
5.4 Component Analysis of LDCE
5.5 Combined Effect / Ablation
```

### 5.1 Setup에서 명시할 것

- LIBERO Spatial, 10 tasks
- formal closed-loop: 500 episodes/method
- latency validation: 150 episodes/method, interleaved
- Action-head timer boundary
- hardware (RTX 4070 Ti)
- exact stopping threshold / Kmax
- Combined semantics

### 5.2 Closed-loop

- Success
- Avg K
- Coda/pred
- paired outcome 해석

### 5.3 Action-head latency

- pooled 150ep official numbers
- interleaved 이유: temporal drift control
- mean + 필요 시 median/P95
- E2E가 아님을 명시

### 5.4 LDCE component analysis

- predictor vs Coda 5.67x
- predictor overhead와 Coda elimination의 관계
- exact stopping criterion 유지

### 5.5 Combined

- Warm과 Combined의 K parity
- Warm -> Combined additional 10.5% Action-head reduction
- paired success labels 500/500 identical
- non-additive effect 설명

## Priority E. Introduction / Method rewrite

- [ ] RD-VLA-specific 문제에서 Adaptive Action Generation 문제로 framing 확장
- [ ] 두 문제를 명확하게 분리: recurrent initialization waste / repeated decode waste
- [ ] 두 해결책 이름을 문서 전체에서 통일: `Midpoint Warm-start`, `LDCE`
- [ ] LDCE를 “convergence predictor”로 오해하지 않도록 wording 고정
- [ ] 수식을 최소화하고 pipeline figure 중심으로 설명

## Priority F. Related Work / Background 정리

- [ ] recurrent-depth / iterative action generation
- [ ] test-time adaptive compute
- [ ] action caching / temporal reuse / warm-start 계열
- [ ] early exit / deferred evaluation / conditional computation 계열
- [ ] RD-VLA는 구현 baseline으로 필요한 만큼만 상세 설명

## Priority G. Appendix / Robustness

Main paper 공간이 부족할 경우 다음을 appendix로 이동한다.

- task별 success table
- latency median / P95
- task-macro latency
- execution-order residual check
- historical Changed-path 결과
- formal 2,000-run latency drift diagnosis
- Coda internal sub-block 1.076 ms
- detailed paired success contingency tables

---

# 14. 추가 실험 필요성 판단

현재 확보된 핵심 evidence:

1. **Closed-loop**: 500 episodes/method, 2,000 total
2. **Action-head latency**: 150 episodes/method, 600 total, interleaved
3. **Coda vs Predictor**: 1,485 predictor calls / 718 Coda calls

따라서 현 시점에서 논문 핵심 주장을 위해 추가 대규모 LIBERO rollout을 수행할 필요성은 낮다.

추가 측정은 아래 조건에서만 고려한다.

- reviewer/advisor가 E2E 또는 policy-query latency를 명시적으로 요구
- 다른 GPU / Jetson에서 portability를 보여줘야 함
- 새로운 method variant를 실제로 논문 method에 추가하기로 결정
- 현재 artifact/provenance에서 재현 불가능한 결함이 발견됨

그 외에는 **결과를 계속 늘리기보다 figure/table/write-up을 완성하는 것이 우선**이다.

---

# 15. 최종 체크리스트

논문 제출 전 다음을 확인한다.

### Method semantics

- [ ] Baseline / Warm / LDCE / Combined 정의가 코드와 본문에서 동일
- [ ] Combined `apply_to_cold=True`
- [ ] LDCE stopping criterion은 exact adjacent action MSE 그대로
- [ ] predictor threshold와 stopping threshold를 혼동하지 않음

### Metrics

- [ ] Success/K/Coda는 formal 2,000-run
- [ ] Action-head latency는 pooled 150ep/method interleaved run
- [ ] Coda/Predictor ratio는 component microprofile
- [ ] E2E latency claim 없음
- [ ] sequential 2k latency를 causal speedup으로 사용하지 않음

### Claims

- [ ] Warm non-inferiority 주장 없음
- [ ] Warm/Combined success label identity와 trajectory identity를 구분
- [ ] Coda elimination denominator 명확
- [ ] Combined effect를 완전 additive라고 주장하지 않음

### Writing

- [ ] Introduction에서 두 문제와 두 방법이 1:1 대응
- [ ] Figure를 먼저 완성하고 본문에서 figure의 구체적인 trend를 직접 설명
- [ ] Experimental Results section 완성
- [ ] contribution paragraph itemized

---

# 16. 현재 논문용 한 문단 요약

현재 최종 결과는 다음과 같이 요약할 수 있다.

> We optimize adaptive recurrent action generation along two complementary compute axes. Midpoint Warm-start reuses a recurrent state from the previous prediction and reduces the average recurrent depth from 7.311 to 5.151 (29.5%). LDCE preserves the original adjacent-action stopping criterion but selectively defers intermediate Coda evaluations, reducing Coda evaluations from 7.311 to 2.694 per prediction while leaving recurrent depth nearly unchanged. In a separate interleaved Action-head latency benchmark with 150 episodes per method, Warm-start and LDCE reduce latency by 26.8% and 15.9%, respectively, while their combination achieves a 34.5% reduction (27.25 ms to 17.86 ms). A component microprofile further shows that a full Coda evaluation costs 1.231 ms on average versus 0.217 ms for the LDCE predictor, a 5.67x per-call cost gap. The 500-episode-per-method closed-loop evaluation yields success rates of 92.0%, 91.8%, 90.8%, and 90.8% for Baseline, LDCE, Warm-start, and Combined, respectively.

이 문단은 abstract/Introduction 대표 결과 작성 시 수치 기준점으로 사용할 수 있다.
