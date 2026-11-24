[← 처음으로](../README.md)

# 지표 사전

| 지표 | 무엇을 보나 | 필요한 것 |
|---|---|---|
| `Faithfulness` | 답의 주장 중 검색된 근거가 뒷받침하는 비율 | 근거 |
| `ContextRelevance` | 근거마다 관련 있음 · 일부 · 없음 판정의 평균 | 근거 |
| `ContextPrecision` | 관련 있는 근거가 위에 있는지(average precision) | 근거 |
| `ContextRecall` | 기준 답의 핵심 주장 중 근거에 있는 비율 | 근거 · 기준 답 |
| `AnswerCorrectness` | 사실 P/R의 F1 × 0.6 + 의미 적합도 × 0.4 | 기준 답 |
| `AnswerRelevancy` | 질문의 의도 중 답이 다룬 비율 - 장황함 감점 | 없음 |

골라 쓰는 지표입니다(`CORE_METRICS` · `ALL_METRICS`에는 없음).

| 지표 | 무엇을 보나 | 필요한 것 | 호출 |
|---|---|---|---:|
| `FactCoverage` | 필수사실마다 있음 · 빠짐 · 어긋남과 답의 근거 문구 | `metadata["expected_facts"]` | 1 |
| `AnsweredRelevancy` | 답했는지 먼저 봄. 답하지 않았으면 0 | 없음 | 1 ~ 2 |
| `FactHit` | 필수사실이 글자 그대로 있는 비율(정답률 아님) | `metadata["expected_facts"]` | 0 |
| `CitedRetrieved` | 답의 출처 가운데 실제로 검색된 자료의 비율 | `metadata["retrieved_ids"]` | 0 |
| `AnswerSimilarity` | 답과 기준 답의 임베딩 유사도 | 기준 답 · 임베딩 | 임베딩 1 |

- 점수는 0 ~ 1입니다. 모델이 채점하는 지표의 `verdict`는 0.8 이상 pass, 0.5 이상 partial, 나머지 fail입니다.
- `FactCoverage`는 필수사실 **밖의** 틀린 내용을 보지 않습니다. `AnswerCorrectness` · `Faithfulness`와 함께 읽습니다. 필수사실에는 대상과 조건을 함께 적습니다.
- `AnsweredRelevancy`가 0이라고 나쁜 답은 아닙니다. 근거가 없을 때의 거절은 옳은 답일 수 있습니다.
- `AnswerSimilarity`는 숫자나 부정이 틀려도 높게 나옵니다. 실제로 정책 질문 40개에서 짧은 기준 답과 비교한 맞은 답(0.43)이 틀린 답(0.65)보다 낮은 일이 있었습니다. 그래서 합격선을 두지 않습니다.
- 채점하는 모델과 답을 만든 모델이 같으면 후하게 채점하는 경향이 있습니다. 중요한 평가는 다른 모델로 교차하고 표본을 사람이 봅니다.
- 판정 모델의 출력이 규칙을 어기면(개수가 음수 등) 점수를 만들지 않고 `error`로 남깁니다.

`diagnose(report)`는 점수를 조합해 의심할 곳(검색 · 생성 · 거절)을 찾아 줍니다. 원인을 확정하지는 않습니다.

<details>
<summary><b>다른 도구와 지표 정의 비교 (RAGAS · DeepEval)</b></summary>

이름이 같아도 계산이 다릅니다. 점수를 도구끼리 평균 내거나 승자를 정하지 마세요.

| 지표 | RAGAS 논문(2309.15217) | RAGAS 0.4.4 | DeepEval 4.3.4 | 이 패키지 |
|---|---|---|---|---|
| Faithfulness | 답의 주장 중 문맥에서 추론되는 비율 | 같은 방식(주장 나누기 → 판정) | 주장마다 YES / BORDERLINE / NO 판정. 기본 `penalize_ambiguous_claims=False` 라 BORDERLINE 도 통과(근거에 없지만 모순도 아닌 주장이 통과할 수 있음), True 면 YES 만 | 주장 지원 비율, 근거 없는 주장 목록 |
| Answer relevancy | 답에서 질문을 거꾸로 만들어 원질문과 임베딩 유사도 | 같은 방식(코사인 평균, 모든 질문이 얼버무림이면 0. 척도가 낮게 나올 수 있음) | 답에서 뽑은 statement 판정 중 YES + BORDERLINE 비율 | 질문 의도 중 다룬 비율 - 장황함 감점(임베딩 안 씀) |
| Context relevance | 문맥에서 필요한 문장 비율 | 두 판정 프롬프트가 낸 0 · 1 · 2 등급을 각각 2로 나눠 평균(0 ~ 1) | (Contextual Relevancy) 모든 문맥 조각의 statement 판정을 펼친 YES 비율 | 문맥 조각마다 3단계 판정 평균 |
| Context precision | (논문에 없음) | reference 있는 판 `ContextPrecision`, 없는 판 `ContextUtilization`(생성된 답을 참고) | 질문 + 기대 답으로 조각마다 유용성을 판정해 만든 average precision | 질문만으로 관련 표시를 만든 average precision |
| Context recall | (논문에 없음) | 정답의 statement 중 문맥에 근거가 있는 비율 | 기대 답의 지원 판정 YES 비율 | 정답 주장 중 근거가 있는 비율, 질문을 함께 줌 |
| Answer correctness | (논문에 없음) | 사실 F-score 0.75 + 의미 유사도 0.25(기본) | G-Eval 등으로 직접 정의 | 사실 P/R 의 F1 0.6 + 의미 적합도 0.4 |

DeepEval은 `eval_mode`에 따라 판정 경로가 달라지므로 비교할 때는 `eval_mode="llm"`으로 고정하세요.
빈 입력 처리도 도구마다 다릅니다(NaN · 0 · 1). 비교 표에는 결측 · 빈 주장 · 호출 실패를 따로 적으세요.

</details>
