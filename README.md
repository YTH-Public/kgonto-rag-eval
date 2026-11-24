# kgonto-rag-eval

지식그래프와 온톨로지에 잘 맞는 **RAG 평가 프레임워크**입니다. (v0.2.1)

LangChain v1 위에서 동작하며, 네 가지 일을 합니다.

- LLM 으로 RAG 답변을 채점합니다 (LLM-as-judge).
- 지식그래프를 순회해 평가셋 초안을 만들고, 누수를 점검합니다.
- 만들어진 지식그래프와 온톨로지의 품질, 검색이 가져온 근거를 LLM 없이 봅니다.
- 답의 인용(출처 표시)을 검사하고, 평가 결과를 기준선과 비교합니다.

의존성은 가볍습니다 (`langchain`, `langchain-openai`, `pydantic`).
평가에 쓰는 LLM 은 직접 넣어 바꿀 수 있습니다.

## 설치

학생 · 수업용(버전 고정):

```bash
pip install "kgonto-rag-eval[kg,pandas] @ git+https://github.com/YTH-Public/kgonto-rag-eval@v0.2.1"
```

코드를 고치며 쓸 때:

```bash
git clone https://github.com/YTH-Public/kgonto-rag-eval.git
cd kgonto-rag-eval
pip install -e ".[kg,pandas]"
```

`uv` 를 쓴다면 `pip` 자리에 `uv pip` 를 그대로 넣으면 됩니다.
기본 평가자는 OpenAI 모델이라 **`OPENAI_API_KEY`** 가 필요합니다 (환경변수 또는 `.env`).

## 왜 만들었나

RAGAS, DeepEval 같은 기존 도구는 훌륭합니다. 다만 2026년에 이런 일이 있었습니다.

- RAGAS 는 0.4.3(2026-01) 뒤로 정식 릴리스가 10개월 비었습니다.
  그동안 `langchain-community` 0.4 에서 `import ragas` 가 깨지고, 새 OpenAI 모델에 `max_tokens` 를 보내 거부당했습니다.
  두 문제는 0.4.4(2026-11-14)에서 고쳐졌습니다.
- DeepEval 은 활발하지만 이름이 계속 바뀝니다(`GPTModel` → `OpenAIModel`, `LLMTestCaseParams` → `SingleTurnParams`).
- 무엇보다 **같은 이름의 지표도 도구마다 정의가 다릅니다**(아래 [지표 정의 비교](#지표-정의-비교)).

그래서 이 패키지는 기존 도구의 좋은 관례(`EvaluationDataset`, metric 객체, `evaluate()`)를 따르되,
**프롬프트와 계산이 다 보이는 짧은 코드**로 다시 구현했습니다. 지표를 읽고 고쳐 보는 교재로 쓰기 좋게 만들었습니다.
기존 도구를 대신하기보다 함께 놓고 비교하는 용도를 권합니다.

## RAG 답변 평가

```python
from langchain.chat_models import init_chat_model
from kgonto_rag_eval import EvaluationDataset, evaluate
from kgonto_rag_eval.metrics import Faithfulness, ContextRecall, AnswerCorrectness

ds = EvaluationDataset.from_list([
    {
        "sample_id": "P001",                       # 기준선 비교 · 원인 추적용
        "user_input": "청년 월세 지원은 한 달에 얼마까지인가요?",
        "retrieved_contexts": ["월 최대 20만 원을 최장 24개월 지원한다."],   # 그때 실제로 검색된 근거
        "response": "월 최대 20만 원입니다.",
        "reference": "월 최대 20만 원",
        "metadata": {"split": "튜닝"},
    },
])

judge = init_chat_model("openai:gpt-6-luna", reasoning_effort="none")
report = evaluate(ds, metrics=[Faithfulness(), ContextRecall(), AnswerCorrectness()], llm=judge)
print(report.summary)   # 지표별 평균(유효 점수만)
print(report.stats)     # 지표별 평균 · 유효 · 누락 · 오류 개수
report.to_pandas()      # 행마다 점수 · 판정 세부(__details) · 오류(__error)
report.explain("P001", "faithfulness")   # 저장된 점수 · 이유 · 세부 (모델을 다시 부르지 않음)
```

문항이 많으면 `evaluate(..., max_workers=4)` 로 샘플을 동시에 채점합니다(기본 1, 행 순서는 입력 순서). 호출 수는 같고 기다리는 시간만 줄어듭니다.

모델을 부르지 않는 지표는 `needs_llm = False` 를 적으면 판정 모델 없이 채점합니다.

```python
from kgonto_rag_eval.metrics import Metric
from kgonto_rag_eval.schema import MetricResult

class HasSource(Metric):
    name = "has_source"
    needs_llm = False

    def score(self, sample, llm):
        return MetricResult(metric=self.name, score=1.0 if "[p." in sample.response else 0.0)

evaluate(ds, [HasSource()])
```

- `retrieved_contexts` 는 문자열 목록이나 langchain `Document` 목록입니다. 문자열 하나를 그대로 넣으면 오류입니다(v0.1 은 글자 단위로 쪼갰습니다).
- judge 출력이 규칙을 어기면(개수가 음수, 번호 중복 등) 점수를 만들지 않고 `error` 로 남깁니다. 호출이 실패해도 전체가 멈추지 않습니다.
- 평가자와 답을 만든 모델이 같으면 후하게 채점하는 경향이 있습니다. 중요한 평가는 다른 · 더 강한 모델로 교차하고 표본을 사람이 봅니다.

### 지표

| 지표 | 무엇을 보나 | reference |
|---|---|---|
| `Faithfulness` | 답의 주장 중 검색 문맥이 뒷받침하는 비율 | 필요 없음 |
| `ContextRelevance` | 검색 문맥마다 relevant / partial / irrelevant 판정 후 평균 | 필요 없음 |
| `ContextPrecision` | 관련 문맥이 위에 있는지(average precision) | 필요 없음 |
| `ContextRecall` | 기준 정답의 핵심 주장 중 검색 문맥에 근거가 있는 비율(질문도 함께 봄) | 필요 |
| `AnswerCorrectness` | 사실 P/R 의 F1 × 0.6 + 의미 적합도 × 0.4 | 필요 |
| `AnswerRelevancy` | 질문의 의도 중 답이 직접 다룬 비율 - 장황함 감점 | 필요 없음 |

점수는 0.0 ~ 1.0, 판정(`verdict`)은 0.8 이상 pass · 0.5 이상 partial · 나머지 fail 입니다.

### 지표 정의 비교

이름이 같아도 계산이 다릅니다. 점수를 도구끼리 평균 내거나 승자를 정하지 마세요.

| 지표 | RAGAS 논문(2309.15217) | RAGAS 0.4.4 | DeepEval 4.3.4 | 이 패키지 |
|---|---|---|---|---|
| Faithfulness | 답의 주장 중 문맥에서 추론되는 비율 | 같은 방식(주장 나누기 → 판정) | 주장마다 YES / BORDERLINE / NO 판정. 기본 `penalize_ambiguous_claims=False` 라 BORDERLINE 도 통과(근거에 없지만 모순도 아닌 주장이 통과할 수 있음), True 면 YES 만 | 주장 지원 비율, 근거 없는 주장 목록 |
| Answer relevancy | 답에서 질문을 거꾸로 만들어 원질문과 임베딩 유사도 | 같은 방식(코사인 평균, 모든 질문이 얼버무림이면 0. 척도가 낮게 나올 수 있음) | 답에서 뽑은 statement 판정 중 YES + BORDERLINE 비율 | 질문 의도 중 다룬 비율 - 장황함 감점(임베딩 안 씀) |
| Context relevance | 문맥에서 필요한 문장 비율 | 두 판정 프롬프트가 낸 0 · 1 · 2 등급을 각각 2로 나눠 평균(0 ~ 1) | (Contextual Relevancy) 모든 문맥 조각의 statement 판정을 펼친 YES 비율 | 문맥 조각마다 3단계 판정 평균 |
| Context precision | (논문에 없음) | reference 있는 판 `ContextPrecision`, 없는 판 `ContextUtilization`(생성된 답을 참고) | 질문 + 기대 답으로 조각마다 유용성을 판정해 만든 average precision | 질문만으로 관련 표시를 만든 average precision |
| Context recall | (논문에 없음) | 정답의 statement 중 문맥에 근거가 있는 비율 | 기대 답의 지원 판정 YES 비율 | 정답 주장 중 근거가 있는 비율, 질문을 함께 줌 |
| Answer correctness | (논문에 없음) | 사실 F-score 0.75 + 의미 유사도 0.25(기본) | G-Eval 등으로 직접 정의 | 사실 P/R 의 F1 0.6 + 의미 적합도 0.4 |

DeepEval 은 `eval_mode` 에 따라 판정 경로가 달라지므로 비교할 때는 `eval_mode="llm"` 으로 고정하세요.
빈 입력 처리도 도구마다 다릅니다(NaN · 0 · 1). 비교 표에는 결측 · 빈 주장 · 호출 실패를 따로 적으세요.

## 지식그래프 기반 평가셋 만들기

```python
from kgonto_rag_eval.kg import (
    load_kg, generate_testset_from_kg, validate_item, leakage_report, save_jsonl, to_eval_dataset,
)

g = load_kg(entities="entities.json", relations="relations.json")   # subject/object 칸도 받음
items = generate_testset_from_kg(g, judge, max_per_type=6)          # review_status='pending' 초안

# (1) 자동 검사 (2) 누수 점검 (3) 사람이 keep / edit / drop 으로 검수
print([validate_item(it) for it in items])
print(leakage_report(items))
items[0].review_status = "keep"
save_jsonl(items, "evalset.jsonl")

# 승인된(keep · edit) 문항만 RAG 를 실행해 평가 데이터로
ds = to_eval_dataset(items, rag_fn=my_rag)   # my_rag(question) -> (answer, 실제로 검색된 contexts)
```

- `load_kg` 는 같은 두 노드 사이의 관계를 모두 남깁니다(MultiDiGraph).
- `validate_item` 을 통과해도 진짜 multi-hop 이라는 보장은 아닙니다. 핵심 엣지 하나를 가려도 답할 수 있는지 사람이 확인하세요.
- `leakage_report` 는 정답이 질문에 그대로 든 문항과, 같은 근거 경로가 여러 split 에 들어간 경우를 찾습니다.
  평가셋을 만든 KG 와 채점 정답이 같은 KG 라서 생기는 순환은 코드로 잡을 수 없습니다.

## 검색 · 근거 · 구조 평가 (LLM 없이)

```python
from kgonto_rag_eval.kg import (
    path_entity_recall, triple_coverage, path_triples,
    source_traceability, compare_kg, check_ontology, schema_violation_rate, non_isolated_node_ratio,
)

path_entity_recall(item.evidence_path, retrieved)            # 이름만 봄(방향이 틀려도 1.0)
reference = path_triples(item.evidence_path, item.evidence_relations)
triple_coverage(reference, retrieved_triples)                # 관계 · 방향까지 봄

# 추출 KG vs 정답: 같은 정리 함수를 양쪽에, 문서 범위 안에서만 비교
compare_kg(pred_rows, gold_rows, normalize=lambda s: s.replace("(주)", ""), scope=("rcept_no",))

# 스키마: 허용 타입 · 관계 + (주어 타입, 관계, 목적어 타입) 패턴
schema_violation_rate(entities, relations, allowed_types={"회사", "품목"},
                      allowed_patterns={("회사", "공급", "회사"), ("회사", "매입", "품목")})
```

| 함수 | 보는 것 | 근거 · 성격 |
|---|---|---|
| `compare_kg` | 추출 KG vs 정답 트리플 P / R / F1, 누락 · 추가 | 트리플 P/R/F1 (Mondal et al. 2021, arXiv:2106.01167) |
| `check_ontology` · `schema_violation_rate` | 허용 타입 · 관계 · 패턴 · dangling 위반 | W3C SHACL 의 동기. 추론 엔진 아님 |
| `non_isolated_node_ratio` | 엣지를 가진 노드 비율 | 이 프로젝트의 구조 보조 지표 |
| `path_entity_recall` | 정답 경로 이름이 검색 결과에 나온 비율 | Recall@k 를 경로 이름에 옮긴 이 프로젝트의 지표 |
| `triple_coverage` | 정답 근거 트리플을 검색 트리플이 덮은 비율 | RAGAS context recall 의 생각을 트리플에 옮긴 이 프로젝트의 지표 |
| `source_traceability` | 출처 칸(page · chunk · provenance 등)이 있는 비율 | 출처가 **붙었는지**만 봄. 관계의 `source` 칸(주어)은 출처로 세지 않음 |

`path_recall` 은 v0.1 이름으로 남겨 두었습니다(= `path_entity_recall`).

## 인용 검사

```python
from kgonto_rag_eval import split_cited_sentences, resolve_citations, quote_in_source, citation_support

corpus = {"35": "...35쪽 원문...", "36": "...36쪽 원문..."}
answer = "월 최대 20만 원입니다 [p.35]. 최장 24개월입니다 [p.36]."

split_cited_sentences(answer)            # 문장마다 출처 ID
resolve_citations(["35", "99"], corpus)  # ① 출처 ID 가 실제 자료로 풀리는가
quote_in_source("월 최대 20만 원", corpus["35"])   # ② 인용 문구가 그대로 있는가(공백 무시)
citation_support(answer, corpus, judge, question="...")  # ③ 그 문장을 출처가 뒷받침하는가(LLM)
```

세 검사는 서로 다릅니다. 같은 쪽에 숫자가 있어도 다른 대상에 대한 숫자일 수 있습니다.
`citation_support` 는 ALCE(Gao et al. 2023, arXiv:2305.14627)의 citation recall / precision 아이디어를
줄인 **수업용 지표**입니다. 원 논문과 다른 점:
- 판정을 NLI 모델(AutoAIS) 대신 LLM 이 합니다.
- recall 분모는 **출처를 단 문장**만입니다(ALCE 는 출처 없는 문장도 0 으로 세어 전체 문장을 분모로 씀). 출처 없는 문장 수는 `uncited` 로 따로 봅니다.
- 풀리지 않는 ID 는 빼고 나머지 출처로 판정합니다. precision 분모는 출처 표시 횟수입니다.
- 판정 호출이 실패한 문장은 0 점이 아니라 `errors` 로 셉니다.

## 두 답 비교 (pairwise)

```python
from kgonto_rag_eval import pairwise_judge
pairwise_judge(question, answer_a, answer_b, judge, criteria="금액 · 기간 · 대상 조건이 정확한가")
# {"ab": "A", "ba": "B", "consistent": False, "final": "inconsistent"}
```

순서를 바꿔 두 번 묻고 답 ID 로 되돌립니다. 두 결과가 다르면 승자를 정하지 않습니다(위치 편향, MT-Bench arXiv:2306.05685).

## 기준선 · 회귀 테스트

```python
from kgonto_rag_eval import save_baseline, compare_baseline
save_baseline(report, "baseline.json", meta={"dataset": "policy-v1", "judge": "gpt-6-luna", "prompt": "v3"})
result = compare_baseline(new_report, "baseline.json", tolerance=0.1)
assert result["ok"], result
```

같은 `sample_id` 끼리, 양쪽 모두 점수가 있는 행만 짝지어 평균을 비교합니다. 지표 점수는 반올림하지 않은 값이라 작은 퇴행도 보입니다.
기준선에서 점수가 있던 칸이 지금 비면(judge 실패 · 행 누락) `lost` 에 적히고 기본으로 실패합니다(`allow_missing=True` 로 풀 수 있음).
`sample_id` 가 겹치면 저장하지 않습니다.
LLM 판정은 실행마다 흔들리므로 허용 폭(`tolerance`)과 반복 횟수를 정해 두세요.
CI 에서는 API 없이 도는 테스트(`tests/test_*.py`)만 돌리고, 실호출 테스트(`tests/live/`)는 따로 실행합니다.

## 한계

- 점수는 그 평가자 · 프롬프트 아래의 상대 지표입니다. 변경 전후 비교와 사람 검수 표본으로 쓰세요.
- 작은 그래프는 의미 있는 multi-hop 질문 수가 적습니다.
- `normalize` 를 넓게 잡으면(모든 괄호 · 공백 제거) 서로 다른 대상이 합쳐집니다.

## 근거 (논문 · 표준)

- **RAGAS** - Es et al., *Ragas: Automated Evaluation of Retrieval Augmented Generation*. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
  원 논문이 정의한 것은 faithfulness · answer relevance · context relevance 셋입니다.
  context precision / recall · answer correctness 는 그 뒤 라이브러리가 더한 지표이고,
  이 패키지의 프롬프트 · 가중치(0.6 / 0.4 등)는 이 프로젝트가 정한 것입니다.
- **ALCE** - Gao et al., *Enabling Large Language Models to Generate Text with Citations*. [arXiv:2305.14627](https://arxiv.org/abs/2305.14627) - `citation_support`
- **MT-Bench** - Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) - `pairwise_judge`
- **KG 구축 P/R/F1** - Mondal et al., *End-to-End NLP Knowledge Graph Construction*. [arXiv:2106.01167](https://arxiv.org/abs/2106.01167) - `compare_kg`
- **W3C SHACL** - [w3.org/TR/shacl](https://www.w3.org/TR/shacl/) - 스키마 검사의 동기
- **GraphRAG-Bench** - [arXiv:2506.02404](https://arxiv.org/abs/2506.02404) - 구축 · 검색 · 생성을 나눠 보는 관점

## v0.1 에서 옮겨 올 때

- `load_kg` 는 MultiDiGraph 를 돌려줍니다. `g[u][v]["relation"]` 대신 `for u, v, rel in g.edges(keys=True)` 처럼 관계를 모두 읽으세요.
- `EvalSample.retrieved_contexts` 에 문자열 하나를 주면 오류입니다. `[텍스트]` 로 넣으세요.
- 지표 점수가 반올림되지 않습니다. 표시용은 `report.summary` 를 쓰세요.
- `compare_kg(..., scope=...)` 에서 범위 칸이 빈 행은 오류입니다.
- 사업보고서 정답처럼 `@제출회사` 치환 · 블록 범위 · 부분 정답(`partial`)이 있는 자료는 이 함수 앞에서 직접 전처리하세요(패키지가 해 주지 않음).

## 변경 기록

[CHANGELOG.md](CHANGELOG.md)

## 라이선스

MIT.
