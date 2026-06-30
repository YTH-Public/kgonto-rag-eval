# kgonto-rag-eval

지식그래프와 온톨로지에 잘 맞는 **RAG 평가 프레임워크**입니다.

LangChain v1 위에서 동작하며, 세 가지 일을 합니다.

- LLM 으로 RAG 답변을 채점합니다 (LLM-as-judge).
- 지식그래프를 순회해 평가셋을 자동으로 만듭니다.
- 만들어진 지식그래프와 온톨로지의 품질까지 봅니다.

의존성은 가볍습니다 (`langchain-openai`, `pydantic`).
평가에 쓰는 LLM 은 직접 넣어 바꿀 수 있습니다.

## 핵심 기능

- **RAG 답변 평가**
  LLM 으로 채점하는 지표 6종입니다 (faithfulness, context relevance / precision / recall, answer correctness / relevancy).
  점수와 함께 판정(verdict)과 실패 원인을 돌려줍니다.

- **지식그래프 기반 평가셋 생성**
  그래프를 순회해 single-hop / multi-hop 질문과 정답을 자동으로 만듭니다.
  페르소나를 적용하며, 자동 생성물은 초안이므로 사람이 검수해 확정합니다.

- **검색·근거·구조 평가 (LLM 없이 계산)**
  검색이 정답 근거를 실제로 가져왔는지(path recall, triple coverage),
  근거가 원문으로 추적되는지(source traceability),
  그래프가 스키마를 지키고 잘 연결됐는지(schema violation rate, non-isolated node ratio)를 봅니다.
  각 지표는 논문·표준에 근거를 둡니다 (아래 [근거](#근거-논문표준) 참고).

- **KG / 온톨로지 평가**
  추출한 그래프를 정답과 비교해 precision / recall / F1 을 내고,
  허용 타입·관계 같은 스키마 적합성을 검사합니다.

- **평가자 LLM 교체**
  기본은 OpenAI 모델이지만, `evaluate(..., llm=...)` 로 어떤 LangChain 챗모델이든 넣을 수 있습니다.

## 왜 만들었나

RAGAS, DeepEval 같은 기존 도구는 훌륭합니다.
다만 최신 LangChain v1.x 와 신형 OpenAI 모델 조합에서 의존성·파라미터 충돌을 겪기도 합니다.

2026-06 에 실제로 겪은 예시입니다.

- `ragas 0.4.3` 이 이미 제거된 `langchain_community.chat_models.vertexai` 를 import 합니다.
- 내부 평가자가 신형 모델이 거부하는 `max_tokens` 를 보냅니다.

그래서 이 패키지는 기존 도구의 좋은 관례(`EvaluationDataset`, metric 객체, `evaluate()`)는 참고하되,
**함수형(SystemMessage / HumanMessage) 호출과 직접 주입하는 LLM** 으로 가볍게 다시 구현했습니다.

특정 RAGAS 버전이 항상 깨진다는 뜻은 아닙니다.
환경 충돌을 피하고, 평가자 모델과 프롬프트를 직접 통제하고 싶을 때 쓰기 좋은 경량 대안입니다.

## 설치

```bash
git clone https://github.com/YTH-Public/kgonto-rag-eval.git
cd kgonto-rag-eval

pip install .                 # 코어 (RAG 답변 평가 지표)
pip install ".[kg]"           # + 지식그래프 평가셋 생성 / KG·온톨로지 평가 (networkx)
pip install ".[kg,pandas]"    # + report.to_pandas()
```

코드를 고치며 쓸 거라면 editable 설치를 권장합니다.

```bash
pip install -e ".[kg,pandas]"
```

`uv` 를 쓴다면 `pip` 자리에 `uv pip` 를 그대로 넣으면 됩니다.

평가자 LLM 은 기본이 OpenAI 모델이라 **`OPENAI_API_KEY`** 가 필요합니다 (환경변수 또는 `.env`).
다른 모델을 쓰려면 `evaluate(..., llm=...)` 로 LangChain 챗모델을 넣으면 됩니다.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."   # 또는 python-dotenv 로 .env 로드
```

## RAG 답변 평가

```python
from kgonto_rag_eval import EvaluationDataset, evaluate
from kgonto_rag_eval.metrics import (
    Faithfulness, ContextRelevance, ContextPrecision,
    ContextRecall, AnswerCorrectness, AnswerRelevancy,
)

ds = EvaluationDataset.from_list([
    {
        "user_input": "이 제품의 보증 기간은?",
        "retrieved_contexts": ["본 제품의 보증 기간은 구매일로부터 2년이다."],
        "response": "구매일로부터 2년입니다.",
        "reference": "구매일로부터 2년",
    },
])

report = evaluate(ds, metrics=[Faithfulness(), ContextRecall(), AnswerCorrectness()])
print(report.summary)        # 지표별 평균
report.to_pandas()           # 샘플별 점수표 ([pandas] extra)
```

평가자 LLM 을 직접 넣을 수도 있습니다 (피평가 모델과 같거나 더 강한 모델을 권장합니다).

```python
from langchain.chat_models import init_chat_model

judge = init_chat_model("openai:gpt-5.4-mini")
report = evaluate(ds, metrics=[Faithfulness()], llm=judge)
```

`retrieved_contexts` 는 문자열 리스트와 langchain `Document` 리스트를 모두 받습니다.

### 지표

| 지표 | 무엇을 보나 | reference 필요 |
|---|---|---|
| `Faithfulness` | 답변의 주장이 검색 문맥에 근거하는가 (환각 탐지) | 아니오 |
| `ContextRelevance` | 검색된 문맥이 질문에 관련 있는가 (문맥별 평균) | 아니오 |
| `ContextPrecision` | 관련 문맥이 상위에 랭크됐는가 (순위 반영) | 아니오 |
| `ContextRecall` | 기준 정답의 핵심 근거가 검색 문맥에 들어있는가 | 예 |
| `AnswerCorrectness` | 답변이 기준 정답과 의미·사실이 일치하는가 (factual F1 + semantic) | 예 |
| `AnswerRelevancy` | 답변이 질문 의도·범위에 직접 답했는가 (사실성과 별개) | 아니오 |

점수는 모두 0.0 부터 1.0 사이입니다.

판정(`verdict`: pass / partial / fail)과 실패 위치(`failed_component`: retriever / generator / reference)도 함께 나옵니다.

`reference` 가 없으면 reference 가 필요한 지표는 `None` 으로 둡니다.
`evaluate(..., keep_reason=True)`(기본값)이면 지표별 판단 이유(`reason`)도 행에 남습니다.

## 지식그래프 기반 평가셋 생성

문서에서 평가 질문을 일일이 손으로 만들기는 어렵습니다.
지식그래프를 순회하면 single-hop / multi-hop 질문과 정답을 자동으로 만들 수 있습니다.
자동 생성물은 초안이므로, 사람이 검수해 확정합니다.

```python
from langchain.chat_models import init_chat_model
from kgonto_rag_eval.kg import (
    load_kg, generate_testset_from_kg, validate_item, save_jsonl, to_eval_dataset,
)

# (1) KG 적재: {name, type} / {source, relation, target} JSON, 또는 (s, rel, o) 트리플
g = load_kg(entities="entities.json", relations="relations.json")
# g = load_kg(triples=[("A", "HAS_PART", "B"), ...])

# (2) KG 순회 -> 페르소나 / 질문유형 -> LLM 으로 질문·정답 생성 (사실은 KG 경로에서만)
llm = init_chat_model("openai:gpt-5.4-mini")
items = generate_testset_from_kg(g, llm, max_per_type=6)

# (3) 자동 검증 -> 사람이 keep / edit / drop 으로 검수 -> 저장
keep = [it for it in items if not validate_item(it)]
save_jsonl(keep, "evalset.jsonl")

# (4) 검수된 셋에 RAG 를 실행해 평가용 데이터셋으로
ds = to_eval_dataset(keep, rag_fn=my_rag)   # my_rag(question) -> (answer, contexts)
```

각 샘플은 근거 경로(`evidence_path`)를 보존합니다.
그래서 나중에 답이 왜 맞고 틀렸는지 되짚어 보기 좋습니다.

페르소나는 `DEFAULT_PERSONAS` 를 쓰거나,
`auto_personas_from_graph(g, llm)` 로 후보를 뽑아 골라 쓰면 됩니다.

## KG / 온톨로지 평가

추출한 지식그래프를 정답과 비교하거나, 스키마 적합성을 검사합니다.
LLM 없이 계산하는 결정적 방식입니다.

```python
from kgonto_rag_eval.kg import compare_kg, check_ontology

# 추출 KG vs 정답: 트리플 precision / recall / F1 + 누락 / 추가
print(compare_kg(pred_relations, reference_relations))

# 온톨로지 적합성: 허용 타입·관계 화이트리스트, dangling 관계, 클래스 계층
report = check_ontology(entities, relations,
                        allowed_types={"Company", "Product", ...},
                        allowed_relations={"PRODUCES", "HAS_PART", ...})
print(report["ok"], report["issues"])
```

`check_ontology` 의 `ok` 는 "허용 타입·관계 화이트리스트와 dangling 검사를 통과했다"는 뜻입니다.
관계별 domain / range 추론까지 하지는 않습니다.

## 검색·근거·구조 평가

지식그래프 RAG 는 답변 점수만 보면 충분하지 않습니다.

- 필요한 근거(경로·트리플)를 검색이 실제로 가져왔는지 (검색 품질)
- 그래프 자체가 잘 만들어졌는지 (구축 품질)

이 두 가지를 LLM 없이 결정적으로 봅니다.
평가셋의 근거 경로(`evidence_path`)를 정답 근거로 사용합니다.

```python
from kgonto_rag_eval.kg import (
    path_recall, triple_coverage, path_triples,         # 검색 품질
    source_traceability,                                # 근거 추적성
    schema_violation_rate, non_isolated_node_ratio,     # 구축 품질
)

# 검색이 정답 경로의 엔티티를 가져왔나 (텍스트=벡터 RAG / 트리플=그래프 RAG 동일 기준)
path_recall(item.evidence_path, retrieved_contexts)     # {"path_recall": 0.667, ...}

# 검색 트리플이 정답 근거 트리플을 덮었나
reference = path_triples(item.evidence_path, item.evidence_relations)
triple_coverage(reference, retrieved_triples)           # {"triple_coverage": 0.5, ...}

# 검색·KG 항목이 원문 출처(page / chunk)로 추적되나
source_traceability(retrieved_docs)                     # {"traceability_rate": 0.92, ...}

# 그래프가 스키마를 지켰나 / 고립된 노드는 없나
schema_violation_rate(entities, relations, allowed_types, allowed_relations)
non_isolated_node_ratio(entities, relations)
```

| 함수 | 보는 것 | 근거 |
|---|---|---|
| `compare_kg` | 추출 KG vs 정답 트리플 P / R / F1 | KG 구축 P/R/F1 (Mondal et al. 2021) |
| `schema_violation_rate` | 허용 타입·관계·dangling 위반 비율 | RDF 제약 검증 (W3C SHACL) |
| `non_isolated_node_ratio` | 엣지를 가진 노드 비율 (구조 보조 지표) | GraphRAG-Bench |
| `path_recall` | 정답 경로 엔티티를 검색이 회수한 비율 | 정보검색 Recall@k 를 KG 에 적용 |
| `triple_coverage` | 정답 근거 트리플을 검색이 덮은 비율 | RAGAS context recall + 트리플 recall (조합) |
| `source_traceability` | 출처 메타데이터 보유 비율 | ALCE attribution 을 결정적으로 축소 |

모두 0.0 부터 1.0 사이의 비율과 함께, 누락·추가 목록 같은 상세 정보를 dict 로 돌려줍니다.

답변 지표(faithfulness 등)와 합치면 "검색 단계 -> 답변 단계" 를 나눠서 볼 수 있습니다.

표준 지표가 없는 조합 지표(`path_recall` 의 경로 적용, `triple_coverage`, `non_isolated_node_ratio`)는
이 프로젝트에서 정의한 지표임을 함수 docstring 에 적어 두었습니다.

## 한계

- 평가자(judge)와 피평가 모델이 같으면 편향이 생깁니다.
  자기 답을 후하게 보거나, 긴 답을 선호하거나, 같은 지식 공백을 공유합니다.
  점수는 절대 품질이 아니라, 그 평가자 프롬프트 아래에서의 상대 지표로 보는 것이 맞습니다.
  중요한 평가는 `evaluate(..., llm=<더 강한 모델>)` 로 교차검증하고, 표본을 사람이 검수해야 합니다.

- LLM 평가자는 빠르고 방향은 맞지만, 절대값보다는 변경 전후의 비교로 쓰는 편이 안전합니다.

- 작은 그래프는 의미 있는 multi-hop 질문 수가 적습니다.
  관계를 더 촘촘히 하거나 원문 청크 근거를 붙이면 나아집니다.

## 근거 (논문·표준)

지표는 임의로 고르지 않고 아래 표준·논문에 맞췄습니다.

- **RAGAS**
  Es et al., *Automated Evaluation of Retrieval Augmented Generation*. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
  faithfulness, context relevance / precision / recall, answer correctness / relevancy 의 근거입니다.

- **ALCE**
  Gao et al., *Enabling Large Language Models to Generate Text with Citations*. [arXiv:2305.14627](https://arxiv.org/abs/2305.14627)
  `source_traceability`(attribution) 의 근거입니다.

- **KG 구축 P/R/F1**
  Mondal et al., *End-to-End NLP Knowledge Graph Construction*. [arXiv:2106.01167](https://arxiv.org/abs/2106.01167)
  `compare_kg` 의 트리플 P/R/F1 근거입니다.

- **W3C SHACL**
  *Shapes Constraint Language*. [w3.org/TR/shacl](https://www.w3.org/TR/shacl/)
  `schema_violation_rate`(스키마 제약 검증)의 동기입니다.

- **GraphRAG-Bench**
  *Challenging Domain-Specific Reasoning for Evaluating Graph RAG*. [arXiv:2506.02404](https://arxiv.org/abs/2506.02404)
  구축·검색·생성을 나눠 평가하는 관점과 `non_isolated_node_ratio` 의 근거입니다.

- **정보검색 Recall@k / MRR / nDCG**
  정보검색의 표준 검색 지표입니다.
  `path_recall` 은 이를 KG 근거 경로에 적용한 형태입니다.

표준 지표는 그대로 쓰고,
표준이 없는 조합 지표(`path_recall` 경로 적용, `triple_coverage`, `non_isolated_node_ratio`)는
이 프로젝트에서 정의한 지표임을 함수 docstring 에 적었습니다.

## 로드맵

- 출처 무결성에서 신선도(freshness)와 계보(lineage)까지 보는 메타데이터 검증기로 확장합니다.
  (현재 `source_traceability` 가 출처 유무를 결정적으로 확인합니다.)
- ALCE 방식의 답변 인용 평가(`citation_precision` / `citation_recall`)를 추가합니다.
- 같은 두 노드 사이에 여러 관계를 두는 `MultiDiGraph` 지원과, 별칭·엔티티 링킹을 추가합니다.
- 방해 문맥이 섞였을 때의 견고함을 보는 지표(`noise_sensitivity`)를 추가합니다.

## 라이선스

MIT.
