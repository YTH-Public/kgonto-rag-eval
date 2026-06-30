# kgonto-rag-eval

지식그래프·온톨로지 친화 **RAG 평가 프레임워크**. LangChain v1 위에서 LLM-as-judge 로 RAG 답변을 채점하고, 지식그래프를 순회해 평가셋을 만들며, 추출된 KG/온톨로지의 품질까지 본다. 의존성은 가볍고(`langchain-openai`·`pydantic`), 평가자 LLM 은 주입식이다.

## 핵심 기능

- **RAG 답변 평가**: LLM-judge 지표 6종(faithfulness, context relevance/precision/recall, answer correctness/relevancy). 점수 + verdict + 실패 원인.
- **KG 기반 평가셋 생성**: 지식그래프를 순회해 single/multi-hop 질문·정답을 자동 생성(페르소나 적용, 사람 검수 전제).
- **검색·근거·구조 평가 (결정적, LLM 불필요)**: path recall·triple coverage(검색이 정답 근거를 가져왔나), source traceability(원문 추적), schema violation rate·non-isolated node ratio(그래프 구축 품질). 각 지표는 논문·표준에 근거(아래 [근거](#근거-논문표준)).
- **KG/온톨로지 평가**: 추출 KG vs gold의 precision/recall/F1, 스키마(허용 타입·관계) 적합성 검사.
- **주입식 LLM**: 기본은 OpenAI 모델, `evaluate(..., llm=...)` 로 어떤 LangChain 챗모델이든 교체.

## 왜 만들었나

RAGAS·DeepEval 같은 기존 도구는 훌륭하지만, 최신 LangChain v1.x + 신형 OpenAI 모델 조합에서 의존성·파라미터 충돌을 겪을 수 있다(2026-06 실측 예: `ragas 0.4.3` 이 제거된 `langchain_community.chat_models.vertexai` 를 import, 내부 평가자가 신형 모델이 거부하는 `max_tokens` 를 전송). 그래서 이 패키지는 그 도구들의 좋은 관례(`EvaluationDataset` / metric 객체 / `evaluate()`)를 참고하되, **함수형(SystemMessage/HumanMessage) + 주입식 LLM** 으로 가볍게 다시 구현했다.

특정 RAGAS 버전이 항상 깨진다는 뜻은 아니다. 환경 충돌을 피하고, 평가자 모델·프롬프트를 직접 통제하고 싶을 때 쓰기 좋은 경량 대안이다.

## 설치

```bash
git clone https://github.com/YTH-Public/kgonto-rag-eval.git
cd kgonto-rag-eval

pip install .                 # 코어 (RAG 평가 지표)
pip install ".[kg]"           # + 지식그래프 평가셋 생성 / KG·온톨로지 평가 (networkx)
pip install ".[kg,pandas]"    # + report.to_pandas()
```

> 코드를 수정하며 쓸 거면 `pip install -e ".[kg,pandas]"` (editable). `uv` 를 쓰면 `pip` 자리에 `uv pip` 를 그대로 쓰면 된다.

LLM 평가자는 기본이 OpenAI 모델이라 **`OPENAI_API_KEY`** 가 필요하다(환경변수 또는 `.env`). 다른 모델을 쓰려면 `evaluate(..., llm=...)` 로 LangChain 챗모델을 주입한다.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."   # 또는 python-dotenv 로 .env 로드
```

## RAG 평가

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

평가자 LLM 주입(권장: 피평가 모델보다 같거나 강한 모델):

```python
from langchain.chat_models import init_chat_model
judge = init_chat_model("openai:gpt-5.4-mini")
report = evaluate(ds, metrics=[Faithfulness()], llm=judge)
```

`retrieved_contexts` 는 문자열 리스트 또는 langchain `Document` 리스트 모두 받는다.

### 지표

| 지표 | 무엇을 보나 | reference 필요 |
|---|---|---|
| `Faithfulness` | 답변의 주장이 검색 문맥에 근거하는가 (환각 탐지) | 아니오 |
| `ContextRelevance` | 검색된 문맥이 질문에 관련 있는가 (per-context 평균) | 아니오 |
| `ContextPrecision` | 관련 문맥이 **상위에** 랭크됐는가 (순위 인식) | 아니오 |
| `ContextRecall` | 기준 정답의 핵심 근거가 검색 문맥에 들어있는가 | 예 |
| `AnswerCorrectness` | 답변이 기준 정답과 의미·사실 일치하는가 (factual F1 + semantic) | 예 |
| `AnswerRelevancy` | 답변이 질문 의도·범위에 직접 답했는가 (사실성과 별개) | 아니오 |

점수는 모두 0.0~1.0, `verdict`(pass/partial/fail)와 `failed_component`(retriever/generator/reference)도 함께 나온다. `reference` 가 없으면 reference 필요 지표는 `None`. `evaluate(..., keep_reason=True)`(기본)면 지표별 `reason` 도 행에 보존된다.

## 지식그래프 기반 평가셋 생성

문서에서 평가 질문을 손으로 다 만들기 어려울 때, 지식그래프를 순회해 single/multi-hop 질문과 정답을 자동 생성한다(자동은 초안, 사람이 검수해 확정).

```python
from langchain.chat_models import init_chat_model
from kgonto_rag_eval.kg import (
    load_kg, generate_testset_from_kg, validate_item, save_jsonl, to_eval_dataset,
)

# (1) KG 적재: {name,type} / {source,relation,target} JSON, 또는 (s,rel,o) 트리플
g = load_kg(entities="entities.json", relations="relations.json")
# g = load_kg(triples=[("A", "HAS_PART", "B"), ...])

# (2) KG 순회 -> 페르소나/질문유형 -> LLM 으로 질문·정답 생성 (사실은 KG path 에서만)
llm = init_chat_model("openai:gpt-5.4-mini")
items = generate_testset_from_kg(g, llm, max_per_type=6)

# (3) 자동 검증 -> 사람이 keep/edit/drop 으로 검수 -> 저장
keep = [it for it in items if not validate_item(it)]
save_jsonl(keep, "gold_evalset.jsonl")

# (4) 검수된 셋에 RAG 를 실행해 평가용 데이터셋으로
ds = to_eval_dataset(keep, rag_fn=my_rag)   # my_rag(question) -> (answer, contexts)
```

각 샘플은 `evidence_path`(근거 경로)를 보존하므로, 나중에 답이 왜 맞고 틀렸는지 추적하기 좋다. 페르소나는 `DEFAULT_PERSONAS` 를 쓰거나 `auto_personas_from_graph(g, llm)` 로 후보를 뽑아 큐레이션한다.

## KG / 온톨로지 평가

추출한 지식그래프를 정답(gold)과 비교하거나, 스키마 적합성을 검사한다(LLM 불필요, 결정적).

```python
from kgonto_rag_eval.kg import compare_kg, check_ontology

# 추출 KG vs gold: 트리플 precision/recall/F1 + 누락/추가
print(compare_kg(pred_relations, gold_relations))

# 온톨로지 적합성: 허용 타입·관계 화이트리스트, dangling 관계, 클래스 계층
report = check_ontology(entities, relations,
                        allowed_types={"Company", "Product", ...},
                        allowed_relations={"PRODUCES", "HAS_PART", ...})
print(report["ok"], report["issues"])
```

`check_ontology` 의 `ok` 는 "허용 타입·관계 화이트리스트와 dangling 검사를 통과"라는 뜻이다(관계별 domain/range 추론은 포함하지 않는다).

## 검색·근거·구조 평가 (결정적, LLM 불필요)

KG-RAG 는 답변 점수만 보면 안 된다. "필요한 근거(경로/트리플)를 검색이 실제로 가져왔는가"(검색 품질)와 "그래프 자체가 잘 만들어졌는가"(구축 품질)를 LLM 없이 결정적으로 본다. 평가셋의 `evidence_path` 를 정답 근거로 쓴다.

```python
from kgonto_rag_eval.kg import (
    path_recall, triple_coverage, path_triples,         # 검색 품질
    source_traceability,                                # 근거 추적성
    schema_violation_rate, non_isolated_node_ratio,     # 구축 품질
)

# 검색이 정답 경로의 엔티티를 가져왔나 (텍스트=벡터RAG / 트리플=그래프RAG 동일 기준)
path_recall(item.evidence_path, retrieved_contexts)     # {"path_recall": 0.667, ...}

# 검색 트리플이 gold 근거 트리플을 덮었나
gold = path_triples(item.evidence_path, item.evidence_relations)
triple_coverage(gold, retrieved_triples)                # {"triple_coverage": 0.5, ...}

# 검색/KG 항목이 원문 출처(page/chunk)로 추적되나
source_traceability(retrieved_docs)                     # {"traceability_rate": 0.92, ...}

# 그래프가 스키마를 지켰나 / 고립 노드는 없나
schema_violation_rate(entities, relations, allowed_types, allowed_relations)
non_isolated_node_ratio(entities, relations)
```

| 함수 | 보는 것 | 근거 |
|---|---|---|
| `compare_kg` | 추출 KG vs gold 트리플 P/R/F1 | KG 구축 P/R/F1 (Mondal et al. 2021) |
| `schema_violation_rate` | 허용 타입·관계·dangling 위반 비율 | RDF 제약 검증 (W3C SHACL) |
| `non_isolated_node_ratio` | 엣지 가진 노드 비율 (구조 보조 지표) | GraphRAG-Bench |
| `path_recall` | 정답 경로 엔티티를 검색이 회수한 비율 | 정보검색 Recall@k 의 KG 적용 |
| `triple_coverage` | gold 근거 트리플을 검색이 덮은 비율 | RAGAS context recall + triple recall (조합) |
| `source_traceability` | 출처 메타데이터 보유 비율 | ALCE attribution 의 결정적 축소판 |

모두 0.0~1.0 비율과 상세(누락/추가 목록 등)를 dict 로 돌려준다. LLM 답변 지표(faithfulness 등)와 합쳐 "검색 → 답변" 을 층층이 본다. 표준이 없는 조합 지표(`path_recall` 의 경로 적용, `triple_coverage`, `non_isolated_node_ratio`)는 project-defined 임을 함수 docstring 에 명시했다.

## 한계

- 평가자(judge)와 피평가 모델이 같으면 self-preference·verbosity 편향, 같은 지식 격차를 공유한다. 점수는 절대 품질이 아니라 **해당 judge 프롬프트 아래의 상대 지표**다. 중요한 평가는 `evaluate(..., llm=<더 강한 모델>)` 로 교차검증하고 표본을 사람이 검수한다.
- LLM-judge 는 빠르고 방향성은 맞지만, 절대값보다 **변경 전후 상대 비교**로 쓰는 것을 권장한다.
- 작은 KG 는 의미 있는 multi-hop 질문 수가 제한된다. 관계를 촘촘히 하거나 청크 근거를 붙이면 좋아진다.

## 근거 (논문·표준)

지표는 임의로 고르지 않고 아래 표준·논문에 맞췄다.

- **RAGAS**: Es et al., *Automated Evaluation of Retrieval Augmented Generation*. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) · faithfulness / context relevance·precision·recall / answer correctness·relevancy 의 근거.
- **ALCE**: Gao et al., *Enabling Large Language Models to Generate Text with Citations*. [arXiv:2305.14627](https://arxiv.org/abs/2305.14627) · `source_traceability`(attribution) 의 근거.
- **KG 구축 P/R/F1**: Mondal et al., *End-to-End NLP Knowledge Graph Construction*. [arXiv:2106.01167](https://arxiv.org/abs/2106.01167) · `compare_kg` 트리플 P/R/F1 의 근거.
- **W3C SHACL**: *Shapes Constraint Language*. [w3.org/TR/shacl](https://www.w3.org/TR/shacl/) · `schema_violation_rate`(스키마 제약 검증) 의 동기.
- **GraphRAG-Bench**: *Challenging Domain-Specific Reasoning for Evaluating Graph RAG*. [arXiv:2506.02404](https://arxiv.org/abs/2506.02404) · 구축·검색·생성 분리 평가, `non_isolated_node_ratio` 의 관점.
- **정보검색 Recall@k / MRR / nDCG**: IR 표준 검색 지표. `path_recall` 은 이를 KG 근거 경로에 적용한 graph-aware recall.

표준 지표는 그대로 쓰고, 표준이 없는 조합 지표(`path_recall` 경로 적용, `triple_coverage`, `non_isolated_node_ratio`)는 project-defined 임을 함수 docstring 에 표기했다.

## 로드맵

- 출처 무결성의 **freshness/lineage** 는 결정적 `source_traceability`(완료) 위에, 신선도까지 보는 메타데이터 검증기로 확장 예정.
- `citation_precision`/`citation_recall`(ALCE 식 NLI 기반 답변 인용 평가) 추가.
- `MultiDiGraph` 지원(같은 두 노드 사이 다중 관계), 별칭/엔티티 링킹.
- `noise_sensitivity` 등 distractor 강건성 지표.

## 라이선스

MIT.
