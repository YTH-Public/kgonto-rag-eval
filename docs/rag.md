[← 처음으로](../README.md)

# RAG 채점

저장해 둔 답을 채점하거나, 만든 RAG를 실행하며 채점합니다.

## 기록을 채점하기

```python
from langchain.chat_models import init_chat_model
from kgonto_rag_eval import evaluate, to_dataset
from kgonto_rag_eval.metrics import AnswerCorrectness, Faithfulness

records = [{
    "no": "P001", "question": "청년 월세 지원은 한 달에 얼마까지인가요?",
    "answer": "월 최대 20만 원입니다 [p.35].", "reference": "월 최대 20만 원",
    "found": [{"page": 35, "text": "월 최대 20만 원을 최장 24개월 지원한다."}],
    "expected_facts": ["월 최대 20만 원"],
}]

dataset = to_dataset(records, contexts="found", reference="reference", id="no",     # 내 자료의 열 이름을 적습니다
                     keep=["expected_facts"], source="page")
model = init_chat_model("openai:gpt-6-luna", reasoning_effort="none")
report = evaluate(dataset, [Faithfulness(), AnswerCorrectness()], llm=model)

report.summary                              # 지표별 평균
report.stats                                # 유효 · 누락 · 오류 수 - 평균이 몇 건에서 나왔는지
report.to_pandas()                          # 문항마다 점수 · 이유 · 세부 · 오류
report.explain("P001", "faithfulness")      # 한 칸의 이유 (모델을 다시 부르지 않음)
```

- `records`는 pandas 표나 사전 목록입니다. 근거 칸은 글자 목록, LangChain `Document` 목록, `{"text": ...}` 사전 목록을 받습니다.
- 없는 열, 겹친 번호, 잘못된 모양은 **몇 번째 기록의 어느 칸인지** 알려 주고 멈춥니다. 기준 답이 빈 칸인 것은 괜찮습니다.
- `source="page"`를 적으면 근거의 쪽 번호를 `metadata["retrieved_ids"]`에 남깁니다(출처 검사에 씀).
- 문항이 많으면 `evaluate(..., max_workers=4)`로 함께 채점합니다.

## 만든 RAG를 실행하며 채점

```python
from kgonto_rag_eval import evaluate, run_rag

questions = [{"no": "P001", "q": "청년 월세 지원은 얼마까지인가요?", "정답": "월 최대 20만 원"}]
rag_data = run_rag(graph, questions, question="q", reference="정답", id="no",
                   contexts="tool", skip=["관련 근거 없음"])      # 도구가 돌려준 글을 근거로
rag_report = evaluate(rag_data, [Faithfulness()], llm=model)
```

| 내 RAG | 넣는 법 | 근거 |
|---|---|---|
| LangGraph 에이전트(`messages`) | `run_rag(graph, ..., contexts="tool")` | 도구가 돌려준 글. `tools=["search"]`로 도구를 고를 수 있음 |
| 상태에 근거 칸이 있는 그래프 | `run_rag(graph, ..., contexts="evidence")` | 그 칸의 목록 |
| 상태 칸이 다른 그래프 | `run_rag(lambda q: graph.invoke({"question": q}), ..., answer="답 칸", contexts="근거 칸")` | 그 칸의 목록 |
| 체인 · 함수 | `run_rag(chain.invoke, ..., contexts=None)` | 없음(답만 채점) |

- 질문마다 한 번씩 순서대로 실행하고, 기준 답은 RAG에 넘기지 않습니다. 한 문항이라도 실패하면 어느 문항인지 알리고 멈춥니다.
- 체크포인터가 있는 그래프는 문항마다 새 대화방(`thread_id`)으로 실행해 대화가 섞이지 않게 합니다.
- 부른 도구는 `metadata["tools_used"]`에 남습니다. 도구를 안 부른 것과, 불렀는데 못 찾은 것을 구별할 수 있습니다.

## 내 지표 만들기

함수 하나가 지표 하나입니다. 클래스를 만들지 않아도 됩니다.

```python
from kgonto_rag_eval import judge_metric, metric

@metric
def has_source(sample):                      # 모델을 부르지 않는 지표
    return "[p." in sample.response

@metric
def fact_hit(sample):                        # 점수, 이유, 세부를 함께
    facts = sample.metadata["expected_facts"]
    hits = [fact for fact in facts if fact in sample.response]
    return len(hits) / len(facts), f"{len(hits)}개 찾음", {"hits": hits}

polite = judge_metric("친절함", "존댓말로 답했고, 다음에 할 일을 알려 주었으면 통과")   # 모델이 판정

report = evaluate(dataset, [has_source, fact_hit, polite], llm=model)
```

- 돌려줄 수 있는 것: `True` / `False`, 0 ~ 1 숫자, `(점수, 이유)`, `(점수, 이유, 세부 사전)`, `None`(채점 대상 아님). 그 밖의 값이나 예외는 그 칸만 오류로 남습니다.
- 모델을 쓰는 함수 지표는 `@metric(needs_llm=True)` + `def 이름(sample, llm)`.
- `judge_metric`은 기본으로 질문과 답만 봅니다. 기준 답이나 근거를 보여 주려면 `fields=("user_input", "response", "reference")`처럼 고릅니다. 보내는 글은 `polite.messages(sample)`로 볼 수 있습니다.
- 점수식과 프롬프트를 세밀하게 다루려면 `Metric`을 상속해도 됩니다([지표 사전](metrics.md)의 지표들이 그렇게 되어 있습니다).
