# kgonto-rag-eval

**내 RAG가 어느 질문에서, 왜 틀렸는지 표로 보여 줍니다.**

질문과 답이 든 표를 넣으면, 질문마다 점수와 이유가 붙은 표가 나옵니다.

[시작하기](#시작하기) · [하고 싶은 일 고르기](#하고-싶은-일-고르기) · [지표 고르기](#지표-고르기) · [막힐 때](#막힐-때)

---

## 시작하기

### 1. 설치

```bash
pip install "kgonto-rag-eval[pandas] @ git+https://github.com/YTH-Public/kgonto-rag-eval@v0.4.0"
```

### 2. 첫 채점 (API 키 없이)

그대로 복사해서 실행해 보세요. 채점 기준은 내가 쓴 함수입니다.

```python
from kgonto_rag_eval import evaluate, metric, to_dataset

# 내 RAG 가 낸 답 기록
records = [
    {"no": "Q1", "question": "월세 지원은 한 달에 얼마인가요?", "answer": "월 최대 20만 원입니다 [p.35].", "fact": "20만 원"},
    {"no": "Q2", "question": "몇 달까지 받나요?", "answer": "자료에 없습니다.", "fact": "24개월"},
    {"no": "Q3", "question": "누가 받나요?", "answer": "19세부터 34세까지 청년입니다.", "fact": "34세"},
]

# 채점 기준 = 함수 하나
@metric
def 정답이_있나(sample):
    return sample.metadata["fact"] in sample.response

@metric
def 출처를_적었나(sample):
    return "[p." in sample.response

dataset = to_dataset(records, id="no", keep=["fact"])
report = evaluate(dataset, [정답이_있나, 출처를_적었나])

print(report.summary)
report.to_pandas()[["sample_id", "response", "정답이_있나", "출처를_적었나"]]
```

**결과**

`{'정답이_있나': 0.667, '출처를_적었나': 0.333}`

| sample_id | response | 정답이_있나 | 출처를_적었나 |
|---|---|---:|---:|
| Q1 | 월 최대 20만 원입니다 [p.35]. | 1.0 | 1.0 |
| Q2 | 자료에 없습니다. | 0.0 | 0.0 |
| Q3 | 19세부터 34세까지 청년입니다. | 1.0 | 0.0 |

### 3. 모델에게 채점 맡기기 (API 키 필요)

뜻을 읽어야 하는 채점은 모델이 합니다. `OPENAI_API_KEY`를 `.env`나 환경 변수에 넣어 두세요.

```python
from dotenv import load_dotenv            # pip install python-dotenv
from langchain.chat_models import init_chat_model
from kgonto_rag_eval import evaluate, judge_metric, to_dataset
from kgonto_rag_eval.metrics import Faithfulness

load_dotenv()
records = [
    {"no": "Q1", "question": "월세 지원은 한 달에 얼마인가요?",
     "answer": "월 최대 30만 원입니다.",
     "found": ["월 최대 20만 원을 최장 24개월 지원한다."]},
    {"no": "Q2", "question": "몇 달까지 받나요?",
     "answer": "최장 24개월 받을 수 있습니다. 신청은 복지로에서 하세요.",
     "found": ["월 최대 20만 원을 최장 24개월 지원한다."]},
]
dataset = to_dataset(records, contexts="found", id="no")          # found = 그때 검색된 근거

친절함 = judge_metric("친절함", "존댓말로 답했고, 다음에 할 일을 알려 주었으면 통과")
model = init_chat_model("openai:gpt-6-luna", reasoning_effort="none")
report = evaluate(dataset, [Faithfulness(), 친절함], llm=model)
report.to_pandas()[["sample_id", "faithfulness", "faithfulness__reason", "친절함"]]
```

**결과** (실행 예, 모델 판정은 실행마다 조금 다릅니다)

| sample_id | faithfulness | faithfulness__reason | 친절함 |
|---|---:|---|---:|
| Q1 | 0.0 | 참고 문맥에는 월 최대 20만 원을 지원한다고 되어 있습니다. 답변의 30만 원이라는 금액은 문맥에 의해 뒷받침되지 않습니다. | 0.0 |
| Q2 | 0.5 | 문맥은 지원 기간이 최장 24개월이라고 뒷받침합니다. 신청처가 복지로라는 내용은 문맥에 없습니다. | 1.0 |

점수만 보지 말고 **이유**를 읽으세요. Q2는 맞는 말도 했지만 근거에 없는 말을 덧붙였습니다.

---

## 하고 싶은 일 고르기

| 하고 싶은 일 | 쓰는 함수 | 자세히 |
|---|---|---|
| 저장해 둔 답을 채점 | `to_dataset` → `evaluate` | [기록 채점](docs/rag.md) |
| 만든 RAG를 실행하며 채점 | `run_rag` → `evaluate` | [실행하며 채점](docs/rag.md#만든-rag를-실행하며-채점) |
| 내 기준으로 채점 | `@metric` · `judge_metric` | [내 지표 만들기](docs/rag.md#내-지표-만들기) |
| 벡터DB 검색이 잘 되는지 | `evaluate_search` | [검색 채점](docs/search.md) |
| 고친 뒤 나빠지지 않았는지 | `save_baseline` · `compare_baseline` | [전후 비교](docs/baseline.md) |
| 답의 출처가 맞는지 | `citation_support` 등 | [인용 검사](docs/citation.md) |
| 지식그래프로 시험 문제 만들기 | `kgonto_rag_eval.kg` | [지식그래프](docs/kg.md) |
| GraphRAG가 어디서 틀리는지 | `kgonto_rag_eval.graphrag` | [GraphRAG 평가](docs/graphrag.md) |

모두 같은 순서입니다.

**표 넣기 → 채점 → 틀린 질문의 이유 읽기 → 하나만 고치기 → 다시 채점**

---

## 지표 고르기

알고 싶은 것부터 고르세요. 지표를 다 돌릴 필요는 없습니다.

| 알고 싶은 것 | 지표 | 필요한 것 |
|---|---|---|
| 근거에 없는 말을 지어냈나 | `Faithfulness` | 근거 |
| 정답과 같은 내용인가 | `AnswerCorrectness` | 기준 답 |
| 꼭 들어가야 할 사실이 있나 | `FactCoverage` | 필수사실 |
| 질문에 답했나 | `AnswerRelevancy` | 없음 |
| 거절한 답을 골라내고 싶다 | `AnsweredRelevancy` | 없음 |
| 필요한 내용을 찾아 왔나 | `ContextRecall` | 근거 · 기준 답 |
| 답의 출처가 검색된 자료인가 | `CitedRetrieved` | 검색된 자료 번호 |

> **하나만 보면 속습니다.** 틀린 자료를 그대로 옮긴 답도 `Faithfulness`는 높습니다. 정답과 비교하는 지표를 함께 보세요.

전체 지표와 계산 방법, RAGAS · DeepEval과 다른 점은 [지표 사전](docs/metrics.md)에 있습니다.

---

## 막힐 때

| 이런 일이 생기면 | 이렇게 하세요 |
|---|---|
| `'answer' 열이 없습니다` | 내 표의 열 이름을 적습니다. `to_dataset(표, question="질문", answer="답")` |
| `'found' 칸은 목록이어야 합니다` | 근거가 글 하나여도 목록으로 넣습니다. `["근거 글"]` |
| 점수가 빈 칸(NaN)이다 | `지표이름__status`를 봅니다. `not_applicable`은 채점 대상이 아님, `error`는 실패입니다 |
| 왜 이 점수인지 모르겠다 | `report.explain("Q1", "faithfulness")` |
| 오래 걸린다 | `evaluate(..., max_workers=4)` |
| 모델을 안 쓰는데 API 키를 찾는다 | `@metric`으로 만들면 키가 필요 없습니다 |
| 평균은 높은데 답이 이상하다 | `report.stats`에서 몇 건의 평균인지, 오류가 몇 건인지 봅니다 |

점수는 그 모델과 그 프롬프트로 잰 **비교용 값**입니다. 고치기 전과 후를 견주는 데 쓰고, 일부는 사람이 직접 읽어 보세요.

---

## 더 알아보기

- [바뀐 기록](CHANGELOG.md) · [만든 까닭과 근거 논문](docs/about.md) · [옛 판에서 옮겨 올 때](docs/about.md#옮겨-올-때)
- 기능에 따라 덧붙여 설치합니다: 유사도 지표 `similarity`, 지식그래프 `kg`, GraphRAG `graphrag`. 예) `kgonto-rag-eval[pandas,kg,similarity]`
- 파이썬 3.12 이상, LangChain v1 위에서 동작합니다. OpenAI가 아닌 LangChain 모델도 `llm=`에 넣을 수 있습니다.
- MIT 라이선스
