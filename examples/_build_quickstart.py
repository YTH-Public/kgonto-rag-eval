"""examples/01_quickstart.ipynb 생성기."""
import nbformat as nbf
import pathlib

ROOT = pathlib.Path(__file__).parent

def md(t): return nbf.v4.new_markdown_cell(t.strip("\n"))
def code(t): return nbf.v4.new_code_cell(t.strip("\n"))

nb = nbf.v4.new_notebook()
nb.cells = [
    md("""
# kgonto-rag-eval 빠른 시작

LLM-judge 로 RAG 답변을 평가합니다. 손으로 만든 작은 평가셋으로 6개 지표를 돌려봅니다.

```bash
uv pip install -e ./kgonto-rag-eval
```
> `.env` 에 `OPENAI_API_KEY` 필요. 평가자(judge)는 기본 `gpt-5.4-mini`.
"""),

    code('''from dotenv import load_dotenv
load_dotenv()
'''),

    md("""
## 1. 평가 데이터셋

각 샘플은 질문(`user_input`), 검색 문맥(`retrieved_contexts`), RAG 답변(`response`), 기준 정답(`reference`)으로 구성합니다.
"""),
    code('''from kgonto_rag_eval import EvaluationDataset

ds = EvaluationDataset.from_list([
    {   # 근거 있는 좋은 답
        "user_input": "청년월세지원 대상은 누구인가요?",
        "retrieved_contexts": [
            "청년월세지원은 무주택 청년에게 월세 일부를 지원한다. 소득 기준을 충족해야 한다.",
        ],
        "response": "무주택 청년이면서 소득 기준을 충족하면 대상입니다.",
        "reference": "무주택 청년 중 소득 기준을 충족하는 사람",
    },
    {   # 문맥은 맞지만 답이 부실(누락)
        "user_input": "국민취업지원제도는 무엇을 주나요?",
        "retrieved_contexts": [
            "국민취업지원제도는 구직자에게 구직촉진수당을 지급하고 취업지원 서비스를 함께 제공한다.",
        ],
        "response": "취업을 도와줍니다.",
        "reference": "구직촉진수당을 지급하고 취업지원 서비스를 제공한다.",
    },
    {   # 환각: 문맥에 없는 수치를 지어냄
        "user_input": "내일배움카드 지원 내용은?",
        "retrieved_contexts": [
            "내일배움카드는 직업훈련 비용을 한도 안에서 보조한다.",
        ],
        "response": "매달 300만 원을 현금으로 지급하고 해외 연수도 보내줍니다.",
        "reference": "직업훈련 비용을 한도 안에서 보조한다.",
    },
])
print("샘플:", len(ds), "건")
'''),

    md("""
## 2. 평가 실행

6개 지표로 채점합니다. `reference` 가 필요한 지표(context_recall, answer_correctness)는 reference 가 없으면 자동으로 건너뜁니다.
"""),
    code('''from kgonto_rag_eval import evaluate
from kgonto_rag_eval.metrics import (
    Faithfulness, ContextRelevance, ContextPrecision,
    ContextRecall, AnswerCorrectness, AnswerRelevancy,
)

report = evaluate(ds, metrics=[
    Faithfulness(), ContextRelevance(), ContextPrecision(),
    ContextRecall(), AnswerCorrectness(), AnswerRelevancy(),
])

print("=== 지표별 평균 ===")
for name, avg in report.summary.items():
    print(f"  {name}: {avg}")
'''),

    md("""
## 3. 샘플별 점수표
"""),
    code('''cols = ["faithfulness", "context_relevance", "context_precision",
        "context_recall", "answer_correctness", "answer_relevancy"]
header = "질문".ljust(28) + "".join(c[:12].ljust(13) for c in cols)
print(header)
for r in report.rows:
    line = r["user_input"][:26].ljust(28)
    line += "".join(str(r.get(c)).ljust(13) for c in cols)
    print(line)
'''),

    md("""
> 1번(근거 있는 답)은 faithfulness·correctness 가 높고, 3번(환각)은 faithfulness 가 낮게 나옵니다. 2번(부실한 답)은 correctness·answer_relevancy 가 낮습니다.

## 4. 지표로 문제 진단

점수가 낮을 때 어디가 문제인지 나눠 봅니다.

- `faithfulness` 낮음 → **생성** 문제(문맥에 없는 말). 환각.
- `context_recall` / `context_relevance` / `context_precision` 낮음 → **검색** 문제(근거를 못 찾음/순위가 나쁨).
- `answer_correctness` 낮은데 `context_recall` 은 높음 → 검색은 됐는데 **답을 잘못** 정리.
- `answer_relevancy` 낮음 → 질문 의도를 빗나감(사실성과 별개).
"""),
    code('''# verdict(pass/partial/fail) 로 빠르게 훑기
for r in report.rows:
    print(f"\\n[{r['user_input'][:30]}]")
    for c in cols:
        print(f"  {c}: {r.get(c)} ({r.get(c + '__verdict')})")
'''),

    md("""
## 정리

- `EvaluationDataset.from_list` → `evaluate(ds, metrics=[...])` → `report.summary` / `report.rows`
- 지표는 모두 0~1, LLM-judge(gpt-5.4-mini). reference 없으면 reference 필요 지표는 건너뜀
- 점수는 절대값보다 **변경 전후 상대 비교**로 보세요. judge=피평가 동일 모델 한계가 있습니다
- 지식그래프가 있다면 `kgonto_rag_eval.kg` 로 multi-hop 평가셋을 자동 생성할 수도 있습니다 (README 참고)
"""),
]

nbf.write(nb, str(ROOT / "01_quickstart.ipynb"))
print("[OK] 01_quickstart.ipynb")
