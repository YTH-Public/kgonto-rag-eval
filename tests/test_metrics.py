"""스모크 테스트: 근거 있는 답은 높게, 환각 답은 낮게 채점되는지 확인.

실제 gpt-5.4-mini 를 호출하므로 OPENAI_API_KEY 가 필요합니다.
    python tests/test_metrics.py
"""
from kgonto_rag_eval import EvaluationDataset, evaluate
from kgonto_rag_eval.metrics import AnswerCorrectness, ContextRecall, Faithfulness

GROUNDED = {
    "user_input": "청년월세지원 대상은 누구인가요?",
    "retrieved_contexts": [
        "청년월세지원은 무주택 청년에게 월세 일부를 지원한다. 소득 기준을 충족해야 한다.",
    ],
    "response": "무주택 청년이면서 소득 기준을 충족하면 대상입니다.",
    "reference": "무주택 청년 중 소득 기준을 충족하는 사람",
}

HALLUCINATED = {
    "user_input": "청년월세지원 대상은 누구인가요?",
    "retrieved_contexts": [
        "청년월세지원은 무주택 청년에게 월세 일부를 지원한다.",
    ],
    "response": "만 50세 이상 주택 소유자에게 매달 200만 원을 현금으로 지급합니다.",
    "reference": "무주택 청년 중 소득 기준을 충족하는 사람",
}


def test_faithfulness_separates_grounded_and_hallucinated():
    ds = EvaluationDataset.from_list([GROUNDED, HALLUCINATED])
    report = evaluate(ds, metrics=[Faithfulness(), ContextRecall(), AnswerCorrectness()])
    f_grounded = report.rows[0]["faithfulness"]
    f_hallucinated = report.rows[1]["faithfulness"]
    assert f_grounded > f_hallucinated, (f_grounded, f_hallucinated)
    assert f_grounded >= 0.5
    assert f_hallucinated <= 0.5
    # correctness 도 근거 있는 답이 더 높아야
    assert report.rows[0]["answer_correctness"] >= report.rows[1]["answer_correctness"]


if __name__ == "__main__":
    test_faithfulness_separates_grounded_and_hallucinated()
    print("OK: faithfulness/correctness smoke test 통과")
