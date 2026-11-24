import copy
import json
import pathlib

import pytest

from kgonto_rag_eval import EvaluationDataset, to_dataset

REC = [{"sample_id": "a", "question": "q1", "answer": "a1", "reference": "r1", "expected_facts": ["f"],
        "retrieved": [{"page": 3, "text": "t3"}, {"page": 1, "text": "t1"}]},
       {"sample_id": "b", "question": "q2", "answer": "", "reference": None, "expected_facts": [], "retrieved": []}]
ARGS = dict(contexts="retrieved", reference="reference", id="sample_id", keep=["expected_facts"], source="page")


def old_way(records):
    return EvaluationDataset.from_list([
        {"sample_id": r["sample_id"], "user_input": r["question"], "response": r["answer"], "reference": r["reference"],
         "retrieved_contexts": [d["text"] for d in r["retrieved"]],
         "metadata": {"expected_facts": r["expected_facts"], "retrieved_ids": [d["page"] for d in r["retrieved"]]}} for r in records])


def test_same_as_loop_and_input_untouched():
    before = copy.deepcopy(REC)
    assert to_dataset(REC, **ARGS) == old_way(REC)
    assert REC == before


def test_dataframe_same_as_list():
    pd = pytest.importorskip("pandas")
    assert to_dataset(pd.DataFrame(REC), **ARGS) == old_way(REC)
    df = pd.DataFrame(REC)
    df["reference"] = [float("nan"), pd.NA]
    df["expected_facts"] = [float("nan"), "x"]
    out = to_dataset(df, **ARGS).samples
    assert [s.reference for s in out] == [None, None] and out[0].metadata["expected_facts"] is None
    with pytest.raises(ValueError, match="겹칩니다"):
        to_dataset(pd.concat([df, df[["question"]]], axis=1), **ARGS)


def test_real_records_59_8():
    path = pathlib.Path(__file__).parents[3] / "notebooks/data/59-8_정책RAG_평가기록.jsonl"
    if not path.exists():
        pytest.skip("수업 자료가 없는 곳")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 40 and to_dataset(records, **ARGS) == old_way(records)


def test_context_shapes():
    from langchain_core.documents import Document
    items = [["글"], [Document(page_content="글", metadata={"page": 7})],
             [{"page_content": "글", "metadata": {"page": 7}}], ({"text": "글", "page": 7},)]
    for k, one in enumerate(items):
        ds = to_dataset([{"q": "q", "a": "a", "c": one}], question="q", answer="a", contexts="c", source="page" if k else None)
        assert ds.samples[0].retrieved_contexts == ["글"]
        assert ds.samples[0].metadata == ({"retrieved_ids": [7]} if k else {})


@pytest.mark.parametrize("change, args, error", [
    ({}, {"answer": "reply"}, KeyError),
    ({"question": None}, {}, ValueError),
    ({"question": " "}, {}, ValueError),
    ({"answer": float("nan")}, {}, ValueError),
    ({"retrieved": "글 하나"}, {}, ValueError),
    ({"retrieved": None}, {}, ValueError),
    ({"retrieved": [{"page": 1, "body": "x"}]}, {}, ValueError),
    ({"retrieved": [{"page": 1, "text": "x", "page_content": "y"}]}, {}, ValueError),
    ({"retrieved": [{"text": "x"}]}, {}, ValueError),
    ({"retrieved": [1]}, {"source": None}, ValueError),
    ({"sample_id": "b"}, {}, ValueError),
    ({"sample_id": None}, {}, ValueError),
    ({"sample_id": 1.0}, {}, ValueError),
    ({}, {"keep": "expected_facts"}, TypeError),
    ({"retrieved_ids": [1]}, {"keep": ["retrieved_ids"]}, ValueError),
    ({}, {"contexts": None}, ValueError),
])
def test_bad_input_is_error(change, args, error):
    with pytest.raises(error):
        to_dataset([{**REC[0], **change}, REC[1]], **{**ARGS, **args})


def test_not_a_list():
    for bad in ("글", REC[0], [], ["글"]):
        with pytest.raises((TypeError, ValueError)):
            to_dataset(bad, **ARGS)


def test_int_id_and_default_id():
    rows = [{"question": "q", "answer": "a", "no": 12}]
    assert to_dataset(rows, id="no").samples[0].sample_id == "12"
    assert to_dataset(rows).samples[0].sample_id is None
