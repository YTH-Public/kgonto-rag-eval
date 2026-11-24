"""검색 채점: 벡터DB 나 검색 함수를 질문 세트로 바로 재기 (Hit@k · MRR@k · Recall@k)."""
from __future__ import annotations

import inspect
import json
from typing import Optional

from .evaluate import evaluate
from .inputs import _chunk, _missing, as_id, as_records
from .metrics.base import Metric
from .schema import EvalReport, EvalSample, EvaluationDataset, MetricResult


class _Rank(Metric):
    """찾은 ID 의 순서와 정답 ID 로 점수를 내는 지표 (hit · mrr · recall)"""
    needs_llm = False

    def __init__(self, kind: str, k: int, key: Optional[str]):
        self.name, self.kind, self.k, self.key = f"{kind}@{k}", kind, k, key

    def score(self, sample: EvalSample, llm=None) -> MetricResult:
        found, gold, error = (sample.metadata[n] for n in ("retrieved_ids", "gold_ids", "search_error"))
        if not gold:
            return MetricResult(metric=self.name, reason="정답이 없는 질문", reason_code="no_gold")
        if error:
            return MetricResult(metric=self.name, reason=error, error=error, failed_component="retriever")
        ranks = [i for i, one in enumerate(found, start=1) if one in gold]      # 정답이 나온 순위
        if self.kind == "hit":                                                  # 하나라도 찾았나
            score = 1.0 if ranks else 0.0
        elif self.kind == "mrr":                                                # 첫 정답이 몇 번째인가 (1 / 순위)
            score = 1 / ranks[0] if ranks else 0.0
        else:                                                                   # 정답 가운데 몇 개를 찾았나
            score = len(set(found) & set(gold)) / len(set(gold))
        return MetricResult(metric=self.name, score=score, details={"first_rank": ranks[0] if ranks else None})


def _found(result, key: Optional[str], k: int, where: str) -> tuple[list, list]:
    """검색 결과 -> (ID 목록, 글 목록)"""
    if result is None or isinstance(result, (str, bytes, dict)):
        raise TypeError(f"{where} 검색 결과가 목록이 아닙니다: {type(result).__name__}")
    ids, texts = [], []
    for j, item in enumerate(list(result)[:k]):
        if hasattr(item, "page_content"):
            if key is None:
                raise ValueError("검색 결과가 Document 라면 key 에 정답과 맞춰 볼 metadata 이름을 적으세요(예 key='page').")
            text, one = _chunk(item, key, f"{where} {j + 1}위 결과")
            texts.append(text)
        else:
            one = item
        ids.append(as_id(one, f"{where} {j + 1}위 결과"))
    return ids, texts


def evaluate_search(search, questions, *, question: str = "question", gold: str = "gold", key: Optional[str] = None,
                    id: Optional[str] = None, k: int = 3, run_meta: Optional[dict] = None) -> EvalReport:
    """벡터DB 나 검색 함수를 질문마다 한 번 불러, 정답 문서를 위 k개 안에 찾았는지 잽니다.

    search   : similarity_search 가 있는 벡터DB, 또는 검색 함수(질문 -> Document 목록이나 ID 목록, k 인자는 있으면 넘김)
    question · gold · id : questions 의 열 이름. gold 칸은 정답 ID 하나나 목록 (비면 그 질문은 채점하지 않음)
    key      : 찾은 Document 에서 ID 로 쓸 metadata 이름 (ID 목록을 돌려주는 함수면 안 적음)

    hit@k    = 정답을 하나라도 찾았으면 1
    mrr@k    = 첫 정답의 순위가 r 이면 1 / r, 없으면 0 (표의 값은 질문 하나의 1 / r, summary 는 평균)
    recall@k = 정답 가운데 찾은 비율 (정답이 둘 이상인 질문이 있을 때만)
    ID 는 글자로 바꿔 비교합니다(12 와 "12" 는 같고 "012" 는 다름). 같은 문서가 두 번 나와도 순위를 당기지 않습니다.
    검색이 실패한 질문은 0점이 아니라 오류로 남고, 찾은 것이 없으면 0점입니다.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k 는 1 이상의 정수여야 합니다: {k!r}")
    if hasattr(search, "similarity_search"):
        call = lambda q: search.similarity_search(q, k=k)
    elif callable(search):
        params = list(inspect.signature(search).parameters.values())
        if any(p.name == "k" and p.kind is p.POSITIONAL_ONLY for p in params):
            call = lambda q: search(q, k)
        else:
            call = (lambda q: search(q, k=k)) if any(p.name == "k" or p.kind is p.VAR_KEYWORD for p in params) else search
    else:
        raise TypeError("search 에는 similarity_search 가 있는 벡터DB 나 검색 함수를 넣으세요.")

    rows, seen = [], set()
    for i, record in enumerate(as_records(questions)):                          # 검색하기 전에 질문 표부터 검사
        for name in [question, gold, id]:
            if name is not None and name not in record:
                raise KeyError(f"{i}번째 질문에 '{name}' 열이 없습니다. 있는 열: {list(record)}")
        text, answers = record[question], record[gold]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{i}번째 질문의 '{question}' 칸이 비었거나 글자가 아닙니다: {text!r}")
        answers = list(answers) if isinstance(answers, (list, tuple, set)) else [] if _missing(answers) else [answers]
        answers = [as_id(a, f"{i}번째 질문의 '{gold}'") for a in answers]
        sample_id = as_id(record[id], f"{i}번째 질문의 '{id}'") if id is not None else None
        if sample_id is not None and sample_id in seen:
            raise ValueError(f"{i}번째 질문의 '{id}' 값이 앞의 질문과 겹칩니다: {sample_id!r}")
        seen.add(sample_id)
        rows.append((sample_id, text, answers))

    samples = []
    for i, (sample_id, text, answers) in enumerate(rows):
        result, error = [], None
        try:
            result = call(text)
            result = result if result is None or isinstance(result, (str, bytes, dict)) else list(result)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:300]}"
        ids, texts = _found(result, key, k, f"{i}번째 질문의")                     # ID 를 못 읽는 것은 설정 실수라 바로 알림
        samples.append(EvalSample(sample_id=sample_id, user_input=text, reference=json.dumps(sorted(set(answers)), ensure_ascii=False) if answers else None, retrieved_contexts=texts,
                                  metadata={"retrieved_ids": ids, "gold_ids": answers, "search_error": error}))
    kinds = ["hit", "mrr"] + (["recall"] if any(len(set(a)) > 1 for _, _, a in rows) else [])
    return evaluate(EvaluationDataset(samples=samples), [_Rank(kind, k, key) for kind in kinds], run_meta=run_meta)
