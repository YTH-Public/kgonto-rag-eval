[← 처음으로](../README.md)

# 검색 채점

벡터DB나 검색 함수가 정답 문서를 찾는지 잽니다. 모델을 부르지 않습니다.

## 먼저 눈으로 보기

그대로 복사해서 실행할 수 있습니다. `search`는 질문을 받아 찾은 쪽 번호를 순서대로 돌려주는 함수입니다.

```python
from kgonto_rag_eval import evaluate_search

ranks = {"월세를 도와주는 제도": [35, 12, 8], "교육비 지원": [40, 3, 64], "아기 낳으면 받는 돈": [5, 9, 20]}

def search(question):
    return ranks[question]

questions = [{"질문": "월세를 도와주는 제도", "쪽": 35}, {"질문": "교육비 지원", "쪽": 64}, {"질문": "아기 낳으면 받는 돈", "쪽": 127}]
report = evaluate_search(search, questions, question="질문", gold="쪽", k=3)
report.to_pandas()[["user_input", "meta__retrieved_ids", "hit@3", "mrr@3"]]
```

| user_input | meta__retrieved_ids | hit@3 | mrr@3 |
|---|---|---:|---:|
| 월세를 도와주는 제도 | [35, 12, 8] | 1.0 | 1.00 |
| 교육비 지원 | [40, 3, 64] | 1.0 | 0.33 |
| 아기 낳으면 받는 돈 | [5, 9, 20] | 0.0 | 0.00 |

둘째 질문은 정답 64쪽을 3위로 찾아 1 / 3 입니다. 셋째는 못 찾았습니다.

## 벡터DB 채점하기

```python
from kgonto_rag_eval import evaluate_search

questions = [{"질문": "월세를 도와주는 제도", "쪽": 35}, {"질문": "교육비 지원 두 가지", "쪽": [40, 64]}]
report = evaluate_search(store, questions, question="질문", gold="쪽", key="page", k=3)   # 벡터DB 그대로
report.summary        # {"hit@3": ..., "mrr@3": ..., "recall@3": ...}
report.to_pandas()    # 질문마다 찾은 문서와 순위
```

- `store`는 `similarity_search`가 있는 벡터DB(PGVectorStore · Chroma · InMemoryVectorStore 등)나 **검색 함수**(질문 -> `Document` 목록 또는 ID 목록)입니다. BM25, RRF, 리랭커를 함수로 넣으면 같은 표로 비교됩니다.
- `hit@k` = 정답을 하나라도 찾았나, `mrr@k` = 첫 정답이 몇 번째인가(1 / 순위), `recall@k` = 정답 가운데 몇 개를 찾았나(정답이 둘 이상인 질문이 있을 때만).
- 검색이 실패한 질문은 0점이 아니라 **오류**로 남습니다.

질문과 정답이 문서의 낱말을 그대로 쓰면 점수가 높게 나옵니다. 사용자가 실제로 쓰는 말로도 재 보세요.
