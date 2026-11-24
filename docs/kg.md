[← 처음으로](../README.md)

# 지식그래프

지식그래프로 평가 문제를 만들고, 추출한 그래프를 정답과 비교합니다. `pip install "kgonto-rag-eval[kg] @ ..."`이 필요합니다.

```python
from kgonto_rag_eval.kg import (
    compare_kg, generate_testset_from_kg, leakage_report, load_kg, save_jsonl, schema_violation_rate, to_eval_dataset, validate_item,
)

# 평가셋 초안 만들기 -> 자동 검사 -> 사람이 keep / edit / drop
graph = load_kg(entities="entities.json", relations="relations.json")
items = generate_testset_from_kg(graph, model, max_per_type=6)
print([validate_item(item) for item in items], leakage_report(items))
items[0].review_status = "keep"
save_jsonl(items, "evalset.jsonl")
dataset = to_eval_dataset(items, rag_fn=my_rag)             # my_rag(question) -> (answer, 실제로 검색된 contexts)

# 추출한 그래프를 정답과 비교 (모델 없음)
compare_kg(pred_rows, gold_rows, normalize=lambda s: s.replace("(주)", ""), scope=("rcept_no",))
schema_violation_rate(entities, relations, allowed_types={"회사", "품목"},
                      allowed_patterns={("회사", "공급", "회사"), ("회사", "매입", "품목")})
```

| 함수 | 보는 것 | 근거 · 성격 |
|---|---|---|
| `compare_kg` | 추출 KG vs 정답 트리플 P / R / F1, 누락 · 추가 | 트리플 P/R/F1 (Mondal et al. 2021, arXiv:2106.01167) |
| `check_ontology` · `schema_violation_rate` | 허용 타입 · 관계 · 패턴 · dangling 위반 | W3C SHACL의 동기. 추론 엔진 아님 |
| `non_isolated_node_ratio` | 관계를 가진 노드 비율 | 이 프로젝트의 구조 보조 지표 |
| `path_entity_recall` | 정답 경로 이름이 검색 결과에 나온 비율(방향은 안 봄) | Recall@k를 경로 이름에 옮긴 이 프로젝트의 지표 |
| `triple_coverage` | 정답 근거 트리플을 검색 트리플이 덮은 비율(방향까지) | RAGAS context recall의 생각을 트리플에 옮긴 것 |
| `source_traceability` | 출처 칸이 있는 비율 | 출처가 **붙었는지**만 봄 |

- `validate_item`을 통과해도 진짜 multi-hop이라는 보장은 아닙니다. 핵심 관계 하나를 가려도 답할 수 있는지 사람이 확인하세요.
- 평가셋을 만든 KG와 채점 정답이 같은 KG라서 생기는 순환은 코드로 잡을 수 없습니다.
- `normalize`를 넓게 잡으면(모든 괄호 · 공백 제거) 서로 다른 대상이 합쳐집니다.
