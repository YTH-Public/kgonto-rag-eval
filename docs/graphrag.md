[← 처음으로](../README.md)

# GraphRAG 평가

그래프에 묻는 RAG는 답만 봐서는 어디서 틀렸는지 알 수 없습니다. **구축 → 검색 → 답**을 단계마다 봅니다.
모델도 DB 연결도 쓰지 않습니다. Neo4j에서 꺼낸 표를 넣고 표를 받습니다.

| 단계 | 함수 |
|---|---|
| 1. 구축 - 그래프를 제대로 만들었나 | `check_build` · `why_missed` · `check_load` · `break_graph` |
| 2. 검색 - 필요한 것을 찾았나 | `entity_link_recall` · `evaluate_cypher` · `evidence_bundle_coverage` |
| 3. 답 - 맞게 답했나 | `evaluate` ([RAG 채점](rag.md)) |
| 모아 보기 - 문항마다 처음 끊긴 단계 | `stage_table` |

```python
from kgonto_rag_eval.graphrag import break_graph, check_build, check_load, evaluate_cypher, stage_table, why_missed

# 표의 열 이름: 노드 id · label · name / 관계 source · relation · target (+ chunk_id · quote) / 원문 id · text
def table(cypher):
    return driver.execute_query(cypher, result_transformer_=Result.to_df)

nodes = table("MATCH (n:Article) RETURN n.id AS id, 'Article' AS label, n.id AS name")
relations = table("MATCH (a:Article)-[r]->(b:Article) RETURN a.id AS source, type(r) AS relation, b.id AS target")

check_build(nodes, relations, schema=[("Article", "CITES", "Article"), ("Article", "DELEGATES", "Article")])   # 모양 · 중복 · 출처
check_load(approved, relations)                       # 넣기로 한 목록과 대조 (문제가 없으면 빈 표)
why_missed(pred, gold, pred_names=names)              # 못 뽑은 관계마다 가능한 까닭

# 일부러 망가뜨린 그래프에 같은 질문을 던져 보기
broken, changes = break_graph(relations, "wrong_target", 0.1, seed=1, nodes=nodes)

# 만든 Cypher 를 실행해 결과로 비교 - 원본과 반례 데이터에서
cypher_questions = [{
    "question": "계약은 몇 건인가요?",
    "gold_cypher": "MATCH (c:Contract) RETURN count(c)",      # 사람이 쓴 정답
    "cypher": "MATCH (c:Contract) RETURN count(*)",           # 모델이 만든 것
}]
report = evaluate_cypher({"원본": run_real, "반례": run_other}, cypher_questions)
report.summary                                        # {"ex@원본": ..., "ex@반례": ..., "ex@all": ...}

# 문항마다 단계를 한 줄로
stage_table(records, ["graph", "entry", "query", "sent", "answer"], keep=["chars"])
```

| 함수 | 보는 것 | 주의 |
|---|---|---|
| `check_build` | 스키마 밖 관계, 끝점 없음, 고립, 중복, '당사' 같은 이름, 출처 번호와 근거 글이 원문에 있나 | 근거 글이 있다고 그 글이 관계를 **뒷받침한다**는 뜻은 아님 |
| `why_missed` | 개체를 못 뽑음 · 관계를 못 뽑음 · 방향 반대 · 종류 다름 | 뽑은 개체 **전체** 목록이 있어야 앞의 둘을 가름. 없으면 '알 수 없음' |
| `check_load` | 빠짐 · 더 들어감 · 중복 · 값이 다름, 두 번 넣어도 같은가(`again=`) | 같은 범위만 꺼내 비교 |
| `break_graph` | 관계를 지움 · 도착점을 바꿈 · 방향을 뒤집음(실험용) | 정답은 망가뜨리기 전 그래프로 정해 둠 |
| `entity_link_recall` | 질문의 출발점을 찾은 후보 안에 있나 | 이름이 아니라 ID로 |
| `evaluate_cypher` | 실행 결과가 정답과 같은가 | 조건을 빠뜨린 질의가 우연히 맞을 수 있어 반례 데이터로도 실행. 쓰는 말이 든 질의는 막지만 읽기 전용 계정을 함께 쓸 것 |
| `compare_result_rows` | 결과 두 개를 열 종류 · 순서 · 중복 규칙까지 정해 비교 | 실행 오류를 빈 결과로 바꾸지 않음 |
| `evidence_bundle_coverage` | 답 모델에 실제로 보낸 근거가 충분한가(묶음 안은 모두, 묶음끼리는 하나) | 세기 · 전부 · 없음을 묻는 질문은 이것만으로 판단하지 않음 |
| `stage_table` | 처음 끊긴 단계 | 원인이 아니라 먼저 볼 곳. 종합 점수는 내지 않음 |
