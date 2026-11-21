# 변경 기록

## 0.2.0 (2026-11)

잘못된 점수가 나오던 곳부터 고쳤습니다.

### 고친 것
- `EvalSample.retrieved_contexts` 에 문자열 하나를 넣으면 글자 단위로 쪼개지던 문제 → 오류로 막음.
- judge 개수가 음수이거나 부분 > 전체여도 1.0 이 되던 문제(Faithfulness · ContextRecall · AnswerRelevancy) → `error` 로 남기고 점수 없음.
- ContextRelevance 가 같은 번호를 두 번 판정해도 1.0 이 되던 문제, ContextPrecision 이 개수가 모자라면 무관으로 채우던 문제 → judge 오류.
- `source_traceability` 가 관계의 `source`(주어)를 문헌 출처로 세던 문제.
- 허용 목록을 generator 로 주면 정상 항목이 모두 위반이 되던 문제(`schema_violation_rate`).
- 같은 두 노드 사이의 관계가 하나만 남던 문제 → `load_kg` 가 MultiDiGraph, `enumerate_paths` 가 관계마다 경로.
- `to_eval_dataset` 이 drop · pending 문항도 실행하던 문제 → 기본은 keep · edit 만.
- `ContextRecall` 에 질문을 함께 줌(질문 조건을 되풀이한 정답 부분을 주장으로 세지 않게).

### 더한 것
- `evaluate`: 행에 `sample_id` · 입력 · `metadata` · 판정 세부 · 오류, `report.stats`(유효 · 누락 · 오류 개수). judge 호출 실패에도 멈추지 않음.
- `compare_kg` · `triple_coverage`: `normalize`, `scope`(문서 범위), subject/object 칸 이름.
- `check_ontology` · `schema_violation_rate`: `allowed_patterns`(주어 타입, 관계, 목적어 타입).
- `path_entity_recall`(이름만 본다는 것을 이름에 드러냄, `path_recall` 은 호환용으로 남김).
- `leakage_report`, `citation`(인용 검사 세 가지), `pairwise_judge`, `baseline`(기준선 저장 · 비교).
- 오프라인 테스트(가짜 판정자)와 실호출 테스트(`tests/live/`, 키 없으면 건너뜀)를 나눔.

### 바뀐 것
- 기본 judge: `gpt-5.4-mini` → `gpt-6-luna`(reasoning_effort none). 노트북에서는 `evaluate(llm=...)` 로 직접 넣는 것을 권장.
- README: 지표 정의 비교 표(논문 / RAGAS 0.4.4 / DeepEval 4.3.4 / 이 패키지), 근거 절에서 원 논문 지표와 라이브러리 확장 · 자체 정의를 구분.

## 0.1.0 (2026-07)
- 첫 공개.
