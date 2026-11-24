# 변경 기록

## 0.4.0 (2026-11)

GraphRAG 평가 모듈 `kgonto_rag_eval.graphrag` 를 더했습니다(선택 설치 `graphrag`). 모델 · DB 연결 없이 표를 넣고 표를 받습니다. `kgonto_rag_eval.kg` 는 그대로입니다.

- 구축: `check_build`(모양 · 중복 · 출처) · `why_missed`(못 뽑은 까닭) · `check_load`(승인 목록과 적재 대조) · `break_graph`(오류 넣기 실험)
- 검색: `entity_link_recall`(출발점) · `evaluate_cypher`(원본 · 반례에서 실행 결과 비교) · `compare_result_rows` · `evidence_bundle_coverage`(보낸 근거)
- 한 표: `stage_table`(문항마다 처음 끊긴 단계, 종합 점수 없음)

## 0.3.0 (2026-11)

초보가 쓰기 쉽게 고치고 새 지표를 더했습니다. 기존 여섯 지표의 이름 · 프롬프트 · 점수식은 0.2 와 같습니다. 새 지표는 골라 쓰는 것이라 `CORE_METRICS` · `ALL_METRICS` 에 넣지 않았습니다.

### 쓰기 쉽게
- `to_dataset(표나 기록, question=, answer=, contexts=, ...)`: 내 자료의 열 이름만 적으면 평가 입력이 됩니다. 글자 · `Document` · `{"text": ...}` 근거를 받고, 없는 열 · 겹친 번호 · 값 충돌은 몇 번째 기록의 어느 칸인지 알립니다.
- `@metric`: 함수 하나가 지표 하나. 참/거짓 · 0 ~ 1 · `(점수, 이유)` · `(점수, 이유, 세부)` · `None` 을 돌려주면 됩니다.
- `judge_metric(이름, 기준 문장, fields=)`: 기준 문장 하나로 통과 / 실패 지표. 보내는 글은 `.messages(sample)`.
- `evaluate_search(벡터DB나 검색 함수, 질문 표)`: hit@k · mrr@k(정답이 여럿이면 recall@k).
- `run_rag(그래프나 함수, 질문 표, contexts=)`: RAG 를 질문마다 한 번 실행해 평가 입력으로. 체크포인터가 있으면 문항마다 새 대화방.
- `AnswerSimilarity(embeddings=)`: 답과 기준 답의 코사인 유사도(선택 설치 `similarity`).

### 더한 지표
- `AnsweredRelevancy`: 답했는지 먼저 보고, 답하지 않았으면 0, 답했으면 `AnswerRelevancy` 그대로(호출 1 ~ 2번).
- `FactCoverage`: 필수사실(`metadata["expected_facts"]`)마다 있음 · 빠짐 · 어긋남과 답의 근거 문구(호출 1번). 있다고 한 문구가 답에 없으면 오류, 답하지 않은 답은 코드로 0점.
- `FactHit` · `CitedRetrieved`: 모델을 부르지 않는 코드 지표.

### 더한 것
- `diagnose(report)`: 점수를 조합해 의심할 곳(검색 · 생성 · 거절 · 다른 자료)을 찾음. 원인을 확정하지 않음.
- `extract_citations`, `split_cited_sentences` · `citation_support` 의 `ranges=True`(출처 범위 [p.1-3] 펼치기, 기본은 전과 같음).
- `report.meta`(패키지 판 · 지표 코드와 설정의 지문 · 판정 모델과 옵션 · 문항 해시 · `run_meta`), `compare_baseline(strict_meta=True)`. `run_meta` 에 비밀 값으로 보이는 이름이 있으면 ValueError.
- 행마다 `__status`(ok · not_applicable · error) · `__reason_code`, `stats` 에 `not_applicable`.

### 바뀐 것 (옮겨 올 때 볼 것)
- 지표 점수가 0 ~ 1 밖이거나 `score` 와 `error` 를 함께 주면 그 칸은 오류입니다(직접 만든 지표 확인).
- 지표 이름으로 표의 칸 이름(`response` · `sample_id` 등)이나 `__` 가 든 이름을 쓰면 ValueError.
- 기준 답이 공백뿐이면 기준 답이 없는 것으로 봅니다(`requires_reference` 지표).
- `evaluate` 에 같은 이름의 지표나 같은 `sample_id`(순번으로 채운 것 포함)를 주면 ValueError.
- `failed_component`: 모델 호출 안에서 난 오류는 judge, 코드 지표나 지표 코드의 실수는 metric(전에는 모두 judge).

## 0.2.1 (2026-11)

쓰기 편하게 고쳤습니다. **지표의 프롬프트 · 점수식은 0.2.0 과 같습니다.**

### 더한 것
- `Metric.needs_llm`(기본 True): 코드 지표에 `needs_llm = False` 를 적으면 `evaluate` 가 기본 판정 모델을 만들지 않습니다.
- `evaluate(..., max_workers=4)`: 샘플을 동시에 채점합니다(기본 1, 행 순서는 입력 순서).
- `report.explain(sample_id, metric)`: 저장된 점수 · 이유 · 세부를 꺼냅니다(모델을 다시 부르지 않음).
- `citation_support`: 문장마다 `reason` · `judgements`(판정별 이유), 전체에 `attempted`. 기존 칸과 점수는 그대로.
- `ESG_PERSONAS`.

### 고친 것
- Faithfulness 의 빈 답 · 빈 근거가 0점인데 `verdict` 가 비어 있던 문제.
- `pairwise_judge`: AB 순서가 실패하면 BA 순서를 묻지 않던 문제 → 둘을 따로 묻고, 하나라도 실패하면 `final="error"`.

### 바뀐 것
- `DEFAULT_PERSONAS` 가 ESG 예시에서 분야를 가리지 않는 세 가지로 바뀌었습니다.
  전처럼 쓰려면 `generate_testset_from_kg(..., personas=ESG_PERSONAS)`.

## 0.2.0 (2026-11)

잘못된 점수가 나오던 곳부터 고쳤습니다.

### 고친 것
- `citation_support`: precision 은 결합 판정이 True 일 때만 출처의 쓸모를 셈(ALCE 와 같게), 문장의 모든 판정 실패를 오류로 남김, 원 논문 분모의 `citation_recall_all` 추가
- `EvalSample.retrieved_contexts` 에 문자열 하나를 넣으면 글자 단위로 쪼개지던 문제 → 오류로 막음.
- judge 개수가 음수이거나 부분 > 전체여도 1.0 이 되던 문제(Faithfulness · ContextRecall · AnswerRelevancy) → `error` 로 남기고 점수 없음.
- ContextRelevance 가 같은 번호를 두 번 판정해도 1.0 이 되던 문제, ContextPrecision 이 개수가 모자라면 무관으로 채우던 문제 → judge 오류.
- `source_traceability` 가 관계의 `source`(주어)를 문헌 출처로 세던 문제.
- 허용 목록을 generator 로 주면 정상 항목이 모두 위반이 되던 문제(`schema_violation_rate`).
- 같은 두 노드 사이의 관계가 하나만 남던 문제 → `load_kg` 가 MultiDiGraph, `enumerate_paths` 가 관계마다 경로.
- `to_eval_dataset` 이 drop · pending 문항도 실행하던 문제 → 기본은 keep · edit 만.
- `ContextRecall` 에 질문을 함께 줌(질문 조건을 되풀이한 정답 부분을 주장으로 세지 않게).

- 지표 점수를 반올림하지 않음(기준선이 작은 퇴행을 놓치던 문제). 표시용 summary 만 반올림.
- 기준선: judge 실패 · 행 누락으로 비는 칸이 있으면 실패(`lost`), `sample_id` 중복 · 빈 지표는 오류.
- 인용: 마침표 뒤에 붙은 출처가 다음 문장으로 넘어가던 문제, 공백뿐인 인용이 일치로 나오던 문제, 판정 실패를 0점으로 세던 문제.
- `compare_kg(scope=...)` 에서 범위 칸이 빈 행은 오류(문서가 섞이던 문제). subject 칸이 있으면 source 칸보다 우선.
- `to_eval_dataset` 이 승인된 문항도 자동 검사를 하고, 근거를 문자열 하나로 주면 막음.
- `evaluate`: 오류 개수가 `keep_details=False` 에서도 맞게 세짐, 근거 원문(`retrieved_contexts`)과 `idx` 열을 행에 남김.
- `pairwise_judge`: 두 순서의 이유를 남기고, 한쪽 호출이 실패하면 `final="error"` 와 남은 결과를 돌려줌.

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
