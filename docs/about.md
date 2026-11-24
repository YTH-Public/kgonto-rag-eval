[← 처음으로](../README.md)

# 만든 까닭과 근거

## 만든 까닭

RAGAS, DeepEval은 좋은 도구입니다. 다만 수업에서 쓰다 보니 이런 일이 있었습니다.

- RAGAS는 0.4.3(2026-01) 뒤로 정식 릴리스가 10개월 비었습니다. 그동안 `import ragas`가 깨지고 새 OpenAI 모델에서 거부당했습니다(0.4.4, 2026-11-14에서 고쳐짐).
- DeepEval은 이름이 계속 바뀝니다(`GPTModel` → `OpenAIModel`, `LLMTestCaseParams` → `SingleTurnParams`).
- **같은 이름의 지표도 도구마다 정의가 다릅니다.**

그래서 두 도구의 좋은 관례(`EvaluationDataset`, metric 객체, `evaluate()`)를 따르되, 프롬프트와 계산이 다 보이는 짧은 코드로 다시 썼습니다.
기존 도구를 대신하기보다 함께 놓고 비교하는 용도를 권합니다. 의존성은 `langchain`, `langchain-openai`, `pydantic`입니다.

## 한계

- 점수는 그 평가자 · 프롬프트 아래의 상대 지표입니다. 변경 전후 비교와 사람 검수 표본으로 쓰세요.
- 검색 채점의 질문과 정답이 문서의 낱말을 그대로 쓰면 점수가 높게 나옵니다. 사용자가 실제로 쓰는 말로도 재 보세요.
- 작은 그래프는 의미 있는 multi-hop 질문 수가 적습니다.

## 근거 논문과 표준

- **RAGAS** - Es et al., [arXiv:2309.15217](https://arxiv.org/abs/2309.15217). 원 논문이 정의한 것은 faithfulness · answer relevance · context relevance 셋입니다. context precision / recall · answer correctness는 그 뒤 라이브러리가 더한 지표이고, 이 패키지의 프롬프트 · 가중치(0.6 / 0.4 등)는 이 프로젝트가 정한 것입니다.
- **ALCE** - Gao et al., [arXiv:2305.14627](https://arxiv.org/abs/2305.14627) - `citation_support`
- **MT-Bench** - Zheng et al., [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) - `pairwise_judge`
- **KG 구축 P/R/F1** - Mondal et al., [arXiv:2106.01167](https://arxiv.org/abs/2106.01167) - `compare_kg`
- **W3C SHACL** - [w3.org/TR/shacl](https://www.w3.org/TR/shacl/) - 스키마 검사의 동기
- **GraphRAG-Bench** - [arXiv:2506.02404](https://arxiv.org/abs/2506.02404) - 구축 · 검색 · 생성을 나눠 보는 관점
- **Dissecting GraphRAG** - [TACL 2026](https://aclanthology.org/2026.tacl-1.29.pdf) 5.2절 - 정답 관계를 넣은 그래프와 추출한 그래프를 같은 질문으로 비교하는 설계
- **TRIAGE** - [arXiv:2607.03447](https://arxiv.org/html/2607.03447v1) - 추출 · 그래프 · 사용을 나눠 보는 흐름만 빌림(추출 지표 · 합격선은 쓰지 않음)
- **test-suite accuracy** - Zhong et al. 2020, [EMNLP](https://aclanthology.org/2020.emnlp-main.29/) - Text-to-SQL 연구의 "다른 데이터에서도 맞는지"를 Cypher에 옮겨 쓴 것
- **MissDiag** - [arXiv:2608.18489](https://arxiv.org/html/2608.18489v1), **GRACE** - [arXiv:2611.27350](https://arxiv.org/html/2611.27350v1) - 질문을 고정하고 그래프만 바꾸는 실험 방식. 이 패키지의 오류 3종은 두 논문의 구현이 아님

## 옮겨 올 때

기존 지표의 이름 · 프롬프트 · 점수식은 0.2와 같습니다. 판마다 바뀐 것은 [CHANGELOG.md](../CHANGELOG.md)에 있습니다.

<details>
<summary><b>v0.1에서 옮겨 올 때</b></summary>

- `load_kg`는 MultiDiGraph를 돌려줍니다. `g[u][v]["relation"]` 대신 `for u, v, rel in g.edges(keys=True)`처럼 관계를 모두 읽으세요.
- `EvalSample.retrieved_contexts`에 문자열 하나를 주면 오류입니다. `[텍스트]`로 넣으세요.
- 지표 점수가 반올림되지 않습니다. 표시용은 `report.summary`를 쓰세요.
- `compare_kg(..., scope=...)`에서 범위 칸이 빈 행은 오류입니다.
- `path_recall`은 `path_entity_recall`의 옛 이름입니다.
- 사업보고서 정답처럼 `@제출회사` 치환 · 블록 범위 · 부분 정답(`partial`)이 있는 자료는 이 함수 앞에서 직접 전처리하세요.

</details>

## 개발

```bash
git clone https://github.com/YTH-Public/kgonto-rag-eval.git
cd kgonto-rag-eval
pip install -e ".[kg,pandas,similarity]" pytest
pytest                 # API 없이 도는 테스트. 실호출 테스트는 tests/live/ (OPENAI_API_KEY 필요)
```

MIT 라이선스입니다.
