[← 처음으로](../README.md)

# 인용 검사

```python
from kgonto_rag_eval import citation_support, quote_in_source, resolve_citations, split_cited_sentences

corpus = {"35": "...35쪽 원문...", "36": "...36쪽 원문..."}
answer = "월 최대 20만 원입니다 [p.35]. 최장 24개월입니다 [p.36]."

split_cited_sentences(answer)                             # 문장마다 출처 ID
resolve_citations(["35", "99"], corpus)                   # 1) 출처가 실제 자료에 있나
quote_in_source("월 최대 20만 원", corpus["35"])            # 2) 인용 문구가 원문에 그대로 있나
citation_support(answer, corpus, model, question="...")   # 3) 그 자료가 문장을 뒷받침하나 (모델)
```

세 검사는 서로 다릅니다. 같은 쪽에 같은 숫자가 있어도 다른 대상의 숫자일 수 있습니다.

<details>
<summary><b><code>citation_support</code>가 ALCE 논문과 다른 점</b></summary>

ALCE(Gao et al. 2023, arXiv:2305.14627)의 citation recall / precision 아이디어를 줄인 수업용 지표입니다.
- 판정을 NLI 모델(AutoAIS) 대신 LLM이 합니다.
- recall 분모는 **출처를 단 문장**만입니다(ALCE는 출처 없는 문장도 0으로 세어 전체 문장을 분모로 씀). 출처 없는 문장 수는 `uncited`로 따로 봅니다.
- 풀리지 않는 ID는 빼고 나머지 출처로 판정합니다. precision 분모는 출처 표시 횟수입니다.
- 판정 호출이 실패한 문장은 0점이 아니라 `errors`로 셉니다.
- 잘못 적은 출처 범위([p.9-3], 50쪽을 넘는 범위)는 `CitedRetrieved`가 맞지 않은 출처로 셉니다.

</details>

## 두 답 비교

```python
from kgonto_rag_eval import pairwise_judge
pairwise_judge(question, answer_a, answer_b, model, criteria="금액 · 기간 · 대상 조건이 정확한가")
# {"ab": "A", "ba": "B", "consistent": False, "final": "inconsistent"}
```

순서를 바꿔 두 번 묻고, 두 결과가 다르면 승자를 정하지 않습니다(위치 편향, MT-Bench arXiv:2306.05685).
