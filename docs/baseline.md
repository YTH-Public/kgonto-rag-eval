[← 처음으로](../README.md)

# 전후 비교

고친 뒤 점수가 떨어진 질문이 없는지 봅니다.

```python
from kgonto_rag_eval import compare_baseline, save_baseline

save_baseline(report, "baseline.json", meta={"dataset": "policy-v1"})
result = compare_baseline(new_report, "baseline.json", tolerance=0.1)
assert result["ok"], result
```

- 같은 `sample_id`끼리, 양쪽 모두 점수가 있는 문항만 짝지어 비교합니다. 점수가 있던 칸이 비면(`lost`) 실패합니다.
- `strict_meta=True`를 주면 **채점 조건이 같은지** 먼저 봅니다 - 패키지 판, 판정 모델, 지표의 코드와 설정, 문항과 정답. 함수 지표는 함수 본문과, 함수가 가져다 쓴 바깥의 글자 · 숫자 · 함수까지 봅니다. 바깥의 객체 속은 보지 못하니 `run_meta={"conditions": {...}}`에 판 이름을 적으세요. 바깥의 목록 · 사전은 규칙으로 보므로, 채점하면서 바깥 목록에 기록을 쌓지 말고 `details`나 `logging`에 남기세요. `run_meta`에 키나 비밀번호를 넣지 마세요.
- 모델 판정은 실행마다 흔들립니다. 허용 폭(`tolerance`)과 반복 횟수를 정해 두세요.
