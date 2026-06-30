"""페르소나: 데이터 기반 자동 후보 + 수동 큐레이션. 질문 말투를 다양화한다."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class PersonaSpec(BaseModel):
    name: str
    role: str = ""
    style: str = ""


# 기본 페르소나 3종 (KG 노드/관계 타입에서 도출한 예시, 필요하면 직접 정의)
DEFAULT_PERSONAS = [
    PersonaSpec(name="투자자", role="재무 성과와 ESG 리스크가 기업 가치에 주는 영향을 확인한다.",
                style="수치, 비교, 근거를 요구한다."),
    PersonaSpec(name="ESG담당자", role="환경·사회·지배구조 활동이 어떤 영역과 지표에 연결되는지 점검한다.",
                style="이니셔티브, 측정 지표, 보고서 섹션을 묻는다."),
    PersonaSpec(name="협력회사", role="공급망과 Scope3, 협력회사 대상 활동을 확인한다.",
                style="자기에게 영향을 주는 활동과 요구사항 중심으로 묻는다."),
]


class _PersonaList(BaseModel):
    personas: list[PersonaSpec] = Field(description="페르소나 후보 목록")


_SYSTEM = """당신은 평가셋용 페르소나 설계자입니다.
주어진 지식그래프의 노드 타입과 관계 타입을 보고, 이 데이터에 자연스러운 질문자 페르소나 후보를 만듭니다.
각 페르소나는 name(짧은 역할명), role(관심사 1문장), style(질문 말투 1문장)을 갖습니다."""


def auto_personas_from_graph(graph, llm, n: int = 5) -> list[PersonaSpec]:
    """KG 의 노드 타입·관계 타입을 요약해 LLM 으로 페르소나 후보 n개를 만든다.

    자동 결과는 '후보'일 뿐, 최종은 사람이 골라 DEFAULT_PERSONAS 처럼 고정하는 것을 권장.
    """
    node_types = sorted({d.get("type", "") for _, d in graph.nodes(data=True) if d.get("type")})
    rel_types = sorted({d["relation"] for _, _, d in graph.edges(data=True) if "relation" in d})
    out = llm.with_structured_output(_PersonaList).invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=(
            f"노드 타입: {node_types}\n관계 타입: {rel_types}\n"
            f"페르소나 후보 {n}개를 한국어로 제안하세요."
        )),
    ])
    return out.personas[:n]
