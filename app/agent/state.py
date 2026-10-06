from typing import Literal, NotRequired, TypedDict


SceneIntent = Literal["consult_qa", "troubleshooting", "maintenance", "maintenance_report",
                      "configuration_recommendation", "clarify", "human_handoff"]
Intent = SceneIntent


class AgentState(TypedDict):
    question: str
    conversation_context: NotRequired[list[dict[str, str]]]
    intent: NotRequired[Intent | None]
    answer: NotRequired[str | None]
    route_reason: NotRequired[str | None]
    qa_result: NotRequired[dict[str, str | bool] | None]


class ReactAgentState(AgentState):
    troubleshooting_status: NotRequired[Literal["ready", "needs_clarification"]]
    symptom_summary: NotRequired[str | None]
    error_code: NotRequired[str | None]
    missing_information: NotRequired[list[str]]
    clarifying_question: NotRequired[str | None]
    scene_policy: NotRequired[dict]
    long_term_memory: NotRequired[list[str]]
    messages: NotRequired[list[dict]]
    pending_tool_calls: NotRequired[list[dict] | None]
    tool_call_count: NotRequired[int]
    draft_answer: NotRequired[str | None]
    evidence: NotRequired[list[dict]]
    manual_observation: NotRequired[dict]
