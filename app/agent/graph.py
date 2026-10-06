from langgraph.graph import END, START, StateGraph

from app.agent.router import classify_intent
from app.agent.state import AgentState
from app.agent.router import ALLOWED_INTENTS
from app.agent.ambiguity import needs_router_clarification
from app.agent.scene_policy import BUSINESS_SCENES
from app.agent.tools import build_t8_knowledge_qa_tool
from app.agent.llm_router import IntentDecision, LLMIntentRouter


CLARIFY_ANSWER = (
    "请再描述一下具体异常现象，例如是否有语音提示、错误码，"
    "或者出现了什么行为异常。"
)
HANDOFF_ANSWER = "已为你转入人工处理流程。"


def build_agent(qa_service=None, llm_router=None):
    """Build an Agent graph with injectable QA service and intent classifier."""
    t8_knowledge_tool = build_t8_knowledge_qa_tool(qa_service)
    intent_router = build_intent_router(llm_router)

    def rag_node(state: AgentState) -> dict:
        result = t8_knowledge_tool.invoke({"question": state["question"]})
        return {
            "answer": None,
            "qa_result": result,
        }

    def clarify_node(_state: AgentState) -> dict:
        return {"answer": CLARIFY_ANSWER}

    def handoff_node(_state: AgentState) -> dict:
        # User-facing support escalation is distinct from internal draft review.
        return {"answer": HANDOFF_ANSWER}

    def select_node(state: AgentState) -> str:
        intent = state.get("intent")
        if intent not in ALLOWED_INTENTS:
            raise ValueError("Router produced an unsupported intent.")
        return intent

    graph = StateGraph(AgentState)
    graph.add_node("intent_router", intent_router)
    graph.add_node("rag_node", rag_node)
    graph.add_node("clarify_node", clarify_node)
    graph.add_node("handoff_node", handoff_node)
    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        select_node,
        {
            **{scene: "rag_node" for scene in BUSINESS_SCENES},
            "clarify": "clarify_node",
            "human_handoff": "handoff_node",
        },
    )
    graph.add_edge("rag_node", END)
    graph.add_edge("clarify_node", END)
    graph.add_edge("handoff_node", END)
    return graph.compile()


def build_intent_router(llm_router=None):
    """Share the unchanged V2 intent decision with the independent V3 graph."""
    intent_classifier = llm_router if llm_router is not None else LLMIntentRouter()

    def intent_router(state: AgentState) -> dict:
        question = state["question"]
        context = state.get("conversation_context") or []
        fallback_intent, fallback_reason = classify_intent(question)

        # Explicit human-service requests bypass the model as a hard rule.
        if fallback_intent == "human_handoff":
            return {"intent": fallback_intent, "route_reason": fallback_reason}

        try:
            raw_decision = (
                intent_classifier.classify(question, conversation_context=context)
                if context else intent_classifier.classify(question)
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, IntentDecision)
                else IntentDecision.model_validate(raw_decision)
            )
            return {"intent": decision.intent, "route_reason": decision.reason}
        except Exception:
            # If the LLM router is unavailable, preserve the same narrow
            # ambiguity safety rule used after a valid model decision. Recent
            # user context deliberately disables this guard and keeps the
            # existing deterministic fallback behavior.
            if needs_router_clarification(question, context):
                return {
                    "intent": "clarify",
                    "route_reason": "问题过于模糊，缺少识别问题所需的信息。",
                }
            # API errors, timeouts, malformed JSON, and invalid schemas all use
            # the unchanged deterministic router as fallback.
            return {"intent": fallback_intent, "route_reason": fallback_reason}

    return intent_router
