import re

from app.agent.state import Intent
from app.agent.scene_policy import BUSINESS_SCENES


ALLOWED_INTENTS = BUSINESS_SCENES | {"clarify", "human_handoff"}

_HANDOFF_PHRASES = (
    "人工客服",
    "我要人工",
    "转人工",
    "找人工",
    "人工服务",
    "真人客服",
    "联系人工",
)
_VAGUE_FAILURE = re.compile(r"(?:它|这个|那个).{0,8}(?:坏了|不工作|有问题|怎么办|异常)")


def classify_intent(question: str) -> tuple[Intent, str]:
    """Route obvious escalation and vague fault requests; default to knowledge QA."""
    normalized = question.strip().lower()

    if any(phrase in normalized for phrase in _HANDOFF_PHRASES):
        return "human_handoff", "用户明确要求人工客服。"

    if _VAGUE_FAILURE.search(normalized):
        return "clarify", "问题只描述了模糊异常，尚不足以定位具体故障。"

    return "consult_qa", "问题可先尝试使用 T8 说明书知识库回答。"
