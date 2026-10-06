"""Conservative full-sentence ambiguity guard, not a business keyword classifier."""

import re

_GENERIC_REQUEST = re.compile(
    r"(?:这个|那个)?(?:怎么弄|怎么处理|怎么办|怎么回事|怎么解决|帮我看看)(?:这个|那个)?"
)
_GENERIC_FAILURE = re.compile(
    r"(?:(?:它|这个|那个|机器人|机器)(?:好像|似乎)?)?"
    r"(?:坏了|有问题|不对劲|不正常|用不了|不对)(?:了)?"
    r"(?:那?(?:怎么办|怎么处理|怎么解决|帮我看看))?"
)


def needs_router_clarification(question: str, conversation_context=None) -> bool:
    # Recent user context may resolve the omitted object/goal. Do not override it.
    if any(message.get("role") == "user" and str(message.get("content", "")).strip()
           for message in conversation_context or []):
        return False
    text = re.sub(r"[\s，,。.!！?？；;]", "", question)
    if not text or len(text) > 20:
        return False
    # Fullmatch excludes explicit parts, symptoms, error codes, reports and settings.
    return bool(_GENERIC_REQUEST.fullmatch(text) or _GENERIC_FAILURE.fullmatch(text))
