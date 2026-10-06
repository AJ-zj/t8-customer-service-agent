"""Narrow deterministic signals for explicit, user-authored troubleshooting facts."""

import re

_ERROR_CODE = re.compile(r"(?<![A-Za-z0-9])(?:E\d+|错误码\s*[:：]?\s*\d+)(?!\d)", re.IGNORECASE)
_FAST_ERROR_CODE = re.compile(
    r"(?P<description>错误码\s*[:：]?\s*(?P<numeric>E?\d+)|(?:(?:提示|显示|出现)\s*)?(?P<letter>E\d+))(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_EXPLICIT_SYMPTOMS = (
    re.compile(r"一直卡住|卡在原地|无法回充|不能回充|回不了充电座|无法回到充电座"),
    re.compile(r"漏水"),
    re.compile(r"(?:主刷|边刷)不转"),
    re.compile(r"扫着扫着(?:会自己)?停(?:了|下来|下来了)?|扫到一半自己停了|清扫中停止|中途停止|清扫时停止|运行中停止"),
    re.compile(r"无法启动|不能启动|启动不了"),
    re.compile(r"异常声音|异响|异常噪音"),
    re.compile(r"反复碰撞|一直撞墙|反复撞墙"),
    re.compile(r"原地打转"),
    re.compile(r"放到充电座不充电|放上充电座不充电|充电座不充电"),
    re.compile(r"Wi-?Fi\s*无法连接|无法连接\s*Wi-?Fi", re.IGNORECASE),
    re.compile(r"吸力(?:突然)?变小|吸力变弱"),
    re.compile(r"拖地时不出水|不出水"),
)


def _negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 5):start]
    return re.search(r"(?:没有|没提示|不显示|未提示|未显示|未出现|未发生|不曾|不是|并非|不会|不再|避免|比如|例如)\s*$", prefix) is not None


def find_explicit_error_code(question: str, conversation_context=None) -> tuple[str | None, str | None]:
    """Return the exact user-authored code description and normalized code."""
    latest = next((m.get("content", "").strip() for m in reversed(conversation_context or [])
                   if m.get("role") == "user" and isinstance(m.get("content"), str)
                   and m.get("content", "").strip()), "")
    for text in (question, latest):
        match = _FAST_ERROR_CODE.search(text)
        if not match or _negated(text, match.start()):
            continue
        code = match.group("letter") or match.group("numeric")
        normalized = code.upper()
        return match.group("description").strip(), normalized
    return None, None


def find_explicit_symptom(question: str, conversation_context=None) -> tuple[str | None, str | None, str | None]:
    """Return exact user-authored evidence; assistant messages are never inspected."""
    latest = next((m.get("content", "").strip() for m in reversed(conversation_context or [])
                   if m.get("role") == "user" and isinstance(m.get("content"), str)
                   and m.get("content", "").strip()), "")
    for text in (question, latest):
        if not text:
            continue
        match = _ERROR_CODE.search(text)
        if match and not _negated(text, match.start()):
            raw = match.group(0)
            code = re.search(r"E\d+|\d+$", raw, re.IGNORECASE)
            return raw, (code.group(0).upper() if code else None), "EXPLICIT_ERROR_CODE"
        for pattern in _EXPLICIT_SYMPTOMS:
            match = pattern.search(text)
            if match and not _negated(text, match.start()):
                return match.group(0), None, "EXPLICIT_SYMPTOM"
    return None, None, None
