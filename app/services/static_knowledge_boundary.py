"""Conservative question-level limits of a static product manual.

These rules identify requests for an actual device instance's state/history,
not whether a generated claim is grounded. Unmatched questions still need
Audit and the semantic Gate. No network, model, retrieval or database access.
"""

from dataclasses import dataclass
import re
from typing import Literal


BoundaryReason = Literal[
    "STATIC_MANUAL_METHOD", "NO_PERSONAL_RUNTIME_QUERY",
    "PERSONAL_BATTERY_VALUE", "PERSONAL_DEVICE_LOCATION",
    "PERSONAL_CHARGING_STATE", "PERSONAL_NETWORK_STATE",
    "PERSONAL_CLEANING_PROGRESS", "PERSONAL_DEVICE_STATE",
    "PERSONAL_RUNTIME_DURATION", "PERSONAL_DEVICE_HISTORY",
]


@dataclass(frozen=True)
class KnowledgeBoundaryDecision:
    status: Literal["within_static_manual", "personal_runtime_unavailable"]
    reason_code: BoundaryReason


_METHOD = re.compile(
    r"怎么看|在哪(?:里)?看|怎么办|"
    r"(?:怎么|如何|怎样)(?:查看|查询|检查|判断|设置|清理|清洁|操作|处理|回充|充电)|"
    r"在哪(?:里)?(?:充电|购买)"
)
_PERSONAL = re.compile(r"我家|我的|这台|那台|这部|它")
_ACTOR = re.compile(r"机器人|机器|设备|它")
_CURRENT = re.compile(r"现在|当前|目前|此刻|实时")
_HISTORY = re.compile(r"昨晚|昨天|上次|上一次|前一晚|最近一次|实际|已经")
_VALUE = re.compile(r"多少|百分之几|几[%％]|几成|多久|多长|几分钟|几小时")
_CAPACITY = re.compile(r"续航|额定|标称|(?:能|可以|可)运行")
_CLAUSE_SEPARATOR = re.compile(
    r"[，,。！？!?；;\n]+|以及|并且|同时|另外|顺便|而且|"
    r"还有(?=(?:怎么看|如何|怎么|怎样|在哪看|怎么办|我家|我的|这台|那台|它|机器人|设备|现在|当前|昨晚|上次))"
)


def _runtime_reason(clause: str) -> BoundaryReason | None:
    personal = bool(_PERSONAL.search(clause))
    current = bool(_CURRENT.search(clause))
    historical = bool(_HISTORY.search(clause))
    actor = bool(_ACTOR.search(clause))
    instance_context = personal or current or historical

    # A battery threshold/specification question alone is not a live reading.
    if (instance_context and _VALUE.search(clause)
            and re.search(r"电量|(?:还剩|剩余|剩|还有).{0,8}电", clause)):
        return "PERSONAL_BATTERY_VALUE"

    # Direct location questions; where to view/check is handled as a method.
    direct_location = re.search(r"(?:在哪里|在哪(?:儿|里|个房间)?)(?:了|呢|呀|啊|吗)?$", clause)
    location_value = "位置" in clause and re.search(r"在哪|在哪里|是什么", clause)
    if (personal or current or actor) and (direct_location or location_value):
        return "PERSONAL_DEVICE_LOCATION"

    active_charge = re.search(
        r"(?:是否|是不是|有没有).{0,4}(?:正在|在)?充电(?:中)?$|"
        r"(?:正在|在)充电(?:中)?(?:吗|没有|没)$|充电状态.{0,4}(?:是什么|怎么样)$",
        clause,
    )
    if (instance_context or actor) and active_charge:
        return "PERSONAL_CHARGING_STATE"

    if (instance_context and re.search(r"联网|在线|联网状态|(?:连上|连接).{0,4}(?:WiFi|wifi|网络)", clause)
            and re.search(r"是否|有没有|是不是|了吗|了没|是什么|怎么样", clause)):
        return "PERSONAL_NETWORK_STATE"

    if (instance_context and re.search(r"清扫进度|(?:清扫|扫|打扫)到哪|清扫.{0,5}(?:完成|结束)", clause)
            and re.search(r"多少|百分之几|到哪|是否|了吗|了没|几[%％]", clause)):
        return "PERSONAL_CLEANING_PROGRESS"

    if (instance_context and re.search(r"当前状态|实时状态|故障状态|(?:现在|当前|目前|此刻|实时).{0,5}(?:状态|故障)", clause)
            and re.search(r"什么|怎么样|有没有|是否", clause)):
        return "PERSONAL_DEVICE_STATE"

    if (instance_context and re.search(r"清扫|运行|工作", clause)
            and re.search(r"多久|多长|几分钟|几小时|时长|时间是多少", clause)
            and not _CAPACITY.search(clause)):
        return "PERSONAL_RUNTIME_DURATION"

    if (instance_context and re.search(r"清扫记录|运行记录|历史记录|清扫历史|运行历史|错误日志|故障日志", clause)
            and re.search(r"是什么|有哪些|有没有|多少|多久|内容", clause)):
        return "PERSONAL_DEVICE_HISTORY"
    return None


def assess_static_knowledge_boundary(question: str) -> KnowledgeBoundaryDecision:
    """Method clauses stay eligible; a separate direct runtime query cannot
    be exempted by appending an unrelated how-to clause to the same question.
    This is a small explicit rule set, not a general Chinese semantic parser.
    """
    text = re.sub(r"\s+", "", question)
    method_seen = False
    # Split only on punctuation and a small set of explicit coordinating
    # connectors. This lets each independent request be checked on its own:
    # a method clause cannot exempt a neighboring live-state question.
    for clause in _CLAUSE_SEPARATOR.split(text):
        if not clause:
            continue
        if _METHOD.search(clause):
            method_seen = True
            continue
        reason = _runtime_reason(clause)
        if reason is not None:
            return KnowledgeBoundaryDecision("personal_runtime_unavailable", reason)
    return KnowledgeBoundaryDecision(
        "within_static_manual", "STATIC_MANUAL_METHOD" if method_seen else "NO_PERSONAL_RUNTIME_QUERY"
    )
