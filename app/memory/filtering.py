"""Conservative deterministic gate for proposed long-term user facts.

These patterns catch obvious cases, not every sensitive or inferred fact.
Candidates are not persisted by this module.
"""

import re
from dataclasses import dataclass

from app.memory.schemas import ExtractedMemoryFact


ALLOWED_CATEGORIES = frozenset({"device_profile", "home_environment", "usage_preference"})
SENSITIVE_PATTERNS = (
    re.compile(r"密码|口令|password|api[ _-]?key|access[ _-]?key|secret|token|验证码", re.I),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"银行卡|卡号|支付宝|微信支付|支付账户|收款账户|信用卡|身份证|护照"),
    re.compile(r"详细地址|住址|门牌号|(?:路|街|巷)\s*\d+\s*号|小区.{0,20}(?:栋|幢|单元|室)|(?:省|市|区|县).{0,30}(?:路|街|巷|小区|门牌)"),
    re.compile(r"精确位置|经纬度|GPS|坐标|定位地址", re.I),
    re.compile(r"姓名|我叫|全名|真实姓名|病史|疾病|诊断|药物|医院|医疗|健康状况|残疾|过敏|怀孕|血型"),
    re.compile(r"种族|民族|宗教|信仰|政治立场|党派|工会|性取向|性生活|犯罪|前科|人脸|指纹|声纹|虹膜"),
)
TRANSIENT_PATTERN = re.compile(
    r"今天|昨天|昨晚|刚刚|刚才|这次|本次|临时|突然|故障|错误码|卡住|断网|充不了电|坏了|当前电量|现在电量|现在坏|现在无法"
)
NON_FACT_PATTERN = re.compile(r"怎么|如何|为什么|哪里|哪个|多少|多久|是否|能不能|可以吗|怎么办|请问|[?？]")
DEVICE_OBJECT = re.compile(r"石头|T8|扫地机器人|机器人|主刷|边刷|滤网|尘盒|拖布|设备", re.I)
DEVICE_RELATION = re.compile(r"(?:我|我们|我家).{0,16}(?:用|使用|有|拥有|配置|装了)")
HOME_RELATION = re.compile(r"我家|家里|家中|我家的")
HOME_FACT = re.compile(r"地毯|木地板|地板|宠物|猫|狗|毛发")
HOME_PREDICATE = re.compile(r"有|铺|养|是|多|比较多")
PREFERENCE_RELATION = re.compile(r"(?:我|我们).{0,16}(?:喜欢|偏好|习惯|更关心|倾向|希望)")
CLEANING_CONTEXT = re.compile(r"清扫|扫地|拖地|静音模式|主刷维护|机器人")


def _source_supports_category(quote: str, category: str) -> bool:
    """Require an explicit user relationship and category fact in the quote itself."""
    if category == "device_profile":
        return bool(DEVICE_OBJECT.search(quote) and DEVICE_RELATION.search(quote))
    if category == "home_environment":
        return bool(HOME_RELATION.search(quote) and HOME_FACT.search(quote)
                    and HOME_PREDICATE.search(quote))
    if category == "usage_preference":
        return bool(PREFERENCE_RELATION.search(quote) and CLEANING_CONTEXT.search(quote))
    return False


@dataclass(frozen=True)
class FilteredMemoryCandidate:
    fact_text: str
    category: str
    source_quote: str
    accepted: bool
    reason: str
    memory_text: str | None


def filter_candidates(user_message: str, candidates: list[ExtractedMemoryFact]):
    """Return (accepted, rejected); only literal current-message quotes qualify."""
    if not isinstance(user_message, str):
        raise ValueError("user_message 必须是字符串。")
    accepted, rejected = [], []
    for candidate in candidates:
        if not isinstance(candidate, ExtractedMemoryFact):
            raise ValueError("candidate 必须通过严格 Schema 校验。")
        quote = candidate.source_quote.strip()
        text = f"{quote} {candidate.fact_text}"
        if quote not in user_message:
            reason = "REJECTED_SOURCE"
        elif any(pattern.search(text) for pattern in SENSITIVE_PATTERNS):
            reason = "REJECTED_SENSITIVE"
        elif TRANSIENT_PATTERN.search(quote):
            reason = "REJECTED_TRANSIENT"
        elif NON_FACT_PATTERN.search(quote):
            reason = "REJECTED_NON_FACT"
        elif candidate.category not in ALLOWED_CATEGORIES or not _source_supports_category(quote, candidate.category):
            reason = "REJECTED_CATEGORY"
        else:
            reason = "ACCEPTED"
        item = FilteredMemoryCandidate(candidate.fact_text, candidate.category,
                                       candidate.source_quote, reason == "ACCEPTED", reason,
                                       quote if reason == "ACCEPTED" else None)
        (accepted if item.accepted else rejected).append(item)
    return accepted, rejected
