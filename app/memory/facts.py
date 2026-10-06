"""Conservative fact normalization and validated cosine similarity."""

import math
import re
import unicodedata


def normalize_memory_fact(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("fact_text 必须是字符串。")
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())
    if not normalized:
        raise ValueError("fact_text 不能为空。")
    return normalized


def _checked_vector(vector):
    if not isinstance(vector, list) or not vector:
        raise ValueError("embedding 必须是非空列表。")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in vector):
        raise ValueError("embedding 包含无效数值。")
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("embedding 范数必须是有限非零值。")
    return values, norm


def cosine_similarity(a, b) -> float:
    left, left_norm = _checked_vector(a)
    right, right_norm = _checked_vector(b)
    if len(left) != len(right):
        raise ValueError("embedding 维度不一致。")
    return sum(x * y for x, y in zip(left, right)) / (left_norm * right_norm)
