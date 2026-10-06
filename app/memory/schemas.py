"""Strict candidate schema; a valid extraction is not trusted memory."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MemoryFactCategory = Literal["device_profile", "home_environment", "usage_preference"]


class ExtractedMemoryFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_text: str = Field(min_length=1, max_length=500)
    category: MemoryFactCategory
    source_quote: str = Field(min_length=1, max_length=500)

    @field_validator("fact_text", "source_quote")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("字段不能为空白。")
        return value


class MemoryExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[ExtractedMemoryFact] = Field(max_length=3)
