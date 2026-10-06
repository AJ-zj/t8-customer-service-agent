"""Reusable Agent tools for the existing Roborock T8 QA service."""

import threading

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool


TOOL_NAME = "t8_knowledge_qa"
TOOL_DESCRIPTION = (
    "使用石头 T8 官方说明书知识库回答产品知识问题，包括使用方法、维护保养及说明书能够支持的内容。"
    "该工具不联网、不查询实时信息、不控制机器人，也不查询用户设备状态；"
    "回答仍经过现有自动审核并保持未发布状态。"
)
SAFE_TOOL_NOTICE = "结果已进入现有内部审核流程，未作为正式客服答案发布。"
SEARCH_TOOL_NAME = "search_t8_manual"
SEARCH_TOOL_DESCRIPTION = (
    "只读检索石头 T8 官方说明书中的使用方法、维护保养和故障知识，返回带页码的原文证据。"
    "不生成客服回答、不查询设备状态、不执行审核或数据库写入。"
)


class T8KnowledgeQAInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question 不能为空。")
        return value


class SearchT8ManualInput(T8KnowledgeQAInput):
    """Use the same 1–500 character validation while preserving the raw question."""


def build_search_t8_manual_tool(retrieval_module=None) -> StructuredTool:
    """Create a read-only manual search Tool backed by production retrieval."""
    if retrieval_module is None:
        from app.retrieval import t8_retrieval as retrieval_module

    from app.services.qa_service import normalize_retrieval_query

    pipeline = None
    lock = threading.Lock()

    def get_pipeline():
        nonlocal pipeline
        if pipeline is None:
            with lock:
                if pipeline is None:
                    pipeline = retrieval_module.create_pipeline()
        return pipeline

    def run_search_t8_manual(question: str) -> dict:
        retrieval_query = normalize_retrieval_query(question)
        top_records = retrieval_module.retrieve(retrieval_query, get_pipeline())
        return {
            "retrieval_query": retrieval_query,
            "evidence": [
                {
                    "evidence_id": evidence_id,
                    "record_index": record_index,
                    "pdf_page": record.get("pdf_page"),
                    "title": record.get("title"),
                    "section": record.get("section"),
                    "text": record.get("text", ""),
                }
                for evidence_id, record_index, record in top_records[:3]
            ],
        }

    return StructuredTool.from_function(
        func=run_search_t8_manual,
        name=SEARCH_TOOL_NAME,
        description=SEARCH_TOOL_DESCRIPTION,
        args_schema=SearchT8ManualInput,
    )


def build_t8_knowledge_qa_tool(qa_service=None) -> StructuredTool:
    """Create an independently invokable Tool backed by GroundedQAService."""
    if qa_service is None:
        from app.services.qa_service import GroundedQAService

        qa_service = GroundedQAService()

    from app.services.qa_service import ALLOWED_AUDIT_STATUSES

    def run_t8_knowledge_qa(question: str) -> dict[str, str | bool]:
        # Pass the original question through; normalization belongs to the QA service.
        result = qa_service.process_question(question)
        if not isinstance(result, dict):
            raise RuntimeError("QA service returned an invalid result.")

        generation_status = result.get("generation_status")
        audit_status = result.get("audit_status")
        if generation_status != "generated" or audit_status not in ALLOWED_AUDIT_STATUSES:
            raise RuntimeError("QA service returned an unsupported status.")

        # Allowlist only safe status metadata; never forward a draft or evidence.
        return {
            "generation_status": generation_status,
            "audit_status": audit_status,
            "published": False,
            "safety_notice": SAFE_TOOL_NOTICE,
        }

    return StructuredTool.from_function(
        func=run_t8_knowledge_qa,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=T8KnowledgeQAInput,
    )
