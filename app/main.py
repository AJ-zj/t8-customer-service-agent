import asyncio
import json
import os
import secrets
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ChatRequest,
    ChatResponse,
    ReviewDecisionRequest,
    ReviewDecisionResponse,
    ReviewDetail,
    ReviewListItem,
    PublicationResponse,
    PublishedAnswerResponse,
)
from app.services.review_service import (
    ReviewDecisionConflict,
    ReviewRecordNotFound,
    ReviewRecordService,
    ReviewServiceError,
    ReviewPublicationConflict,
)
from app.services.qa_service import GroundedQAService, QAServiceError
from app.services.answer_release_policy import (
    AnswerReleasePolicy,
    GateOutcome,
    SAFE_ABSTENTION_ANSWER,
)
from app.services.semantic_grounding_gate import SemanticGroundingGate
from app.services.static_knowledge_boundary import assess_static_knowledge_boundary
from app.agent.react_graph import build_react_agent
from app.agent.scene_policy import BUSINESS_SCENES
from app.agent.memory import SessionMemoryStore
from app.agent.troubleshooting import TROUBLESHOOTING_CLARIFICATION
from app.memory.runtime import LongTermMemoryRuntime
from app.config import ENABLE_LONG_TERM_MEMORY_INJECTION
from app.services.readiness import check_readiness, ReadinessError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


SAFETY_NOTICES = {
    "PROGRAM_BLOCKED_INVALID_CITATION": "回答因引用编号异常已拦截，不可作为客服答案发布。",
    "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM": "回答因金额缺少证据已拦截，不可作为客服答案发布。",
    "PROGRAM_BLOCKED_UNSUPPORTED_PERSONAL_RUNTIME": "回答因个人历史数据缺少证据已拦截，不可作为客服答案发布。",
    "PROGRAM_BLOCKED_MISSING_CITATION": "回答因缺少证据引用已拦截，不可作为客服答案发布。",
    "MODEL_ABSTAINED_NEEDS_REVIEW": "模型表示资料不足；该拒答仍需人工审核，未作为正式答案发布。",
    "NEEDS_HUMAN_SEMANTIC_REVIEW": "模型已生成回答，但语义依据未自动核验；需人工审核，未作为正式答案发布。",
}

SSE_HEARTBEAT_SECONDS = 9.0
SSE_CHUNK_CHARACTERS = 14


def _sse_event(event: str, payload: dict) -> str:
    """Encode one fixed-name SSE event as a single-line UTF-8 JSON frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _sse_done_metadata(response: AgentChatResponse) -> dict:
    metadata = {
        "intent": response.intent,
        "delivery_status": response.delivery_status,
        "published": response.published,
    }
    if response.generation_status is not None:
        metadata["generation_status"] = response.generation_status
    if response.audit_status is not None:
        metadata["audit_status"] = response.audit_status
    return metadata


def _consume_background_exception(task: asyncio.Task) -> None:
    """Retrieve a worker exception if the browser disconnected before completion."""
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


def require_review_admin(
    x_review_token: str | None = Header(default=None, alias="X-Review-Token"),
):
    expected_token = os.getenv("REVIEW_ADMIN_TOKEN")
    if not expected_token or not x_review_token:
        raise HTTPException(status_code=401, detail="未授权访问内部审核接口。")

    try:
        matches = secrets.compare_digest(
            x_review_token.encode("utf-8"), expected_token.encode("utf-8")
        )
    except (AttributeError, UnicodeError):
        matches = False
    if not matches:
        raise HTTPException(status_code=401, detail="未授权访问内部审核接口。")


def create_app(qa_service=None, review_service=None, agent_graph=None, memory_store=None,
               long_term_memory_runtime=None, answer_release_policy=None,
               semantic_grounding_gate=None):
    service = qa_service or GroundedQAService()
    agent = agent_graph if agent_graph is not None else build_react_agent(qa_service=service)
    memory = memory_store if memory_store is not None else SessionMemoryStore()
    release_policy = answer_release_policy or AnswerReleasePolicy()
    grounding_gate = semantic_grounding_gate or SemanticGroundingGate()
    review_service = (
        review_service if review_service is not None else ReviewRecordService()
    )
    app = FastAPI(title="智扫通内部审核 API", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(QAServiceError)
    async def qa_service_error_handler(request: Request, exc: QAServiceError):
        return JSONResponse(
            status_code=503,
            content={"detail": "问答服务暂不可用，请稍后重试。"},
        )

    @app.exception_handler(ReviewServiceError)
    async def review_service_error_handler(request: Request, exc: ReviewServiceError):
        return JSONResponse(
            status_code=503,
            content={"detail": "审核记录暂时不可用，请稍后重试。"},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        locations = [error.get("loc", ()) for error in exc.errors()]
        if any("limit" in location for location in locations):
            detail = "请求参数无效：limit 须为 1–50。"
        elif any("record_id" in location for location in locations):
            detail = "请求参数无效：record_id 必须为整数。"
        elif any("session_id" in location for location in locations):
            detail = "请求参数无效：session_id 须为 1–64 位字母、数字、下划线或连字符。"
        elif any("memory_owner_id" in location for location in locations):
            detail = "请求参数无效：memory_owner_id 须为 1–64 位字母、数字、下划线或连字符。"
        elif any(
            "decision" in location or "review_note" in location
            for location in locations
        ):
            detail = "请求参数无效：decision 须为 approved 或 rejected，review_note 须为 1–1000 字。"
        else:
            detail = "请求参数无效：question 须为去除首尾空格后 1–500 字的非空字符串。"
        return JSONResponse(
            status_code=422,
            content={"detail": detail},
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        try:
            check_readiness()
        except ReadinessError:
            raise HTTPException(status_code=503, detail="本地依赖尚未就绪，请稍后重试。") from None
        return {"status": "ready"}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest):
        try:
            result = service.process_question(payload.question)
        except QAServiceError:
            raise
        except Exception:
            raise QAServiceError("问答服务暂不可用，请稍后重试。") from None
        status = result["audit_status"]
        return ChatResponse(
            generation_status=result["generation_status"],
            audit_status=status,
            answer=None,
            published=False,
            safety_notice=SAFETY_NOTICES[status],
        )

    def process_agent_chat(payload: AgentChatRequest) -> AgentChatResponse:
        try:
            # memory_owner_id is an unauthenticated local project label, distinct
            # from the short-lived session_id; production needs real identity binding.
            runtime = None
            if payload.memory_owner_id:
                try:
                    runtime = long_term_memory_runtime or LongTermMemoryRuntime()
                except Exception:
                    pass  # Optional Memory must never fail the main chat request.
            guard = (
                memory.session_guard(payload.session_id)
                if payload.session_id else nullcontext()
            )
            with guard:
                graph_input = {"question": payload.question}
                if payload.session_id:
                    context = memory.get(payload.session_id)
                    if context:
                        graph_input["conversation_context"] = context
                if runtime is not None and ENABLE_LONG_TERM_MEMORY_INJECTION:
                    try:
                        facts = runtime.retrieve_for_agent(payload.memory_owner_id, payload.question)
                        if facts:
                            graph_input["long_term_memory"] = facts
                    except Exception:
                        pass
                state = agent.invoke(graph_input)
                intent = state.get("intent")

                if intent == "troubleshooting" and state.get("troubleshooting_status") == "needs_clarification":
                    if state.get("qa_result") is not None:
                        raise QAServiceError("问答服务暂不可用，请稍后重试。")
                    response = AgentChatResponse(
                        intent="troubleshooting", answer=TROUBLESHOOTING_CLARIFICATION, published=False,
                        delivery_status="clarification",
                    )
                    if payload.session_id:
                        memory.append_exchange(payload.session_id, payload.question, response.answer)
                    return response

                if intent in BUSINESS_SCENES:
                    if intent == "troubleshooting" and state.get("troubleshooting_status") != "ready":
                        raise QAServiceError("问答服务暂不可用，请稍后重试。")
                    qa_result = state.get("qa_result")
                    if not isinstance(qa_result, dict):
                        raise QAServiceError("问答服务暂不可用，请稍后重试。")
                    audit_status = qa_result.get("audit_status")
                    if (
                        qa_result.get("generation_status") != "generated"
                        or audit_status not in SAFETY_NOTICES
                    ):
                        raise QAServiceError("问答服务暂不可用，请稍后重试。")
                    draft_answer = state.get("draft_answer")
                    evidence = state.get("evidence")
                    knowledge_boundary = assess_static_knowledge_boundary(payload.question)
                    decision = release_policy.decide(
                        scene=intent,
                        generation_status=qa_result.get("generation_status"),
                        audit_status=audit_status,
                        draft_answer=draft_answer,
                        evidence=evidence,
                        citation_status=qa_result.get("citation_status"),
                        knowledge_boundary=knowledge_boundary,
                    )
                    if decision.reason_code == "SEMANTIC_GATE_REQUIRED":
                        try:
                            gate_outcome = grounding_gate.evaluate(
                                question=payload.question,
                                draft_answer=draft_answer,
                                evidence=evidence,
                            )
                        except Exception:
                            # Gate failure is a delivery review, never a service success.
                            gate_outcome = GateOutcome("failed", "GATE_UNAVAILABLE")
                        decision = release_policy.decide(
                            scene=intent,
                            generation_status=qa_result.get("generation_status"),
                            audit_status=audit_status,
                            draft_answer=draft_answer,
                            evidence=evidence,
                            citation_status=qa_result.get("citation_status"),
                            gate_outcome=gate_outcome,
                            knowledge_boundary=knowledge_boundary,
                        )

                    # Live delivery is separate from approval/public publication.
                    # Auto-released answers remain pending review and unpublished.
                    answer = None
                    safety_notice = None
                    if decision.status == "auto_released":
                        answer = draft_answer
                    elif decision.status == "safe_abstention":
                        answer = SAFE_ABSTENTION_ANSWER
                    else:
                        safety_notice = SAFETY_NOTICES[audit_status]
                    response = AgentChatResponse(
                        intent=intent,
                        generation_status="generated",
                        audit_status=audit_status,
                        answer=answer,
                        delivery_status=decision.status,
                        published=False,
                        **({"safety_notice": safety_notice} if safety_notice else {}),
                    )
                    if payload.session_id:
                        memory.append_exchange(
                            payload.session_id,
                            payload.question,
                            response.answer or response.safety_notice or "",
                        )
                    if runtime is not None:
                        try:
                            runtime.process_successful_user_message(
                                payload.memory_owner_id, payload.question
                            )
                        except Exception:
                            pass
                    return response

                if intent in {"clarify", "human_handoff"}:
                    answer = state.get("answer")
                    if not isinstance(answer, str) or not answer:
                        raise QAServiceError("问答服务暂不可用，请稍后重试。")
                    response = AgentChatResponse(
                        intent=intent,
                        answer=answer,
                        delivery_status=(
                            "clarification" if intent == "clarify" else "human_handoff"
                        ),
                        published=False,
                    )
                    if payload.session_id:
                        memory.append_exchange(payload.session_id, payload.question, answer)
                    if runtime is not None:
                        try:
                            runtime.process_successful_user_message(
                                payload.memory_owner_id, payload.question
                            )
                        except Exception:
                            pass
                    return response

                raise QAServiceError("问答服务暂不可用，请稍后重试。")
        except QAServiceError:
            raise
        except Exception:
            raise QAServiceError("问答服务暂不可用，请稍后重试。") from None

    @app.post(
        "/api/agent/chat",
        response_model=AgentChatResponse,
        response_model_exclude_unset=True,
    )
    def agent_chat(payload: AgentChatRequest):
        return process_agent_chat(payload)

    @app.post("/api/agent/chat/stream")
    async def agent_chat_stream(payload: AgentChatRequest):
        async def event_stream():
            # These are transport-level states only. The synchronous graph does
            # not expose trustworthy live retrieval or audit sub-stages.
            yield _sse_event("status", {"stage": "received", "message": "已收到问题"})
            yield _sse_event("status", {"stage": "processing", "message": "正在分析问题并核验说明书依据…"})

            # Keep the exact same orchestration used by the JSON endpoint. A
            # disconnected browser stops consuming bytes; the worker may still
            # finish its existing model/database work.
            task = asyncio.create_task(asyncio.to_thread(process_agent_chat, payload))
            task.add_done_callback(_consume_background_exception)
            try:
                while True:
                    try:
                        response = await asyncio.wait_for(
                            asyncio.shield(task), timeout=SSE_HEARTBEAT_SECONDS
                        )
                        break
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            except asyncio.CancelledError:
                raise
            except Exception:
                yield _sse_event("error", {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "问答服务暂不可用，请稍后重试。",
                })
                return

            metadata = _sse_done_metadata(response)
            if response.delivery_status == "review_required":
                yield _sse_event("review_required", {
                    "intent": response.intent,
                    "delivery_status": response.delivery_status,
                    "message": response.safety_notice or "回答已转入质量审核。",
                    "published": False,
                })
                yield _sse_event("done", metadata)
                return

            if not isinstance(response.answer, str) or not response.answer:
                yield _sse_event("error", {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "问答服务暂不可用，请稍后重试。",
                })
                return

            # All release checks have completed before the first answer byte is
            # yielded. Deltas are slices of the exact public JSON answer, never
            # provider tokens or reconstructed text.
            yield _sse_event("answer_start", {
                "intent": response.intent,
                "delivery_status": response.delivery_status,
                "published": response.published,
            })
            for offset in range(0, len(response.answer), SSE_CHUNK_CHARACTERS):
                yield _sse_event("answer_delta", {
                    "text": response.answer[offset:offset + SSE_CHUNK_CHARACTERS]
                })
            yield _sse_event("done", metadata)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(
        "/internal/reviews",
        response_model=list[ReviewListItem],
        dependencies=[Depends(require_review_admin)],
    )
    def list_pending_reviews(limit: int = Query(default=20, ge=1, le=50),
                             scope: Literal["pending", "all"] = "pending"):
        return review_service.list_all(limit) if scope == "all" else review_service.list_pending(limit)

    @app.get(
        "/internal/reviews/{record_id}",
        response_model=ReviewDetail,
        dependencies=[Depends(require_review_admin)],
    )
    def get_review_detail(record_id: int):
        record = review_service.get_by_id(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="审核记录不存在。")
        return record

    @app.post(
        "/internal/reviews/{record_id}/decision",
        response_model=ReviewDecisionResponse,
        dependencies=[Depends(require_review_admin)],
    )
    def decide_review(record_id: int, payload: ReviewDecisionRequest):
        try:
            return review_service.decide(
                record_id,
                payload.decision,
                payload.review_note,
            )
        except ReviewRecordNotFound:
            raise HTTPException(
                status_code=404, detail="审核记录不存在。"
            ) from None
        except ReviewDecisionConflict:
            raise HTTPException(
                status_code=409,
                detail="该记录已完成审核，不能重复修改。",
            ) from None

    @app.post(
        "/internal/reviews/{record_id}/publish",
        response_model=PublicationResponse,
        dependencies=[Depends(require_review_admin)],
    )
    def publish_review(record_id: int):
        try:
            return review_service.publish(record_id)
        except ReviewRecordNotFound:
            raise HTTPException(status_code=404, detail="审核记录不存在。") from None
        except ReviewPublicationConflict:
            raise HTTPException(status_code=409, detail="该记录当前不能发布。") from None

    @app.get("/api/published/{record_id}", response_model=PublishedAnswerResponse)
    def get_published_answer(record_id: int):
        record = review_service.get_published(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="已发布答案不存在。")
        return record

    return app


app = create_app()
