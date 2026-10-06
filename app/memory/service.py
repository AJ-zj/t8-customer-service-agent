"""Owner-scoped candidate fact persistence and two-stage deduplication.

Callers must supply only explicitly stated, reusable customer-service facts.
This store does not extract or screen facts; it must not be fed drafts, evidence,
secrets, payment data, precise addresses, or model inferences. memory_owner_id
is an internal project identifier, not an authenticated user credential.
"""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from sqlalchemy import case, delete, or_, select, update
from sqlalchemy.exc import IntegrityError

from app.memory.embeddings import embed_fact
from app.memory.facts import cosine_similarity, normalize_memory_fact, _checked_vector
from app.memory.lifecycle import (CATEGORY_POLICIES, DEFAULT_POLICY, as_utc, checked_now,
                                 effective_score, expiration_for, importance_for, policy,
                                 utc_now)
from app.models import AgentMemoryFact


MEMORY_SEMANTIC_DUPLICATE_THRESHOLD = 0.97  # Conservative starting value; not tuned.
LONG_TERM_MEMORY_MIN_SIMILARITY = 0.75  # Initial engineering threshold, not tuned.
LONG_TERM_MEMORY_TOP_K = 3


@dataclass(frozen=True)
class SaveFactResult:
    status: str
    fact_id: int
    fact_text: str
    seen_count: int
    fact_category: str | None = None
    importance: float = 0.5
    last_seen_at: datetime | None = None
    expires_at: datetime | None = None
    effective_score: float | None = None


@dataclass(frozen=True)
class RetrievedMemoryFact:
    fact_id: int
    fact_text: str
    similarity: float
    seen_count: int


def _owner_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 64:
        raise ValueError("memory_owner_id 须为 1–64 个非空字符。")
    return value.strip()


def _result(status, row, now=None):
    last_seen = as_utc(row.last_seen_at)
    expires = as_utc(row.expires_at) if row.expires_at is not None else None
    score = effective_score(importance=row.importance, seen_count=row.seen_count,
                            last_seen_at=last_seen, fact_category=row.fact_category,
                            now=now, expires_at=expires) if now is not None else None
    return SaveFactResult(status, row.id, row.fact_text, row.seen_count,
                          row.fact_category, row.importance, last_seen, expires, score)


def _increment_seen_count(session, fact_id: int, status: str, *, owner: str,
                          now: datetime) -> SaveFactResult:
    """Atomically refresh lifecycle using the stored category and new SQL count.

    Do not recategorize an old fact from a new model candidate. Concurrent calls
    increment the database value and cannot move lifecycle timestamps backwards.
    """
    category = AgentMemoryFact.fact_category
    count = AgentMemoryFact.seen_count + 1
    base = case(*[(category == name, values[0]) for name, values in CATEGORY_POLICIES.items()],
                else_=DEFAULT_POLICY[0])
    bonus = case((count >= 6, 0.10), else_=(count - 1) * 0.02)
    importance = case((base + bonus >= 1.0, 1.0), else_=base + bonus)
    expiry = case(*[(category == name, expiration_for(name, now))
                    for name in CATEGORY_POLICIES], else_=expiration_for(None, now))
    row = session.execute(
        update(AgentMemoryFact)
        .where(AgentMemoryFact.id == fact_id, AgentMemoryFact.memory_owner_id == owner)
        .values(seen_count=count, importance=importance,
                updated_at=case((AgentMemoryFact.updated_at > now, AgentMemoryFact.updated_at), else_=now),
                last_seen_at=case((AgentMemoryFact.last_seen_at > now, AgentMemoryFact.last_seen_at), else_=now),
                expires_at=case((AgentMemoryFact.expires_at > expiry, AgentMemoryFact.expires_at), else_=expiry))
        .returning(AgentMemoryFact.id, AgentMemoryFact.fact_text,
                   AgentMemoryFact.seen_count, AgentMemoryFact.fact_category,
                   AgentMemoryFact.importance, AgentMemoryFact.last_seen_at,
                   AgentMemoryFact.expires_at)
    ).one()
    session.commit()
    return _result(status, row, now)


class MemoryFactService:
    def __init__(self, *, session_factory=None, embedding_provider=None, now_provider=None):
        if session_factory is None:
            from app.database import SessionLocal, configured_database_name
            if configured_database_name() != "roborock":
                raise RuntimeError("长期 Memory 只能写入 roborock。")
            session_factory = SessionLocal
        self._session_factory = session_factory
        self._embedding_provider = embedding_provider or embed_fact
        self._now_provider = now_provider or utc_now

    def save_fact(self, memory_owner_id: str, fact_text: str, *,
                  fact_category: str | None = None) -> SaveFactResult:
        owner = _owner_id(memory_owner_id)
        policy(fact_category)  # Validate before opening a transaction or embedding.
        now = checked_now(self._now_provider)
        normalized = normalize_memory_fact(fact_text)
        fact_hash = sha256(normalized.encode("utf-8")).hexdigest()
        session = self._session_factory()
        try:
            existing = session.scalar(select(AgentMemoryFact).where(
                AgentMemoryFact.memory_owner_id == owner,
                AgentMemoryFact.fact_hash == fact_hash,
            ))
            if existing is not None:
                return _increment_seen_count(session, existing.id, "EXACT_DUPLICATE", owner=owner, now=now)

            vector = self._embedding_provider(normalized)
            _checked_vector(vector)
            owner_facts = session.scalars(select(AgentMemoryFact).where(
                AgentMemoryFact.memory_owner_id == owner,
            ).order_by(AgentMemoryFact.id)).all()
            if owner_facts:
                nearest = max(owner_facts,
                              key=lambda row: cosine_similarity(vector, row.embedding))
                if cosine_similarity(vector, nearest.embedding) >= MEMORY_SEMANTIC_DUPLICATE_THRESHOLD:
                    return _increment_seen_count(session, nearest.id, "SEMANTIC_DUPLICATE", owner=owner, now=now)

            # UNIQUE protects same owner/hash; concurrent new, different-hash facts
            # with similar meaning are not strictly serialized in this version.
            created = AgentMemoryFact(memory_owner_id=owner, fact_text=normalized,
                                      fact_hash=fact_hash, embedding=vector, seen_count=1,
                                      fact_category=fact_category,
                                      importance=importance_for(fact_category, 1),
                                      last_seen_at=now, expires_at=expiration_for(fact_category, now),
                                      created_at=now, updated_at=now)
            session.add(created)
            try:
                session.commit()
            except IntegrityError:
                # Another transaction inserted this exact owner/hash concurrently.
                session.rollback()
                winner = session.scalar(select(AgentMemoryFact).where(
                    AgentMemoryFact.memory_owner_id == owner,
                    AgentMemoryFact.fact_hash == fact_hash,
                ))
                if winner is None:
                    raise
                return _increment_seen_count(session, winner.id, "EXACT_DUPLICATE", owner=owner, now=now)
            return _result("CREATED", created, now)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_facts(self, memory_owner_id: str) -> list[SaveFactResult]:
        """Administrative history, including expired rows; use list_active_facts for active data."""
        owner = _owner_id(memory_owner_id)
        session = self._session_factory()
        try:
            rows = session.scalars(select(AgentMemoryFact).where(
                AgentMemoryFact.memory_owner_id == owner,
            ).order_by(AgentMemoryFact.id)).all()
            return [_result("STORED", row) for row in rows]
        finally:
            session.close()

    def list_active_facts(self, memory_owner_id: str) -> list[SaveFactResult]:
        owner = _owner_id(memory_owner_id)
        now = checked_now(self._now_provider)
        session = self._session_factory()
        try:
            rows = session.scalars(select(AgentMemoryFact).where(
                AgentMemoryFact.memory_owner_id == owner,
                or_(AgentMemoryFact.expires_at.is_(None), AgentMemoryFact.expires_at > now),
            )).all()
            results = [_result("ACTIVE", row, now) for row in rows]
            return sorted(results, key=lambda row: (-row.effective_score, row.fact_id))
        finally:
            session.close()

    def cleanup_expired_facts(self, memory_owner_id: str | None = None) -> int:
        """One SQL DELETE, optionally owner-scoped; never log facts or vectors."""
        owner = _owner_id(memory_owner_id) if memory_owner_id is not None else None
        now = checked_now(self._now_provider)
        session = self._session_factory()
        try:
            statement = delete(AgentMemoryFact).where(AgentMemoryFact.expires_at <= now)
            if owner is not None:
                statement = statement.where(AgentMemoryFact.memory_owner_id == owner)
            result = session.execute(statement)
            count = result.rowcount
            if count < 0:
                raise RuntimeError("Database did not report a deletion count.")
            session.commit()
            return count
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def search_facts(self, memory_owner_id: str, query: str,
                     limit: int = LONG_TERM_MEMORY_TOP_K) -> list[RetrievedMemoryFact]:
        owner = _owner_id(memory_owner_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空。")
        if type(limit) is not int or not 1 <= limit <= LONG_TERM_MEMORY_TOP_K:
            raise ValueError("limit 须为 1–3。")
        now = checked_now(self._now_provider)
        session = self._session_factory()
        try:
            rows = session.scalars(select(AgentMemoryFact).where(
                AgentMemoryFact.memory_owner_id == owner,
                or_(AgentMemoryFact.expires_at.is_(None), AgentMemoryFact.expires_at > now),
            ).order_by(AgentMemoryFact.id)).all()
            if not rows:
                return []
            query_vector = self._embedding_provider(query)
            _checked_vector(query_vector)
            matches = [RetrievedMemoryFact(
                fact_id=row.id, fact_text=row.fact_text,
                similarity=cosine_similarity(query_vector, row.embedding),
                seen_count=row.seen_count,
            ) for row in rows]
            return sorted(
                (item for item in matches
                 if item.similarity >= LONG_TERM_MEMORY_MIN_SIMILARITY),
                key=lambda item: (-item.similarity, item.fact_id),
            )[:limit]
        finally:
            session.close()
