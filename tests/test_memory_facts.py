"""Isolated DB tests with mocked embeddings; no external model or PostgreSQL writes."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.memory.facts import cosine_similarity, normalize_memory_fact
from app.memory import service as memory_service
from app.memory.service import MemoryFactService
from app.models import AgentMemoryFact, Base


@compiles(JSONB, "sqlite")
def compile_jsonb_for_isolated_tests(_type, _compiler, **_kw):
    return "JSON"


class MemoryFactTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine, tables=[AgentMemoryFact.__table__])
        self.factory = sessionmaker(self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def service(self, vectors):
        provider = Mock(side_effect=vectors)
        return MemoryFactService(session_factory=self.factory,
                                 embedding_provider=provider), provider

    def test_normalization_is_conservative_and_empty_rejected(self):
        self.assertEqual(normalize_memory_fact("  Ａ  用户家中   有地毯  "),
                         "A 用户家中 有地毯")
        self.assertNotEqual(normalize_memory_fact("用户家里有猫"),
                            normalize_memory_fact("用户家里没有猫"))
        with self.assertRaises(ValueError):
            normalize_memory_fact(" \t ")

    def test_exact_duplicate_skips_second_embedding_call(self):
        service, provider = self.service([[1.0, 0.0]])
        first = service.save_fact("owner-a", "用户家中有地毯")
        second = service.save_fact("owner-a", "用户家中有地毯")
        self.assertEqual((first.status, second.status, second.seen_count),
                         ("CREATED", "EXACT_DUPLICATE", 2))
        self.assertEqual(len(service.list_facts("owner-a")), 1)
        provider.assert_called_once()

    def test_two_concurrent_exact_duplicates_increment_atomically(self):
        with TemporaryDirectory() as directory:
            engine = create_engine(f"sqlite:///{Path(directory) / 'facts.db'}",
                                   connect_args={"timeout": 10})
            Base.metadata.create_all(engine, tables=[AgentMemoryFact.__table__])
            factory = sessionmaker(engine, expire_on_commit=False)
            provider = Mock(return_value=[1.0, 0.0])
            MemoryFactService(session_factory=factory,
                              embedding_provider=provider).save_fact(
                                  "owner-a", "用户家中有地毯")
            both_read = Barrier(2)

            def concurrent_factory():
                session = factory()
                original_scalar = session.scalar

                def scalar_then_wait(statement, *args, **kwargs):
                    row = original_scalar(statement, *args, **kwargs)
                    if isinstance(row, AgentMemoryFact):
                        both_read.wait(timeout=5)
                    return row

                session.scalar = scalar_then_wait
                return session

            service = MemoryFactService(session_factory=concurrent_factory,
                                        embedding_provider=provider)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(service.save_fact, "owner-a", "用户家中有地毯")
                           for _ in range(2)]
                results = [future.result() for future in futures]
            with factory() as session:
                rows = session.query(AgentMemoryFact).all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].seen_count, 3)
            self.assertEqual([row.status for row in results],
                             ["EXACT_DUPLICATE", "EXACT_DUPLICATE"])
            provider.assert_called_once()
            engine.dispose()

    def test_semantic_duplicate_preserves_original_text(self):
        service, _ = self.service([[1.0, 0.0], [0.99, 0.01]])
        first = service.save_fact("owner-a", "用户家里养了一只猫")
        with patch.object(memory_service, "_increment_seen_count",
                          wraps=memory_service._increment_seen_count) as increment:
            second = service.save_fact("owner-a", "用户家中养猫")
        increment.assert_called_once()
        self.assertEqual(increment.call_args.args[2], "SEMANTIC_DUPLICATE")
        self.assertEqual(second.status, "SEMANTIC_DUPLICATE")
        self.assertEqual(second.fact_id, first.fact_id)
        self.assertEqual(second.fact_text, "用户家里养了一只猫")
        self.assertEqual(second.seen_count, 2)
        self.assertEqual(len(service.list_facts("owner-a")), 1)

    def test_distinct_and_contradictory_facts_are_not_merged(self):
        service, _ = self.service([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(service.save_fact("owner-a", "用户家里有猫").status, "CREATED")
        self.assertEqual(service.save_fact("owner-a", "用户家里没有猫").status, "CREATED")
        self.assertEqual(len(service.list_facts("owner-a")), 2)

    def test_same_fact_isolated_by_owner(self):
        service, provider = self.service([[1.0, 0.0], [1.0, 0.0]])
        first = service.save_fact("owner-a", "用户家中有地毯")
        second = service.save_fact("owner-b", "用户家中有地毯")
        self.assertNotEqual(first.fact_id, second.fact_id)
        self.assertEqual(len(service.list_facts("owner-a")), 1)
        self.assertEqual(len(service.list_facts("owner-b")), 1)
        self.assertEqual(provider.call_count, 2)

    def test_invalid_vectors_and_empty_fact_fail(self):
        for a, b in [([0.0, 0.0], [1.0, 0.0]),
                     ([1.0], [1.0, 0.0]), ([], [1.0])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                cosine_similarity(a, b)
        service, provider = self.service([])
        with self.assertRaises(ValueError):
            service.save_fact("owner-a", "   ")
        provider.assert_not_called()

    def test_unique_race_recovers_existing_row(self):
        provider = Mock(return_value=[1.0, 0.0])
        factory = self.factory
        raced = False

        def racing_factory():
            nonlocal raced
            session = factory()
            original_commit = session.commit

            def commit_with_race():
                nonlocal raced
                if not raced and session.new:
                    raced = True
                    session.rollback()
                    with factory() as other:
                        other.add(AgentMemoryFact(
                            memory_owner_id="owner-a", fact_text="用户家中有地毯",
                            fact_hash=sha256("用户家中有地毯".encode()).hexdigest(),
                            embedding=[1.0, 0.0], seen_count=1))
                        other.commit()
                    raise IntegrityError("insert", {}, Exception("unique constraint"))
                return original_commit()

            session.commit = commit_with_race
            return session

        service = MemoryFactService(session_factory=racing_factory,
                                    embedding_provider=provider)
        with patch.object(memory_service, "_increment_seen_count",
                          wraps=memory_service._increment_seen_count) as increment:
            result = service.save_fact("owner-a", "用户家中有地毯")
        increment.assert_called_once()
        self.assertEqual(increment.call_args.args[2], "EXACT_DUPLICATE")
        self.assertEqual((result.status, result.seen_count), ("EXACT_DUPLICATE", 2))
        self.assertEqual(len(service.list_facts("owner-a")), 1)
