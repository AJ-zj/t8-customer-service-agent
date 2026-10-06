"""Fixed-clock lifecycle tests; isolated SQLite and mock embeddings only."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib.util
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import MagicMock, Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests.test_memory_facts import compile_jsonb_for_isolated_tests
from app.main import create_app
from app.memory.lifecycle import effective_score, importance_for
from app.memory.runtime import LongTermMemoryRuntime
from app.memory.schemas import MemoryExtractionResult
from app.memory.service import MemoryFactService
from app.models import AgentMemoryFact, Base

ROOT = Path(__file__).resolve().parents[1]
DAY0 = datetime(2030, 1, 1, tzinfo=timezone.utc)


def script_module(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine, tables=[AgentMemoryFact.__table__])
        self.factory = sessionmaker(self.engine, expire_on_commit=False)
        self.now = DAY0
        self.embedding = Mock(return_value=[1.0, 0.0])
        self.service = MemoryFactService(session_factory=self.factory,
            embedding_provider=self.embedding, now_provider=lambda: self.now)

    def tearDown(self):
        self.engine.dispose()

    def insert(self, owner, label, expires):
        with self.factory() as session:
            row = AgentMemoryFact(memory_owner_id=owner, fact_text=label,
                fact_hash=sha256(label.encode()).hexdigest(), embedding=[1.0, 0.0],
                seen_count=1, fact_category="home_environment", importance=0.75,
                created_at=DAY0, updated_at=DAY0, last_seen_at=DAY0, expires_at=expires)
            session.add(row)
            session.commit()
            return row.id

    def test_new_fact_has_category_base_importance(self):
        for category, expected in (("device_profile", .9), ("home_environment", .75),
                                   ("usage_preference", .65)):
            result = self.service.save_fact(category, "安全测试事实", fact_category=category)
            self.assertEqual(result.importance, expected)
            self.assertEqual(result.fact_category, category)
            self.assertEqual(result.last_seen_at, DAY0)
            self.assertEqual(result.seen_count, 1)

    def test_new_fact_has_category_ttl(self):
        for category, days in (("device_profile", 180), ("home_environment", 90),
                               ("usage_preference", 60)):
            result = self.service.save_fact(category, "安全测试事实", fact_category=category)
            self.assertEqual(result.expires_at, DAY0 + timedelta(days=days))

    def test_unknown_category_uses_conservative_defaults(self):
        result = self.service.save_fact("a", "旧接口安全事实")
        self.assertIsNone(result.fact_category)
        self.assertEqual(result.importance, .5)
        self.assertEqual(result.expires_at, DAY0 + timedelta(days=30))

    def test_exact_duplicate_refreshes_atomic_count_and_lifecycle(self):
        first = self.service.save_fact("a", "我家有地毯", fact_category="home_environment")
        self.now += timedelta(days=30)
        second = self.service.save_fact("a", "我家有地毯", fact_category="home_environment")
        self.assertEqual((first.seen_count, second.seen_count), (1, 2))
        self.assertEqual(second.status, "EXACT_DUPLICATE")
        self.assertEqual(second.last_seen_at, self.now)
        self.assertEqual(second.expires_at, self.now + timedelta(days=90))
        self.assertAlmostEqual(second.importance, .77)
        self.embedding.assert_called_once()

    def test_semantic_duplicate_refreshes_existing_category_and_lifecycle(self):
        self.service.save_fact("a", "我家有地毯", fact_category="home_environment")
        self.now += timedelta(days=10)
        result = self.service.save_fact("a", "家里铺着地毯", fact_category="usage_preference")
        self.assertEqual(result.status, "SEMANTIC_DUPLICATE")
        self.assertEqual(result.seen_count, 2)
        self.assertEqual(result.fact_category, "home_environment")
        self.assertEqual(result.last_seen_at, self.now)
        self.assertEqual(result.expires_at, self.now + timedelta(days=90))

    def test_repeat_importance_increases_and_caps(self):
        scores = [importance_for("device_profile", n) for n in (1, 2, 6, 100)]
        self.assertEqual(scores[0], .9)
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(scores[-1], 1.0)
        self.assertTrue(all(score <= 1 for score in scores))
        self.service.save_fact("a", "我使用T8", fact_category="device_profile")
        for _ in range(10):
            result = self.service.save_fact("a", "我使用T8")
        self.assertEqual(result.importance, 1.0)

    def test_effective_score_decreases_at_fixed_days(self):
        scores = [effective_score(importance=.9, seen_count=1, last_seen_at=DAY0,
                  fact_category="device_profile", now=DAY0 + timedelta(days=days))
                  for days in (0, 30, 60, 180)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], scores[-1])

    def test_repetition_factor_improves_same_age_score(self):
        inputs = dict(importance=.75, last_seen_at=DAY0,
                      fact_category="home_environment", now=DAY0 + timedelta(days=30))
        self.assertGreater(effective_score(seen_count=6, **inputs),
                           effective_score(seen_count=1, **inputs))

    def test_invalid_importance_rejected_by_formula_and_database(self):
        for value in (-.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                effective_score(importance=value, seen_count=1, last_seen_at=DAY0,
                                fact_category=None, now=DAY0)
        for value in (-.1, 1.1):
            with self.factory() as session, self.assertRaises(IntegrityError):
                session.add(AgentMemoryFact(memory_owner_id="bad", fact_text="安全测试",
                    fact_hash=str(value), embedding=[1.0], importance=value))
                session.commit()

    def test_invalid_category_and_naive_clock_rejected_before_embedding(self):
        with self.assertRaises(ValueError):
            self.service.save_fact("a", "事实", fact_category="made_up")
        bad_clock = MemoryFactService(session_factory=self.factory,
            embedding_provider=self.embedding, now_provider=lambda: datetime(2030, 1, 1))
        with self.assertRaises(ValueError):
            bad_clock.save_fact("a", "事实")
        self.embedding.assert_not_called()

    def test_active_list_excludes_expired_at_exact_boundary(self):
        self.insert("a", "expired", DAY0)
        active = self.insert("a", "active", DAY0 + timedelta(days=1))
        self.assertEqual([row.fact_id for row in self.service.list_active_facts("a")], [active])
        self.assertEqual(len(self.service.list_facts("a")), 2)

    def test_search_excludes_expired_without_query_embedding(self):
        self.insert("a", "expired", DAY0)
        self.assertEqual(self.service.search_facts("a", "test"), [])
        self.embedding.assert_not_called()

    def test_active_list_sorts_dynamic_score_without_writing_importance(self):
        first = self.insert("a", "first", DAY0 + timedelta(days=90))
        second = self.insert("a", "second", DAY0 + timedelta(days=90))
        with self.factory() as session:
            row = session.get(AgentMemoryFact, second)
            row.last_seen_at = DAY0 + timedelta(days=20)
            session.commit()
        self.now += timedelta(days=30)
        results = self.service.list_active_facts("a")
        self.assertEqual([row.fact_id for row in results], [second, first])
        with self.factory() as session:
            self.assertEqual(session.get(AgentMemoryFact, first).importance, .75)

    def test_expired_duplicate_is_refreshed_and_active_again(self):
        self.service.save_fact("a", "我家有地毯", fact_category="home_environment")
        self.now += timedelta(days=100)
        self.assertEqual(self.service.list_active_facts("a"), [])
        result = self.service.save_fact("a", "我家有地毯")
        self.assertEqual(result.seen_count, 2)
        self.assertEqual(len(self.service.list_active_facts("a")), 1)

    def test_owner_cleanup_deletes_only_expired_owner_rows(self):
        self.insert("a", "a-expired", DAY0)
        self.insert("a", "a-active", DAY0 + timedelta(days=1))
        self.insert("b", "b-expired", DAY0)
        self.assertEqual(self.service.cleanup_expired_facts("a"), 1)
        self.assertEqual(len(self.service.list_facts("a")), 1)
        self.assertEqual(len(self.service.list_facts("b")), 1)

    def test_global_cleanup_keeps_all_active_rows(self):
        for owner in ("a", "b"):
            self.insert(owner, owner + "-expired", DAY0)
            self.insert(owner, owner + "-active", DAY0 + timedelta(days=1))
        self.assertEqual(self.service.cleanup_expired_facts(), 2)
        self.assertEqual(self.service.cleanup_expired_facts(), 0)
        self.assertEqual(len(self.service.list_facts("a")), 1)
        self.assertEqual(len(self.service.list_facts("b")), 1)

    def test_cleanup_failure_rolls_back_and_closes(self):
        session = Mock()
        session.execute.side_effect = RuntimeError("internal")
        service = MemoryFactService(session_factory=lambda: session)
        with self.assertRaises(RuntimeError):
            service.cleanup_expired_facts("a")
        session.rollback.assert_called_once()
        session.close.assert_called_once()
        session.commit.assert_not_called()

    def test_concurrent_duplicates_keep_count_and_latest_timestamps(self):
        with TemporaryDirectory() as directory:
            engine = create_engine(f"sqlite:///{Path(directory) / 'lifecycle.db'}",
                                   connect_args={"timeout": 10})
            Base.metadata.create_all(engine, tables=[AgentMemoryFact.__table__])
            factory = sessionmaker(engine, expire_on_commit=False)
            MemoryFactService(session_factory=factory, embedding_provider=lambda _: [1.0],
                now_provider=lambda: DAY0).save_fact("a", "我家有地毯", fact_category="home_environment")
            barrier = Barrier(2)
            def concurrent_factory():
                session = factory()
                scalar = session.scalar
                def read(statement, *args, **kwargs):
                    row = scalar(statement, *args, **kwargs)
                    if isinstance(row, AgentMemoryFact):
                        barrier.wait(timeout=5)
                    return row
                session.scalar = read
                return session
            provider = Mock(side_effect=AssertionError("Exact duplicate must not embed"))
            def duplicate(day):
                return MemoryFactService(session_factory=concurrent_factory,
                    embedding_provider=provider,
                    now_provider=lambda: DAY0 + timedelta(days=day)).save_fact("a", "我家有地毯")
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(duplicate, (1, 2)))
            with factory() as session:
                rows = session.scalars(select(AgentMemoryFact)).all()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].seen_count, 3)
                self.assertEqual(rows[0].last_seen_at.replace(tzinfo=timezone.utc), DAY0 + timedelta(days=2))
                self.assertEqual(rows[0].expires_at.replace(tzinfo=timezone.utc), DAY0 + timedelta(days=92))
                self.assertAlmostEqual(rows[0].importance, .79)
            provider.assert_not_called()
            engine.dispose()

    @patch("app.main.ENABLE_LONG_TERM_MEMORY_INJECTION", False)
    def test_disabled_injection_never_reads_lifecycle_facts(self):
        self.service.save_fact("a", "我家有地毯", fact_category="home_environment")
        extractor = Mock()
        extractor.extract_candidates.return_value = MemoryExtractionResult(facts=[])
        runtime = LongTermMemoryRuntime(fact_service=self.service, extractor=extractor)
        graph = Mock()
        graph.invoke.return_value = {"intent": "clarify", "answer": "请补充信息。"}
        with patch.object(self.service, "search_facts") as search:
            client = TestClient(create_app(agent_graph=graph, long_term_memory_runtime=runtime))
            response = client.post("/api/agent/chat", json={"question": "测试问题", "memory_owner_id": "a"})
        self.assertEqual(response.status_code, 200)
        search.assert_not_called()
        graph.invoke.assert_called_once_with({"question": "测试问题"})


class MemoryLifecycleScriptTests(unittest.TestCase):
    def test_migration_is_additive_idempotent_and_transactional(self):
        migration = script_module("40_upgrade_memory_lifecycle.py")
        constraints, statements = set(), []
        connection = Mock()
        def scalar(statement, parameters=None):
            sql = str(statement)
            return "roborock" if "current_database" in sql else ((parameters or {}).get("name") in constraints)
        def execute(statement):
            sql = str(statement)
            statements.append(sql)
            if "ADD CONSTRAINT" in sql:
                constraints.add(sql.split("ADD CONSTRAINT ")[1].split()[0])
        connection.scalar.side_effect = scalar
        connection.execute.side_effect = execute
        engine = MagicMock()
        engine.begin.return_value.__enter__.return_value = connection
        migration.upgrade_memory_lifecycle(engine, "roborock")
        migration.upgrade_memory_lifecycle(engine, "roborock")
        self.assertEqual(engine.begin.call_count, 2)
        self.assertEqual(sum("ADD CONSTRAINT" in sql for sql in statements), 2)
        self.assertFalse(any("DELETE" in sql or "DROP" in sql for sql in statements))
        self.assertTrue(any("COALESCE(last_seen_at, updated_at, created_at)" in sql for sql in statements))
        self.assertFalse(any("SET fact_category" in sql for sql in statements))

    def test_non_roborock_migration_rejected_before_ddl(self):
        migration = script_module("40_upgrade_memory_lifecycle.py")
        engine = MagicMock()
        with self.assertRaises(RuntimeError):
            migration.upgrade_memory_lifecycle(engine, "shop_demo")
        engine.begin.assert_not_called()

    def test_connected_non_roborock_migration_rejected_before_ddl(self):
        migration = script_module("40_upgrade_memory_lifecycle.py")
        engine = MagicMock()
        connection = engine.begin.return_value.__enter__.return_value
        connection.scalar.return_value = "shop_demo"
        with self.assertRaises(RuntimeError):
            migration.upgrade_memory_lifecycle(engine, "roborock")
        connection.execute.assert_not_called()

    def test_cleanup_script_outputs_only_count_and_owner_scope(self):
        script = script_module("41_cleanup_expired_memory.py")
        service = Mock()
        service.cleanup_expired_facts.return_value = 2
        output = StringIO()
        with redirect_stdout(output):
            result = script.main(["--memory-owner-id", "test-owner"],
                service_factory=lambda: service, database_check=lambda: ("roborock", "roborock"))
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "deleted_count=2\n")
        service.cleanup_expired_facts.assert_called_once_with("test-owner")

    def test_cleanup_error_does_not_expose_memory_or_credentials(self):
        script = script_module("41_cleanup_expired_memory.py")
        service = Mock()
        service.cleanup_expired_facts.side_effect = RuntimeError("private fact_text embedding password")
        output = StringIO()
        with redirect_stdout(output):
            result = script.main([], service_factory=lambda: service,
                                 database_check=lambda: ("roborock", "roborock"))
        self.assertEqual(result, 1)
        self.assertNotIn("private", output.getvalue())
        self.assertNotIn("embedding", output.getvalue())

    def test_cleanup_script_rejects_other_database(self):
        script = script_module("41_cleanup_expired_memory.py")
        factory = Mock()
        with redirect_stdout(StringIO()):
            result = script.main([], service_factory=factory,
                                 database_check=lambda: ("shop_demo", "shop_demo"))
        self.assertEqual(result, 1)
        factory.assert_not_called()
