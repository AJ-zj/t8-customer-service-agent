import importlib.util
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app.main import create_app
from app.services import readiness
from app.models import QAReviewRecord, AgentMemoryFact


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(agent_graph=MagicMock()))
        self.engine = MagicMock()
        self.engine.connect.return_value.__enter__.return_value.scalar.return_value = "roborock"
        self.inspector = MagicMock()
        self.inspector.has_table.return_value = True
        self.inspector.get_columns.side_effect = lambda name: [
            {"name": c.name} for model in (QAReviewRecord, AgentMemoryFact)
            if model.__tablename__ == name for c in model.__table__.columns]

    def request(self, *, db_error=None, missing=False, chroma_error=None):
        if db_error:
            self.engine.connect.side_effect = db_error
        if missing:
            self.inspector.has_table.return_value = False
        with patch("app.database.engine", self.engine), patch("app.database.configured_database_name", return_value="roborock"), patch(
            "app.services.readiness.inspect", return_value=self.inspector), patch(
            "app.retrieval.t8_vector_store.T8VectorStore") as store, patch(
            "app.retrieval.t8_retrieval.request_embeddings", side_effect=AssertionError("external call")), patch(
            "app.agent.tool_calling.ManualToolCallingLLM.complete", side_effect=AssertionError("external call")):
            store.return_value.validate_collection.return_value = {
                "stored_record_count": 41, "embedding_dimension": 1024}
            if chroma_error:
                store.side_effect = chroma_error
            return self.client.get("/ready")

    def test_ready_success(self):
        self.assertEqual(self.request().json(), {"status": "ready"})

    def test_ready_database_failure(self):
        self.assertEqual(self.request(db_error=RuntimeError("password=secret")).status_code, 503)

    def test_ready_missing_table(self):
        self.assertEqual(self.request(missing=True).status_code, 503)

    def test_ready_chroma_failure(self):
        self.assertEqual(self.request(chroma_error=RuntimeError("private/path")).status_code, 503)

    def test_ready_no_external_models(self):
        self.assertEqual(self.request().status_code, 200)

    def test_ready_does_not_leak(self):
        response = self.request(db_error=RuntimeError("password=secret DATABASE_URL private/path"))
        self.assertNotIn("secret", response.text)
        self.assertNotIn("DATABASE_URL", response.text)
        self.assertNotIn("private/path", response.text)

    def load_init(self):
        path = Path(__file__).resolve().parents[1] / "scripts/50_initialize_database.py"
        spec = importlib.util.spec_from_file_location("deployment_initializer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_init_rejects_other_database(self):
        module = self.load_init()
        with patch("app.database.configured_database_name", return_value="shop_demo"), patch.object(module, "load_step") as step:
            with self.assertRaises(RuntimeError):
                module.initialize()
            step.assert_not_called()

    def test_init_repeatable_order(self):
        module = self.load_init()
        with patch("app.database.configured_database_name", return_value="roborock"), patch(
            "app.database.current_database_name", return_value="roborock"), patch.object(module, "load_step") as step:
            step.return_value.main.return_value = 0
            module.initialize()
            module.initialize()
            self.assertEqual([c.args[0] for c in step.call_args_list],
                ["14_create_review_table.py", "21_create_memory_facts_table.py",
                 "32_upgrade_review_publication.py", "40_upgrade_memory_lifecycle.py"] * 2)

    def test_compose_disables_injection(self):
        compose = (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
        self.assertIn('ENABLE_LONG_TERM_MEMORY_INJECTION: "false"', compose)


if __name__ == "__main__":
    unittest.main()
