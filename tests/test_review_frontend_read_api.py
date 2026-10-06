import os
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from app.main import create_app
from app.services.review_service import ReviewRecordService


class ReviewFrontendReadTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.list_pending.return_value = []
        self.service.list_all.return_value = []
        self.client = TestClient(create_app(review_service=self.service, agent_graph=Mock()))
        self.env = patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": "unit-only-token"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_default_retains_pending_behavior(self):
        response = self.client.get('/internal/reviews', headers={'X-Review-Token': 'unit-only-token'})
        self.assertEqual(response.status_code, 200)
        self.service.list_pending.assert_called_once_with(20)
        self.service.list_all.assert_not_called()

    def test_all_scope_uses_same_authenticated_service(self):
        response = self.client.get('/internal/reviews?scope=all&limit=50', headers={'X-Review-Token': 'unit-only-token'})
        self.assertEqual(response.status_code, 200)
        self.service.list_all.assert_called_once_with(50)

    def test_all_scope_does_not_bypass_auth(self):
        self.assertEqual(self.client.get('/internal/reviews?scope=all').status_code, 401)
        self.service.list_all.assert_not_called()

    def test_invalid_scope_rejected(self):
        self.assertEqual(self.client.get('/internal/reviews?scope=secret', headers={'X-Review-Token': 'unit-only-token'}).status_code, 422)
        self.service.list_all.assert_not_called()

    def test_all_query_is_read_only_and_ordered(self):
        session = Mock()
        session.scalars.return_value.all.return_value = []
        self.assertEqual(ReviewRecordService(Mock(return_value=session)).list_all(50), [])
        statement = str(session.scalars.call_args.args[0])
        self.assertNotIn('WHERE', statement)
        self.assertIn('created_at DESC', statement)
        self.assertIn('id DESC', statement)
        session.commit.assert_not_called()
        session.close.assert_called_once()

    def test_publication_fields_are_read_from_saved_record(self):
        stamp = datetime.now(timezone.utc)
        record = Mock(id=101, question='测试问题', draft_answer='内部草稿', audit_status='NEEDS_HUMAN_SEMANTIC_REVIEW', review_status='approved', publication_status='published', evidence=[], review_note='审核意见', created_at=stamp, reviewed_at=stamp, published_at=stamp)
        session = Mock()
        session.scalars.return_value.all.return_value = [record]
        session.get.return_value = record
        service = ReviewRecordService(Mock(return_value=session))
        item = service.list_all(50)[0]
        self.assertEqual(item['publication_status'], 'published')
        self.assertNotIn('draft_answer', item)
        self.assertNotIn('evidence', item)
        detail = service.get_by_id(101)
        self.assertEqual(detail['publication_status'], 'published')
        self.assertEqual(detail['published_at'], stamp)
        session.commit.assert_not_called()

if __name__ == '__main__':
    unittest.main()
