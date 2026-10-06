"""Insert, read, and remove one identifiable ORM smoke-test record."""

import sys
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import (
    SessionLocal,
    configured_database_name,
    current_database_name,
)
from app.models import QAReviewRecord


TARGET_DATABASE = "roborock"


def main() -> int:
    marker = f"ORM_REVIEW_RECORD_SMOKE_TEST::{uuid.uuid4()}"
    inserted_id = None

    if configured_database_name() != TARGET_DATABASE:
        print("停止：DATABASE_URL 配置的数据库不是 roborock。")
        return 1

    try:
        connected_database = current_database_name()
    except SQLAlchemyError as exc:
        print(f"连接数据库失败，错误类型：{type(exc).__name__}")
        return 1

    if connected_database != TARGET_DATABASE:
        print("停止：当前连接的数据库不是 roborock。")
        return 1

    try:
        with SessionLocal() as session:
            record = QAReviewRecord(
                question=marker,
                draft_answer="仅用于 ORM 插入与查询验证。",
                evidence=[{"source": "smoke-test", "page": 1}],
                audit_status="NEEDS_HUMAN_SEMANTIC_REVIEW",
                review_note=None,
            )
            session.add(record)
            session.commit()
            inserted_id = record.id

        with SessionLocal() as session:
            result = session.execute(
                select(QAReviewRecord).where(QAReviewRecord.id == inserted_id)
            ).scalar_one()
            assert result.question == marker
            assert result.evidence == [{"source": "smoke-test", "page": 1}]
            assert result.review_status == "pending_review"
            assert result.created_at.tzinfo is not None

        print("通过：审核记录已插入并查询，JSONB、默认状态和时区时间均有效。")
        return 0
    except (AssertionError, SQLAlchemyError) as exc:
        print(f"失败：审核记录 ORM 验证错误类型：{type(exc).__name__}")
        return 1
    finally:
        # Delete by the unique marker even if verification failed after commit.
        try:
            with SessionLocal() as session:
                records = session.execute(
                    select(QAReviewRecord).where(
                        QAReviewRecord.question == marker
                    )
                ).scalars()
                for record in records:
                    session.delete(record)
                session.commit()
        except SQLAlchemyError as exc:
            print(f"清理测试记录失败，数据库错误类型：{type(exc).__name__}")
            return 1

        if inserted_id is not None:
            print("已清理本次测试记录。")


if __name__ == "__main__":
    sys.exit(main())
