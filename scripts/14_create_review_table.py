"""Create qa_review_records in the configured roborock database, if absent."""

import sys
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import (
    configured_database_name,
    current_database_name,
    engine,
)
from app.models import Base, QAReviewRecord


TARGET_DATABASE = "roborock"


def main() -> int:
    if configured_database_name() != TARGET_DATABASE:
        print("停止：DATABASE_URL 配置的数据库不是 roborock。")
        return 1

    try:
        connected_database = current_database_name()
        if connected_database != TARGET_DATABASE:
            print("停止：当前连接的数据库不是 roborock，未执行建表。")
            return 1

        Base.metadata.create_all(
            bind=engine,
            tables=[QAReviewRecord.__table__],
            checkfirst=True,
        )
    except SQLAlchemyError as exc:
        print(f"建表失败，数据库错误类型：{type(exc).__name__}")
        return 1

    print("完成：roborock.qa_review_records 已存在或已创建。")
    print("已有表不会被删除或重建。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
