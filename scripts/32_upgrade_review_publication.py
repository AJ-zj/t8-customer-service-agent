"""Idempotently add publication state to roborock.qa_review_records."""

import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database import configured_database_name, engine


def main():
    if configured_database_name() != "roborock":
        print("停止：配置数据库不是 roborock。")
        return 1
    try:
        with engine.begin() as connection:
            if connection.scalar(text("SELECT current_database()")) != "roborock":
                print("停止：连接数据库不是 roborock。")
                return 1
            connection.execute(text("""
                ALTER TABLE qa_review_records
                ADD COLUMN IF NOT EXISTS publication_status VARCHAR(32)
                    NOT NULL DEFAULT 'unpublished',
                ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ NULL
            """))
            # Check constraints are additive; old rows remain unpublished with NULL time.
            for name, expression in (
                ("ck_review_publication_status",
                 "publication_status IN ('unpublished', 'published')"),
                ("ck_review_published_approved",
                 "publication_status != 'published' OR "
                 "(review_status = 'approved' AND published_at IS NOT NULL "
                 "AND draft_answer IS NOT NULL)"),
            ):
                exists = connection.scalar(text("""
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'qa_review_records'::regclass AND conname = :name
                """), {"name": name})
                if not exists:
                    # Both strings come exclusively from the fixed literals above.
                    connection.execute(text(
                        f"ALTER TABLE qa_review_records ADD CONSTRAINT {name} CHECK ({expression})"
                    ))
    except SQLAlchemyError as exc:
        print(f"升级失败：{type(exc).__name__}；事务已回滚。")
        return 1
    print("完成：roborock.qa_review_records 发布字段与约束已就绪；已有记录未删除。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
