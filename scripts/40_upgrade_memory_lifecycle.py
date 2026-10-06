"""Add lifecycle fields to roborock.agent_memory_facts without deleting rows."""

import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def upgrade_memory_lifecycle(target_engine, configured_name: str) -> None:
    if configured_name != "roborock":
        raise RuntimeError("Configured database must be roborock.")
    with target_engine.begin() as connection:
        if connection.scalar(text("SELECT current_database()")) != "roborock":
            raise RuntimeError("Connected database must be roborock.")
        connection.execute(text("""
            ALTER TABLE agent_memory_facts
            ADD COLUMN IF NOT EXISTS fact_category VARCHAR(32) NULL,
            ADD COLUMN IF NOT EXISTS importance DOUBLE PRECISION NULL,
            ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ NULL,
            ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ NULL
        """))
        # Old rows have no validated category. Never classify them from fact_text.
        # Backfill once; a later intentional NULL expires_at is not overwritten.
        connection.execute(text("""
            UPDATE agent_memory_facts SET
                importance = COALESCE(importance, 0.5),
                last_seen_at = COALESCE(last_seen_at, updated_at, created_at),
                expires_at = COALESCE(expires_at,
                    COALESCE(last_seen_at, updated_at, created_at) + INTERVAL '30 days')
            WHERE importance IS NULL OR last_seen_at IS NULL
        """))
        connection.execute(text("""
            ALTER TABLE agent_memory_facts
            ALTER COLUMN importance SET DEFAULT 0.5,
            ALTER COLUMN importance SET NOT NULL,
            ALTER COLUMN last_seen_at SET DEFAULT NOW(),
            ALTER COLUMN last_seen_at SET NOT NULL
        """))
        for name, expression in (
            ("ck_memory_importance", "importance >= 0 AND importance <= 1"),
            ("ck_memory_fact_category", "fact_category IS NULL OR fact_category IN "
             "('device_profile', 'home_environment', 'usage_preference')"),
        ):
            exists = connection.scalar(text("""
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'agent_memory_facts'::regclass AND conname = :name
            """), {"name": name})
            if not exists:
                # Both identifiers/expressions are fixed program constants.
                connection.execute(text(
                    f"ALTER TABLE agent_memory_facts ADD CONSTRAINT {name} CHECK ({expression})"
                ))


def main() -> int:
    from app.database import configured_database_name, engine
    try:
        upgrade_memory_lifecycle(engine, configured_database_name())
    except Exception:
        print("生命周期升级失败；事务已回滚。请确认仅连接 roborock。")
        return 1
    print("完成：roborock.agent_memory_facts 生命周期字段已就绪；已有事实未删除。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
