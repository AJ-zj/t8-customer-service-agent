"""Create agent_memory_facts only in the configured roborock database."""

import sys
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import configured_database_name, current_database_name, engine
from app.models import AgentMemoryFact, Base


def main() -> int:
    if configured_database_name() != "roborock":
        print("停止：DATABASE_URL 配置的数据库不是 roborock。")
        return 1
    try:
        if current_database_name() != "roborock":
            print("停止：当前连接的数据库不是 roborock。")
            return 1
        Base.metadata.create_all(bind=engine, tables=[AgentMemoryFact.__table__],
                                 checkfirst=True)
    except SQLAlchemyError as error:
        print(f"建表失败，数据库错误类型：{type(error).__name__}")
        return 1
    print("完成：roborock.agent_memory_facts 已存在或已创建；未重建已有表。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
