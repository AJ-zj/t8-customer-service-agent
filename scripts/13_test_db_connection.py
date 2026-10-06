import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


# 1. 从项目根目录读取 .env
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise SystemExit("连接失败：.env 中没有找到 DATABASE_URL")


# 2. 创建 ORM 数据库连接引擎
engine = create_engine(database_url, pool_pre_ping=True)

try:
    # 3. 执行只读 SQL，确认连接的是哪个数据库、哪个账号
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user")
        ).one()

        print("数据库连接成功！")
        print(f"数据库名称：{result[0]}")
        print(f"连接账号：{result[1]}")

except SQLAlchemyError as exc:
    print(f"数据库连接失败，错误类型：{type(exc).__name__}")
    print("请检查 .env 中的地址、端口、账号和密码。")
    raise SystemExit(1)

finally:
    engine.dispose()