"""Explicit maintenance command; no scheduler, Agent or model calls."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None, *, service_factory=None, database_check=None) -> int:
    parser = argparse.ArgumentParser(description="Clean expired memory in roborock only.")
    parser.add_argument("--memory-owner-id", default=None)
    args = parser.parse_args(argv)
    try:
        if database_check is None:
            from app.database import configured_database_name, current_database_name
            database_check = lambda: (configured_database_name(), current_database_name())
        if database_check() != ("roborock", "roborock"):
            raise RuntimeError("Database must be roborock.")
        if service_factory is None:
            from app.memory.service import MemoryFactService
            service_factory = MemoryFactService
        deleted_count = service_factory().cleanup_expired_facts(args.memory_owner_id)
    except Exception:
        print("清理失败；请检查 roborock 配置及数据库服务。")
        return 1
    print(f"deleted_count={deleted_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
