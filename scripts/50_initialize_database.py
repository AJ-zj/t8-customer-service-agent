"""Compose one-shot schema initialization; additive, ordered and repeatable.

Each existing step owns its transaction. A failed step stops initialization;
rerunning resumes safely because all steps are idempotent.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_step(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize():
    from app.database import configured_database_name, current_database_name
    if configured_database_name() != "roborock" or current_database_name() != "roborock":
        raise RuntimeError("初始化仅允许 roborock。")
    for filename in ("14_create_review_table.py", "21_create_memory_facts_table.py",
                     "32_upgrade_review_publication.py", "40_upgrade_memory_lifecycle.py"):
        if load_step(filename).main() != 0:
            raise RuntimeError("数据库初始化步骤失败。")


if __name__ == "__main__":
    try:
        initialize()
    except Exception:
        print("数据库初始化失败；未删除或重建已有表。请检查 roborock 配置。")
        sys.exit(1)
    print("Schema ready: roborock")
