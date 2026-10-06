"""兼容命令行入口；Grounded QA 生产实现位于 app.services.grounded_qa。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.grounded_qa import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
