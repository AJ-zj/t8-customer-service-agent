"""Shared application configuration loaded from the project-root .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

CHAT_URL = "https://api.siliconflow.cn/v1/chat/completions"
ROUTER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHAT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
BACKUP_CHAT_MODEL = "Qwen/Qwen3.8-27B"
SEMANTIC_GATE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Automatic injection is experimental; safe extraction/storage remain enabled.
# Only an explicit true enables the previous retrieval path.
ENABLE_LONG_TERM_MEMORY_INJECTION = (
    os.getenv("ENABLE_LONG_TERM_MEMORY_INJECTION", "false").strip().lower() == "true"
)


def get_siliconflow_api_key() -> str:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "未配置 SILICONFLOW_API_KEY。请在项目根目录 .env 中配置 API Key。"
        )
    return api_key
