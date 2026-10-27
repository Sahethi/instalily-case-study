"""Settings and a tiny .env loader (python-dotenv isn't a dependency)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from .env without overriding real env vars."""
    for path in (_REPO_ROOT / ".env", _REPO_ROOT / "backend" / ".env"):
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

AGENT_MODEL = os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
ROUTER_MODEL = os.environ.get("OPENAI_ROUTER_MODEL") or AGENT_MODEL
MAX_STEPS = int(os.environ.get("CHAT_MAX_STEPS", "10"))
MAX_TOOL_OUTPUT_CHARS = 12_000  # what the model sees; the log keeps the full result
MAX_ACTIVE_TOOLS = 24


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    # Retries cover transient 429/5xx; the SDK honours Retry-After.
    return OpenAI(timeout=45, max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "2")))
