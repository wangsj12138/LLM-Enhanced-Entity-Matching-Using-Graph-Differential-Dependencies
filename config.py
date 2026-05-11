from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str | None
    model: str
    temperature: float
    self_consistency: int
    timeout_seconds: float


def neo4j_config() -> Neo4jConfig:
    load_dotenv()
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "xxx"),
    )


def llm_config() -> LLMConfig:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL") or None,
        model=os.getenv("LLM_MODEL", "xxx"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        self_consistency=int(os.getenv("LLM_SELF_CONSISTENCY", "1")),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
    )
