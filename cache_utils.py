from __future__ import annotations

import json
import threading
from pathlib import Path


def is_failed(row: dict) -> bool:
    return str(row.get("reason", "")).startswith("LLM request failed:") or (
        not str(row.get("raw", "")) and float(row.get("confidence", 0.0)) == 0.0
    )


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row
    return rows
