from __future__ import annotations

import csv
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower().replace("\\", " ")
    return " ".join(TOKEN_RE.findall(value))


def tokens(value: str | None) -> set[str]:
    return set(clean_text(value).split())


def phone_digits(value: str | None) -> str:
    return "".join(ch for ch in (value or "") if ch.isdigit())


def jaccard(left: str | None, right: str | None) -> float:
    a = tokens(left)
    b = tokens(right)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_truth(path: Path, table_b_offset: int | None = None) -> set[tuple[int, int]]:
    truth: set[tuple[int, int]] = set()
    if path.suffix.lower() != ".csv":
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not lines:
            return truth
        if all(len(line.split()) == 2 for line in lines):
            for line in lines:
                left, right = (int(value) for value in line.split())
                truth.add(tuple(sorted((left, right))))
        else:
            values = [int(line) for line in lines]
            for left, right in zip(values[0::2], values[1::2]):
                truth.add(tuple(sorted((left, right))))
        return truth

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            left = int(row["ltable_id"])
            right = int(row["rtable_id"])
            if table_b_offset is not None:
                right += table_b_offset
            truth.add(tuple(sorted((left, right))))
    return truth


def evaluate(predicted: set[tuple[int, int]], truth: set[tuple[int, int]]) -> dict[str, float]:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
