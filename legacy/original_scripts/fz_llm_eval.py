from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import llm_config
from er_utils import evaluate, jaccard, load_truth, phone_digits
from gaplink_pipeline import fetch_candidates
from llm_matcher import LLMMatcher, parse_decision


EXAMPLES = [
    {
        "label": True,
        "restaurant_1": "art delicatessen | 12224 ventura blvd | studio city | 818/762 -1221 | american",
        "restaurant_2": "art deli | 12224 ventura blvd | studio city | 818-762-1221 | delis",
    },
    {
        "label": False,
        "restaurant_1": "stars | 150 redwood alley | san francisco | 415/861 -7827 | american",
        "restaurant_2": "stars | 500 van ness ave | san francisco | 415/861 -4344 | american",
    },
    {
        "label": True,
        "restaurant_1": "cafe bizou | 14016 ventura blvd | sherman oaks | 818/788 -3536 | french",
        "restaurant_2": "cafe bizou | 14016 ventura blvd | sherman oaks | 818-788-3536 | french bistro",
    },
]

PROMPT_VERSION = "v2-address-name-primary"


def format_restaurant(candidate, side: str) -> str:
    node = candidate.left if side == "left" else candidate.right
    address = candidate.left_address if side == "left" else candidate.right_address
    return (
        f"name={node.get('name', '')}; address={address}; city_context={candidate.city}; "
        f"phone={node.get('phone', '')}; cuisine={node.get('cuisine', '')}"
    )


def prompt(candidate, mode: str) -> str:
    examples = ""
    if mode in {"few-shot", "self-consistency"}:
        examples = "\nExamples:\n" + "\n".join(
            [
                json.dumps(
                    {
                        "restaurant_1": item["restaurant_1"],
                        "restaurant_2": item["restaurant_2"],
                        "match": item["label"],
                    },
                    ensure_ascii=False,
                )
                for item in EXAMPLES
            ]
        )
    return f"""
You are doing restaurant entity resolution for the Fodors-Zagats benchmark.
Decide whether the two records refer to the same real-world restaurant.
Use name, address, city, phone, cuisine, and the graph-derived GDD rules.
Phone formatting differences and minor spelling/order differences can still be a match.
Treat name plus address/location as the strongest evidence. Phone and cuisine may differ
between sources and should not veto a match when name and address clearly agree.
Same city/cuisine/name alone is not enough when both address and phone clearly conflict.
Return compact JSON only: {{"match": true/false, "confidence": 0.0-1.0, "reason": "..."}}
{examples}

GDD rules triggered: {candidate.rules}
Graph pattern: restaurant_1-LOCATED_AT-address_1-IN_CITY-city_context; restaurant_2-LOCATED_AT-address_2-IN_CITY-city_context
restaurant_1: {format_restaurant(candidate, "left")}
restaurant_2: {format_restaurant(candidate, "right")}
""".strip()


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row
    return cache


def append_cache(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge(candidate, mode: str, votes: int) -> dict:
    matcher = LLMMatcher(llm_config())
    decisions = []
    for _ in range(votes):
        decision = matcher.decide(prompt(candidate, mode))
        decisions.append(decision)
    positives = [d for d in decisions if d.match]
    negatives = [d for d in decisions if not d.match]
    chosen = positives if len(positives) >= len(negatives) else negatives
    return {
        "left_id": candidate.left_id,
        "right_id": candidate.right_id,
        "match": len(positives) >= len(negatives),
        "confidence": sum(d.confidence for d in chosen) / len(chosen),
        "reason": " | ".join(d.reason for d in chosen)[:500],
        "raw": "\n".join(d.raw for d in decisions),
        "votes": len(decisions),
    }


def strong_phone_normalization(candidate) -> bool:
    exact_phone = phone_digits(candidate.left.get("phone")) and phone_digits(candidate.left.get("phone")) == phone_digits(candidate.right.get("phone"))
    name_sim = jaccard(candidate.left.get("name"), candidate.right.get("name"))
    address_sim = jaccard(candidate.left_address, candidate.right_address)
    return bool(exact_phone and (name_sim >= 0.4 or address_sim >= 0.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce FZ LLM prompt performance.")
    parser.add_argument("--mode", choices=["zero-shot", "few-shot", "self-consistency"], default="zero-shot")
    parser.add_argument("--threshold", type=float, default=0.48)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("output_file"))
    parser.add_argument("--phone-normalization", action="store_true")
    parser.add_argument("--one-to-one", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / f"fz_{args.mode}_llm_cache.jsonl"
    cache = load_cache(cache_path)
    lock = threading.Lock()
    votes = 3 if args.mode == "self-consistency" else 1

    candidates = [c for c in fetch_candidates() if c.structural_score >= args.threshold]
    candidates.sort(key=lambda c: c.structural_score, reverse=True)
    if args.limit:
        candidates = candidates[: args.limit]

    truth = load_truth(Path("dataset/fodors_zagats/relational-dataset/fodors-zagats/matches.csv"), table_b_offset=533)
    accepted: list[tuple[float, float, tuple[int, int]]] = []
    start = time.time()

    def run_one(candidate):
        key = f"{PROMPT_VERSION}:{args.mode}:{candidate.left_id}:{candidate.right_id}:{votes}"
        if key in cache:
            return cache[key]
        row = {"key": key, **judge(candidate, args.mode, votes)}
        append_cache(cache_path, row, lock)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, candidate) for candidate in candidates]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            candidate = next(c for c in candidates if c.left_id == int(row["left_id"]) and c.right_id == int(row["right_id"]))
            pair = tuple(sorted((int(row["left_id"]), int(row["right_id"]))))
            if row["match"] or (args.phone_normalization and strong_phone_normalization(candidate)):
                accepted.append((float(row.get("confidence", 0.0)), candidate.structural_score, pair))
            if args.one_to_one:
                predictions = one_to_one_predictions(accepted)
            else:
                predictions = {item[2] for item in accepted}
            if i % 10 == 0 or i == len(futures):
                metrics = evaluate(predictions, truth)
                print(
                    f"{i}/{len(futures)} p={metrics['precision']:.4f} "
                    f"r={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
                )

    predictions = one_to_one_predictions(accepted) if args.one_to_one else {item[2] for item in accepted}
    metrics = evaluate(predictions, truth)
    result = {
        **metrics,
        "mode": args.mode,
        "threshold": args.threshold,
        "candidate_pairs": len(candidates),
        "predicted_matches": len(predictions),
        "votes": votes,
        "phone_normalization": args.phone_normalization,
        "one_to_one": args.one_to_one,
        "runtime_seconds": time.time() - start,
    }
    out = args.output_dir / f"fz_{args.mode}_metrics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def one_to_one_predictions(accepted: list[tuple[float, float, tuple[int, int]]]) -> set[tuple[int, int]]:
    used_left: set[int] = set()
    used_right: set[int] = set()
    predictions: set[tuple[int, int]] = set()
    for _confidence, _score, pair in sorted(accepted, reverse=True):
        if pair[0] in used_left or pair[1] in used_right:
            continue
        predictions.add(pair)
        used_left.add(pair[0])
        used_right.add(pair[1])
    return predictions


if __name__ == "__main__":
    main()
