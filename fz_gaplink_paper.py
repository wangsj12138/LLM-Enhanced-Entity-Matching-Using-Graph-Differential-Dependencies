from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from config import llm_config
from gaplink_local import evaluate, jaccard, load_truth, phone_digits
from gaplink_pipeline import GDD_QUERIES, GraphCandidate, fetch_candidates
from llm_matcher import LLMMatcher


EXAMPLES = [
    {
        "match": True,
        "restaurant_1": "art delicatessen | 12224 ventura blvd | studio city | 818/762 -1221 | american",
        "restaurant_2": "art deli | 12224 ventura blvd | studio city | 818-762-1221 | delis",
    },
    {
        "match": False,
        "restaurant_1": "stars | 150 redwood alley | san francisco | 415/861 -7827 | american",
        "restaurant_2": "stars | 500 van ness ave | san francisco | 415/861 -4344 | american",
    },
    {
        "match": True,
        "restaurant_1": "cafe bizou | 14016 ventura blvd | sherman oaks | 818/788 -3536 | french",
        "restaurant_2": "cafe bizou | 14016 ventura blvd | sherman oaks | 818-788-3536 | french bistro",
    },
]

DEFAULTS = {
    "zero-shot": {"structural_threshold": 0.40, "top_rules": 3, "one_to_one": False},
    "few-shot": {"structural_threshold": 0.36, "top_rules": 3, "one_to_one": False},
    "self-consistency": {"structural_threshold": 0.36, "top_rules": 3, "one_to_one": True},
}


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities.values() if value > 0)


def candidate_probability(candidate: GraphCandidate, probabilities: dict[str, float]) -> float:
    return min(1.0, sum(probabilities[rule] for rule in candidate.rules))


def bayes_update(probabilities: dict[str, float], candidate: GraphCandidate, match: bool, confidence: float) -> dict[str, float]:
    theta = max(0.05, min(0.95, confidence or 0.8))
    p_match = max(0.05, min(0.95, candidate_probability(candidate, probabilities)))
    candidate_rules = set(candidate.rules)
    updated: dict[str, float] = {}
    for rule, old in probabilities.items():
        contains = rule in candidate_rules
        if match and contains:
            numerator = old * theta
            denominator = p_match * theta + (1 - p_match) * (1 - theta)
        elif (not match) and contains:
            numerator = old * (1 - theta)
            denominator = p_match * (1 - theta) + (1 - p_match) * theta
        elif match and not contains:
            numerator = old * (1 - theta)
            denominator = (1 - p_match) * (1 - theta) + p_match * theta
        else:
            numerator = old * theta
            denominator = (1 - p_match) * theta + p_match * (1 - theta)
        updated[rule] = numerator / denominator if denominator else old
    total = sum(updated.values())
    return {rule: value / total for rule, value in updated.items()}


def restaurant_text(candidate: GraphCandidate, side: str) -> str:
    node = candidate.left if side == "left" else candidate.right
    address = candidate.left_address if side == "left" else candidate.right_address
    return (
        f"name={node.get('name', '')}; address={address}; city={candidate.city}; "
        f"phone={node.get('phone', '')}; cuisine={node.get('cuisine', '')}"
    )


def rule_feedback_prompt(candidate: GraphCandidate) -> str:
    payload = {
        "task": "GAPLink Stage 1: give match feedback to refine GDD rule probabilities.",
        "gdd_rules_triggered": candidate.rules,
        "graph_pattern": [
            "(restaurant_1)-[:LOCATED_AT]->(address_1)-[:IN_CITY]->(city_context)",
            "(restaurant_2)-[:LOCATED_AT]->(address_2)-[:IN_CITY]->(city_context)",
        ],
        "restaurant_1": restaurant_text(candidate, "left"),
        "restaurant_2": restaurant_text(candidate, "right"),
        "instruction": 'Return JSON only: {"match": true/false, "confidence": 0.0-1.0, "reason": "..."}',
    }
    return json.dumps(payload, ensure_ascii=False)


def match_prompt(candidate: GraphCandidate, mode: str) -> str:
    examples = ""
    if mode in {"few-shot", "self-consistency"}:
        examples = "\nExamples:\n" + "\n".join(json.dumps(item, ensure_ascii=False) for item in EXAMPLES)
    payload = {
        "gdd_rules_triggered": candidate.rules,
        "graph_pattern": [
            "(restaurant_1)-[:LOCATED_AT]->(address_1)-[:IN_CITY]->(city_context)",
            "(restaurant_2)-[:LOCATED_AT]->(address_2)-[:IN_CITY]->(city_context)",
        ],
        "graph_scores": {
            "structural_score": round(candidate.structural_score, 4),
            "name_jaccard": round(jaccard(candidate.left.get("name"), candidate.right.get("name")), 4),
            "address_jaccard": round(jaccard(candidate.left_address, candidate.right_address), 4),
        },
        "restaurant_1": restaurant_text(candidate, "left"),
        "restaurant_2": restaurant_text(candidate, "right"),
    }
    return f"""
You are doing restaurant entity resolution for Fodors-Zagats.
Decide whether restaurant_1 and restaurant_2 refer to the same real-world restaurant.
Use both attributes and graph/GDD evidence.
Treat name plus address/location as the strongest evidence.
Phone formatting differences and minor spelling differences can still be a match.
Cuisine may be broader/narrower across sources and should not veto a clear name/address match.
Return compact JSON only: {{"match": true/false, "confidence": 0.0-1.0, "reason": "..."}}
{examples}

{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def load_jsonl_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row
    return cache


def load_legacy_stage2_cache(output_dir: Path, mode: str, votes: int) -> dict[tuple[int, int], dict]:
    cache: dict[tuple[int, int], dict] = {}
    path = output_dir / f"fz_{mode}_llm_cache.jsonl"
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("votes", votes)) != votes:
                continue
            reason = str(row.get("reason", ""))
            if reason.startswith("LLM request failed:"):
                continue
            pair = tuple(sorted((int(row["left_id"]), int(row["right_id"]))))
            cache[pair] = row
    return cache


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_failed_decision(row: dict) -> bool:
    return str(row.get("reason", "")).startswith("LLM request failed:") or (
        not str(row.get("raw", "")) and float(row.get("confidence", 0.0)) == 0.0
    )


def optimize_rules(
    candidates: list[GraphCandidate],
    output_dir: Path,
    max_llm_calls: int,
    theta: float,
    use_cache: bool,
) -> tuple[dict[str, float], list[dict]]:
    cache_path = output_dir / "fz_gaplink_stage1_cache.jsonl"
    cache = load_jsonl_cache(cache_path) if use_cache else {}
    lock = threading.Lock()
    matcher = LLMMatcher(llm_config())
    probabilities = {rule: 1 / len(GDD_QUERIES) for rule in GDD_QUERIES}
    asked: set[tuple[int, int]] = set()
    history: list[dict] = []
    attempts = 0
    max_attempts = min(len(candidates), max_llm_calls * 4)
    while len(history) < min(max_llm_calls, len(candidates)) and attempts < max_attempts:
        attempts += 1
        for candidate in candidates:
            candidate.rule_probability = candidate_probability(candidate, probabilities)
        pending = [candidate for candidate in candidates if candidate.pair not in asked]
        if not pending:
            break
        selected = max(
            pending,
            key=lambda item: (
                1 / (1 + abs(item.rule_probability - theta)),
                item.structural_score,
                len(item.rules),
            ),
        )
        key = f"fz-gaplink-stage1-v1:{selected.left_id}:{selected.right_id}:{','.join(selected.rules)}"
        if key in cache:
            row = cache[key]
        else:
            decision = matcher.decide(rule_feedback_prompt(selected))
            row = {
                "key": key,
                "left_id": selected.left_id,
                "right_id": selected.right_id,
                "rules": selected.rules,
                "match": decision.match,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "raw": decision.raw,
            }
        asked.add(selected.pair)
        if is_failed_decision(row):
            print(f"Stage1 skip failed feedback for {selected.left_id}-{selected.right_id}: {row.get('reason', '')[:80]}")
            continue
        append_jsonl(cache_path, row, lock)
        probabilities = bayes_update(probabilities, selected, bool(row["match"]), float(row.get("confidence", 0.8)))
        iteration = len(history) + 1
        history.append(
            {
                "iteration": iteration,
                "left_id": selected.left_id,
                "right_id": selected.right_id,
                "rules": selected.rules,
                "match": row["match"],
                "confidence": row.get("confidence", 0.0),
                "reason": row.get("reason", ""),
                "rule_probabilities": probabilities,
                "rule_entropy": entropy(probabilities),
            }
        )
        print(f"Stage1 {iteration}/{max_llm_calls} entropy={entropy(probabilities):.4f}")
    return probabilities, history


def select_candidates(
    candidates: list[GraphCandidate],
    probabilities: dict[str, float],
    structural_threshold: float,
    top_rules: int,
) -> tuple[list[str], list[GraphCandidate]]:
    ranked_rules = [rule for rule, _value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)]
    selected_rules = ranked_rules[:top_rules]
    filtered = [
        candidate
        for candidate in candidates
        if candidate.structural_score >= structural_threshold and any(rule in candidate.rules for rule in selected_rules)
    ]
    for candidate in filtered:
        candidate.rule_probability = candidate_probability(candidate, probabilities)
    filtered.sort(
        key=lambda item: (
            item.rule_probability,
            item.structural_score,
            len(item.rules),
        ),
        reverse=True,
    )
    return selected_rules, filtered


def stage2_key(mode: str, candidate: GraphCandidate, votes: int) -> str:
    model = llm_config().model
    return f"fz-gaplink-stage2-v2:{model}:{mode}:{candidate.left_id}:{candidate.right_id}:{votes}"


def judge_candidate(candidate: GraphCandidate, mode: str, votes: int) -> dict:
    matcher = LLMMatcher(llm_config())
    decisions = [matcher.decide(match_prompt(candidate, mode)) for _ in range(votes)]
    positives = [decision for decision in decisions if decision.match]
    negatives = [decision for decision in decisions if not decision.match]
    chosen = positives if len(positives) >= len(negatives) else negatives
    return {
        "left_id": candidate.left_id,
        "right_id": candidate.right_id,
        "match": len(positives) >= len(negatives),
        "confidence": sum(decision.confidence for decision in chosen) / len(chosen),
        "reason": " | ".join(decision.reason for decision in chosen if decision.reason)[:500],
        "raw": "\n".join(decision.raw for decision in decisions),
        "votes": votes,
    }


def strong_phone_match(candidate: GraphCandidate) -> bool:
    left_phone = phone_digits(candidate.left.get("phone"))
    right_phone = phone_digits(candidate.right.get("phone"))
    return bool(
        left_phone
        and left_phone == right_phone
        and (
            jaccard(candidate.left.get("name"), candidate.right.get("name")) >= 0.4
            or jaccard(candidate.left_address, candidate.right_address) >= 0.5
        )
    )


def one_to_one_predictions(accepted: list[tuple[float, float, tuple[int, int]]]) -> set[tuple[int, int]]:
    used_left: set[int] = set()
    used_right: set[int] = set()
    predictions: set[tuple[int, int]] = set()
    for confidence, score, pair in sorted(accepted, reverse=True):
        if pair[0] in used_left or pair[1] in used_right:
            continue
        predictions.add(pair)
        used_left.add(pair[0])
        used_right.add(pair[1])
    return predictions


def run(
    mode: str,
    output_dir: Path = Path("output_file"),
    max_llm_calls: int = 20,
    theta: float = 0.5,
    structural_threshold: float | None = None,
    top_rules: int | None = None,
    workers: int = 8,
    use_cache: bool = True,
    use_stage2_cache: bool = True,
    phone_normalization: bool = True,
    one_to_one: bool | None = None,
) -> dict:
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = DEFAULTS[mode]
    structural_threshold = defaults["structural_threshold"] if structural_threshold is None else structural_threshold
    top_rules = defaults["top_rules"] if top_rules is None else top_rules
    one_to_one = defaults["one_to_one"] if one_to_one is None else one_to_one

    all_candidates = fetch_candidates()
    probabilities, history = optimize_rules(all_candidates, output_dir, max_llm_calls, theta, use_cache)
    selected_rules, candidates = select_candidates(all_candidates, probabilities, structural_threshold, top_rules)
    truth = load_truth(Path("dataset/relational-dataset/fodors-zagats/matches.csv"), table_b_offset=533)
    votes = 3 if mode == "self-consistency" else 1
    cache_path = output_dir / f"fz_gaplink_{mode}_stage2_cache.jsonl"
    cache = load_jsonl_cache(cache_path) if use_cache and use_stage2_cache else {}
    legacy_cache = load_legacy_stage2_cache(output_dir, mode, votes) if use_cache and use_stage2_cache else {}
    lock = threading.Lock()

    def run_one(candidate: GraphCandidate) -> dict:
        key = stage2_key(mode, candidate, votes)
        if key in cache:
            return cache[key]
        if candidate.pair in legacy_cache:
            legacy_row = legacy_cache[candidate.pair]
            row = {
                "key": key,
                "left_id": legacy_row["left_id"],
                "right_id": legacy_row["right_id"],
                "match": legacy_row["match"],
                "confidence": legacy_row.get("confidence", 0.0),
                "reason": legacy_row.get("reason", ""),
                "raw": legacy_row.get("raw", ""),
                "votes": legacy_row.get("votes", votes),
                "cache_source": "legacy_fz_llm_eval",
            }
            append_jsonl(cache_path, row, lock)
            return row
        row = {"key": key, **judge_candidate(candidate, mode, votes)}
        append_jsonl(cache_path, row, lock)
        return row

    accepted: list[tuple[float, float, tuple[int, int]]] = []
    by_pair = {candidate.pair: candidate for candidate in candidates}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, candidate) for candidate in candidates]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            pair = tuple(sorted((int(row["left_id"]), int(row["right_id"]))))
            candidate = by_pair[pair]
            if row["match"] or (phone_normalization and strong_phone_match(candidate)):
                accepted.append((float(row.get("confidence", 0.0)), candidate.structural_score, pair))
            predictions = one_to_one_predictions(accepted) if one_to_one else {item[2] for item in accepted}
            if index % 20 == 0 or index == len(futures):
                metrics = evaluate(predictions, truth)
                print(f"Stage2 {index}/{len(futures)} p={metrics['precision']:.4f} r={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

    predictions = one_to_one_predictions(accepted) if one_to_one else {item[2] for item in accepted}
    metrics = evaluate(predictions, truth)
    result = {
        **metrics,
        "dataset": "FZ",
        "mode": mode,
        "llm_rule_feedback_calls": len(history),
        "llm_candidate_calls": len(candidates) * votes,
        "candidate_pairs_after_rule_filter": len(candidates),
        "selected_rules": selected_rules,
        "rule_probabilities": probabilities,
        "rule_entropy": entropy(probabilities),
        "structural_threshold": structural_threshold,
        "top_rules": top_rules,
        "votes": votes,
        "phone_normalization": phone_normalization,
        "one_to_one": one_to_one,
        "predicted_matches": len(predictions),
        "f1_percent": metrics["f1"] * 100,
        "runtime_seconds": time.time() - start,
    }
    (output_dir / f"fz_gaplink_{mode}_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"fz_gaplink_{mode}_rule_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / f"fz_gaplink_{mode}_candidates.jsonl").open("w", encoding="utf-8") as file:
        for candidate in candidates:
            file.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FZ using the paper GAPLink flow.")
    parser.add_argument("--mode", choices=["zero-shot", "few-shot", "self-consistency"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output_file"))
    parser.add_argument("--max-llm-calls", type=int, default=20)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--structural-threshold", type=float)
    parser.add_argument("--top-rules", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-stage2-cache", action="store_true")
    parser.add_argument("--no-phone-normalization", action="store_true")
    parser.add_argument("--one-to-one", action="store_true")
    parser.add_argument("--no-one-to-one", action="store_true")
    args = parser.parse_args()
    one_to_one = None
    if args.one_to_one:
        one_to_one = True
    if args.no_one_to_one:
        one_to_one = False
    run(
        mode=args.mode,
        output_dir=args.output_dir,
        max_llm_calls=args.max_llm_calls,
        theta=args.theta,
        structural_threshold=args.structural_threshold,
        top_rules=args.top_rules,
        workers=args.workers,
        use_cache=not args.no_cache,
        use_stage2_cache=not args.no_stage2_cache,
        phone_normalization=not args.no_phone_normalization,
        one_to_one=one_to_one,
    )


if __name__ == "__main__":
    main()
