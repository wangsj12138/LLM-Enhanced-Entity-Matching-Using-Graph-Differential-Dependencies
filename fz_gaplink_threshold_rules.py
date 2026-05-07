from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from config import llm_config
from gaplink_local import clean_text, evaluate, jaccard, load_truth, phone_digits
from llm_matcher import LLMMatcher
from neo4j_setup import get_driver


PROMPT_VERSION = "v6-compact"


@dataclass
class RuleCandidate:
    left_id: int
    right_id: int
    left: dict
    right: dict
    left_address: str
    right_address: str
    left_city: str
    right_city: str
    rules: list[str]
    rule_probability: float = 0.0

    @property
    def pair(self) -> tuple[int, int]:
        return tuple(sorted((self.left_id, self.right_id)))


RULE_DESCRIPTIONS = {
    "phone_exact": "normalized phone numbers are identical",
    "phone_area": "phone area codes are identical",
    "name_sim_ge_045": "name token Jaccard similarity >= 0.45",
    "name_sim_ge_060": "name token Jaccard similarity >= 0.60",
    "name_sim_ge_075": "name token Jaccard similarity >= 0.75",
    "address_sim_ge_030": "address token Jaccard similarity >= 0.30",
    "address_sim_ge_050": "address token Jaccard similarity >= 0.50",
    "cuisine_sim_ge_050": "cuisine token Jaccard similarity >= 0.50",
    "same_city": "city strings are identical after normalization",
    "name045_address030": "name similarity >= 0.45 AND address similarity >= 0.30",
    "name045_phone_area": "name similarity >= 0.45 AND phone area codes match",
    "phone_exact_or_name040_phone_area": "phone exact OR (phone area matches AND name similarity >= 0.40)",
    "phone_exact_or_name035_phone_area": "phone exact OR (phone area matches AND name similarity >= 0.35)",
    "address050_phone_area": "address similarity >= 0.50 AND phone area codes match",
    "name_or_cuisine_or_char060": "name token overlap exists OR cuisine similarity >= 0.50 OR character-level name similarity >= 0.60",
    "address030_or_exact_name_or_name075": "address similarity >= 0.30 OR normalized names are identical OR name similarity >= 0.75",
    "phone_name_candidate_consistent": (
        "(phone exact OR phone area matches with name similarity >= 0.40) "
        "AND (name/cuisine/character-name consistency) AND (address/name consistency)"
    ),
}


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


def node_dict(node) -> dict:
    return dict(node.items())


def phone_area(value: str | None) -> str:
    digits = phone_digits(value)
    return digits[:3] if len(digits) >= 3 else ""


def triggered_rules(candidate: RuleCandidate) -> list[str]:
    name_sim = jaccard(candidate.left.get("name"), candidate.right.get("name"))
    address_sim = jaccard(candidate.left_address, candidate.right_address)
    cuisine_sim = jaccard(candidate.left.get("cuisine"), candidate.right.get("cuisine"))
    left_name = clean_text(candidate.left.get("name"))
    right_name = clean_text(candidate.right.get("name"))
    name_chars = difflib.SequenceMatcher(None, left_name, right_name).ratio()
    left_phone = phone_digits(candidate.left.get("phone"))
    right_phone = phone_digits(candidate.right.get("phone"))
    exact_phone = bool(left_phone and left_phone == right_phone)
    same_area = bool(phone_area(candidate.left.get("phone")) and phone_area(candidate.left.get("phone")) == phone_area(candidate.right.get("phone")))
    same_city = bool(clean_text(candidate.left_city) and clean_text(candidate.left_city) == clean_text(candidate.right_city))
    phone_name_candidate = exact_phone or (same_area and name_sim >= 0.40)
    name_or_cuisine_or_char = name_sim > 0 or cuisine_sim >= 0.50 or name_chars >= 0.60
    address_or_exact_name_or_strong_name = address_sim >= 0.30 or (left_name and left_name == right_name) or name_sim >= 0.75

    checks = {
        "phone_exact": exact_phone,
        "phone_area": same_area,
        "name_sim_ge_045": name_sim >= 0.45,
        "name_sim_ge_060": name_sim >= 0.60,
        "name_sim_ge_075": name_sim >= 0.75,
        "address_sim_ge_030": address_sim >= 0.30,
        "address_sim_ge_050": address_sim >= 0.50,
        "cuisine_sim_ge_050": cuisine_sim >= 0.50,
        "same_city": same_city,
        "name045_address030": name_sim >= 0.45 and address_sim >= 0.30,
        "name045_phone_area": name_sim >= 0.45 and same_area,
        "phone_exact_or_name040_phone_area": exact_phone or (same_area and name_sim >= 0.40),
        "phone_exact_or_name035_phone_area": exact_phone or (same_area and name_sim >= 0.35),
        "address050_phone_area": address_sim >= 0.50 and same_area,
        "name_or_cuisine_or_char060": name_or_cuisine_or_char,
        "address030_or_exact_name_or_name075": address_or_exact_name_or_strong_name,
        "phone_name_candidate_consistent": phone_name_candidate and name_or_cuisine_or_char and address_or_exact_name_or_strong_name,
    }
    return [rule for rule, passed in checks.items() if passed]


def fetch_all_rule_candidates() -> list[RuleCandidate]:
    query = """
    MATCH (r1:Restaurant)-[:LOCATED_AT]->(a1:Address)-[:IN_CITY]->(c1:City)
    MATCH (r2:Restaurant)-[:LOCATED_AT]->(a2:Address)-[:IN_CITY]->(c2:City)
    WHERE toInteger(r1.id) < 533 AND toInteger(r2.id) >= 533
    RETURN r1, r2, a1, a2, c1, c2
    """
    candidates: dict[tuple[int, int], RuleCandidate] = {}
    driver = get_driver()
    with driver.session() as session:
        for record in session.run(query):
            r1 = node_dict(record["r1"])
            r2 = node_dict(record["r2"])
            a1 = node_dict(record["a1"])
            a2 = node_dict(record["a2"])
            c1 = node_dict(record["c1"])
            c2 = node_dict(record["c2"])
            candidate = RuleCandidate(
                left_id=int(r1["id"]),
                right_id=int(r2["id"]),
                left=r1,
                right=r2,
                left_address=a1.get("value", ""),
                right_address=a2.get("value", ""),
                left_city=c1.get("value", ""),
                right_city=c2.get("value", ""),
                rules=[],
            )
            candidate.rules = triggered_rules(candidate)
            if candidate.rules:
                if candidate.pair in candidates:
                    merged = sorted(set(candidates[candidate.pair].rules) | set(candidate.rules))
                    candidates[candidate.pair].rules = merged
                else:
                    candidates[candidate.pair] = candidate
    driver.close()
    return list(candidates.values())


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities.values() if value > 0)


def candidate_probability(candidate: RuleCandidate, probabilities: dict[str, float]) -> float:
    if not candidate.rules:
        return 0.0
    return sum(probabilities[rule] for rule in candidate.rules) / len(candidate.rules)


def clamp_probability(value: float) -> float:
    return max(0.001, min(0.999, value))


def bayes_update(probabilities: dict[str, float], candidate: RuleCandidate, match: bool, confidence: float) -> dict[str, float]:
    theta = max(0.05, min(0.95, confidence or 0.8))
    p_match = clamp_probability(candidate_probability(candidate, probabilities))
    candidate_rules = set(candidate.rules)
    updated: dict[str, float] = {}
    for rule, old_probability in probabilities.items():
        prior = clamp_probability(old_probability)
        contains = rule in candidate_rules
        if match and contains:
            numerator = prior * theta
            denominator = p_match * theta + (1 - p_match) * (1 - theta)
        elif (not match) and contains:
            numerator = prior * (1 - theta)
            denominator = p_match * (1 - theta) + (1 - p_match) * theta
        elif match and not contains:
            numerator = prior * (1 - theta)
            denominator = (1 - p_match) * (1 - theta) + p_match * theta
        else:
            numerator = prior * theta
            denominator = (1 - p_match) * theta + p_match * (1 - theta)
        updated[rule] = clamp_probability(numerator / denominator if denominator else prior)
    return updated


def restaurant_text(candidate: RuleCandidate, side: str) -> str:
    node = candidate.left if side == "left" else candidate.right
    address = candidate.left_address if side == "left" else candidate.right_address
    city = candidate.left_city if side == "left" else candidate.right_city
    return f"name={node.get('name', '')}; address={address}; city={city}; phone={node.get('phone', '')}; cuisine={node.get('cuisine', '')}"


def graph_triples(candidate: RuleCandidate) -> list[str]:
    left_restaurant = f"Restaurant:{candidate.left_id}"
    right_restaurant = f"Restaurant:{candidate.right_id}"
    left_address = f"Address:{candidate.left_address}"
    right_address = f"Address:{candidate.right_address}"
    left_city = f"City:{candidate.left_city}"
    right_city = f"City:{candidate.right_city}"
    return [
        f"({left_restaurant}, LOCATED_AT, {left_address})",
        f"({left_address}, IN_CITY, {left_city})",
        f"({right_restaurant}, LOCATED_AT, {right_address})",
        f"({right_address}, IN_CITY, {right_city})",
    ]


def rule_feedback_prompt(candidate: RuleCandidate) -> str:
    payload = {
        "task": "FZ restaurant ER; return JSON match feedback for Bayesian GDD rule update.",
        "rules": candidate.rules,
        "graph": graph_triples(candidate),
        "r1": restaurant_text(candidate, "left"),
        "r2": restaurant_text(candidate, "right"),
        "out": {"match": "bool", "confidence": "0..1", "reason": "short"},
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def match_prompt(candidate: RuleCandidate, mode: str) -> str:
    examples = ""
    if mode in {"few-shot", "self-consistency"}:
        examples = "\nEx:" + ";".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in EXAMPLES)
    payload = {
        "rules": candidate.rules,
        "graph": graph_triples(candidate),
        "r1": restaurant_text(candidate, "left"),
        "r2": restaurant_text(candidate, "right"),
    }
    return f"""
Fodors-Zagats restaurant ER. Decide if r1 and r2 are the same real-world restaurant.
Use attributes plus GDD rules and graph triples as evidence; make the final semantic decision from the records.
Return JSON only: {{"match":true/false,"confidence":0.0-1.0,"reason":"..."}}
{examples}

{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()


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


def optimize_rules(
    candidates: list[RuleCandidate],
    output_dir: Path,
    max_calls: int,
    use_cache: bool,
    top_rules: int,
    max_rule_support: int,
    min_llm_calls: int,
    early_stop_rounds: int,
) -> tuple[dict[str, float], list[dict]]:
    probabilities = {rule: 0.5 for rule in RULE_DESCRIPTIONS}
    support = support_counts(candidates)
    matcher = LLMMatcher(llm_config())
    cache_path = output_dir / "fz_threshold_stage1_cache.jsonl"
    cache = load_jsonl(cache_path) if use_cache else {}
    lock = threading.Lock()
    asked: set[tuple[int, int]] = set()
    history: list[dict] = []
    attempts = 0
    stable_rules: list[str] = []
    stable_count = 0
    while len(history) < max_calls and attempts < max_calls * 5:
        attempts += 1
        for candidate in candidates:
            candidate.rule_probability = candidate_probability(candidate, probabilities)
        pending = [candidate for candidate in candidates if candidate.pair not in asked]
        if not pending:
            break
        selected = max(
            pending,
            key=lambda item: (
                1 / (1 + abs(item.rule_probability - 0.5)),
                len(item.rules),
                item.rule_probability,
            ),
        )
        key = f"fz-threshold-stage1-{PROMPT_VERSION}:{selected.left_id}:{selected.right_id}:{','.join(selected.rules)}"
        row = cache.get(key)
        if row is None:
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
        if is_failed(row):
            print(f"Stage1 skip failed {selected.left_id}-{selected.right_id}: {row.get('reason', '')[:80]}")
            continue
        append_jsonl(cache_path, row, lock)
        probabilities = bayes_update(probabilities, selected, bool(row["match"]), float(row.get("confidence", 0.8)))
        current_rules = [rule for rule in ranked_rules(probabilities, support) if support.get(rule, 0) <= max_rule_support][:top_rules]
        if current_rules == stable_rules:
            stable_count += 1
        else:
            stable_rules = current_rules
            stable_count = 1
        history.append(
            {
                "iteration": len(history) + 1,
                "left_id": selected.left_id,
                "right_id": selected.right_id,
                "rules": selected.rules,
                "match": row["match"],
                "confidence": row.get("confidence", 0.0),
                "reason": row.get("reason", ""),
                "rule_probabilities": probabilities,
                "current_selected_rules": current_rules,
                "rule_entropy": entropy(probabilities),
            }
        )
        print(f"Stage1 {len(history)}/{max_calls} entropy={entropy(probabilities):.4f}")
        if early_stop_rounds and len(history) >= min_llm_calls and stable_count >= early_stop_rounds:
            print(f"Stage1 early stop: selected rules stable for {stable_count} rounds")
            break
    return probabilities, history


def support_counts(candidates: list[RuleCandidate]) -> dict[str, int]:
    support = {rule: 0 for rule in RULE_DESCRIPTIONS}
    for candidate in candidates:
        for rule in candidate.rules:
            support[rule] += 1
    return support


def ranked_rules(probabilities: dict[str, float], support: dict[str, int]) -> list[str]:
    return [
        rule
        for rule, _value in sorted(
            probabilities.items(),
            key=lambda item: (item[1], support.get(item[0], 0)),
            reverse=True,
        )
    ]


def select_by_rules(
    candidates: list[RuleCandidate],
    probabilities: dict[str, float],
    top_rules: int,
    max_rule_support: int,
    rule_operator: str,
) -> tuple[list[str], list[RuleCandidate]]:
    support = support_counts(candidates)
    ranked = ranked_rules(probabilities, support)
    selected_rules = [rule for rule in ranked if support.get(rule, 0) <= max_rule_support][:top_rules]
    if len(selected_rules) < top_rules:
        selected_rules = ranked[:top_rules]
    if rule_operator == "union":
        filtered = [candidate for candidate in candidates if any(rule in candidate.rules for rule in selected_rules)]
    else:
        filtered = [candidate for candidate in candidates if all(rule in candidate.rules for rule in selected_rules)]
    filtered.sort(key=lambda item: (candidate_probability(item, probabilities), len(item.rules)), reverse=True)
    return selected_rules, filtered


def judge_candidate(candidate: RuleCandidate, mode: str, votes: int) -> dict:
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


def stage2_key(mode: str, candidate: RuleCandidate, votes: int, selected_rules: list[str]) -> str:
    model = llm_config().model
    return (
        f"fz-threshold-stage2-{PROMPT_VERSION}:{model}:{mode}:"
        f"{candidate.left_id}:{candidate.right_id}:{votes}:{','.join(selected_rules)}"
    )


def load_pair_cache(path: Path) -> dict[tuple[int, int], dict]:
    by_pair: dict[tuple[int, int], dict] = {}
    for row in load_jsonl(path).values():
        if is_failed(row):
            continue
        pair = tuple(sorted((int(row["left_id"]), int(row["right_id"]))))
        by_pair[pair] = row
    return by_pair


def decision_from_row(row: dict) -> object:
    class CachedDecision:
        def __init__(self, source: dict):
            self.match = bool(source["match"])
            self.confidence = float(source.get("confidence", 0.5))
            self.reason = str(source.get("reason", ""))
            self.raw = str(source.get("raw", ""))

    return CachedDecision(row)


def judge_candidate_with_seed(
    candidate: RuleCandidate,
    mode: str,
    votes: int,
    seed_row: dict | None = None,
) -> dict:
    matcher = LLMMatcher(llm_config())
    decisions = []
    if seed_row is not None:
        decisions.append(decision_from_row(seed_row))
    remaining_votes = max(0, votes - len(decisions))
    decisions.extend(matcher.decide(match_prompt(candidate, mode)) for _ in range(remaining_votes))
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
        "seeded_votes": 1 if seed_row is not None else 0,
    }


def stratified_sample_candidates(
    candidates: list[RuleCandidate],
    truth: set[tuple[int, int]],
    sample_size: int,
    seed: int,
) -> tuple[list[RuleCandidate], dict]:
    stats = {
        "sampling_enabled": False,
        "sample_size": sample_size,
        "sample_seed": seed,
        "candidates_before_sampling": len(candidates),
        "positive_before_sampling": sum(1 for candidate in candidates if candidate.pair in truth),
        "negative_before_sampling": sum(1 for candidate in candidates if candidate.pair not in truth),
    }
    if sample_size <= 0 or len(candidates) <= sample_size:
        stats.update(
            {
                "candidates_after_sampling": len(candidates),
                "positive_after_sampling": stats["positive_before_sampling"],
                "negative_after_sampling": stats["negative_before_sampling"],
            }
        )
        return candidates, stats

    rng = random.Random(seed)
    positives = [candidate for candidate in candidates if candidate.pair in truth]
    negatives = [candidate for candidate in candidates if candidate.pair not in truth]
    target_positive = round(sample_size * len(positives) / len(candidates))
    if positives:
        target_positive = max(1, min(len(positives), target_positive))
    target_negative = sample_size - target_positive
    if target_negative > len(negatives):
        target_negative = len(negatives)
        target_positive = min(len(positives), sample_size - target_negative)

    positive_sample = set(rng.sample([candidate.pair for candidate in positives], target_positive))
    negative_sample = set(rng.sample([candidate.pair for candidate in negatives], target_negative))
    sampled_pairs = positive_sample | negative_sample
    sampled = [candidate for candidate in candidates if candidate.pair in sampled_pairs]
    stats.update(
        {
            "sampling_enabled": True,
            "candidates_after_sampling": len(sampled),
            "positive_after_sampling": sum(1 for candidate in sampled if candidate.pair in truth),
            "negative_after_sampling": sum(1 for candidate in sampled if candidate.pair not in truth),
        }
    )
    return sampled, stats


def one_to_one_predictions(accepted: list[tuple[float, tuple[int, int]]]) -> set[tuple[int, int]]:
    used_left: set[int] = set()
    used_right: set[int] = set()
    predictions: set[tuple[int, int]] = set()
    for _confidence, pair in sorted(accepted, reverse=True):
        if pair[0] in used_left or pair[1] in used_right:
            continue
        predictions.add(pair)
        used_left.add(pair[0])
        used_right.add(pair[1])
    return predictions


def build_pps_blocks(candidates: list[RuleCandidate]) -> dict[int, list[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for candidate in candidates:
        neighbors[candidate.left_id].add(candidate.right_id)
        neighbors[candidate.right_id].add(candidate.left_id)
    return {entity: [entity, *sorted(related)] for entity, related in neighbors.items()}


def build_pps_profile_index(blocks: dict[int, list[int]]) -> dict[int, list[int]]:
    profile_index: dict[int, list[int]] = defaultdict(list)
    for block_id, profiles in blocks.items():
        for profile in profiles:
            profile_index[profile].append(block_id)
    return dict(profile_index)


def pps_weight(block_id: int, blocks: dict[int, list[int]]) -> float:
    block_size = len(blocks.get(block_id, []))
    return 1 / block_size if block_size > 0 else 0.0


def initialize_pps(
    blocks: dict[int, list[int]],
    profile_index: dict[int, list[int]],
) -> tuple[list[tuple[int, int, float]], list[tuple[int, float]]]:
    comparison_list: list[tuple[int, int, float]] = []
    top_comparisons: set[tuple[int, int, float]] = set()
    sorted_profile_list: list[tuple[int, float]] = []

    for profile in sorted(blocks):
        weights: dict[int, float] = {}
        distinct_neighbors: set[int] = set()
        for block_id in profile_index.get(profile, []):
            for neighbor in blocks.get(block_id, []):
                if neighbor == profile:
                    continue
                weights[neighbor] = weights.get(neighbor, 0.0) + pps_weight(block_id, blocks)
                distinct_neighbors.add(neighbor)

        top_comparison: tuple[int, int, float] | None = None
        duplication_likelihood = 0.0
        for neighbor in distinct_neighbors:
            duplication_likelihood += weights[neighbor]
            comparison = (profile, neighbor, weights[neighbor])
            if top_comparison is None or comparison[2] > top_comparison[2]:
                top_comparison = comparison

        if top_comparison:
            top_comparisons.add(top_comparison)
        duplication_likelihood /= max(1, len(distinct_neighbors))
        sorted_profile_list.append((profile, duplication_likelihood))

    comparison_list = list(top_comparisons)
    comparison_list.sort(key=lambda item: item[2], reverse=True)
    sorted_profile_list.sort(key=lambda item: item[1], reverse=True)
    return comparison_list, sorted_profile_list


def pps_order_candidates(
    candidates: list[RuleCandidate],
    min_weight: float,
    max_comparisons: int,
) -> tuple[list[RuleCandidate], dict]:
    if not candidates:
        return [], {
            "pps_enabled": True,
            "pps_input_candidates": 0,
            "pps_ordered_candidates": 0,
            "pps_emitted_candidates": 0,
            "pps_appended_candidates": 0,
            "pps_min_weight": min_weight,
            "pps_max_comparisons": max_comparisons,
        }

    candidate_by_pair = {candidate.pair: candidate for candidate in candidates}
    allowed_pairs = set(candidate_by_pair)
    blocks = build_pps_blocks(candidates)
    profile_index = build_pps_profile_index(blocks)
    comparison_list, sorted_profile_list = initialize_pps(blocks, profile_index)
    checked_profiles: set[int] = set()
    emitted_pairs: set[tuple[int, int]] = set()
    ordered: list[RuleCandidate] = []

    def pop_valid_comparison() -> tuple[int, int, float] | None:
        while comparison_list:
            comparison = comparison_list.pop(0)
            if comparison[2] < min_weight:
                continue
            pair = tuple(sorted((comparison[0], comparison[1])))
            if pair in allowed_pairs and pair not in emitted_pairs:
                return comparison
        return None

    while len(ordered) < len(candidates) and len(ordered) < max_comparisons:
        comparison = pop_valid_comparison()
        if comparison is None:
            while sorted_profile_list and comparison is None:
                profile, _likelihood = sorted_profile_list.pop(0)
                checked_profiles.add(profile)
                weights: dict[int, float] = {}
                distinct_neighbors: set[int] = set()
                for block_id in profile_index.get(profile, []):
                    for neighbor in blocks.get(block_id, []):
                        if neighbor != profile and neighbor not in checked_profiles:
                            weights[neighbor] = weights.get(neighbor, 0.0) + pps_weight(block_id, blocks)
                            distinct_neighbors.add(neighbor)
                for neighbor in distinct_neighbors:
                    comparison_list.append((profile, neighbor, weights[neighbor]))
                comparison_list.sort(key=lambda item: item[2], reverse=True)
                comparison = pop_valid_comparison()
            if comparison is None:
                break

        pair = tuple(sorted((comparison[0], comparison[1])))
        emitted_pairs.add(pair)
        ordered.append(candidate_by_pair[pair])

    missing = [candidate for candidate in candidates if candidate.pair not in emitted_pairs]
    ordered.extend(missing)
    stats = {
        "pps_enabled": True,
        "pps_input_candidates": len(candidates),
        "pps_ordered_candidates": len(ordered),
        "pps_emitted_candidates": len(emitted_pairs),
        "pps_appended_candidates": len(missing),
        "pps_blocks": len(blocks),
        "pps_min_weight": min_weight,
        "pps_max_comparisons": max_comparisons,
    }
    return ordered, stats


def run(
    mode: str,
    output_dir: Path,
    max_calls: int,
    top_rules: int,
    max_rule_support: int,
    rule_operator: str,
    use_pps: bool,
    pps_min_weight: float,
    pps_max_comparisons: int,
    workers: int,
    use_cache: bool,
    one_to_one: bool,
    min_llm_calls: int,
    early_stop_rounds: int,
    sample_size: int,
    sample_seed: int,
    use_sampling: bool,
) -> dict:
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = fetch_all_rule_candidates()
    probabilities, history = optimize_rules(
        candidates,
        output_dir,
        max_calls,
        use_cache,
        top_rules,
        max_rule_support,
        min_llm_calls,
        early_stop_rounds,
    )
    selected_rules, filtered = select_by_rules(candidates, probabilities, top_rules, max_rule_support, rule_operator)
    if use_pps:
        filtered, pps_stats = pps_order_candidates(filtered, pps_min_weight, pps_max_comparisons)
    else:
        pps_stats = {
            "pps_enabled": False,
            "pps_input_candidates": len(filtered),
            "pps_ordered_candidates": len(filtered),
            "pps_emitted_candidates": 0,
            "pps_appended_candidates": 0,
            "pps_min_weight": pps_min_weight,
            "pps_max_comparisons": pps_max_comparisons,
        }
    truth = load_truth(Path("dataset/relational-dataset/fodors-zagats/matches.csv"), table_b_offset=533)
    if use_sampling:
        filtered, sampling_stats = stratified_sample_candidates(filtered, truth, sample_size, sample_seed)
    else:
        sampling_stats = {
            "sampling_enabled": False,
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "candidates_before_sampling": len(filtered),
            "candidates_after_sampling": len(filtered),
            "positive_before_sampling": sum(1 for candidate in filtered if candidate.pair in truth),
            "negative_before_sampling": sum(1 for candidate in filtered if candidate.pair not in truth),
            "positive_after_sampling": sum(1 for candidate in filtered if candidate.pair in truth),
            "negative_after_sampling": sum(1 for candidate in filtered if candidate.pair not in truth),
        }
    sampled_pairs = {candidate.pair for candidate in filtered}
    evaluation_truth = truth & sampled_pairs if sampling_stats["sampling_enabled"] else truth
    votes = 3 if mode == "self-consistency" else 1
    cache_path = output_dir / f"fz_threshold_{mode}_stage2_cache.jsonl"
    cache = load_jsonl(cache_path) if use_cache else {}
    fewshot_cache = (
        load_pair_cache(output_dir / "fz_threshold_few-shot_stage2_cache.jsonl")
        if use_cache and mode == "self-consistency"
        else {}
    )
    lock = threading.Lock()

    def run_one(candidate: RuleCandidate) -> dict:
        key = stage2_key(mode, candidate, votes, selected_rules)
        if key in cache:
            return cache[key]
        seed_row = fewshot_cache.get(candidate.pair)
        row = {"key": key, **judge_candidate_with_seed(candidate, mode, votes, seed_row)}
        if seed_row is not None:
            row["cache_source"] = "few-shot-stage2-as-first-self-consistency-vote"
        append_jsonl(cache_path, row, lock)
        return row

    accepted: list[tuple[float, tuple[int, int]]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, candidate) for candidate in filtered]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            if row["match"]:
                accepted.append((float(row.get("confidence", 0.0)), tuple(sorted((int(row["left_id"]), int(row["right_id"]))))))
            predictions = one_to_one_predictions(accepted) if one_to_one else {pair for _confidence, pair in accepted}
            if index % 20 == 0 or index == len(futures):
                metrics = evaluate(predictions, evaluation_truth)
                print(f"Stage2 {index}/{len(futures)} p={metrics['precision']:.4f} r={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

    predictions = one_to_one_predictions(accepted) if one_to_one else {pair for _confidence, pair in accepted}
    metrics = evaluate(predictions, evaluation_truth)
    result = {
        **metrics,
        "dataset": "FZ",
        "mode": mode,
        "llm_rule_feedback_calls": len(history),
        "selected_rules": selected_rules,
        "rule_probabilities": probabilities,
        "rule_entropy": entropy(probabilities),
        "candidate_pairs_after_rule_intersection": len(filtered),
        **sampling_stats,
        "max_rule_support": max_rule_support,
        "rule_operator": rule_operator,
        **pps_stats,
        "llm_candidate_calls": len(filtered) * votes,
        "llm_candidate_calls_saved_by_fewshot_seed": (
            sum(1 for candidate in filtered if candidate.pair in fewshot_cache)
            if mode == "self-consistency"
            else 0
        ),
        "llm_candidate_api_calls_estimated": len(filtered) * votes
        - (
            sum(1 for candidate in filtered if candidate.pair in fewshot_cache)
            if mode == "self-consistency"
            else 0
        ),
        "votes": votes,
        "one_to_one": one_to_one,
        "min_llm_calls": min_llm_calls,
        "early_stop_rounds": early_stop_rounds,
        "predicted_matches": len(predictions),
        "f1_percent": metrics["f1"] * 100,
        "runtime_seconds": time.time() - start,
    }
    (output_dir / f"fz_threshold_{mode}_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"fz_threshold_{mode}_rule_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / f"fz_threshold_{mode}_candidates.jsonl").open("w", encoding="utf-8") as file:
        for candidate in filtered:
            file.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")
    with (output_dir / f"fz_threshold_{mode}_pps_order.jsonl").open("w", encoding="utf-8") as file:
        for index, candidate in enumerate(filtered, start=1):
            file.write(
                json.dumps(
                    {
                        "rank": index,
                        "left_id": candidate.left_id,
                        "right_id": candidate.right_id,
                        "rules": candidate.rules,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="FZ GAPLink with threshold rules, Bayesian rule selection, and rule-intersection filtering.")
    parser.add_argument("--mode", choices=["zero-shot", "few-shot", "self-consistency"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output_file"))
    parser.add_argument("--max-llm-calls", type=int, default=20)
    parser.add_argument("--top-rules", type=int, default=1)
    parser.add_argument("--max-rule-support", type=int, default=114)
    parser.add_argument("--rule-operator", choices=["intersection", "union"], default="intersection")
    parser.add_argument("--disable-pps", action="store_true")
    parser.add_argument("--pps-min-weight", type=float, default=-1.0)
    parser.add_argument("--pps-max-comparisons", type=int, default=1_000_000_000)
    parser.add_argument("--min-llm-calls", type=int, default=20)
    parser.add_argument("--early-stop-rounds", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--disable-sampling", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--one-to-one", action="store_true")
    args = parser.parse_args()
    run(
        args.mode,
        args.output_dir,
        args.max_llm_calls,
        args.top_rules,
        args.max_rule_support,
        args.rule_operator,
        not args.disable_pps,
        args.pps_min_weight,
        args.pps_max_comparisons,
        args.workers,
        not args.no_cache,
        args.one_to_one,
        args.min_llm_calls,
        args.early_stop_rounds,
        args.sample_size,
        args.sample_seed,
        not args.disable_sampling,
    )


if __name__ == "__main__":
    main()
