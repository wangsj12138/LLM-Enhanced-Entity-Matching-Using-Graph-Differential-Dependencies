from __future__ import annotations

import math
import threading
from pathlib import Path

from cache_utils import append_jsonl, is_failed, load_jsonl
from config import llm_config
from dataset_config import DatasetConfig, PROMPT_VERSION
from llm_matcher import LLMMatcher
from prompts import rule_feedback_prompt
from rule_candidates import RuleCandidate, rule_descriptions


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


def support_counts(candidates: list[RuleCandidate], config: DatasetConfig) -> dict[str, int]:
    support = {rule: 0 for rule in rule_descriptions(config)}
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


def allowed_rule_names(config: DatasetConfig | None = None) -> set[str]:
    allowed = config.stage1_allowed_rules if config is not None else ()
    return set(allowed)


def select_by_rules(
    candidates: list[RuleCandidate],
    probabilities: dict[str, float],
    top_rules: int,
    max_rule_support: int,
    rule_operator: str,
    config: DatasetConfig | None = None,
) -> tuple[list[str], list[RuleCandidate]]:
    if config is None:
        raise ValueError("Dataset config is required for rule selection.")
    support = support_counts(candidates, config)
    allowed = allowed_rule_names(config)
    ranked = [rule for rule in ranked_rules(probabilities, support) if not allowed or rule in allowed]
    selected_rules = [rule for rule in ranked if support.get(rule, 0) <= max_rule_support][:top_rules]
    if len(selected_rules) < top_rules:
        selected_rules = ranked[:top_rules]
    if rule_operator == "union":
        filtered = [candidate for candidate in candidates if any(rule in candidate.rules for rule in selected_rules)]
    else:
        filtered = [candidate for candidate in candidates if all(rule in candidate.rules for rule in selected_rules)]
    filtered.sort(key=lambda item: (candidate_probability(item, probabilities), len(item.rules)), reverse=True)
    return selected_rules, filtered


def optimize_rules(
    candidates: list[RuleCandidate],
    config: DatasetConfig,
    output_dir: Path,
    max_calls: int,
    use_cache: bool,
    top_rules: int,
    max_rule_support: int,
    min_llm_calls: int,
    early_stop_rounds: int,
) -> tuple[dict[str, float], list[dict]]:
    probabilities = {rule: 0.5 for rule in rule_descriptions(config)}
    support = support_counts(candidates, config)
    matcher = LLMMatcher(llm_config())
    cache_path = output_dir / f"{config.cache_prefix}_stage1_cache.jsonl"
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
        prompt_version = getattr(config, "prompt_version", PROMPT_VERSION)
        key = f"{config.cache_prefix}-stage1-{prompt_version}:{selected.left_id}:{selected.right_id}:{','.join(selected.rules)}"
        row = cache.get(key)
        if row is None:
            decision = matcher.decide(rule_feedback_prompt(selected, config))
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
        allowed = allowed_rule_names(config)
        current_rules = [
            rule
            for rule in ranked_rules(probabilities, support)
            if (not allowed or rule in allowed) and support.get(rule, 0) <= max_rule_support
        ][:top_rules]
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
