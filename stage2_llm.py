from __future__ import annotations

from config import llm_config
from dataset_config import DatasetConfig, PROMPT_VERSION
from llm_matcher import LLMMatcher
from prompts import match_prompt
from rule_candidates import RuleCandidate
from cache_utils import is_failed, load_jsonl


def judge_candidate(candidate: RuleCandidate, mode: str, votes: int, config: DatasetConfig) -> dict:
    matcher = LLMMatcher(llm_config())
    decisions = [matcher.decide(match_prompt(candidate, mode, config)) for _ in range(votes)]
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


def stage2_key(mode: str, candidate: RuleCandidate, votes: int, selected_rules: list[str], config: DatasetConfig) -> str:
    model = llm_config().model
    prompt_version = getattr(config, "prompt_version", PROMPT_VERSION)
    return (
        f"{config.cache_prefix}-stage2-{prompt_version}:{model}:{mode}:"
        f"{candidate.left_id}:{candidate.right_id}:{votes}:{','.join(selected_rules)}"
    )


def load_pair_cache(path) -> dict[tuple[int, int], dict]:
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
    config: DatasetConfig,
    seed_row: dict | None = None,
) -> dict:
    matcher = LLMMatcher(llm_config())
    decisions = []
    if seed_row is not None:
        decisions.append(decision_from_row(seed_row))
    remaining_votes = max(0, votes - len(decisions))
    decisions.extend(matcher.decide(match_prompt(candidate, mode, config)) for _ in range(remaining_votes))
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
