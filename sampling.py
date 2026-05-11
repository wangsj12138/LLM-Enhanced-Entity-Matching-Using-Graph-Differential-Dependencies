from __future__ import annotations

import random

from rule_candidates import RuleCandidate


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
