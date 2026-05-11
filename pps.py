from __future__ import annotations

from collections import defaultdict

from rule_candidates import RuleCandidate


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
