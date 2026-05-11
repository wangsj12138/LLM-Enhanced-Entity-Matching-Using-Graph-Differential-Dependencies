"""Local, dependency-light GAPLink runner for the bundled Fodors-Zagats data.

The original scripts expect a Neo4j instance. This module keeps the same
rule-filtering spirit but runs directly from CSV files so the repository has a
working demo without external services.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from er_utils import clean_text, evaluate, jaccard, load_truth, phone_digits, tokens


@dataclass(frozen=True)
class Restaurant:
    source: str
    local_id: int
    entity_id: int
    name: str
    address: str
    city: str
    phone: str
    cuisine: str


@dataclass(frozen=True)
class CandidatePair:
    left: Restaurant
    right: Restaurant
    rules: tuple[str, ...]
    score: float

    @property
    def ids(self) -> tuple[int, int]:
        return tuple(sorted((self.left.entity_id, self.right.entity_id)))


def load_restaurants(path: Path, source: str, id_offset: int = 0) -> list[Restaurant]:
    restaurants: list[Restaurant] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            restaurants.append(
                Restaurant(
                    source=source,
                    local_id=int(row["id"]),
                    entity_id=int(row["id"]) + id_offset,
                    name=row.get("name", ""),
                    address=row.get("addr", ""),
                    city=row.get("city", ""),
                    phone=row.get("phone", ""),
                    cuisine=row.get("type", ""),
                )
            )
    return restaurants


def build_rule_blocks(left: list[Restaurant], right: list[Restaurant]) -> dict[str, dict[str, list[Restaurant]]]:
    blocks: dict[str, dict[str, list[Restaurant]]] = {
        "same_city": defaultdict(list),
        "phone_area": defaultdict(list),
        "name_token": defaultdict(list),
    }

    for restaurant in itertools.chain(left, right):
        city = clean_text(restaurant.city)
        if city:
            blocks["same_city"][city].append(restaurant)

        digits = phone_digits(restaurant.phone)
        if len(digits) >= 3:
            blocks["phone_area"][digits[:3]].append(restaurant)

        for token in tokens(restaurant.name):
            if len(token) > 2:
                blocks["name_token"][token].append(restaurant)

    return blocks


def pair_score(left: Restaurant, right: Restaurant) -> float:
    name = jaccard(left.name, right.name)
    address = jaccard(left.address, right.address)
    cuisine = jaccard(left.cuisine, right.cuisine)
    city = 1.0 if clean_text(left.city) == clean_text(right.city) and clean_text(left.city) else 0.0
    phone = 1.0 if phone_digits(left.phone) and phone_digits(left.phone) == phone_digits(right.phone) else 0.0
    return 0.45 * name + 0.2 * address + 0.15 * cuisine + 0.1 * city + 0.1 * phone


def generate_candidates(left: list[Restaurant], right: list[Restaurant]) -> list[CandidatePair]:
    right_ids = {restaurant.entity_id for restaurant in right}
    raw_pairs: dict[tuple[int, int], set[str]] = defaultdict(set)

    for rule_name, block_map in build_rule_blocks(left, right).items():
        for members in block_map.values():
            left_members = [item for item in members if item.source == "A"]
            right_members = [item for item in members if item.source == "B"]
            for left_item, right_item in itertools.product(left_members, right_members):
                if right_item.entity_id in right_ids:
                    raw_pairs[(left_item.entity_id, right_item.entity_id)].add(rule_name)

    by_id = {item.entity_id: item for item in itertools.chain(left, right)}
    candidates = [
        CandidatePair(
            left=by_id[left_id],
            right=by_id[right_id],
            rules=tuple(sorted(rules)),
            score=pair_score(by_id[left_id], by_id[right_id]),
        )
        for (left_id, right_id), rules in raw_pairs.items()
    ]
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def select_matches(candidates: list[CandidatePair], threshold: float) -> list[CandidatePair]:
    return [candidate for candidate in candidates if candidate.score >= threshold]


def rule_entropy(candidates: list[CandidatePair]) -> float:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        counts.update(candidate.rules)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def write_outputs(output_dir: Path, candidates: list[CandidatePair], matches: list[CandidatePair], metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "query_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["left_id", "right_id", "score", "rules", "left_name", "right_name"])
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.left.entity_id,
                    candidate.right.entity_id,
                    f"{candidate.score:.4f}",
                    "|".join(candidate.rules),
                    candidate.left.name,
                    candidate.right.name,
                ]
            )

    with (output_dir / "filtered_pattern.txt").open("w", encoding="utf-8") as file:
        for candidate in candidates:
            file.write(
                f"Restaurant 1: (ID: {candidate.left.entity_id}, Name: {candidate.left.name}, "
                f"Cuisine: {candidate.left.cuisine}, Phone: {candidate.left.phone})\n"
            )
            file.write(
                f"Restaurant 2: (ID: {candidate.right.entity_id}, Name: {candidate.right.name}, "
                f"Cuisine: {candidate.right.cuisine}, Phone: {candidate.right.phone})\n"
            )
            file.write(f"Value Similarity: {jaccard(candidate.left.name, candidate.right.name):.2f}\n")
            file.write(f"Rules: {', '.join(candidate.rules)}\n")
            file.write("--------------------------------------------\n")

    neighbors: dict[int, set[int]] = defaultdict(set)
    for candidate in candidates:
        neighbors[candidate.left.entity_id].add(candidate.right.entity_id)
        neighbors[candidate.right.entity_id].add(candidate.left.entity_id)
    with (output_dir / "blocks.txt").open("w", encoding="utf-8") as file:
        for entity_id in sorted(neighbors):
            file.write(f"target entity: {entity_id}\n")
            file.write(f"similar entities: {' '.join(map(str, sorted(neighbors[entity_id])))}\n\n")

    with (output_dir / "results.txt").open("w", encoding="utf-8") as file:
        for match in matches:
            file.write(f"{match.left.entity_id}|{match.right.entity_id}\n")

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value:.6f}\n")


def run(
    dataset_dir: Path = Path("dataset/fodors_zagats/relational-dataset/fodors-zagats"),
    output_dir: Path = Path("output_file"),
    threshold: float = 0.48,
) -> dict[str, float]:
    start = time.time()
    left = load_restaurants(dataset_dir / "tableA.csv", "A")
    right = load_restaurants(dataset_dir / "tableB.csv", "B", id_offset=len(left))
    truth = load_truth(dataset_dir / "matches.csv", table_b_offset=len(left))
    candidates = generate_candidates(left, right)
    matches = select_matches(candidates, threshold)
    predicted = {match.ids for match in matches}
    metrics = evaluate(predicted, truth)
    metrics["candidate_pairs"] = float(len(candidates))
    metrics["predicted_matches"] = float(len(matches))
    metrics["rule_entropy"] = rule_entropy(candidates)
    metrics["runtime_seconds"] = time.time() - start
    write_outputs(output_dir, candidates, matches, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAPLink locally on the bundled Fodors-Zagats CSV data.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/fodors_zagats/relational-dataset/fodors-zagats"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_file"))
    parser.add_argument("--threshold", type=float, default=0.48)
    args = parser.parse_args()

    metrics = run(args.dataset_dir, args.output_dir, args.threshold)
    print("GAPLink local run complete")
    for key, value in metrics.items():
        if key.endswith("seconds"):
            print(f"{key}: {value:.4f}")
        elif key in {"candidate_pairs", "predicted_matches", "true_positive", "false_positive", "false_negative"}:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
