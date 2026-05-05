from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gaplink_local import clean_text, evaluate, jaccard, load_truth, phone_digits, tokens
from llm_matcher import LLMMatcher
from neo4j_setup import get_driver


@dataclass
class GraphCandidate:
    left_id: int
    right_id: int
    left: dict
    right: dict
    left_address: str
    right_address: str
    city: str
    rules: list[str]
    structural_score: float = 0.0
    rule_probability: float = 0.0
    llm_decision: bool | None = None
    llm_confidence: float = 0.0
    llm_reason: str = ""

    @property
    def pair(self) -> tuple[int, int]:
        return tuple(sorted((self.left_id, self.right_id)))


GDD_QUERIES = {
    "name_token": """
        MATCH (r1:Restaurant)-[:LOCATED_AT]->(a1:Address)-[:IN_CITY]->(c1:City)
        MATCH (r2:Restaurant)-[:LOCATED_AT]->(a2:Address)-[:IN_CITY]->(c2:City)
        WHERE toInteger(r1.id) < 533 AND toInteger(r2.id) >= 533
          AND any(w IN split(toLower(r1.name), ' ') WHERE size(w) > 2 AND w IN split(toLower(r2.name), ' '))
        RETURN r1, r2, a1, a2, c1, c2
    """,
    "same_city_cuisine": """
        MATCH (r1:Restaurant)-[:LOCATED_AT]->(a1:Address)-[:IN_CITY]->(c1:City)
        MATCH (r2:Restaurant)-[:LOCATED_AT]->(a2:Address)-[:IN_CITY]->(c1)
        WHERE toInteger(r1.id) < 533 AND toInteger(r2.id) >= 533
          AND r1.cuisine <> '' AND toLower(r1.cuisine) = toLower(r2.cuisine)
        RETURN r1, r2, a1, a2, c1, c1 AS c2
    """,
    "phone_area": """
        MATCH (r1:Restaurant)-[:LOCATED_AT]->(a1:Address)-[:IN_CITY]->(c1:City)
        MATCH (r2:Restaurant)-[:LOCATED_AT]->(a2:Address)-[:IN_CITY]->(c2:City)
        WHERE toInteger(r1.id) < 533 AND toInteger(r2.id) >= 533
        RETURN r1, r2, a1, a2, c1, c2
    """,
    "address_token": """
        MATCH (r1:Restaurant)-[:LOCATED_AT]->(a1:Address)-[:IN_CITY]->(c1:City)
        MATCH (r2:Restaurant)-[:LOCATED_AT]->(a2:Address)-[:IN_CITY]->(c2:City)
        WHERE toInteger(r1.id) < 533 AND toInteger(r2.id) >= 533
          AND any(w IN split(toLower(a1.value), ' ') WHERE size(w) > 2 AND w IN split(toLower(a2.value), ' '))
        RETURN r1, r2, a1, a2, c1, c2
    """,
}


def node_dict(node) -> dict:
    return dict(node.items())


def structural_score(r1: dict, r2: dict, a1_value: str, a2_value: str, city_context: str) -> float:
    name = jaccard(r1.get("name"), r2.get("name"))
    address = jaccard(a1_value, a2_value)
    cuisine = jaccard(r1.get("cuisine"), r2.get("cuisine"))
    phone = 1.0 if phone_digits(r1.get("phone")) and phone_digits(r1.get("phone")) == phone_digits(r2.get("phone")) else 0.0
    city = 0.0 if " / " in city_context else 1.0
    return 0.45 * name + 0.2 * address + 0.15 * cuisine + 0.1 * city + 0.1 * phone


def passes_rule(rule: str, r1: dict, r2: dict, a1: dict, a2: dict) -> bool:
    if rule == "phone_area":
        p1 = phone_digits(r1.get("phone"))
        p2 = phone_digits(r2.get("phone"))
        return len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3]
    if rule == "name_token":
        return bool({t for t in tokens(r1.get("name")) if len(t) > 2} & {t for t in tokens(r2.get("name")) if len(t) > 2})
    if rule == "address_token":
        return bool({t for t in tokens(a1.get("value")) if len(t) > 2} & {t for t in tokens(a2.get("value")) if len(t) > 2})
    if rule == "same_city_cuisine":
        return clean_text(r1.get("cuisine")) == clean_text(r2.get("cuisine")) and bool(clean_text(r1.get("cuisine")))
    return True


def fetch_candidates(limit_per_rule: int | None = None) -> list[GraphCandidate]:
    by_pair: dict[tuple[int, int], GraphCandidate] = {}
    driver = get_driver()
    with driver.session() as session:
        for rule, query in GDD_QUERIES.items():
            query_text = query
            if limit_per_rule:
                query_text += f"\nLIMIT {int(limit_per_rule)}"
            for record in session.run(query_text):
                r1 = node_dict(record["r1"])
                r2 = node_dict(record["r2"])
                a1 = node_dict(record["a1"])
                a2 = node_dict(record["a2"])
                if not passes_rule(rule, r1, r2, a1, a2):
                    continue
                left_id = int(r1["id"])
                right_id = int(r2["id"])
                pair = tuple(sorted((left_id, right_id)))
                candidate = by_pair.get(pair)
                c1 = node_dict(record["c1"])
                c2 = node_dict(record["c2"])
                if candidate is None:
                    left_city = c1.get("value", "") if left_id == pair[0] else c2.get("value", "")
                    right_city = c2.get("value", "") if right_id == pair[1] else c1.get("value", "")
                    left_address = a1.get("value", "") if left_id == pair[0] else a2.get("value", "")
                    right_address = a2.get("value", "") if right_id == pair[1] else a1.get("value", "")
                    left_node = r1 if left_id == pair[0] else r2
                    right_node = r2 if right_id == pair[1] else r1
                    candidate = GraphCandidate(
                        left_id=pair[0],
                        right_id=pair[1],
                        left=left_node,
                        right=right_node,
                        left_address=left_address,
                        right_address=right_address,
                        city=left_city if left_city == right_city else f"{left_city} / {right_city}",
                        rules=[],
                    )
                    candidate.structural_score = structural_score(
                        candidate.left,
                        candidate.right,
                        candidate.left_address,
                        candidate.right_address,
                        candidate.city,
                    )
                    by_pair[pair] = candidate
                if rule not in candidate.rules:
                    candidate.rules.append(rule)
    driver.close()
    return list(by_pair.values())


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities.values() if p > 0)


def update_rule_probabilities(probabilities: dict[str, float], candidate: GraphCandidate, decision: bool, lr: float = 0.35) -> dict[str, float]:
    updated = probabilities.copy()
    candidate_rules = set(candidate.rules)
    for rule in updated:
        if rule in candidate_rules:
            updated[rule] *= 1 + lr if decision else 1 - lr
        else:
            updated[rule] *= 1 - lr / 3 if decision else 1 + lr / 3
    total = sum(updated.values())
    return {rule: value / total for rule, value in updated.items()}


def candidate_probability(candidate: GraphCandidate, probabilities: dict[str, float]) -> float:
    return min(1.0, sum(probabilities[rule] for rule in candidate.rules))


def prompt_for(candidate: GraphCandidate) -> str:
    payload = {
        "graph_pattern": [
            "(restaurant_1)-[:LOCATED_AT]->(address_1)-[:IN_CITY]->(city)",
            "(restaurant_2)-[:LOCATED_AT]->(address_2)-[:IN_CITY]->(city)",
        ],
        "gdd_rules_triggered": candidate.rules,
        "restaurant_1": {
            "id": candidate.left_id,
            "name": candidate.left.get("name", ""),
            "phone": candidate.left.get("phone", ""),
            "cuisine": candidate.left.get("cuisine", ""),
            "address": candidate.left_address,
            "city_context": candidate.city,
        },
        "restaurant_2": {
            "id": candidate.right_id,
            "name": candidate.right.get("name", ""),
            "phone": candidate.right.get("phone", ""),
            "cuisine": candidate.right.get("cuisine", ""),
            "address": candidate.right_address,
            "city_context": candidate.city,
        },
    }
    return (
        "Decide whether restaurant_1 and restaurant_2 refer to the same real-world restaurant. "
        "Use both attributes and the graph pattern/GDD rules. Return JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def run_pipeline(
    output_dir: Path = Path("output_file"),
    max_llm_calls: int = 20,
    theta: float = 0.5,
    limit_per_rule: int | None = None,
    structural_threshold: float = 0.48,
) -> dict:
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = fetch_candidates(limit_per_rule=limit_per_rule)
    probabilities = {rule: 1 / len(GDD_QUERIES) for rule in GDD_QUERIES}
    matcher = LLMMatcher()
    asked: set[tuple[int, int]] = set()

    for _ in range(min(max_llm_calls, len(candidates))):
        for candidate in candidates:
            candidate.rule_probability = candidate_probability(candidate, probabilities)
        pending = [candidate for candidate in candidates if candidate.pair not in asked]
        selected = max(
            pending,
            key=lambda item: (
                1 / (1 + abs(item.rule_probability - theta)),
                item.structural_score,
            ),
        )
        decision = matcher.decide(prompt_for(selected))
        selected.llm_decision = decision.match
        selected.llm_confidence = decision.confidence
        selected.llm_reason = decision.reason
        asked.add(selected.pair)
        probabilities = update_rule_probabilities(probabilities, selected, decision.match)

    predictions: set[tuple[int, int]] = set()
    for candidate in candidates:
        if candidate.llm_decision is True:
            predictions.add(candidate.pair)
        elif candidate.llm_decision is False:
            continue
        elif candidate.llm_decision is None:
            candidate.rule_probability = candidate_probability(candidate, probabilities)
            if candidate.structural_score >= structural_threshold and candidate.rule_probability >= 0.15:
                predictions.add(candidate.pair)

    truth = load_truth(Path("dataset/relational-dataset/fodors-zagats/matches.csv"), table_b_offset=533)
    metrics = evaluate(predictions, truth)
    result = {
        **metrics,
        "candidate_pairs": len(candidates),
        "predicted_matches": len(predictions),
        "llm_calls": len(asked),
        "structural_threshold": structural_threshold,
        "rule_entropy": entropy(probabilities),
        "rule_probabilities": probabilities,
        "runtime_seconds": time.time() - start,
    }
    write_outputs(output_dir, candidates, predictions, result)
    return result


def write_outputs(output_dir: Path, candidates: list[GraphCandidate], predictions: set[tuple[int, int]], metrics: dict) -> None:
    with (output_dir / "gaplink_llm_candidates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["left_id", "right_id", "rules", "structural_score", "rule_probability", "llm_decision", "llm_confidence", "llm_reason"])
        for candidate in candidates:
            writer.writerow([
                candidate.left_id,
                candidate.right_id,
                "|".join(candidate.rules),
                f"{candidate.structural_score:.6f}",
                f"{candidate.rule_probability:.6f}",
                candidate.llm_decision,
                candidate.llm_confidence,
                candidate.llm_reason,
            ])

    with (output_dir / "gaplink_llm_results.txt").open("w", encoding="utf-8") as file:
        for left, right in sorted(predictions):
            file.write(f"{left}|{right}\n")

    with (output_dir / "gaplink_llm_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with (output_dir / "gaplink_llm_reviewed.jsonl").open("w", encoding="utf-8") as file:
        for candidate in candidates:
            if candidate.llm_decision is not None:
                file.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-style GAPLink pipeline with Neo4j and LLM feedback.")
    parser.add_argument("--output-dir", type=Path, default=Path("output_file"))
    parser.add_argument("--max-llm-calls", type=int, default=20)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--limit-per-rule", type=int)
    parser.add_argument("--structural-threshold", type=float, default=0.48)
    args = parser.parse_args()
    metrics = run_pipeline(args.output_dir, args.max_llm_calls, args.theta, args.limit_per_rule, args.structural_threshold)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
