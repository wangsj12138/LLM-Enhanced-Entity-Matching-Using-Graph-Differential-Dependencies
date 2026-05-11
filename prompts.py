from __future__ import annotations

import json

from dataset_config import DATASET_CONFIGS, DatasetConfig
from rule_candidates import RuleCandidate


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


def record_text(candidate: RuleCandidate, side: str, config: DatasetConfig) -> str:
    node = candidate.left if side == "left" else candidate.right
    address = candidate.left_address if side == "left" else candidate.right_address
    city = candidate.left_city if side == "left" else candidate.right_city
    if config.graph_kind == "paper_author_venue":
        return f"title={node.get('name', '')}; year={node.get('year', node.get('phone', ''))}; authors={address}; venue={city}"
    if config.graph_kind == "software_manufacturer":
        return f"title={node.get('name', '')}; manufacturer={address}; price={node.get('price', node.get('phone', ''))}"
    if config.graph_kind == "author_paper":
        return f"author_name={node.get('name', '')}; paper_title={address}"
    return f"name={node.get('name', '')}; address={address}; city={city}; phone={node.get('phone', '')}; cuisine={node.get('cuisine', '')}"


def restaurant_text(candidate: RuleCandidate, side: str) -> str:
    return record_text(candidate, side, DATASET_CONFIGS["fz"])


def graph_triples(candidate: RuleCandidate, config: DatasetConfig) -> list[str]:
    left_restaurant = f"{config.entity_label}:{candidate.left_id}"
    right_restaurant = f"{config.entity_label}:{candidate.right_id}"
    left_address = f"{config.address_label}:{candidate.left_address}"
    right_address = f"{config.address_label}:{candidate.right_address}"
    left_city = f"{config.city_label}:{candidate.left_city}"
    right_city = f"{config.city_label}:{candidate.right_city}"
    if config.graph_kind == "paper_author_venue":
        return [
            f"({left_restaurant}, {config.located_rel}, {left_address})",
            f"({left_restaurant}, {config.city_rel}, {left_city})",
            f"({right_restaurant}, {config.located_rel}, {right_address})",
            f"({right_restaurant}, {config.city_rel}, {right_city})",
        ]
    if config.graph_kind == "software_manufacturer":
        return [
            f"({left_restaurant}, {config.located_rel}, {left_address})",
            f"({right_restaurant}, {config.located_rel}, {right_address})",
        ]
    if config.graph_kind == "author_paper":
        return [
            f"({left_restaurant}, {config.located_rel}, {left_address})",
            f"({right_restaurant}, {config.located_rel}, {right_address})",
        ]
    return [
        f"({left_restaurant}, {config.located_rel}, {left_address})",
        f"({left_address}, {config.city_rel}, {left_city})",
        f"({right_restaurant}, {config.located_rel}, {right_address})",
        f"({right_address}, {config.city_rel}, {right_city})",
    ]


def graph_pattern_text(candidate: RuleCandidate, config: DatasetConfig) -> str:
    triples = "; ".join(graph_triples(candidate, config))
    attributes = (
        f"Attributes: "
        f"r1({record_text(candidate, 'left', config)}); "
        f"r2({record_text(candidate, 'right', config)})"
    )
    return f"Graph pattern: {triples}\n{attributes}"


def rule_feedback_prompt(candidate: RuleCandidate, config: DatasetConfig) -> str:
    payload = {
        "task": f"{config.task_name}; return JSON match feedback for Bayesian GDD rule update.",
        "rules": candidate.rules,
        "graph_pattern": graph_pattern_text(candidate, config),
        "out": {"match": "bool", "confidence": "0..1", "reason": "short"},
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def match_prompt(candidate: RuleCandidate, mode: str, config: DatasetConfig) -> str:
    examples = ""
    if mode in {"few-shot", "self-consistency"}:
        examples = "\nEx:" + ";".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in EXAMPLES)
    dataset_note = ""
    if config.graph_kind == "software_manufacturer":
        dataset_note = (
            "For software products, judge primarily by product title, version, edition, platform, bundle size, and price. "
            "Manufacturer graph links can be noisy; use them only as weak evidence. "
            "Same brand/category or same price alone is not enough; different version/edition usually means non-match."
        )
    elif config.graph_kind == "author_paper":
        dataset_note = (
            "For author records, judge by name variants, initials, surname consistency, and paper-title context. "
            "Different spellings or abbreviated initials may still refer to the same real-world author."
        )
    payload = {
        "rules": candidate.rules,
        "graph_pattern": graph_pattern_text(candidate, config),
    }
    return f"""
{config.task_name}. Decide if r1 and r2 are the same real-world {config.entity_noun}.
Use attributes plus GDD rules and the graph pattern as evidence; make the final semantic decision from the records.
{dataset_note}
Return JSON only: {{"match":true/false,"confidence":0.0-1.0,"reason":"..."}}
{examples}

{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()
