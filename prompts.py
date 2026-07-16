from __future__ import annotations

import json
import os

from dataset_config import DATASET_CONFIGS, DatasetConfig
from rule_candidates import RuleCandidate





def prompt_profile() -> str:
    explicit = os.getenv("LLM_PROMPT_PROFILE", "").strip().lower()
    if explicit in {"small", "large"}:
        return explicit
    model = os.getenv("LLM_MODEL", "").lower()
    if any(marker in model for marker in ("pro", "v4", "gpt-4", "claude-3", "gemini")):
        return "large"
    return "small"


def fz_decision_guidance() -> str:
    return FZ_LARGE_MODEL_GUIDANCE if prompt_profile() == "large" else FZ_SMALL_MODEL_GUIDANCE


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
        dataset_note = AG_LARGE_MODEL_GUIDANCE if prompt_profile() == "large" else AG_SMALL_MODEL_GUIDANCE
    elif config.graph_kind == "paper_author_venue":
        dataset_note = PAPER_LARGE_MODEL_GUIDANCE if prompt_profile() == "large" else PAPER_SMALL_MODEL_GUIDANCE
    elif config.graph_kind == "fodors_zagats":
        dataset_note = fz_decision_guidance()
    elif config.graph_kind == "author_paper":
        dataset_note = AUTHOR_LARGE_MODEL_GUIDANCE if prompt_profile() == "large" else AUTHOR_SMALL_MODEL_GUIDANCE
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
FZ_SMALL_MODEL_GUIDANCE = (
    "Fodors-Zagats restaurant matching guidance: phone punctuation, street abbreviations "
    "(st/street, rd/road, ave/avenue, ln/lane), cross-street suffixes, and city variants "
    "such as New York vs New York City are usually formatting differences, not conflicts. "
    "Cuisine/category labels are noisy and should be weaker than name, phone, and address. "
    "Do not treat different graph node IDs as evidence of non-match; every candidate pair "
    "starts as two different records. A normalized phone match is very strong evidence: "
    "if the core restaurant name is the same or one name is a location-expanded version of "
    "the other, prefer match even when cuisine differs, address has extra detail, or the "
    "street address differs because one source uses a hotel/landmark address. A missing or "
    "added style word such as bistro, deli, or cafe can be compatible when the core name is "
    "unique. However, for hotel or venue brands that contain several named subvenues at the "
    "same address/phone, subvenue tokens are identity-bearing: cafe, dining room, restaurant, "
    "grill, bar, bistro, deli, delicatessen, palace/court, and similar venue descriptors "
    "should agree or be clearly compatible. For hotel brands such as Ritz-Carlton, a bare "
    "hotel/location name is not enough to match a specific cafe, dining room, or restaurant. "
    "Example: a Ritz-Carlton dining room should not match a Ritz-Carlton cafe; a Ritz-Carlton "
    "restaurant should not match a generic Ritz-Carlton location entry unless the restaurant "
    "subvenue is explicit on both sides."
)

FZ_LARGE_MODEL_GUIDANCE = (
    "Fodors-Zagats restaurant matching guidance: judge normalized name, phone, address, "
    "city, and cuisine together. Treat phone punctuation, street abbreviations, cross-street "
    "suffixes, and New York/New York City style variants as formatting differences. Cuisine "
    "is noisy and weaker than name, phone, and address. Be careful with hotel or venue brands "
    "that contain multiple restaurants at the same address/phone: cafe, dining room, restaurant, "
    "grill, bar, bistro, deli, and similar subvenue tokens may distinguish different entities."
)

PAPER_SMALL_MODEL_GUIDANCE = (
    "Publication matching guidance for DBLP-ACM: decide whether the two records are the same "
    "paper, not merely papers by the same authors on the same topic. Title identity or a clear "
    "typo/abbreviation of the same full title is the strongest signal. Author order may differ, "
    "initials may replace full names, accents or HTML entities may appear, and one source may "
    "list an extra or missing coauthor. Venue names are often aliases: SIGMOD Conference, "
    "International Conference on Management of Data, and ACM SIGMOD are compatible; VLDB and "
    "Very Large Data Bases are compatible; SIGMOD Record and ACM SIGMOD Record are compatible. "
    "Do not treat different graph node IDs as evidence of non-match. If title matches exactly "
    "and authors substantially overlap, prefer match even when venue strings differ. If titles "
    "share only topic words but are not the same full title, prefer non-match even with the same "
    "authors. Be conservative for recurring or generic titles such as book review column, book "
    "reviews, editorial, foreword, introduction, or column names; they may refer to different "
    "installments. A year mismatch is weak when title is exact, but important when title is only "
    "similar."
)

PAPER_LARGE_MODEL_GUIDANCE = (
    "Publication matching guidance: compare normalized title, author set, year, and venue. "
    "Treat author order, initials, accents, and common venue abbreviations/full names as "
    "formatting differences when title and authors strongly agree."
)

AG_SMALL_MODEL_GUIDANCE = (
    "Amazon-Google software matching guidance: product titles are the primary identity signal. "
    "Treat missing words, OCR-like typos, punctuation, bracketed media markers such as [CD] or "
    "[DVD], seller suffixes, and word-order changes as weak differences when the core product, "
    "version number, edition, and platform agree. Manufacturer links in this benchmark can be "
    "retailer, distributor, or noisy catalog context, so do not reject a pair only because the "
    "manufacturer differs, especially when one side is a generic/repeated catalog maker. Price "
    "differences are weak evidence. Prefer match for strong title overlap unless there is a clear "
    "conflict in major version, edition tier, platform, license count, bundle contents, or a "
    "different product family."
)

AG_LARGE_MODEL_GUIDANCE = (
    "Amazon-Google software matching guidance: prioritize normalized product title, version, "
    "edition, platform, and bundle/license cues. Manufacturer and price are secondary because "
    "they may reflect sellers, distributors, or noisy catalog links. For high-title-similarity "
    "candidates, lean match unless a concrete version/edition/platform/bundle conflict indicates "
    "a different product."
)

AUTHOR_SMALL_MODEL_GUIDANCE = (
    "Author disambiguation guidance: decide whether two author records refer to the same person. "
    "Different graph node IDs are never evidence of non-match; every candidate starts as two "
    "different records. Different paper IDs or paper titles are also not conflicts because the "
    "same author can publish multiple papers. Exact normalized author names are very strong "
    "evidence for match. Initials and full names can be compatible, e.g. i_a with igor, a_a with "
    "arkady, or first initials replacing given names, as long as surname and initials do not "
    "contradict. Prefer non-match when surnames differ, or when full given names clearly refer "
    "to different people."
)

AUTHOR_LARGE_MODEL_GUIDANCE = (
    "Author disambiguation guidance: compare surname, given names, initials, and publication "
    "context. Treat different record IDs and different paper IDs as expected, not as conflicts. "
    "Exact names or compatible initials/full-name variants are strong match evidence; conflicting "
    "surnames or incompatible full given names indicate non-match."
)