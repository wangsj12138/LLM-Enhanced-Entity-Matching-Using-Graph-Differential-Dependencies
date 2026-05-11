from __future__ import annotations

import difflib
import ast
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dataset_config import DatasetConfig
from er_utils import clean_text, jaccard, phone_digits, tokens
from neo4j_setup import get_driver


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


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    description: str
    expression: str


DEFAULT_RULES_PATH = Path(__file__).with_name("rules.txt")
ALLOWED_RULE_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
)


def validate_rule_expression(expression: str) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_RULE_AST_NODES):
            raise ValueError(f"Unsupported rule expression syntax in {expression!r}: {type(node).__name__}")
    return tree


@lru_cache(maxsize=None)
def load_rule_definitions(path: Path = DEFAULT_RULES_PATH) -> tuple[RuleDefinition, ...]:
    definitions: list[RuleDefinition] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid rule format at {path}:{line_number}; expected 3 tab-separated columns.")
            rule_id, description, expression = (part.strip() for part in parts)
            validate_rule_expression(expression)
            definitions.append(RuleDefinition(rule_id, description, expression))
    if not definitions:
        raise ValueError(f"No rules loaded from {path}.")
    return tuple(definitions)


@lru_cache(maxsize=None)
def load_rule_policies(path: Path = DEFAULT_RULES_PATH) -> dict[tuple[str, str], tuple[str, ...]]:
    policies: dict[tuple[str, str], tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#") or not line.startswith("@"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid rule policy at {path}:{line_number}; expected 3 tab-separated columns.")
            policy_name = parts[0].lstrip("@").strip()
            dataset_key = parts[1].strip()
            rule_ids = tuple(rule_id.strip() for rule_id in parts[2].split(",") if rule_id.strip())
            policies[(policy_name, dataset_key)] = rule_ids
    known_rules = {rule.rule_id for rule in load_rule_definitions(path)}
    for (policy_name, dataset_key), rule_ids in policies.items():
        unknown = [rule_id for rule_id in rule_ids if rule_id not in known_rules]
        if unknown:
            raise ValueError(f"Unknown rule ids in policy {policy_name}/{dataset_key}: {', '.join(unknown)}")
    return policies


def rules_for_policy(policy_name: str, config: DatasetConfig) -> tuple[str, ...]:
    return load_rule_policies(config.rules_path).get((policy_name, config.key), ())


def rule_descriptions(config: DatasetConfig | None = None) -> dict[str, str]:
    path = config.rules_path if config is not None else DEFAULT_RULES_PATH
    return {rule.rule_id: rule.description for rule in load_rule_definitions(path)}


def rules_signature(config: DatasetConfig) -> str:
    return hashlib.sha1(config.rules_path.read_bytes()).hexdigest()[:10]


@lru_cache(maxsize=None)
def compiled_rule_expressions(path: Path = DEFAULT_RULES_PATH) -> tuple[tuple[str, object], ...]:
    return tuple(
        (rule.rule_id, compile(validate_rule_expression(rule.expression), str(path), "eval"))
        for rule in load_rule_definitions(path)
    )


RULE_DESCRIPTIONS = rule_descriptions()


def node_dict(node) -> dict:
    return dict(node.items())


def phone_area(value: str | None) -> str:
    digits = phone_digits(value)
    return digits[:3] if len(digits) >= 3 else ""


def triggered_rules(candidate: RuleCandidate, config: DatasetConfig) -> list[str]:
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
    context = {
        "name_sim": name_sim,
        "address_sim": address_sim,
        "cuisine_sim": cuisine_sim,
        "name_chars": name_chars,
        "exact_phone": exact_phone,
        "same_area": same_area,
        "same_city": same_city,
        "exact_name": bool(left_name and left_name == right_name),
    }
    triggered: list[str] = []
    for rule_id, expression in compiled_rule_expressions(config.rules_path):
        if bool(eval(expression, {"__builtins__": {}}, context)):
            triggered.append(rule_id)
    return triggered


def fetch_all_rule_candidates(config: DatasetConfig) -> list[RuleCandidate]:
    if config.graph_kind == "paper_author_venue":
        return fetch_paper_rule_candidates(config)
    if config.graph_kind == "software_manufacturer":
        return fetch_product_rule_candidates(config)
    if config.graph_kind == "author_paper":
        return fetch_author_rule_candidates(config)

    query = f"""
    MATCH (r1:{config.entity_label})-[:{config.located_rel}]->(a1:{config.address_label})-[:{config.city_rel}]->(c1:{config.city_label})
    MATCH (r2:{config.entity_label})-[:{config.located_rel}]->(a2:{config.address_label})-[:{config.city_rel}]->(c2:{config.city_label})
    WHERE toInteger(r1.id) < {config.left_id_max_exclusive} AND toInteger(r2.id) >= {config.right_id_min_inclusive}
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
            candidate.rules = triggered_rules(candidate, config)
            if candidate.rules:
                if candidate.pair in candidates:
                    merged = sorted(set(candidates[candidate.pair].rules) | set(candidate.rules))
                    candidates[candidate.pair].rules = merged
                else:
                    candidates[candidate.pair] = candidate
    driver.close()
    return list(candidates.values())


TITLE_STOPWORDS = {
    "and", "the", "for", "with", "from", "into", "using", "based", "data", "system",
    "systems", "method", "model", "models", "approach", "analysis", "design",
    "database", "databases", "information", "management",
}


def title_block_tokens(value: str | None) -> set[str]:
    return {token for token in tokens(value) if len(token) >= 4 and token not in TITLE_STOPWORDS}


def author_surname(value: str | None) -> str:
    parts = clean_text((value or "").replace("-", "_")).split()
    return parts[0] if parts else ""


def fetch_paper_rule_candidates(config: DatasetConfig) -> list[RuleCandidate]:
    query = f"""
    MATCH (p:{config.entity_label})
    RETURN p
    """
    driver = get_driver()
    papers: dict[int, dict] = {}
    with driver.session() as session:
        for record in session.run(query):
            paper = node_dict(record["p"])
            papers[int(paper["id"])] = paper
    driver.close()

    left = [paper for paper_id, paper in papers.items() if paper_id < config.left_id_max_exclusive]
    right = [paper for paper_id, paper in papers.items() if paper_id >= config.right_id_min_inclusive]
    right_index: dict[str, list[dict]] = {}
    for paper in right:
        for token in title_block_tokens(paper.get("name")):
            right_index.setdefault(token, []).append(paper)

    candidates: dict[tuple[int, int], RuleCandidate] = {}
    for left_paper in left:
        seen_right: set[int] = set()
        for token in title_block_tokens(left_paper.get("name")):
            for right_paper in right_index.get(token, []):
                right_id = int(right_paper["id"])
                if right_id in seen_right:
                    continue
                seen_right.add(right_id)
                candidate = RuleCandidate(
                    left_id=int(left_paper["id"]),
                    right_id=right_id,
                    left=left_paper,
                    right=right_paper,
                    left_address=left_paper.get("authors", ""),
                    right_address=right_paper.get("authors", ""),
                    left_city=left_paper.get("venue", ""),
                    right_city=right_paper.get("venue", ""),
                    rules=[],
                )
                candidate.rules = triggered_rules(candidate, config)
                if candidate.rules:
                    candidates[candidate.pair] = candidate

    return list(candidates.values())


def fetch_author_rule_candidates(config: DatasetConfig) -> list[RuleCandidate]:
    query = f"""
    MATCH (a:{config.entity_label})
    RETURN a
    """
    driver = get_driver()
    authors: dict[int, dict] = {}
    with driver.session() as session:
        for record in session.run(query):
            author = node_dict(record["a"])
            authors[int(author["id"])] = author
    driver.close()

    blocks: dict[str, list[dict]] = {}
    paper_blocks: dict[str, list[dict]] = {}
    for author in authors.values():
        key = author_surname(author.get("name"))
        if key:
            blocks.setdefault(key, []).append(author)
        paper_key = clean_text(author.get("paper"))
        if paper_key:
            paper_blocks.setdefault(paper_key, []).append(author)

    candidates: dict[tuple[int, int], RuleCandidate] = {}
    def add_block_candidates(block_authors: list[dict]) -> None:
        if len(block_authors) > 250:
            return
        block_authors = sorted(block_authors, key=lambda item: int(item["id"]))
        for index, left_author in enumerate(block_authors):
            for right_author in block_authors[index + 1:]:
                candidate = RuleCandidate(
                    left_id=int(left_author["id"]),
                    right_id=int(right_author["id"]),
                    left=left_author,
                    right=right_author,
                    left_address=left_author.get("paper", ""),
                    right_address=right_author.get("paper", ""),
                    left_city=left_author.get("paper", ""),
                    right_city=right_author.get("paper", ""),
                    rules=[],
                )
                candidate.rules = triggered_rules(candidate, config)
                if candidate.rules:
                    candidates[candidate.pair] = candidate

    for block_authors in blocks.values():
        add_block_candidates(block_authors)
    for block_authors in paper_blocks.values():
        add_block_candidates(block_authors)

    return list(candidates.values())


def fetch_product_rule_candidates(config: DatasetConfig) -> list[RuleCandidate]:
    query = f"""
    MATCH (p:{config.entity_label})
    RETURN p
    """
    driver = get_driver()
    products: dict[int, dict] = {}
    with driver.session() as session:
        for record in session.run(query):
            product = node_dict(record["p"])
            products[int(product["id"])] = product
    driver.close()

    left = [product for product_id, product in products.items() if product_id < config.left_id_max_exclusive]
    right = [product for product_id, product in products.items() if product_id >= config.right_id_min_inclusive]
    right_index: dict[str, list[dict]] = {}
    for product in right:
        for token in title_block_tokens(product.get("name")):
            right_index.setdefault(token, []).append(product)

    candidates: dict[tuple[int, int], RuleCandidate] = {}
    for left_product in left:
        seen_right: set[int] = set()
        for token in title_block_tokens(left_product.get("name")):
            for right_product in right_index.get(token, []):
                right_id = int(right_product["id"])
                if right_id in seen_right:
                    continue
                seen_right.add(right_id)
                candidate = RuleCandidate(
                    left_id=int(left_product["id"]),
                    right_id=right_id,
                    left=left_product,
                    right=right_product,
                    left_address=left_product.get("manufacturer", ""),
                    right_address=right_product.get("manufacturer", ""),
                    left_city=left_product.get("manufacturer", ""),
                    right_city=right_product.get("manufacturer", ""),
                    rules=[],
                )
                candidate.rules = triggered_rules(candidate, config)
                if candidate.rules:
                    candidates[candidate.pair] = candidate

    return list(candidates.values())
