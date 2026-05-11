from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROMPT_VERSION = "v7-graph-pattern"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    task_name: str
    entity_noun: str
    graph_kind: str
    network_dir: Path
    entity_label: str
    address_label: str
    city_label: str
    located_rel: str
    city_rel: str
    left_id_max_exclusive: int
    right_id_min_inclusive: int
    truth_path: Path
    table_b_offset: int | None
    cache_prefix: str
    rules_path: Path
    stage2_accept_confidence: float = 0.0
    prompt_version: str = PROMPT_VERSION
    stage2_accept_any_rules: tuple[str, ...] = ()
    stage1_allowed_rules: tuple[str, ...] = ()


DATASET_CONFIGS = {
    "fz": DatasetConfig(
        key="fz",
        display_name="FZ",
        task_name="Fodors-Zagats restaurant ER",
        entity_noun="restaurant",
        graph_kind="fodors_zagats",
        network_dir=Path("dataset/fodors_zagats/network"),
        entity_label="Restaurant",
        address_label="Address",
        city_label="City",
        located_rel="LOCATED_AT",
        city_rel="IN_CITY",
        left_id_max_exclusive=533,
        right_id_min_inclusive=533,
        truth_path=Path("dataset/fodors_zagats/relational-dataset/fodors-zagats/matches.csv"),
        table_b_offset=533,
        cache_prefix="fz_threshold",
        rules_path=Path("rules/fz.txt"),
        stage2_accept_confidence=0.0,
    ),
    "dblp_acm": DatasetConfig(
        key="dblp_acm",
        display_name="DBLP-ACM",
        task_name="DBLP-ACM publication ER",
        entity_noun="paper",
        graph_kind="paper_author_venue",
        network_dir=Path("dataset/dblp_acm/network"),
        entity_label="Paper",
        address_label="Author",
        city_label="Venue",
        located_rel="WRITTEN_BY",
        city_rel="PUBLISHED_IN",
        left_id_max_exclusive=2616,
        right_id_min_inclusive=2616,
        truth_path=Path("dataset/dblp_acm/relational-dataset/dblp-acm/matches.csv"),
        table_b_offset=2616,
        cache_prefix="dblp_acm_threshold",
        rules_path=Path("rules/fz.txt"),
        stage2_accept_confidence=0.95,
    ),
    "amazon_google": DatasetConfig(
        key="amazon_google",
        display_name="Amazon-Google",
        task_name="Amazon-Google software product ER",
        entity_noun="software product",
        graph_kind="software_manufacturer",
        network_dir=Path("dataset/amazon_google/network"),
        entity_label="Software",
        address_label="Manufacturer",
        city_label="Manufacturer",
        located_rel="MADE_BY",
        city_rel="MADE_BY",
        left_id_max_exclusive=1363,
        right_id_min_inclusive=1363,
        truth_path=Path("dataset/amazon_google/network/ground_truth_amazon_google.txt"),
        table_b_offset=None,
        cache_prefix="amazon_google_threshold",
        rules_path=Path("rules/fz.txt"),
        stage2_accept_confidence=0.8,
        prompt_version="v8-ag-product",
        stage2_accept_any_rules=("address_sim_ge_050", "cuisine_sim_ge_050", "same_city"),
    ),
    "citeseer": DatasetConfig(
        key="citeseer",
        display_name="CiteSeer",
        task_name="CiteSeer author entity resolution",
        entity_noun="author record",
        graph_kind="author_paper",
        network_dir=Path("dataset/citeseer/network"),
        entity_label="Author",
        address_label="Paper",
        city_label="Paper",
        located_rel="AUTHORED",
        city_rel="AUTHORED",
        left_id_max_exclusive=2892,
        right_id_min_inclusive=0,
        truth_path=Path("dataset/citeseer/network/ground_truth_citeseer.txt"),
        table_b_offset=None,
        cache_prefix="citeseer_threshold",
        rules_path=Path("rules/fz.txt"),
        stage2_accept_confidence=0.9,
        prompt_version="v8-citeseer-author",
        stage1_allowed_rules=("name_or_cuisine_or_char060",),
    ),
    "arxiv": DatasetConfig(
        key="arxiv",
        display_name="ArXiv",
        task_name="ArXiv author entity resolution",
        entity_noun="author record",
        graph_kind="author_paper",
        network_dir=Path("dataset/arxiv/network"),
        entity_label="Author",
        address_label="Paper",
        city_label="Paper",
        located_rel="AUTHORED",
        city_rel="AUTHORED",
        left_id_max_exclusive=58515,
        right_id_min_inclusive=0,
        truth_path=Path("dataset/arxiv/network/arxiv_ground_turth.txt"),
        table_b_offset=None,
        cache_prefix="arxiv_threshold",
        rules_path=Path("rules/fz.txt"),
        stage2_accept_confidence=0.8,
        prompt_version="v8-arxiv-author",
        stage2_accept_any_rules=("name_sim_ge_045",),
        stage1_allowed_rules=("name_or_cuisine_or_char060",),
    ),
}
