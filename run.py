from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from bayesian_update import entropy, optimize_rules, select_by_rules
from cache_utils import append_jsonl, load_jsonl
from dataset_config import DATASET_CONFIGS
from er_utils import evaluate, load_truth
from pps import one_to_one_predictions, pps_order_candidates
from rule_candidates import RuleCandidate, fetch_all_rule_candidates, rules_for_policy, rules_signature
from sampling import stratified_sample_candidates
from stage2_llm import judge_candidate_with_seed, load_pair_cache, stage2_key


def candidate_from_row(row: dict) -> RuleCandidate:
    return RuleCandidate(
        left_id=int(row["left_id"]),
        right_id=int(row["right_id"]),
        left=row.get("left", {}),
        right=row.get("right", {}),
        left_address=row.get("left_address", ""),
        right_address=row.get("right_address", ""),
        left_city=row.get("left_city", ""),
        right_city=row.get("right_city", ""),
        rules=list(row.get("rules", [])),
        rule_probability=float(row.get("rule_probability", 0.0)),
    )


def load_or_fetch_rule_candidates(config, output_dir: Path, use_cache: bool) -> list[RuleCandidate]:
    cache_path = output_dir / f"{config.cache_prefix}_{rules_signature(config)}_rule_candidates_cache.jsonl"
    if use_cache and cache_path.exists():
        candidates: list[RuleCandidate] = []
        with cache_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    candidates.append(candidate_from_row(json.loads(line)))
        print(f"Loaded {len(candidates)} rule candidates from {cache_path}")
        return candidates

    candidates = fetch_all_rule_candidates(config)
    with cache_path.open("w", encoding="utf-8") as file:
        for candidate in candidates:
            file.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")
    print(f"Saved {len(candidates)} rule candidates to {cache_path}")
    return candidates


def run(
    mode: str,
    output_dir: Path,
    max_calls: int,
    top_rules: int,
    max_rule_support: int,
    rule_operator: str,
    use_pps: bool,
    pps_min_weight: float,
    pps_max_comparisons: int,
    workers: int,
    use_cache: bool,
    one_to_one: bool,
    min_llm_calls: int,
    early_stop_rounds: int,
    sample_size: int,
    sample_seed: int,
    use_sampling: bool,
    dataset_key: str = "fz",
    confidence_threshold: float | None = None,
) -> dict:
    start = time.time()
    config = DATASET_CONFIGS[dataset_key]
    acceptance_confidence = config.stage2_accept_confidence if confidence_threshold is None else confidence_threshold
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_or_fetch_rule_candidates(config, output_dir, use_cache)
    probabilities, history = optimize_rules(
        candidates,
        config,
        output_dir,
        max_calls,
        use_cache,
        top_rules,
        max_rule_support,
        min_llm_calls,
        early_stop_rounds,
    )
    selected_rules, filtered = select_by_rules(candidates, probabilities, top_rules, max_rule_support, rule_operator, config)
    if use_pps:
        filtered, pps_stats = pps_order_candidates(filtered, pps_min_weight, pps_max_comparisons)
    else:
        pps_stats = {
            "pps_enabled": False,
            "pps_input_candidates": len(filtered),
            "pps_ordered_candidates": len(filtered),
            "pps_emitted_candidates": 0,
            "pps_appended_candidates": 0,
            "pps_min_weight": pps_min_weight,
            "pps_max_comparisons": pps_max_comparisons,
        }
    truth = load_truth(config.truth_path, table_b_offset=config.table_b_offset)
    if use_sampling:
        filtered, sampling_stats = stratified_sample_candidates(filtered, truth, sample_size, sample_seed)
    else:
        sampling_stats = {
            "sampling_enabled": False,
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "candidates_before_sampling": len(filtered),
            "candidates_after_sampling": len(filtered),
            "positive_before_sampling": sum(1 for candidate in filtered if candidate.pair in truth),
            "negative_before_sampling": sum(1 for candidate in filtered if candidate.pair not in truth),
            "positive_after_sampling": sum(1 for candidate in filtered if candidate.pair in truth),
            "negative_after_sampling": sum(1 for candidate in filtered if candidate.pair not in truth),
        }
    sampled_pairs = {candidate.pair for candidate in filtered}
    evaluation_truth = truth & sampled_pairs if sampling_stats["sampling_enabled"] else truth
    votes = 3 if mode == "self-consistency" else 1
    cache_path = output_dir / f"{config.cache_prefix}_{mode}_stage2_cache.jsonl"
    cache = load_jsonl(cache_path) if use_cache else {}
    fewshot_cache = (
        load_pair_cache(output_dir / f"{config.cache_prefix}_few-shot_stage2_cache.jsonl")
        if use_cache and mode == "self-consistency"
        else {}
    )
    lock = threading.Lock()

    def passes_stage2_acceptance(candidate: RuleCandidate, row: dict) -> bool:
        confidence = float(row.get("confidence", 0.0))
        required_any_rules = rules_for_policy("stage2_accept_any", config)
        has_required_rule = not required_any_rules or any(rule in candidate.rules for rule in required_any_rules)
        return bool(row["match"]) and confidence >= acceptance_confidence and has_required_rule

    def run_one(candidate: RuleCandidate) -> dict:
        key = stage2_key(mode, candidate, votes, selected_rules, config)
        if key in cache:
            return cache[key]
        seed_row = fewshot_cache.get(candidate.pair)
        row = {"key": key, **judge_candidate_with_seed(candidate, mode, votes, config, seed_row)}
        if seed_row is not None:
            row["cache_source"] = "few-shot-stage2-as-first-self-consistency-vote"
        append_jsonl(cache_path, row, lock)
        return row

    accepted: list[tuple[float, tuple[int, int]]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, candidate) for candidate in filtered]
        candidate_by_pair = {candidate.pair: candidate for candidate in filtered}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            confidence = float(row.get("confidence", 0.0))
            pair = tuple(sorted((int(row["left_id"]), int(row["right_id"]))))
            candidate = candidate_by_pair[pair]
            if passes_stage2_acceptance(candidate, row):
                accepted.append((confidence, tuple(sorted((int(row["left_id"]), int(row["right_id"]))))))
            predictions = one_to_one_predictions(accepted) if one_to_one else {pair for _confidence, pair in accepted}
            if index % 20 == 0 or index == len(futures):
                metrics = evaluate(predictions, evaluation_truth)
                print(f"Stage2 {index}/{len(futures)} p={metrics['precision']:.4f} r={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

    predictions = one_to_one_predictions(accepted) if one_to_one else {pair for _confidence, pair in accepted}
    metrics = evaluate(predictions, evaluation_truth)
    result = {
        **metrics,
        "dataset": config.display_name,
        "dataset_key": config.key,
        "mode": mode,
        "llm_rule_feedback_calls": len(history),
        "selected_rules": selected_rules,
        "rule_probabilities": probabilities,
        "rule_entropy": entropy(probabilities),
        "candidate_pairs_after_rule_intersection": len(filtered),
        **sampling_stats,
        "max_rule_support": max_rule_support,
        "rule_operator": rule_operator,
        **pps_stats,
        "llm_candidate_calls": len(filtered) * votes,
        "llm_candidate_calls_saved_by_fewshot_seed": (
            sum(1 for candidate in filtered if candidate.pair in fewshot_cache)
            if mode == "self-consistency"
            else 0
        ),
        "llm_candidate_api_calls_estimated": len(filtered) * votes
        - (
            sum(1 for candidate in filtered if candidate.pair in fewshot_cache)
            if mode == "self-consistency"
            else 0
        ),
        "votes": votes,
        "one_to_one": one_to_one,
        "stage2_accept_confidence": acceptance_confidence,
        "stage2_accept_any_rules": list(rules_for_policy("stage2_accept_any", config)),
        "min_llm_calls": min_llm_calls,
        "early_stop_rounds": early_stop_rounds,
        "predicted_matches": len(predictions),
        "f1_percent": metrics["f1"] * 100,
        "runtime_seconds": time.time() - start,
    }
    (output_dir / f"{config.cache_prefix}_{mode}_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{config.cache_prefix}_{mode}_rule_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / f"{config.cache_prefix}_{mode}_candidates.jsonl").open("w", encoding="utf-8") as file:
        for candidate in filtered:
            file.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")
    with (output_dir / f"{config.cache_prefix}_{mode}_pps_order.jsonl").open("w", encoding="utf-8") as file:
        for index, candidate in enumerate(filtered, start=1):
            file.write(
                json.dumps(
                    {
                        "rank": index,
                        "left_id": candidate.left_id,
                        "right_id": candidate.right_id,
                        "rules": candidate.rules,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="GAPLink threshold-rule LLM runner with dataset configuration.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="fz")
    parser.add_argument("--mode", choices=["zero-shot", "few-shot", "self-consistency"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output_file"))
    parser.add_argument("--max-llm-calls", type=int, default=20)
    parser.add_argument("--top-rules", type=int, default=1)
    parser.add_argument("--max-rule-support", type=int, default=114)
    parser.add_argument("--rule-operator", choices=["intersection", "union"], default="intersection")
    parser.add_argument("--disable-pps", action="store_true")
    parser.add_argument("--pps-min-weight", type=float, default=-1.0)
    parser.add_argument("--pps-max-comparisons", type=int, default=1_000_000_000)
    parser.add_argument("--min-llm-calls", type=int, default=20)
    parser.add_argument("--early-stop-rounds", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--disable-sampling", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--one-to-one", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Override the dataset's Stage2 acceptance confidence. By default each dataset uses its configured verification policy.",
    )
    args = parser.parse_args()
    run(
        args.mode,
        args.output_dir,
        args.max_llm_calls,
        args.top_rules,
        args.max_rule_support,
        args.rule_operator,
        not args.disable_pps,
        args.pps_min_weight,
        args.pps_max_comparisons,
        args.workers,
        not args.no_cache,
        args.one_to_one,
        args.min_llm_calls,
        args.early_stop_rounds,
        args.sample_size,
        args.sample_seed,
        not args.disable_sampling,
        args.dataset,
        args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
