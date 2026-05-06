# LLM-Enhanced-Entity-Matching-Using-Graph-Differential-Dependencies
![Pipeline](./pipeline.svg)
## Datasets

The project supports two types of benchmark datasets for entity resolution:

1. **Graph ER Benchmark Datasets:**
   - [LINQS Datasets](https://linqs.org/datasets/)
   - [Neo4j Sandbox](https://neo4j.com/sandbox/)

2. **Relational ER Benchmark Datasets:**
   - [DeepMatcher Datasets](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)

## Quick start

The default runner uses the bundled Fodors-Zagats CSV files and does not require
Neo4j or an LLM API key:

```bash
python3 main1.py --mode local
```

This writes candidate pairs, blocks, predicted matches, and evaluation metrics to
`output_file/`.

Expected local demo output is approximately:

```text
precision: 0.9550
recall: 0.9636
f1: 0.9593
candidate_pairs: 37187
predicted_matches: 111
```

Run tests with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The original Neo4j path is still available for graph-backed experiments:

```bash
python3 -m pip install -r requirements-neo4j.txt
python3 main1.py --mode neo4j
```

## Paper-style GAPLink run

For the full graph + LLM workflow, configure `.env` first:

```bash
cp .env.example .env
```

Set `NEO4J_PASSWORD` to your local Neo4j password and set `LLM_API_KEY`.
OpenAI-compatible providers can be used by also setting `LLM_BASE_URL` and
`LLM_MODEL`.

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Import the bundled property graph into Neo4j:

```bash
.venv/bin/python neo4j_setup.py
```

Run GAPLink with GDD candidate filtering, entropy-driven rule refinement, and
LLM rule-prompt matching:

```bash
.venv/bin/python gaplink_pipeline.py --max-llm-calls 20
```

The same full workflow can also be launched through `main1.py`:

```bash
.venv/bin/python main1.py --mode paper --max-llm-calls 20
```

Outputs are written to:

- `output_file/gaplink_llm_candidates.csv`
- `output_file/gaplink_llm_results.txt`
- `output_file/gaplink_llm_metrics.json`
- `output_file/gaplink_llm_reviewed.jsonl`

## Reproduce FZ prompt table

Run the three FZ prompt settings:

```bash
.venv/bin/python fz_gaplink_threshold_rules.py --mode zero-shot
.venv/bin/python fz_gaplink_threshold_rules.py --mode few-shot
.venv/bin/python fz_gaplink_threshold_rules.py --mode self-consistency
```

Each run follows the paper-style pipeline:

1. Load the graph-backed Fodors-Zagats candidates from Neo4j.
2. Generate threshold-style GDD rules.
3. Refine rule confidence with LLM feedback and the Bayesian update formula.
4. Select rules and filter candidates.
5. Apply PPS ordering.
6. Ask the LLM to verify the remaining candidates.
7. Compare predictions against the ground truth.

The per-mode metrics are saved to:

```text
output_file/fz_threshold_zero-shot_metrics.json
output_file/fz_threshold_few-shot_metrics.json
output_file/fz_threshold_self-consistency_metrics.json
```

## Reporting Notes

- For datasets where some model-prompt settings reach very strong performance
  such as 100% F1, the reported values may be averaged over additional
  independent runs beyond the three primary runs. This reduces the effect of
  run-to-run variance from LLM sampling and rule optimization in near-perfect
  regimes.
- For Table 3, `NGDDF` means **No GDD Filtering**. Since disabling GDD filtering
  can produce an extremely large candidate pool, the NGDDF setting is evaluated
  by averaging results over sampled candidate subsets. This keeps the comparison
  computationally feasible while highlighting the practical usefulness of GDD
  filtering.


## Reproduction Note: Cost-aware Sampling on Large Datasets

Due to the high cost of GPT-4 API calls, the LLM-based evaluation on large datasets in the paper was conducted with a cost-aware sampling strategy.

Specifically, for datasets with more than 1,000 ground-truth matching pairs, candidate pairs were sampled after GDD-based filtering, and the same graph-aware prompting strategy was applied to the sampled candidates. Fixed random seeds were used to make the sampling process reproducible, and the reported results were averaged over three runs.

This sampling strategy only affects the number of LLM API calls on large datasets. The GDD filtering, graph construction, rule selection, graph-aware prompt design, and evaluation procedure remain unchanged.
