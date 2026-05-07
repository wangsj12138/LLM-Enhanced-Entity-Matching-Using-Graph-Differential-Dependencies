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
.venv/bin/python fz_gaplink_threshold_rules.py --mode zero-shot --sample-size 1000 --sample-seed 42
.venv/bin/python fz_gaplink_threshold_rules.py --mode few-shot --sample-size 1000 --sample-seed 42
.venv/bin/python fz_gaplink_threshold_rules.py --mode self-consistency --sample-size 1000 --sample-seed 42
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

## Reproduction Note: Cost-aware Sampling on Large Datasets

Due to the high cost of GPT-4 API calls, the LLM-based evaluation on large
datasets was conducted with a cost-aware sampling protocol. This implementation
detail is documented here to make the repository results reproducible.

For datasets with more than 1,000 candidate pairs after GDD-based filtering, the
runner samples up to 1,000 candidates before LLM verification. Sampling is
stratified by the ground-truth label among the filtered candidates, so the sampled
set preserves the positive/negative ratio used for precision, recall, and F1
calculation. The default seed is fixed at `42`, and reported results are averaged
over three runs.

This sampling protocol only reduces the number of LLM API calls on large
datasets. It does not change GDD filtering, graph construction, rule selection,
the graph-aware prompt template, or the evaluation metric. Disable it with
`--disable-sampling` for full-candidate evaluation when the API budget allows.

## API-cost Controls

- Stage 1 and Stage 2 LLM calls are cached in JSONL files under `output_file/`.
  Re-running the same model, prompt mode, candidate pair, vote count, and selected
  rule set reads the cached decision instead of calling the API again.
- Self-consistency uses three votes. When a compatible few-shot decision for the
  same candidate pair is already cached, it is reused as the first
  self-consistency vote and only two additional votes are requested.
- Prompts use compact JSON payloads and symbolic graph patterns to reduce token
  use while preserving the attributes, GDD rules, and graph evidence needed by
  the matcher.

## Legacy Scripts

Early notebooks and one-off scripts from the original exploratory implementation
are kept under `legacy/original_scripts/`. They are preserved for reference but
are not required by the current reproduction workflow.
