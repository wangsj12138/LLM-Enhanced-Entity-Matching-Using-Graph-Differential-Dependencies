# LLM-Enhanced-Entity-Matching-Using-Graph-Differential-Dependencies
![Pipeline](./pipeline.svg)
## Datasets

The project supports two types of benchmark datasets for entity resolution:

1. **Graph ER Benchmark Datasets:**
   - [LINQS Datasets](https://linqs.org/datasets/)
   - [Neo4j Sandbox](https://neo4j.com/sandbox/)

2. **Relational ER Benchmark Datasets:**
   - [Fodors-Zagat, DBLP-ACM, Amazon-Google Datasets](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)

## Data Conversion for Entity Resolution 
Q:How to Use FastER on Relational Datasets or Run graph datasets on relational datasets-based baselines?

A:https://anonymous.4open.science/r/data-conversion-E737      
This project provides tools for converting between relational data types and graph data types for entity resolution (ER). The goal is to enable efficient integration and processing of data from various sources, facilitating graph-based ER tasks.



## **Rule Mining**
The rule mining feature in FastER is based on the definitions and processes described in the following papers:
1. **Discovering Graph Differential Dependencies**  
2. **Certus: An Effective Entity Resolution Approach with Graph Differential Dependencies (GDDs)**  

For a deeper understanding of rule definitions and mining processes, users are encouraged to read these papers.



## GAPLink FZ run

Configure `.env` first:

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

Import the bundled Fodors-Zagats graph into Neo4j:

```bash
.venv/bin/python neo4j_setup.py --dataset fz
```

Run the three FZ prompt settings:

```bash
.venv/bin/python run.py --dataset fz --mode zero-shot --workers 8
.venv/bin/python run.py --dataset fz --mode few-shot --workers 8
.venv/bin/python run.py --dataset fz --mode self-consistency --workers 8
```

To ignore cached LLM outputs and call the API again, add `--no-cache`:

```bash
.venv/bin/python run.py --dataset fz --mode zero-shot --workers 8 --no-cache
.venv/bin/python run.py --dataset fz --mode few-shot --workers 8 --no-cache
.venv/bin/python run.py --dataset fz --mode self-consistency --workers 8 --no-cache
```

Outputs are written to:

- `output_file/fz_threshold_zero-shot_metrics.json`
- `output_file/fz_threshold_few-shot_metrics.json`
- `output_file/fz_threshold_self-consistency_metrics.json`

Each run follows the pipeline:

1. Load the graph-backed Fodors-Zagats candidates from Neo4j.
2. Generate threshold-style GDD rules.
3. Refine rule confidence with LLM feedback and the Bayesian update formula.
4. Select rules and filter candidates.
5. Apply PPS ordering.
6. Ask the LLM to verify the remaining candidates.
7. Compare predictions against the ground truth.

## Reporting Notes

- For some model-prompt settings with very high performance, such as near or at 100% F1, the reported values may be averaged over additional independent runs beyond the three primary runs. This helps reduce the effect of run-to-run variance caused by LLM sampling and rule optimization in near-perfect performance regimes.

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



