# LLM-Enhanced-Entity-Matching-Using-Graph-Differential-Dependencies
![Pipeline](./pipeline.svg)
## Data Conversion for Entity Resolution 
Q:How to Use FastER on Relational Datasets or Run graph datasets on relational datasets-based baselines?

A:https://anonymous.4open.science/r/data-conversion-E737      
This project provides tools for converting between relational data types and graph data types for entity resolution (ER). The goal is to enable efficient integration and processing of data from various sources, facilitating graph-based ER tasks.



## **Rule Mining**
The rule mining feature in FastER is based on the definitions and processes described in the following papers:
1. **Discovering Graph Differential Dependencies**  
2. **Certus: An Effective Entity Resolution Approach with Graph Differential Dependencies (GDDs)**  

For a deeper understanding of rule definitions and mining processes, users are encouraged to read these papers.



## Reporting Notes

- For some model-prompt settings with very high performance, such as near or at 100% F1（or some orther situation）, the reported values may be averaged over additional independent runs beyond the three primary runs. This helps reduce the effect of run-to-run variance caused by LLM sampling and rule optimization in near-perfect performance regimes.

## Cost-aware Sampling on Large Datasets

Due to the high cost of GPT-4 API calls, the LLM-based evaluation on large
datasets was conducted with a cost-aware sampling protocol. This implementation
detail is documented here to make the repository results reproducible.

For datasets with more than 950 candidate pairs after GDD-based filtering, the
runner samples up to ～950 candidates before LLM verification. Sampling is
stratified by the ground-truth label among the filtered candidates, so the sampled
set preserves the positive/negative ratio used for precision, recall, and F1
calculation. The default seed is fixed at `42`.

This sampling protocol only reduces the number of LLM API calls on large
datasets. It does not change GDD filtering, graph construction, rule selection,
the graph-aware prompt template, or the evaluation metric. Disable it with
`--disable-sampling` for full-candidate evaluation when the API budget allows.


