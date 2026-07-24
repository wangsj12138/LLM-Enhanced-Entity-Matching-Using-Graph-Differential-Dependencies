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

Due to the high cost of GPT-4 API calls, to ensure efficient evaluation, candidate blocks with excessive candidate pairs are down-sampled to control comparison costs. Conversely, candidate blocks with fewer candidate pairs are up-sampled for subsequent matching.
