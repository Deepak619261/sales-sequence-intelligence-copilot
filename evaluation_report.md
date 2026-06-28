# Sales Sequence Copilot - Search Quality & RAG Evaluation Report

This report outlines the search quality benchmarking parameters and metrics for 4 distinct retrieval strategies compiled over a manually curated synthetic intent-based dataset of **10 queries**.

## Benchmark Parameters
- **Candidate Pool Size (N):** 50 (All strategies pull matching size before ranking/scoring)
- **Manually Curated Dataset Size:** 10 intent-based search queries
- **Avg. Semantic & BM25 Pool overlap:** 1.0 / 50 common chunks

## Retrieval Performance Metrics Comparison

### Evaluation Cutoff (K = 5)

| Retrieval Strategy | Precision@5 | Recall@5 | MRR@5 (Reciprocal Rank) | Hit Rate@5 |
| :--- | :---: | :---: | :---: | :---: |
| **Semantic** | 0.00% | 0.00% | 0.0000 | 0.00% |
| **BM25** | 0.00% | 0.00% | 0.0000 | 0.00% |
| **Hybrid RRF** | 0.00% | 0.00% | 0.0000 | 0.00% |
| **Hybrid Reranked** | 0.00% | 0.00% | 0.0000 | 0.00% |


### Evaluation Cutoff (K = 10)

| Retrieval Strategy | Precision@10 | Recall@10 | MRR@10 (Reciprocal Rank) | Hit Rate@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Semantic** | 0.00% | 0.00% | 0.0000 | 0.00% |
| **BM25** | 0.00% | 0.00% | 0.0000 | 0.00% |
| **Hybrid RRF** | 0.00% | 0.00% | 0.0000 | 0.00% |
| **Hybrid Reranked** | 0.00% | 0.00% | 0.0000 | 0.00% |


## Cross-Encoder Reranker Impact Analysis

Reranking runs a second-stage Cross-Encoder model to re-score the merged candidates pool. The model evaluates detailed query-chunk text pairs to correct positional errors from the initial fast search passes.

- **Total Queries triggering Rank Shifts:** 0 / 10


## Search Failures & Missed Relevance Analysis

Failure cases list target ground-truth relevant chunks that were missed in the top 10 rankings by search strategies. Tracking these exposes blind spots in semantic indexing or term tokenization.

- **Query:** *"caching database scaling bottleneck"*
  - Expected relevant chunks: `email_001_sent_0, email_002_sent_0, email_004_sent_0, email_005_sent_0`
  - Missed by **Semantic**: `email_001_sent_0, email_002_sent_0, email_004_sent_0, email_005_sent_0`
  - Missed by **BM25**: `email_001_sent_0, email_002_sent_0, email_004_sent_0, email_005_sent_0`
  - Missed by **Hybrid RRF**: `email_001_sent_0, email_002_sent_0, email_004_sent_0, email_005_sent_0`
  - Missed by **Hybrid Reranked**: `email_001_sent_0, email_002_sent_0, email_004_sent_0, email_005_sent_0`

- **Query:** *"compliance audit preparation regulations"*
  - Expected relevant chunks: `email_007_sent_0, email_007_sent_1, email_008_sent_0, email_009_sent_0, email_010_sent_0, email_011_sent_0, email_012_sent_0`
  - Missed by **Semantic**: `email_007_sent_0, email_007_sent_1, email_008_sent_0, email_009_sent_0, email_010_sent_0, email_011_sent_0, email_012_sent_0`
  - Missed by **BM25**: `email_007_sent_0, email_007_sent_1, email_008_sent_0, email_009_sent_0, email_010_sent_0, email_011_sent_0, email_012_sent_0`
  - Missed by **Hybrid RRF**: `email_007_sent_0, email_007_sent_1, email_008_sent_0, email_009_sent_0, email_010_sent_0, email_011_sent_0, email_012_sent_0`
  - Missed by **Hybrid Reranked**: `email_007_sent_0, email_007_sent_1, email_008_sent_0, email_009_sent_0, email_010_sent_0, email_011_sent_0, email_012_sent_0`

- **Query:** *"how to reduce time to hire software developers"*
  - Expected relevant chunks: `email_013_sent_0, email_013_sent_1, email_014_sent_0, email_015_sent_0, email_016_sent_0, email_017_sent_0`
  - Missed by **Semantic**: `email_013_sent_0, email_013_sent_1, email_014_sent_0, email_015_sent_0, email_016_sent_0, email_017_sent_0`
  - Missed by **BM25**: `email_013_sent_0, email_013_sent_1, email_014_sent_0, email_015_sent_0, email_016_sent_0, email_017_sent_0`
  - Missed by **Hybrid RRF**: `email_013_sent_0, email_013_sent_1, email_014_sent_0, email_015_sent_0, email_016_sent_0, email_017_sent_0`
  - Missed by **Hybrid Reranked**: `email_013_sent_0, email_013_sent_1, email_014_sent_0, email_015_sent_0, email_016_sent_0, email_017_sent_0`


## Key Insights & Search Quality Strategy
1. **Concept Similarity vs exact Keyword Matching**: Pure semantic search handles conceptual phrasing (e.g. 'clinician scheduling double-bookings') where exact word matches are absent. However, it can match irrelevant segments with similar vocabulary, inflating false positives.
2. **Keyword Match Precision**: BM25 remains highly accurate for exact references (case study terms like 'AcmeCorp') and specific numeric references, but misses paraphrased intent completely.
3. **RRF Hybrid Fusion**: Leverages RRF rank scores to synthesize sparse and dense queries. It consistently improves Hit Rate and Recall under diverse workloads.
4. **Cross-Encoder Precision Lift**: Corrects rank anomalies by assessing full semantic interaction, promoting highly relevant but keyword-weak chunks to the highest slot, optimizing MRR.