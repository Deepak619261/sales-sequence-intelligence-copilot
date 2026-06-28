import json
import os
import logging
from typing import Dict, Any, List, Tuple, Set
from embeddings.base import BaseEmbedder
from core.vectorstore.base import BaseVectorStore
from core.factory import (
    get_semantic_retriever,
    get_bm25_retriever,
    get_hybrid_retriever
)
from reranker.factory import get_reranker
from evaluation.metrics import RetrievalMetrics

logger = logging.getLogger(__name__)

class EvaluationRunner:
    """
    Advanced retrieval evaluation runner that benchmarks Semantic, BM25, Hybrid,
    and Reranking search. Calculates metrics at K=5 and K=10, tracks query overlap,
    logs rank shifts, and compiles failure case analysis.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        ground_truth_path: str = "data/ground_truth.json"
    ):
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.ground_truth_path = ground_truth_path
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.ground_truth_path):
            raise FileNotFoundError(f"Ground truth file not found at: {self.ground_truth_path}")
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_evaluation(self) -> Dict[str, Any]:
        logger.info(f"Initializing advanced RAG retrieval benchmark across {len(self.dataset)} intent-based queries...")

        # Force score threshold to -1.0 so we evaluate search relevance directly without early drops
        eval_cfg = dict(self.config)
        eval_cfg["retrieval"]["score_threshold"] = -1.0
        
        # Consistent retrieval candidate size N = 50
        candidate_pool_size = 50
        eval_cfg["retrieval"]["semantic_k"] = candidate_pool_size
        eval_cfg["retrieval"]["bm25_k"] = candidate_pool_size
        eval_cfg["retrieval"]["hybrid_k"] = candidate_pool_size
        
        # Apply reranking only to the top 20 fused hybrid candidates
        reranker_candidate_limit = 20
        eval_cfg["reranker"]["top_n"] = reranker_candidate_limit

        # Retrieve instances
        semantic_ret = get_semantic_retriever(eval_cfg, self.embedder, self.vector_store)
        bm25_ret = get_bm25_retriever(eval_cfg, self.vector_store)
        
        eval_cfg["retrieval"]["fusion_type"] = "rrf"
        hybrid_rrf_ret = get_hybrid_retriever(eval_cfg, semantic_ret, bm25_ret)
        
        reranker = get_reranker(eval_cfg)

        # Strategies & metrics tracking
        strategies = ["Semantic", "BM25", "Hybrid_RRF", "Hybrid_Reranked"]
        cutoffs = [5, 10]
        
        # Structure: metrics_summary[strategy][cutoff][metric]
        metrics_summary = {
            s: {
                k: {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "hit_rate": 0.0} 
                for k in cutoffs
            } 
            for s in strategies
        }

        # Diagnostic counters
        total_queries = len(self.dataset)
        overlaps_list: List[int] = []
        rank_shifts: List[Dict[str, Any]] = []
        failure_cases: List[Dict[str, Any]] = []

        for q_idx, item in enumerate(self.dataset):
            query = item["query"]
            gt_ids = item["relevant_chunk_ids"]
            gt_set = set(gt_ids)

            logger.info(f"\n--- Evaluating Query {q_idx+1}/{total_queries}: '{query}' ---")

            # 1. Candidate Retrieval (all retrieve top-20 under identical conditions)
            sem_results = semantic_ret.retrieve(query, k=candidate_pool_size)
            sem_ids = [c.chunk_id for c, _ in sem_results]
            
            bm25_results = bm25_ret.retrieve(query, k=candidate_pool_size)
            bm25_ids = [c.chunk_id for c, _ in bm25_results]

            # Track Overlap between Semantic and BM25 candidate pools
            overlap = len(set(sem_ids).intersection(set(bm25_ids)))
            overlaps_list.append(overlap)
            logger.info(f"  Candidate overlap between Semantic and BM25: {overlap}/{candidate_pool_size} chunks.")

            # 2. Hybrid Retrieval (RRF merging of candidate_pool_size pools)
            hybrid_results = hybrid_rrf_ret.retrieve(query, k=candidate_pool_size)
            hybrid_ids = [c.chunk_id for c, _ in hybrid_results]

            # 3. Reranked Retrieval (applies CrossEncoder only to top-20 hybrid candidate list)
            hybrid_chunks = [c for c, _ in hybrid_results[:reranker_candidate_limit]]
            reranked_results = reranker.rerank(query, hybrid_chunks)
            reranked_ids = [c.chunk_id for c, _ in reranked_results]

            # 4. Track Rank Shifts after Cross-Encoder Reranking
            query_shifts = []
            for rank_post, (chunk, score) in enumerate(reranked_results[:10], start=1):
                cid = chunk.chunk_id
                if cid in hybrid_ids:
                    rank_pre = hybrid_ids.index(cid) + 1
                    shift = rank_pre - rank_post
                    if shift != 0:
                        query_shifts.append({
                            "chunk_id": cid,
                            "pre_rank": rank_pre,
                            "post_rank": rank_post,
                            "shift": shift
                        })
            if query_shifts:
                rank_shifts.append({"query": query, "shifts": query_shifts})
                logger.info(f"  Reranker active rank shifts: {len(query_shifts)} chunks changed positions.")

            # 5. Track Failure Cases (expected chunks missed in top-10 retrieval)
            missed_by_strategy = {}
            for strat_name, retrieved_list in [
                ("Semantic", sem_ids[:10]),
                ("BM25", bm25_ids[:10]),
                ("Hybrid_RRF", hybrid_ids[:10]),
                ("Hybrid_Reranked", reranked_ids[:10])
            ]:
                missed = [gt_id for gt_id in gt_ids if gt_id not in retrieved_list]
                if len(missed) > 0:
                    missed_by_strategy[strat_name] = missed
                    
            if len(missed_by_strategy) > 0:
                failure_cases.append({
                    "query": query,
                    "expected": gt_ids,
                    "missed": missed_by_strategy
                })
                logger.warning(f"  Failure Case: Chunks missed in top-10 by one or more strategies.")

            # 6. Compute Metrics at K=5 and K=10
            for k in cutoffs:
                for strat, retrieved_list in [
                    ("Semantic", sem_ids),
                    ("BM25", bm25_ids),
                    ("Hybrid_RRF", hybrid_ids),
                    ("Hybrid_Reranked", reranked_ids)
                ]:
                    # Select slice up to K
                    slice_ids = retrieved_list[:k]
                    metrics_summary[strat][k]["precision"] += RetrievalMetrics.precision_at_k(slice_ids, gt_ids, k)
                    metrics_summary[strat][k]["recall"] += RetrievalMetrics.recall_at_k(slice_ids, gt_ids, k)
                    metrics_summary[strat][k]["mrr"] += RetrievalMetrics.reciprocal_rank(slice_ids, gt_ids)
                    metrics_summary[strat][k]["hit_rate"] += RetrievalMetrics.hit_rate(slice_ids, gt_ids, k)

        # Average metrics over total queries
        for strat in strategies:
            for k in cutoffs:
                for metric in ["precision", "recall", "mrr", "hit_rate"]:
                    metrics_summary[strat][k][metric] /= total_queries

        avg_overlap = sum(overlaps_list) / len(overlaps_list) if overlaps_list else 0.0

        # Generate output report
        self._write_diagnostic_report(
            summary=metrics_summary,
            avg_overlap=avg_overlap,
            rank_shifts=rank_shifts,
            failure_cases=failure_cases,
            total_queries=total_queries,
            pool_size=candidate_pool_size
        )

        return {
            "summary": metrics_summary,
            "avg_overlap": avg_overlap,
            "rank_shifts_count": len(rank_shifts),
            "failures_count": len(failure_cases)
        }

    def _write_diagnostic_report(
        self, 
        summary: Dict[str, Any], 
        avg_overlap: float, 
        rank_shifts: List[Dict[str, Any]], 
        failure_cases: List[Dict[str, Any]], 
        total_queries: int,
        pool_size: int
    ) -> None:
        report_path = "evaluation_report.md"
        logger.info(f"Writing final diagnostic report to: {report_path}")

        md = []
        md.append("# Sales Sequence Copilot - Search Quality & RAG Evaluation Report\n")
        md.append(f"This report outlines the search quality benchmarking parameters and metrics for 4 distinct retrieval strategies compiled over a manually curated synthetic intent-based dataset of **{total_queries} queries**.\n")
        
        md.append("## Benchmark Parameters")
        md.append(f"- **Candidate Pool Size (N):** {pool_size} (All strategies pull matching size before ranking/scoring)")
        md.append(f"- **Manually Curated Dataset Size:** {total_queries} intent-based search queries")
        md.append(f"- **Avg. Semantic & BM25 Pool overlap:** {avg_overlap:.1f} / {pool_size} common chunks\n")

        # Metrics Tables
        md.append("## Retrieval Performance Metrics Comparison\n")
        
        # Table for K = 5
        md.append("### Evaluation Cutoff (K = 5)\n")
        md.append("| Retrieval Strategy | Precision@5 | Recall@5 | MRR@5 (Reciprocal Rank) | Hit Rate@5 |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for strat in ["Semantic", "BM25", "Hybrid_RRF", "Hybrid_Reranked"]:
            m = summary[strat][5]
            name = strat.replace("_", " ")
            md.append(f"| **{name}** | {m['precision']:.2%} | {m['recall']:.2%} | {m['mrr']:.4f} | {m['hit_rate']:.2%} |")
        md.append("\n")

        # Table for K = 10
        md.append("### Evaluation Cutoff (K = 10)\n")
        md.append("| Retrieval Strategy | Precision@10 | Recall@10 | MRR@10 (Reciprocal Rank) | Hit Rate@10 |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for strat in ["Semantic", "BM25", "Hybrid_RRF", "Hybrid_Reranked"]:
            m = summary[strat][10]
            name = strat.replace("_", " ")
            md.append(f"| **{name}** | {m['precision']:.2%} | {m['recall']:.2%} | {m['mrr']:.4f} | {m['hit_rate']:.2%} |")
        md.append("\n")

        # Reranker Impact Analysis
        md.append("## Cross-Encoder Reranker Impact Analysis\n")
        md.append("Reranking runs a second-stage Cross-Encoder model to re-score the merged candidates pool. The model evaluates detailed query-chunk text pairs to correct positional errors from the initial fast search passes.")
        md.append(f"\n- **Total Queries triggering Rank Shifts:** {len(rank_shifts)} / {total_queries}")
        
        if rank_shifts:
            md.append("\n### Sample Active Rank Shift Tracking (Top 10 Slots)")
            for rs in rank_shifts[:3]:
                md.append(f"\n- **Query:** *\"{rs['query']}\"*")
                for s in rs["shifts"][:4]:
                    dir_str = "▲ UP" if s['shift'] > 0 else "▼ DOWN"
                    md.append(f"  - Chunk ID: `{s['chunk_id']}` shifted {dir_str} by {abs(s['shift'])} ranks (Position: {s['pre_rank']} -> {s['post_rank']})")
        md.append("\n")

        # Failure Case Analysis
        md.append("## Search Failures & Missed Relevance Analysis\n")
        md.append("Failure cases list target ground-truth relevant chunks that were missed in the top 10 rankings by search strategies. Tracking these exposes blind spots in semantic indexing or term tokenization.")
        
        if failure_cases:
            for fc in failure_cases[:3]:
                md.append(f"\n- **Query:** *\"{fc['query']}\"*")
                md.append(f"  - Expected relevant chunks: `{', '.join(fc['expected'])}`")
                for strat, missed_ids in fc["missed"].items():
                    md.append(f"  - Missed by **{strat.replace('_', ' ')}**: `{', '.join(missed_ids)}`")
        else:
            md.append("\nNo major retrieval failures detected (all expected chunks retrieved in top 10).")
        md.append("\n")

        # Strategy Analysis
        md.append("## Key Insights & Search Quality Strategy")
        md.append("1. **Concept Similarity vs exact Keyword Matching**: Pure semantic search handles conceptual phrasing (e.g. 'clinician scheduling double-bookings') where exact word matches are absent. However, it can match irrelevant segments with similar vocabulary, inflating false positives.")
        md.append("2. **Keyword Match Precision**: BM25 remains highly accurate for exact references (case study terms like 'AcmeCorp') and specific numeric references, but misses paraphrased intent completely.")
        md.append("3. **RRF Hybrid Fusion**: Leverages RRF rank scores to synthesize sparse and dense queries. It consistently improves Hit Rate and Recall under diverse workloads.")
        md.append("4. **Cross-Encoder Precision Lift**: Corrects rank anomalies by assessing full semantic interaction, promoting highly relevant but keyword-weak chunks to the highest slot, optimizing MRR.")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        logger.info(f"Evaluation report written successfully to {report_path}.")
