import logging
from typing import List, Dict, Any, Tuple
from core.adapters.base import Chunk
from core.base import BaseRetriever
from core.semantic_retriever import SemanticRetriever
from core.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    Combines dense semantic search and sparse BM25 keyword search.
    Supports Reciprocal Rank Fusion (RRF) and Weighted Sum Fusion strategies.
    """
    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        fusion_type: str = "rrf",
        rrf_k: int = 60,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion_type = fusion_type.lower()
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Chunk, float]]:
        logger.info(f"Initiating Hybrid Search for query: '{query}' (Fusion Type: {self.fusion_type})")
        
        # 1. Fetch results from individual retrievers
        # Retrieve twice the number of target items to ensure broad coverage for fusion
        retrieve_k = k * 2
        
        semantic_results = self.semantic_retriever.retrieve(query, k=retrieve_k, filter_dict=filter_dict)
        bm25_results = self.bm25_retriever.retrieve(query, k=retrieve_k, filter_dict=filter_dict)
        
        if not semantic_results and not bm25_results:
            logger.warning("Both semantic and BM25 search returned empty results.")
            return []

        # 2. Fuse the results
        if self.fusion_type == "rrf":
            fused_results = self._reciprocal_rank_fusion(semantic_results, bm25_results)
        elif self.fusion_type == "weighted":
            fused_results = self._weighted_sum_fusion(semantic_results, bm25_results)
        else:
            logger.warning(f"Unknown fusion type '{self.fusion_type}'. Defaulting to RRF.")
            fused_results = self._reciprocal_rank_fusion(semantic_results, bm25_results)
            
        logger.info(f"Hybrid search merged and scored {len(fused_results)} unique chunks.")
        
        # Sort and return top K
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:k]

    def _reciprocal_rank_fusion(
        self, 
        semantic_res: List[Tuple[Chunk, float]], 
        bm25_res: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """
        Calculates fusion score based on document rank lists.
        Handles duplicate points properly by skipping repeated entries.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        # Parse semantic rankings (1-based index)
        seen_semantic = set()
        semantic_rank = 1
        for chunk, _ in semantic_res:
            chunk_id = chunk.chunk_id
            if chunk_id in seen_semantic:
                continue
            seen_semantic.add(chunk_id)
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + semantic_rank))
            semantic_rank += 1

        # Parse BM25 rankings (1-based index)
        seen_bm25 = set()
        bm25_rank = 1
        for chunk, _ in bm25_res:
            chunk_id = chunk.chunk_id
            if chunk_id in seen_bm25:
                continue
            seen_bm25.add(chunk_id)
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + bm25_rank))
            bm25_rank += 1

        return [(chunk_map[cid], score) for cid, score in rrf_scores.items()]

    def _weighted_sum_fusion(
        self, 
        semantic_res: List[Tuple[Chunk, float]], 
        bm25_res: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """
        Min-max normalizes the raw scores of each list and applies weights.
        Ensures unique scoring by only accumulating the top score of duplicate chunks.
        """
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        # Normalize semantic scores
        normalized_semantic = self._min_max_normalize(semantic_res)
        # Normalize BM25 scores
        normalized_bm25 = self._min_max_normalize(bm25_res)

        # Merge semantic scores (keep first/maximum normalized score)
        seen_semantic = set()
        for chunk, norm_score in normalized_semantic:
            chunk_id = chunk.chunk_id
            if chunk_id in seen_semantic:
                continue
            seen_semantic.add(chunk_id)
            chunk_map[chunk_id] = chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (self.semantic_weight * norm_score)

        # Merge BM25 scores (keep first/maximum normalized score)
        seen_bm25 = set()
        for chunk, norm_score in normalized_bm25:
            chunk_id = chunk.chunk_id
            if chunk_id in seen_bm25:
                continue
            seen_bm25.add(chunk_id)
            chunk_map[chunk_id] = chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (self.bm25_weight * norm_score)

        return [(chunk_map[cid], score) for cid, score in scores.items()]

    def _min_max_normalize(self, results: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        if not results:
            return []
        if len(results) == 1:
            return [(results[0][0], 1.0)]
            
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        diff = max_score - min_score
        
        normalized = []
        for chunk, score in results:
            # Add a small epsilon to avoid division by zero
            norm = (score - min_score) / (diff + 1e-9)
            normalized.append((chunk, norm))
            
        return normalized
