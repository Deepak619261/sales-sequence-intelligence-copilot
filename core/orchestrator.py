import logging
import time
from typing import List, Dict, Any, Tuple
from config import get_config
from core.adapters.base import Chunk
from embeddings.factory import get_embedder
from core.vectorstore.factory import get_vector_store
from core.factory import (
    get_semantic_retriever,
    get_bm25_retriever,
    get_hybrid_retriever
)
from reranker.factory import get_reranker
from core.llm.prompt_builder import PromptBuilder
from core.llm.generator import LLMGenerator
from utils.logging import time_it, request_metrics

logger = logging.getLogger(__name__)

class QueryOrchestrator:
    """
    Coordinates RAG execution: retrieves, fuses, reranks context,
    tracks latencies and rank shifts, and generates grounded responses.
    """
    def __init__(self, config_dict: Dict[str, Any] = None):
        # Allow passing custom overrides, default to central configuration
        self.config_dict = config_dict or get_config().to_dict()
        
        # Load sub-components
        self.embedder = get_embedder(self.config_dict)
        self.vector_store = get_vector_store(self.config_dict)
        
        self.semantic_ret = get_semantic_retriever(self.config_dict, self.embedder, self.vector_store)
        self.bm25_ret = get_bm25_retriever(self.config_dict, self.vector_store)
        self.hybrid_ret = get_hybrid_retriever(self.config_dict, self.semantic_ret, self.bm25_ret)
        
        self.reranker = get_reranker(self.config_dict)
        self.prompt_builder = PromptBuilder()
        self.generator = LLMGenerator(self.config_dict)

    @time_it("total_pipeline")
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Runs the end-to-end RAG query pipeline.
        """
        # Ensure a clean dict for metrics accumulation
        request_metrics.set({
            "query_text": query_text,
            "timestamp": time.time()
        })
        
        # 1. Fetch retrieval candidates (Hybrid RRF / Weighted)
        candidate_k = self.config_dict.get("retrieval", {}).get("hybrid_k", 20)
        fused_candidates = self._hybrid_retrieve(query_text, k=candidate_k)
        
        # 2. Slice to reranker input limit (top_n input)
        reranker_input_limit = self.config_dict.get("reranker", {}).get("top_n", 10)
        chunks_to_rerank = [chunk for chunk, _ in fused_candidates[:reranker_input_limit]]
        
        # 3. Rerank the chunks
        reranked_results = self._rerank(query_text, chunks_to_rerank)
        final_context_chunks = [chunk for chunk, _ in reranked_results]
        
        # Log top chunk IDs
        metrics = request_metrics.get()
        metrics["top_retrieved_chunk_ids"] = [chunk.chunk_id for chunk in final_context_chunks]
        request_metrics.set(metrics)
        
        # 4. Get all chunks for aggregation-aware prompting
        all_chunks = self.vector_store.get_all_chunks()
        
        # 5. Build prompt (passes all_chunks for global summary if aggregation query detected)
        prompt = self.prompt_builder.build_prompt(query_text, final_context_chunks, all_chunks=all_chunks)
        
        # 6. Generate Response
        response = self._generate(prompt)
        return response

    def retrieve_debug(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieves raw chunks from Semantic, BM25, and Hybrid pipelines for debugging.
        """
        semantic_results = self._semantic_retrieve(query_text, k=k)
        bm25_results = self._bm25_retrieve(query_text, k=k)
        hybrid_results = self._hybrid_retrieve(query_text, k=k)
        
        return {
            "query": query_text,
            "semantic": [{"chunk_id": c.chunk_id, "content": c.content, "score": float(s)} for c, s in semantic_results],
            "bm25": [{"chunk_id": c.chunk_id, "content": c.content, "score": float(s)} for c, s in bm25_results],
            "hybrid": [{"chunk_id": c.chunk_id, "content": c.content, "score": float(s)} for c, s in hybrid_results]
        }

    @time_it("semantic_retrieval")
    def _semantic_retrieve(self, query_text: str, k: int) -> List[Tuple[Chunk, float]]:
        return self.semantic_ret.retrieve(query_text, k=k)

    @time_it("bm25_retrieval")
    def _bm25_retrieve(self, query_text: str, k: int) -> List[Tuple[Chunk, float]]:
        return self.bm25_ret.retrieve(query_text, k=k)

    @time_it("hybrid_fusion")
    def _hybrid_retrieve(self, query_text: str, k: int) -> List[Tuple[Chunk, float]]:
        return self.hybrid_ret.retrieve(query_text, k=k)

    @time_it("reranking")
    def _rerank(self, query_text: str, chunks: List[Chunk]) -> List[Tuple[Chunk, float]]:
        reranked_results = self.reranker.rerank(query_text, chunks)
        
        # Calculate rank shifts
        rank_shifts = []
        hybrid_ids = [c.chunk_id for c in chunks]
        for post_rank, (chunk, _) in enumerate(reranked_results, start=1):
            cid = chunk.chunk_id
            if cid in hybrid_ids:
                pre_rank = hybrid_ids.index(cid) + 1
                shift = pre_rank - post_rank
                if shift != 0:
                    rank_shifts.append({
                        "chunk_id": cid,
                        "pre_rank": pre_rank,
                        "post_rank": post_rank,
                        "shift": shift
                    })
                    
        metrics = request_metrics.get()
        metrics["rank_shifts"] = rank_shifts
        request_metrics.set(metrics)
        
        return reranked_results

    @time_it("llm_generation")
    def _generate(self, prompt: str) -> Dict[str, Any]:
        return self.generator.generate(prompt)
