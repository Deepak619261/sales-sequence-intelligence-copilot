from typing import Dict, Any
from core.base import BaseRetriever
from core.semantic_retriever import SemanticRetriever
from core.bm25_retriever import BM25Retriever
from core.hybrid_retriever import HybridRetriever
from embeddings.base import BaseEmbedder
from core.vectorstore.base import BaseVectorStore

def get_semantic_retriever(
    config: Dict[str, Any], 
    embedder: BaseEmbedder, 
    vector_store: BaseVectorStore
) -> SemanticRetriever:
    """
    Builds and returns a SemanticRetriever configured according to config options.
    """
    ret_cfg = config.get("retrieval", {})
    threshold = ret_cfg.get("score_threshold", 0.0)
    return SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        score_threshold=threshold
    )

def get_bm25_retriever(
    config: Dict[str, Any],
    vector_store: BaseVectorStore
) -> BM25Retriever:
    """
    Builds and returns a BM25Retriever configured according to config options.
    """
    # Okapi BM25 defaults
    return BM25Retriever(
        vector_store=vector_store,
        k1=1.5,
        b=0.75
    )

def get_hybrid_retriever(
    config: Dict[str, Any],
    semantic_retriever: SemanticRetriever,
    bm25_retriever: BM25Retriever
) -> HybridRetriever:
    """
    Builds and returns a HybridRetriever configured according to config options.
    """
    ret_cfg = config.get("retrieval", {})
    return HybridRetriever(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
        fusion_type=ret_cfg.get("fusion_type", "rrf"),
        rrf_k=ret_cfg.get("rrf_k", 60),
        semantic_weight=ret_cfg.get("semantic_weight", 0.7),
        bm25_weight=ret_cfg.get("bm25_weight", 0.3)
    )
