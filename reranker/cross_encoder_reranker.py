import re
import logging
from typing import List, Tuple
from core.adapters.base import Chunk
from reranker.base import BaseReranker

logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseReranker):
    """
    Reranker that uses a local SentenceTransformers CrossEncoder model.
    Falls back to pass-through mode if sentence-transformers is not installed,
    or runs a deterministic simulation if simulate_fallback is enabled.
    """
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n: int = 3, 
        enabled: bool = True,
        simulate_fallback: bool = False
    ):
        self.model_name = model_name
        self.top_n = top_n
        self.enabled = enabled
        self.simulate_fallback = simulate_fallback
        self.model = None
        self.has_cross_encoder = False
        
        if not self.enabled:
            logger.info("Cross-Encoder Reranker is disabled in config. Operating in pass-through mode.")
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading local Cross-Encoder model '{self.model_name}'...")
            self.model = CrossEncoder(self.model_name)
            self.has_cross_encoder = True
            logger.info("Cross-Encoder model loaded successfully.")
        except ImportError:
            logger.warning(
                f"package 'sentence-transformers' is not installed. "
                f"CrossEncoderReranker will operate in fallback mode (simulate_fallback={self.simulate_fallback})."
            )
            self.has_cross_encoder = False

    def rerank(
        self, 
        query: str, 
        chunks: List[Chunk]
    ) -> List[Tuple[Chunk, float]]:
        if not chunks:
            return []

        # 1. Fallback mode if library is missing or disabled
        if not self.has_cross_encoder or not self.model:
            if self.simulate_fallback:
                logger.info("Executing mock cross-encoder simulation (calculating term overlap scores).")
                results = []
                query_words = set(re.findall(r'\b\w+\b', query.lower()))
                for chunk in chunks:
                    chunk_words = set(re.findall(r'\b\w+\b', chunk.content.lower()))
                    # Basic Jaccard overlap heuristic to score chunks differently
                    intersection = query_words.intersection(chunk_words)
                    score = float(len(intersection)) + (len(chunk.content) * 0.0001)
                    results.append((chunk, score))
                # Sort descending based on calculated mock score
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:self.top_n]
            else:
                logger.info("Executing pass-through reranking (no scoring, order preserved).")
                results = []
                for idx, chunk in enumerate(chunks):
                    dummy_score = 1.0 / (idx + 1.0)
                    results.append((chunk, dummy_score))
                return results[:self.top_n]

        # 2. Local Cross-Encoder model scoring
        logger.info(f"Reranking {len(chunks)} candidate chunks using Cross-Encoder model...")
        pairs = [(query, chunk.content) for chunk in chunks]
        
        try:
            scores = self.model.predict(pairs)
            scored_chunks = list(zip(chunks, [float(s) for s in scores]))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            logger.info("Cross-Encoder reranking completed successfully.")
            return scored_chunks[:self.top_n]
        except Exception as e:
            logger.error(f"Error occurred during Cross-Encoder prediction: {e}. Falling back to input order.")
            results = []
            for idx, chunk in enumerate(chunks):
                results.append((chunk, 0.0))
            return results[:self.top_n]
