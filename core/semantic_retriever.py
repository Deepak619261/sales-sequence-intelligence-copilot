import logging
from typing import List, Dict, Any, Tuple
from core.adapters.base import Chunk
from embeddings.base import BaseEmbedder
from core.vectorstore.base import BaseVectorStore
from core.base import BaseRetriever

logger = logging.getLogger(__name__)

class SemanticRetriever(BaseRetriever):
    """
    Retriever that uses dense vector embeddings to perform similarity searches.
    Filters out results with scores below the configured threshold.
    """
    def __init__(
        self, 
        embedder: BaseEmbedder, 
        vector_store: BaseVectorStore, 
        score_threshold: float = 0.0
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.score_threshold = score_threshold

    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Chunk, float]]:
        logger.info(f"Retrieving top {k} semantic matches for query: '{query}'")
        
        # 1. Embed the search query
        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        # 2. Query the vector store
        raw_results = self.vector_store.similarity_search(
            query_embedding=query_vector, 
            k=k, 
            filter_dict=filter_dict
        )
        
        # 3. Apply score threshold filtering
        filtered_results = []
        for chunk, score in raw_results:
            if score >= self.score_threshold:
                filtered_results.append((chunk, score))
            else:
                logger.debug(f"Discarding chunk {chunk.chunk_id} with score {score:.4f} below threshold {self.score_threshold}")
                
        logger.info(f"Semantic search returned {len(filtered_results)} hits (discarded {len(raw_results) - len(filtered_results)} below threshold).")
        return filtered_results
