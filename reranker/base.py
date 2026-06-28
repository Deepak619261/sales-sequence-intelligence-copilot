import abc
from typing import List, Tuple
from core.adapters.base import Chunk

class BaseReranker(abc.ABC):
    """
    Interface for document/chunk rerankers.
    Re-scores and re-ranks retrieved chunks using high-precision models.
    """
    @abc.abstractmethod
    def rerank(
        self, 
        query: str, 
        chunks: List[Chunk]
    ) -> List[Tuple[Chunk, float]]:
        """
        Reranks a list of candidate chunks against the search query.
        Returns a sorted list of (Chunk, score) tuples.
        """
        pass
