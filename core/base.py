import abc
from typing import List, Dict, Any, Tuple
from core.adapters.base import Chunk

class BaseRetriever(abc.ABC):
    """
    Interface for document/chunk retrieval strategies.
    """
    @abc.abstractmethod
    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieves top K chunks matching the query string.
        Returns a list of (Chunk, score) tuples.
        """
        pass
