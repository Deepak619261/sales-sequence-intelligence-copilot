import abc
from typing import List, Dict, Any, Tuple
from core.adapters.base import Chunk

class BaseVectorStore(abc.ABC):
    """
    Interface for Vector Databases.
    """
    @abc.abstractmethod
    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Stores chunks and their corresponding embedding vectors.
        """
        pass

    @abc.abstractmethod
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Queries the database using vector similarity.
        Returns a list of (Chunk, score) tuples.
        """
        pass
        
    @abc.abstractmethod
    def clear(self) -> None:
        """
        Clears all items in the vector store collection.
        """
        pass

    @abc.abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        """
        Retrieves all chunks currently stored in the database.
        """
        pass
