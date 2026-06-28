import abc
from typing import List
from core.adapters.base import SalesEmail, Chunk

class BaseChunker(abc.ABC):
    """
    Interface for data chunkers. Splits a SalesEmail into multiple Chunks,
    preserving metadata for RAG traceability and filtering.
    """
    @abc.abstractmethod
    def chunk(self, email: SalesEmail) -> List[Chunk]:
        """
        Takes a SalesEmail and returns a list of Chunk objects.
        """
        pass
