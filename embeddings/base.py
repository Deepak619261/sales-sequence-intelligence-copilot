import abc
from typing import List

class BaseEmbedder(abc.ABC):
    """
    Interface for text embedding generators.
    """
    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single user query.
        """
        pass

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of document texts/chunks.
        """
        pass
