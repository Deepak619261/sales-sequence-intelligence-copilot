import logging
from typing import List
from embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder wrapper for local SentenceTransformers models.
    Produces continuous, learned high-dimensional semantic vectors.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local SentenceTransformer model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer model loaded successfully.")
        except ImportError as e:
            logger.error(
                "Failed to import 'sentence_transformers'. "
                "Ensure the package is installed via 'pip install sentence-transformers'."
            )
            raise e

    def embed_query(self, text: str) -> List[float]:
        if not self.model:
            self._load_model()
        vector = self.model.encode(text, convert_to_numpy=True)
        return [float(x) for x in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            self._load_model()
        if not texts:
            return []
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return [[float(x) for x in vector] for vector in vectors]
