import os
import json
import urllib.request
import urllib.error
import logging
from typing import List
from embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)

class CohereEmbedder(BaseEmbedder):
    """
    Client for Cohere Embeddings using direct HTTP REST requests.
    """
    def __init__(self, model_name: str = "embed-english-v3.0", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            logger.warning("No Cohere API key found in credentials. CohereEmbedder calls may fail.")

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text], input_type="search_query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts, input_type="search_document")

    def _embed_batch(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not configured.")
        if not texts:
            return []
            
        url = "https://api.cohere.ai/v1/embed"
        payload = {
            "texts": texts,
            "model": self.model_name,
            "input_type": input_type,
            "embedding_types": ["float"]
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req) as res:
                data = json.loads(res.read().decode("utf-8"))
                return data["embeddings"]["float"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            logger.error(f"Cohere Embedding API HTTP error: {e.code} - {err_body}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch Cohere embedding: {e}")
            raise
