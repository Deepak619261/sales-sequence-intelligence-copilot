import os
import json
import urllib.request
import urllib.error
import logging
from typing import List
from embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)

class GeminiEmbedder(BaseEmbedder):
    """
    Client for Gemini Embeddings using direct HTTP REST requests.
    """
    def __init__(self, model_name: str = "models/text-embedding-004", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No Gemini API key found in credentials. GeminiEmbedder calls may fail.")

    def embed_query(self, text: str) -> List[float]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:embedContent?key={self.api_key}"
        payload = {
            "model": self.model_name,
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req) as res:
                data = json.loads(res.read().decode("utf-8"))
                return data["embedding"]["values"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            logger.error(f"Gemini Embedding API HTTP error: {e.code} - {err_body}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch Gemini embedding: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        if not texts:
            return []
            
        # The batch API endpoint is:
        # https://generativelanguage.googleapis.com/v1beta/{model_name}:batchEmbedContents?key={api_key}
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:batchEmbedContents?key={self.api_key}"
        
        # Batch sizes are capped. We will send requests in chunks of 50 to prevent size limits.
        chunk_size = 50
        all_embeddings = []
        
        for i in range(0, len(texts), chunk_size):
            batch_texts = texts[i:i+chunk_size]
            requests = [
                {
                    "model": self.model_name,
                    "content": {"parts": [{"text": t}]}
                }
                for t in batch_texts
            ]
            payload = {"requests": requests}
            
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req) as res:
                    data = json.loads(res.read().decode("utf-8"))
                    chunk_embeddings = [emb["values"] for emb in data["embeddings"]]
                    all_embeddings.extend(chunk_embeddings)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8")
                logger.error(f"Gemini Batch Embedding API HTTP error: {e.code} - {err_body}")
                raise
            except Exception as e:
                logger.error(f"Failed to fetch Gemini batch embeddings: {e}")
                raise
                
        return all_embeddings
