import os
import json
import uuid
import urllib.request
import urllib.error
import logging
from typing import List, Dict, Any, Tuple
from core.adapters.base import Chunk
from core.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant Cloud/Local Vector Store Client using direct HTTP REST requests.
    """
    def __init__(
        self, 
        url: str = None, 
        api_key: str = None, 
        collection_name: str = "sales_sequences",
        vector_size: int = 768
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Load credentials from config keys or environment variables
        self.url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        
        # Strip trailing slash from URL
        if self.url.endswith("/"):
            self.url = self.url[:-1]
            
        self._ensure_collection()

    def _request(self, path: str, method: str = "GET", payload: Any = None) -> Dict[str, Any]:
        url = f"{self.url}{path}"
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["api-key"] = self.api_key

        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            logger.error(f"Qdrant API HTTP Error [{method} {path}]: {e.code} - {body}")
            raise
        except Exception as e:
            logger.error(f"Qdrant API Connection Error [{method} {path}]: {e}")
            raise

    def _ensure_collection(self) -> None:
        try:
            # Check if collection exists
            logger.info(f"Checking if Qdrant collection '{self.collection_name}' exists...")
            self._request(f"/collections/{self.collection_name}")
            logger.info(f"Collection '{self.collection_name}' already exists.")
        except urllib.error.HTTPError as e:
            if e.code == 444 or e.code == 404: # Qdrant returns 404 if not found
                logger.info(f"Collection '{self.collection_name}' not found. Creating collection...")
                create_payload = {
                    "vectors": {
                        "size": self.vector_size,
                        "distance": "Cosine"
                    }
                }
                self._request(f"/collections/{self.collection_name}", method="PUT", payload=create_payload)
                logger.info(f"Collection '{self.collection_name}' created successfully.")
            else:
                raise

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Mismatched dimensions between chunks and embeddings.")

        points = []
        for chunk, emb in zip(chunks, embeddings):
            # Qdrant requires IDs to be UUIDs or 64-bit unsigned integers.
            # Generate a stable UUID based on chunk_id so upserts overwrite existing records.
            pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            
            points.append({
                "id": pt_id,
                "vector": emb,
                "payload": chunk.to_dict()
            })

        payload = {"points": points}
        self._request(f"/collections/{self.collection_name}/points?wait=true", method="PUT", payload=payload)
        logger.info(f"Successfully upserted {len(chunks)} points to Qdrant collection '{self.collection_name}'.")

    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Chunk, float]]:
        
        search_payload: Dict[str, Any] = {
            "vector": query_embedding,
            "limit": k,
            "with_payload": True
        }

        # Build Qdrant metadata filters if provided
        if filter_dict:
            must_filters = []
            for key, val in filter_dict.items():
                must_filters.append({
                    "key": key,
                    "match": {"value": val}
                })
            search_payload["filter"] = {"must": must_filters}

        response = self._request(f"/collections/{self.collection_name}/points/search", method="POST", payload=search_payload)
        
        results = []
        for hit in response.get("result", []):
            payload = hit["payload"]
            score = hit["score"]
            
            chunk = Chunk(
                chunk_id=payload["chunk_id"],
                content=payload["content"],
                email_id=payload["email_id"],
                sequence_id=payload["sequence_id"],
                step=payload["step"],
                persona=payload["persona"],
                industry=payload["industry"],
                stage=payload["stage"],
                open_rate=payload["open_rate"],
                reply_rate=payload["reply_rate"],
                source_metadata=payload["source_metadata"]
            )
            results.append((chunk, score))
            
        return results

    def clear(self) -> None:
        # Recreate collection to wipe all items
        logger.info(f"Clearing collection '{self.collection_name}'...")
        try:
            self._request(f"/collections/{self.collection_name}", method="DELETE")
        except Exception:
            pass
        self._ensure_collection()

    def get_all_chunks(self) -> List[Chunk]:
        scroll_payload = {
            "limit": 10000,
            "with_payload": True,
            "with_vector": False
        }
        response = self._request(f"/collections/{self.collection_name}/points/scroll", method="POST", payload=scroll_payload)
        
        chunks = []
        points = response.get("result", {}).get("points", [])
        for pt in points:
            payload = pt.get("payload", {})
            if not payload:
                continue
            chunk = Chunk(
                chunk_id=payload["chunk_id"],
                content=payload["content"],
                email_id=payload["email_id"],
                sequence_id=payload["sequence_id"],
                step=payload["step"],
                persona=payload["persona"],
                industry=payload["industry"],
                stage=payload["stage"],
                open_rate=payload["open_rate"],
                reply_rate=payload["reply_rate"],
                source_metadata=payload["source_metadata"]
            )
            chunks.append(chunk)
        return chunks
