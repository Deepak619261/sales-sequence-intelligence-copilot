import os
import json
import math
import logging
from typing import List, Dict, Any, Tuple
from core.adapters.base import Chunk
from core.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

class LocalVectorStore(BaseVectorStore):
    """
    Offline vector database that persists records to a JSON file.
    Uses pure-Python cosine similarity calculations.
    """
    def __init__(self, persist_path: str = "data/local_store.json"):
        self.persist_path = persist_path
        self.data: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logger.info(f"Loaded {len(self.data)} vectors from local store: {self.persist_path}")
            except Exception as e:
                logger.error(f"Failed to load local store JSON: {e}")
                self.data = []
        else:
            # Create directories if missing
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            self.data = []

    def _save(self) -> None:
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            logger.debug(f"Saved store to {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to save local store JSON: {e}")

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Mismatched dimensions between chunks and embeddings.")
            
        # Remove existing IDs if they exist (override)
        existing_ids = {item["chunk"]["chunk_id"] for item in self.data}
        
        for chunk, emb in zip(chunks, embeddings):
            record = {
                "chunk": chunk.to_dict(),
                "embedding": emb
            }
            # Evict existing if present
            if chunk.chunk_id in existing_ids:
                self.data = [item for item in self.data if item["chunk"]["chunk_id"] != chunk.chunk_id]
            self.data.append(record)
            
        self._save()
        logger.info(f"Upserted {len(chunks)} chunks into LocalVectorStore.")

    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Chunk, float]]:
        
        results = []
        filter_dict = filter_dict or {}

        for item in self.data:
            chunk_dict = item["chunk"]
            chunk_emb = item["embedding"]
            
            # Apply metadata filters (logical AND)
            match = True
            for filter_key, filter_val in filter_dict.items():
                if chunk_dict.get(filter_key) != filter_val:
                    match = False
                    break
            
            if not match:
                continue

            # Calculate cosine similarity
            sim = self._cosine_similarity(query_embedding, chunk_emb)
            
            # Reconstruct Chunk object
            chunk = Chunk(
                chunk_id=chunk_dict["chunk_id"],
                content=chunk_dict["content"],
                email_id=chunk_dict["email_id"],
                sequence_id=chunk_dict["sequence_id"],
                step=chunk_dict["step"],
                persona=chunk_dict["persona"],
                industry=chunk_dict["industry"],
                stage=chunk_dict["stage"],
                open_rate=chunk_dict["open_rate"],
                reply_rate=chunk_dict["reply_rate"],
                source_metadata=chunk_dict["source_metadata"]
            )
            results.append((chunk, sim))

        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def clear(self) -> None:
        self.data = []
        self._save()
        logger.info("Cleared all collections in LocalVectorStore.")

    def get_all_chunks(self) -> List[Chunk]:
        chunks = []
        for item in self.data:
            chunk_dict = item["chunk"]
            chunk = Chunk(
                chunk_id=chunk_dict["chunk_id"],
                content=chunk_dict["content"],
                email_id=chunk_dict["email_id"],
                sequence_id=chunk_dict["sequence_id"],
                step=chunk_dict["step"],
                persona=chunk_dict["persona"],
                industry=chunk_dict["industry"],
                stage=chunk_dict["stage"],
                open_rate=chunk_dict["open_rate"],
                reply_rate=chunk_dict["reply_rate"],
                source_metadata=chunk_dict["source_metadata"]
            )
            chunks.append(chunk)
        return chunks

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm_v1 = math.sqrt(sum(x * x for x in v1))
        norm_v2 = math.sqrt(sum(x * x for x in v2))
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)
