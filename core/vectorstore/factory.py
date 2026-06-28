from typing import Dict, Any
from core.vectorstore.base import BaseVectorStore
from core.vectorstore.local_store import LocalVectorStore
from core.vectorstore.qdrant_store import QdrantVectorStore
from core.vectorstore.azure_store import AzureVectorStore
# Global cache to reuse vector store instances
_vector_store_cache: Dict[str, BaseVectorStore] = {}

def get_vector_store(config: Dict[str, Any]) -> BaseVectorStore:
    """
    Returns the configured VectorStore instance based on settings dict.
    Caches the instance to prevent recreating database clients on every call.
    """
    global _vector_store_cache
    vs_cfg = config.get("vectorstore", {})
    store_type = vs_cfg.get("type", "local")
    
    if store_type == "qdrant":
        q_cfg = vs_cfg.get("qdrant", {})
        collection_name = q_cfg.get("collection_name", "sales_sequences")
        cache_key = f"qdrant:{collection_name}"
    elif store_type == "azure":
        a_cfg = vs_cfg.get("azure", {})
        index_name = a_cfg.get("index_name", "sales_sequences")
        cache_key = f"azure:{index_name}"
    else:
        l_cfg = vs_cfg.get("local", {})
        persist_path = l_cfg.get("persist_path", "data/local_store.json")
        cache_key = f"local:{persist_path}"
        
    if cache_key in _vector_store_cache:
        return _vector_store_cache[cache_key]
        
    if store_type == "qdrant":
        q_cfg = vs_cfg.get("qdrant", {})
        dim = config.get("embeddings", {}).get("dimension", 768)
        instance = QdrantVectorStore(
            collection_name=q_cfg.get("collection_name", "sales_sequences"),
            vector_size=dim
        )
    elif store_type == "azure":
        a_cfg = vs_cfg.get("azure", {})
        dim = config.get("embeddings", {}).get("dimension", 768)
        instance = AzureVectorStore(
            index_name=a_cfg.get("index_name", "sales_sequences"),
            vector_size=dim
        )
    else:
        l_cfg = vs_cfg.get("local", {})
        instance = LocalVectorStore(
            persist_path=l_cfg.get("persist_path", "data/local_store.json")
        )
        
    _vector_store_cache[cache_key] = instance
    return instance
