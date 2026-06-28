from typing import Dict, Any
from embeddings.base import BaseEmbedder
from embeddings.mock_embedder import MockEmbedder
from embeddings.gemini_embedder import GeminiEmbedder
from embeddings.cohere_embedder import CohereEmbedder
from embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

# Global cache to reuse embedder instances (avoid reloading weights)
_embedder_cache: Dict[str, BaseEmbedder] = {}

def get_embedder(config: Dict[str, Any]) -> BaseEmbedder:
    """
    Returns the configured Embedder instance based on settings dict.
    Caches the instance to prevent reloading model weights on every call.
    """
    global _embedder_cache
    emb_cfg = config.get("embeddings", {})
    provider = emb_cfg.get("provider", "mock")
    
    if provider == "gemini":
        model_name = emb_cfg.get("gemini", {}).get("model_name", "models/text-embedding-004")
    elif provider == "cohere":
        model_name = emb_cfg.get("cohere", {}).get("model_name", "embed-english-v3.0")
    elif provider in ("sentence-transformers", "local"):
        model_name = emb_cfg.get("sentence-transformers", {}).get("model_name", "all-MiniLM-L6-v2")
    else:
        model_name = "mock"
        
    cache_key = f"{provider}:{model_name}"
    if cache_key in _embedder_cache:
        return _embedder_cache[cache_key]
        
    if provider == "gemini":
        gem_cfg = emb_cfg.get("gemini", {})
        instance = GeminiEmbedder(
            model_name=gem_cfg.get("model_name", "models/text-embedding-004")
        )
    elif provider == "cohere":
        coh_cfg = emb_cfg.get("cohere", {})
        instance = CohereEmbedder(
            model_name=coh_cfg.get("model_name", "embed-english-v3.0")
        )
    elif provider in ("sentence-transformers", "local"):
        st_cfg = emb_cfg.get("sentence-transformers", {})
        instance = SentenceTransformerEmbedder(
            model_name=st_cfg.get("model_name", "all-MiniLM-L6-v2")
        )
    else:
        instance = MockEmbedder(dimension=emb_cfg.get("dimension", 768))
        
    _embedder_cache[cache_key] = instance
    return instance

