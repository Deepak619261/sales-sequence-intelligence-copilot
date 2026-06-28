from typing import Dict, Any
from reranker.base import BaseReranker
from reranker.cross_encoder_reranker import CrossEncoderReranker

# Global cache to reuse reranker instances (avoid reloading weights)
_reranker_cache: Dict[str, BaseReranker] = {}

def get_reranker(config: Dict[str, Any]) -> BaseReranker:
    """
    Builds and returns a Reranker instance based on the configuration.
    Caches the instance to prevent reloading CrossEncoder model weights on every call.
    """
    global _reranker_cache
    r_cfg = config.get("reranker", {})
    enabled = r_cfg.get("enabled", True)
    model_name = r_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_n = r_cfg.get("top_n", 3)
    simulate_fallback = r_cfg.get("simulate_fallback", False)
    
    cache_key = f"{enabled}:{model_name}:{top_n}:{simulate_fallback}"
    if cache_key in _reranker_cache:
        return _reranker_cache[cache_key]
        
    instance = CrossEncoderReranker(
        model_name=model_name,
        top_n=top_n,
        enabled=enabled,
        simulate_fallback=simulate_fallback
    )
    _reranker_cache[cache_key] = instance
    return instance
