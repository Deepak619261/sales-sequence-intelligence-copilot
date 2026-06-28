import os
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

# Ensure setup for loading yaml config works
try:
    from utils.config_loader import load_config as load_yaml_config
except ImportError:
    # Handle absolute import paths for root/core structures
    from core.utils.config_loader import load_config as load_yaml_config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

class SystemConfig(BaseModel):
    project_name: str = "sales-sequence-copilot"
    log_level: str = "INFO"

class DataConfig(BaseModel):
    raw_csv_path: str = "data/raw_sequences.csv"
    processed_output_dir: str = "data/processed/"

class ChunkingConfig(BaseModel):
    strategy: str = "sentence"
    chunk_size: int = 500
    chunk_overlap: int = 50
    semantic_threshold: float = 0.7

class GeminiEmbedderConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = "models/text-embedding-004"
    api_key_env: str = "GEMINI_API_KEY"

class CohereEmbedderConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = "embed-english-v3.0"
    api_key_env: str = "COHERE_API_KEY"

class STEmbedderConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = "all-MiniLM-L6-v2"

class EmbeddingsConfig(BaseModel):
    provider: str = "sentence-transformers"
    gemini: GeminiEmbedderConfig = Field(default_factory=GeminiEmbedderConfig)
    cohere: CohereEmbedderConfig = Field(default_factory=CohereEmbedderConfig)
    sentence_transformers: STEmbedderConfig = Field(default_factory=STEmbedderConfig, alias="sentence-transformers")
    dimension: int = 384

    class Config:
        populate_by_name = True

class QdrantConfig(BaseModel):
    url_env: str = "QDRANT_URL"
    api_key_env: str = "QDRANT_API_KEY"
    collection_name: str = "sales_sequences"

class LocalStoreConfig(BaseModel):
    persist_path: str = "data/local_store.json"
    collection_name: str = "sales_sequences"

class AzureConfig(BaseModel):
    endpoint_env: str = "AZURE_AI_SEARCH_URL"
    api_key_env: str = "AZURE_AI_SEARCH_KEY"
    index_name: str = "sales-sequences"

class VectorStoreConfig(BaseModel):
    type: str = "local"
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    local: LocalStoreConfig = Field(default_factory=LocalStoreConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)

class RetrievalConfig(BaseModel):
    semantic_k: int = 20
    bm25_k: int = 20
    hybrid_k: int = 20
    fusion_type: str = "rrf"
    rrf_k: int = 60
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    score_threshold: float = 0.1

class RerankerConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 10
    simulate_fallback: bool = True

class LLMConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    provider: str = "gemini"
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_output_tokens: int = 1024
    grounding_threshold: float = 0.6

class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def to_dict(self) -> Dict[str, Any]:
        # Return dict representation for backwards compatibility with dict-based factories
        return self.model_dump(by_alias=True)

_config_cache: Optional[AppConfig] = None

def get_config(config_path: str = "config/config.yaml") -> AppConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    yaml_data = {}
    if os.path.exists(config_path):
        try:
            yaml_data = load_yaml_config(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config file {config_path}: {e}. Using defaults.")

    # Convert 'sentence-transformers' to 'sentence_transformers' if present
    if "embeddings" in yaml_data and "sentence-transformers" in yaml_data["embeddings"]:
        yaml_data["embeddings"]["sentence_transformers"] = yaml_data["embeddings"].pop("sentence-transformers")

    config = AppConfig(**yaml_data)

    # Environment variable overrides
    if "LOG_LEVEL" in os.environ:
        config.system.log_level = os.environ["LOG_LEVEL"]

    if "EMBEDDINGS_PROVIDER" in os.environ:
        config.embeddings.provider = os.environ["EMBEDDINGS_PROVIDER"]
    if "EMBEDDINGS_DIMENSION" in os.environ:
        config.embeddings.dimension = int(os.environ["EMBEDDINGS_DIMENSION"])

    if "VECTORSTORE_TYPE" in os.environ:
        config.vectorstore.type = os.environ["VECTORSTORE_TYPE"]

    if "RERANKER_MODEL_NAME" in os.environ:
        config.reranker.model_name = os.environ["RERANKER_MODEL_NAME"]
    if "RERANKER_TOP_N" in os.environ:
        config.reranker.top_n = int(os.environ["RERANKER_TOP_N"])
    if "RERANKER_ENABLED" in os.environ:
        config.reranker.enabled = os.environ["RERANKER_ENABLED"].lower() in ("true", "1")

    if "LLM_PROVIDER" in os.environ:
        config.llm.provider = os.environ["LLM_PROVIDER"]
    if "LLM_MODEL_NAME" in os.environ:
        config.llm.model_name = os.environ["LLM_MODEL_NAME"]

    _config_cache = config
    return config
