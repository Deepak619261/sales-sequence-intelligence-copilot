import os
import logging
from typing import List, Dict, Any, Tuple
from core.vectorstore.base import BaseVectorStore
from core.adapters.base import Chunk
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

logger = logging.getLogger(__name__)

class AzureVectorStore(BaseVectorStore):
    """
    Azure AI Search Vector Store client using the official Azure SDK.
    """
    def __init__(
        self,
        endpoint: str = None,
        api_key: str = None,
        index_name: str = "sales-sequences",
        vector_size: int = 384
    ):
        self.endpoint = endpoint or os.environ.get("AZURE_AI_SEARCH_URL")
        self.api_key = api_key or os.environ.get("AZURE_AI_SEARCH_KEY")
        
        # Azure index names must start with a letter, contain only lowercase letters, digits, or dashes, and cannot end with a dash
        self.index_name = index_name.replace("_", "-").lower()
        self.vector_size = vector_size

        if not self.endpoint or not self.api_key:
            logger.warning("Azure AI Search credentials (URL or Key) are missing from environment.")

        self.credential = AzureKeyCredential(self.api_key) if self.api_key else None
        
        if self.endpoint and self.credential:
            self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
            self.client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)
            self._ensure_index()
        else:
            self.index_client = None
            self.client = None

    def _ensure_index(self) -> None:
        """
        Creates the search index if it doesn't exist, configuring schema and HNSW vector parameters.
        """
        if not self.index_client:
            return

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="email_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="sequence_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="step", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="persona", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="industry", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="stage", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="open_rate", type=SearchFieldDataType.Double, filterable=True),
            SimpleField(name="reply_rate", type=SearchFieldDataType.Double, filterable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=self.vector_size,
                vector_search_profile_name="my-vector-profile"
            )
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="my-hnsw-config")
            ],
            profiles=[
                VectorSearchProfile(
                    name="my-vector-profile",
                    algorithm_configuration_name="my-hnsw-config"
                )
            ]
        )

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search
        )

        try:
            self.index_client.create_or_update_index(index)
            logger.info(f"Azure Search Index '{self.index_name}' is ready.")
        except Exception as e:
            logger.error(f"Failed to create or update Azure Search Index: {e}")
            raise

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Uploads chunks and their vector embeddings to the Azure index.
        """
        if not self.client:
            logger.error("Azure Search client is not initialized.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError("Mismatched dimensions between chunks and embeddings.")

        documents = []
        for chunk, emb in zip(chunks, embeddings):
            doc = {
                "id": chunk.chunk_id,
                "content": chunk.content,
                "email_id": chunk.email_id,
                "sequence_id": chunk.sequence_id,
                "step": int(chunk.step),
                "persona": chunk.persona,
                "industry": chunk.industry,
                "stage": chunk.stage,
                "open_rate": float(chunk.open_rate),
                "reply_rate": float(chunk.reply_rate),
                "embedding": emb
            }
            documents.append(doc)

        try:
            self.client.upload_documents(documents=documents)
            logger.info(f"Successfully uploaded {len(documents)} documents to Azure Search.")
        except Exception as e:
            logger.error(f"Failed to upload documents to Azure Search: {e}")
            raise

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Searches the Azure index using the query vector embedding.
        """
        if not self.client:
            logger.error("Azure Search client is not initialized.")
            return []

        # Construct OData filter string from filter_dict if provided
        filter_str = None
        if filter_dict:
            filter_clauses = []
            for key, val in filter_dict.items():
                if isinstance(val, str):
                    filter_clauses.append(f"{key} eq '{val}'")
                elif isinstance(val, (int, float)):
                    filter_clauses.append(f"{key} eq {val}")
            if filter_clauses:
                filter_str = " and ".join(filter_clauses)

        from azure.search.documents.models import VectorizedQuery
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=k,
            fields="embedding"
        )

        try:
            results = self.client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=filter_str,
                select=[
                    "id", "content", "email_id", "sequence_id", "step",
                    "persona", "industry", "stage", "open_rate", "reply_rate"
                ]
            )

            chunk_scores = []
            for res in results:
                # Azure return score is @search.score
                score = res.get("@search.score", 0.0)
                chunk = Chunk(
                    chunk_id=res["id"],
                    content=res["content"],
                    email_id=res["email_id"],
                    sequence_id=res["sequence_id"],
                    step=res["step"],
                    persona=res["persona"],
                    industry=res["industry"],
                    stage=res["stage"],
                    open_rate=res["open_rate"],
                    reply_rate=res["reply_rate"],
                    source_metadata={"source": "azure_ai_search"}
                )
                chunk_scores.append((chunk, score))
            return chunk_scores
        except Exception as e:
            logger.error(f"Azure Search similarity search failed: {e}")
            return []

    def clear(self) -> None:
        """
        Deletes and recreates the search index to clean up all documents.
        This resets storage utilization back to zero (ideal for free plan).
        """
        if not self.index_client:
            logger.error("Azure Index Client is not initialized.")
            return

        try:
            logger.info(f"Deleting Azure Search Index '{self.index_name}'...")
            self.index_client.delete_index(self.index_name)
            logger.info(f"Index '{self.index_name}' deleted. Recreating clean index...")
            self._ensure_index()
        except Exception as e:
            logger.error(f"Failed to clear/reset Azure Search Index: {e}")
            # If deletion failed because it doesn't exist, ensure it's created anyway
            self._ensure_index()

    def get_all_chunks(self) -> List[Chunk]:
        """
        Retrieves all documents currently in the index.
        """
        if not self.client:
            logger.error("Azure Search client is not initialized.")
            return []

        try:
            results = self.client.search(
                search_text="*",
                select=[
                    "id", "content", "email_id", "sequence_id", "step",
                    "persona", "industry", "stage", "open_rate", "reply_rate"
                ]
            )

            chunks = []
            for res in results:
                chunk = Chunk(
                    chunk_id=res["id"],
                    content=res["content"],
                    email_id=res["email_id"],
                    sequence_id=res["sequence_id"],
                    step=res["step"],
                    persona=res["persona"],
                    industry=res["industry"],
                    stage=res["stage"],
                    open_rate=res["open_rate"],
                    reply_rate=res["reply_rate"],
                    source_metadata={"source": "azure_ai_search"}
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            logger.error(f"Failed to retrieve all chunks from Azure Search: {e}")
            return []