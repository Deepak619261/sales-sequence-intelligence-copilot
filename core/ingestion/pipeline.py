import logging
from typing import List, Dict, Any
from config import get_config
from utils.logger import setup_logger
from core.adapters.csv_adapter import CSVAdapter
from core.ingestion.loader import DataLoader
from core.chunking.sentence_chunker import SentenceChunker
from core.chunking.sliding_window import SlidingWindowChunker
from core.chunking.semantic_chunker import SemanticChunker
from embeddings.factory import get_embedder
from core.vectorstore.factory import get_vector_store

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Orchestrates the entire data ingestion, chunking, embedding, 
    and vector DB storage workflow.
    """
    def __init__(self, config_path: str = "config/config.yaml", embedder: Any = None, vector_store: Any = None):
        self.config = get_config(config_path).to_dict()
        setup_logger("ingestion_pipeline", self.config["system"].get("log_level", "INFO"))
        
        # Initialize Embedder & Vector Store if not passed
        self.embedder = embedder or get_embedder(self.config)
        self.vector_store = vector_store or get_vector_store(self.config)
        
        # Initialize Chunker based on strategy
        chunking_cfg = self.config["chunking"]
        strategy = chunking_cfg.get("strategy", "sentence")
        
        if strategy == "sentence":
            self.chunker = SentenceChunker(
                sentences_per_chunk=2, 
                prepend_subject=True
            )
        elif strategy == "sliding_window":
            self.chunker = SlidingWindowChunker(
                chunk_size=chunking_cfg.get("chunk_size", 100),
                chunk_overlap=chunking_cfg.get("chunk_overlap", 20),
                prepend_subject=True
            )
        elif strategy == "semantic":
            # Semantic chunker gets initialized with the configured embedder
            self.chunker = SemanticChunker(
                embedder=self.embedder,
                threshold_percentile=chunking_cfg.get("semantic_threshold", 0.7),
                prepend_subject=True
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
            
        # Loader setup (lazy initialized in run)
        self.loader = None

    def run(self, csv_path: str = None) -> int:
        target_path = csv_path or self.config["data"]["raw_csv_path"]
        logger.info(f"Running Ingestion Pipeline on: {target_path}")
        
        # 1. Setup adapter dynamically based on file extension
        if target_path.lower().endswith('.pdf'):
            from core.adapters.pdf_adapter import PDFAdapter
            adapter = PDFAdapter()
            logger.info("Selected PDF adapter for ingestion.")
        else:
            from core.adapters.csv_adapter import CSVAdapter
            adapter = CSVAdapter()
            logger.info("Selected CSV adapter for ingestion.")
            
        self.loader = DataLoader(adapter=adapter)
        
        # 2. Load and Clean
        emails = self.loader.load(target_path)
        if not emails:
            logger.warning("No emails loaded. Ingestion stopped.")
            return 0
            
        # 2. Chunk
        all_chunks = []
        for email in emails:
            chunks = self.chunker.chunk(email)
            all_chunks.extend(chunks)
            
        logger.info(f"Generated {len(all_chunks)} chunks from {len(emails)} emails.")
        if not all_chunks:
            logger.warning("No chunks generated. Ingestion stopped.")
            return 0
            
        # 3. Embed
        logger.info("Generating embeddings for all chunks...")
        contents = [c.content for c in all_chunks]
        embeddings = self.embedder.embed_documents(contents)
        
        # 4. Upsert
        logger.info(f"Upserting chunks to Vector Store...")
        self.vector_store.upsert_chunks(all_chunks, embeddings)
        
        logger.info("Ingestion Pipeline run completed successfully.")
        return len(all_chunks)
