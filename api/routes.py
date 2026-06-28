import logging
import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field

from core.orchestrator import QueryOrchestrator
from evaluation.runner import EvaluationRunner
from embeddings.factory import get_embedder
from core.vectorstore.factory import get_vector_store
from core.ingestion.pipeline import IngestionPipeline
from utils.logging import setup_json_logger, request_metrics
from config import get_config

# Set up JSON logger for API request profiling
logger = setup_json_logger("api.routes", level="INFO")
router = APIRouter()

# Schema definitions
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query text for sales sequence evaluation")

class QueryResponse(BaseModel):
    response_type: str = Field(..., description="The classification mode of the query: 'factual' or 'diagnostic'")
    direct_answer: Optional[str] = Field(None, description="Direct text response for factual lookups")
    drop_off: Optional[str] = Field(None, description="The step where sequence performance drops (diagnostic only)")
    insights: Optional[List[str]] = Field(None, description="Root cause insights (diagnostic only)")
    fixes: Optional[List[str]] = Field(None, description="Actionable improvement suggestions (diagnostic only)")
    improved_email: Optional[str] = Field(None, description="The rewritten improved email body (diagnostic only)")
    retrieved_chunk_ids: List[str] = Field(..., description="IDs of chunks referenced for grounding")

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="The query string to retrieve matches for")
    k: int = Field(5, description="Number of results to retrieve per strategy")

class ChunkMatch(BaseModel):
    chunk_id: str
    content: str
    score: float

class RetrieveResponse(BaseModel):
    query: str
    semantic: List[ChunkMatch]
    bm25: List[ChunkMatch]
    hybrid: List[ChunkMatch]

class EvaluateResponse(BaseModel):
    status: str
    metrics: Dict[str, Any]
    report_file: str

# Regex or substring blacklists for prompt injection protection
PROMPT_INJECTION_BLACKLIST = [
    "ignore previous",
    "ignore system",
    "ignore instruction",
    "ignore rule",
    "system prompt",
    "system instruction",
    "jailbreak",
    "you must now act as",
    "you are now a",
    "override rules",
    "bypass limits",
    "dan mode",
    "developer mode"
]

@router.post("/query", response_model=QueryResponse)
async def query_pipeline(req: QueryRequest):
    """
    Executes the hybrid retrieval, cross-encoder reranking, prompt building,
    and LLM generation pipeline. Logs complete timing and rank shifts as structured JSON.
    """
    # 1. Prompt Injection Validator Check
    normalized_query = req.query.lower()
    for blacklist_word in PROMPT_INJECTION_BLACKLIST:
        if blacklist_word in normalized_query:
            logger.warning(f"Prompt injection pattern detected: '{blacklist_word}' in query '{req.query}'")
            raise HTTPException(
                status_code=400, 
                detail="Security Warning: Potential prompt injection or system override attempt detected. Access denied."
            )

    try:
        orchestrator = QueryOrchestrator()
        response_data = orchestrator.query(req.query)
        
        # Pull request-scoped timing and rank metrics accumulated via decorators
        metrics = request_metrics.get()
        logger.info("Processed query request", extra={"extra_data": metrics})
        
        return response_data
    except Exception as e:
        logger.error(f"Error handling query request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_pipeline(req: RetrieveRequest):
    """
    Exposes raw candidate chunks and similarity scores across retrieval pipelines for debugging.
    """
    try:
        orchestrator = QueryOrchestrator()
        debug_results = orchestrator.retrieve_debug(req.query, k=req.k)
        return debug_results
    except Exception as e:
        logger.error(f"Error handling retrieve request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_pipeline():
    """
    Triggers the offline evaluation runner to compute metrics over the ground truth dataset.
    """
    try:
        config = get_config()
        config_dict = config.to_dict()
        
        # Load embedder and store config
        embedder = get_embedder(config_dict)
        vector_store = get_vector_store(config_dict)
        
        runner = EvaluationRunner(
            config=config_dict,
            embedder=embedder,
            vector_store=vector_store,
            ground_truth_path="data/ground_truth.json"
        )
        
        metrics = runner.run_evaluation()
        
        return {
            "status": "success",
            "metrics": metrics["summary"],
            "report_file": "evaluation_report.md"
        }
    except Exception as e:
        logger.error(f"Evaluation trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a CSV or PDF file, wipes the current database,
    and runs the IngestionPipeline on it.
    """
    filename = file.filename
    if not (filename.endswith('.csv') or filename.endswith('.pdf')):
        raise HTTPException(status_code=400, detail="Only CSV and PDF files are supported.")
        
    try:
        # Save to a temporary file
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        # Get DB config and clear vector store first
        config = get_config()
        config_dict = config.to_dict()
        
        embedder = get_embedder(config_dict)
        vector_store = get_vector_store(config_dict)
        vector_store.clear()
        
        # Run Ingestion Pipeline
        pipeline = IngestionPipeline(
            config_path="config/config.yaml",
            embedder=embedder,
            vector_store=vector_store
        )
        
        seeded_count = pipeline.run(tmp_path)
        
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return {
            "status": "success",
            "filename": filename,
            "type": "PDF" if filename.endswith('.pdf') else "CSV",
            "chunks_ingested": seeded_count
        }
    except Exception as e:
        logger.error(f"Failed to ingest file via upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chunks")
async def list_chunks():
    """
    Lists all ingested chunks currently stored in the vector database.
    """
    try:
        config = get_config()
        config_dict = config.to_dict()
        vector_store = get_vector_store(config_dict)
        chunks = vector_store.get_all_chunks()
        
        # Sort chunks by chunk_id or sequence/step to keep it clean
        sorted_chunks = sorted(chunks, key=lambda x: (x.sequence_id, x.step, x.chunk_id))
        
        return {
            "count": len(sorted_chunks),
            "chunks": [c.to_dict() for c in sorted_chunks]
        }
    except Exception as e:
        logger.error(f"Failed to list chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_database():
    """
    Wipes the active vector store database completely.
    """
    try:
        config = get_config()
        config_dict = config.to_dict()
        vector_store = get_vector_store(config_dict)
        vector_store.clear()
        return {"status": "success", "message": "Database wiped successfully."}
    except Exception as e:
        logger.error(f"Error wiping database: {e}")
        raise HTTPException(status_code=500, detail=str(e))
