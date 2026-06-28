from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from api.routes import router
from config import get_config
from core.ingestion.pipeline import IngestionPipeline
from embeddings.factory import get_embedder
from core.vectorstore.factory import get_vector_store
from utils.logging import setup_json_logger

# Initialize structured logger for service entrypoint
logger = setup_json_logger("main", level="INFO")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Validates configuration and ensures the vector database is populated.
    Seeds the database automatically from the configured CSV if empty.
    """
    logger.info("Initializing Sales Sequence Intelligence Copilot service...")
    
    config = get_config()
    config_dict = config.to_dict()
    
    try:
        # Check if database has elements
        embedder = get_embedder(config_dict)
        vector_store = get_vector_store(config_dict)
        
        chunks = vector_store.get_all_chunks()
        if not chunks:
            logger.info("Vector database is empty. Ready for document uploads via the UI.")
        else:
            logger.info(f"Vector database has {len(chunks)} pre-existing chunks. Ready to serve.")
            
    except Exception as e:
        logger.error(f"Service initialization checks encountered an error: {e}")
        
    yield

app = FastAPI(
    title="Sales Sequence Intelligence Copilot API",
    description="Production-ready REST API service for hybrid sales email retrieval and analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)

# Ensure static folder exists and mount it
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    """
    Serves the dashboard UI.
    """
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Sales Sequence Intelligence Copilot API is running. Dashboard index.html not found."}

@app.get("/health")
def health_check():
    """
    Health check JSON endpoint.
    """
    config = get_config()
    return {
        "status": "healthy",
        "project": config.system.project_name,
        "embeddings": {
            "provider": config.embeddings.provider,
            "dimension": config.embeddings.dimension
        },
        "vectorstore": {
            "type": config.vectorstore.type
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
