"""
RSE Embedding Service
Provides local sentence-transformers embeddings via FastAPI for the RSE system.
Single implementation with transparent error handling and no fallbacks.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio
import os

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingRequest(BaseModel):
    """Request model for text embedding generation."""
    text: str = Field(..., description="Text to embed", min_length=1)
    model_name: Optional[str] = Field(None, description="Optional model override")

class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embedding: List[float] = Field(..., description="Generated embedding vector")
    model_used: str = Field(..., description="Model that generated the embedding")
    dimension: int = Field(..., description="Embedding dimension")

class BatchEmbeddingRequest(BaseModel):
    """Request model for batch text embedding generation."""
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1)
    model_name: Optional[str] = Field(None, description="Optional model override")
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate that all texts are non-empty."""
        if any(not text.strip() for text in v):
            raise ValueError("All texts must be non-empty")
        return v

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    chroma_connected: bool
    version: str = "1.0.0"

class EmbeddingService:
    """
    RSE Embedding Service using sentence-transformers.
    Provides local embedding generation with no external dependencies.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chroma_persist_dir: Optional[Path] = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Sentence-transformers model to use
            chroma_persist_dir: Directory for ChromaDB persistence
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.chroma_persist_dir = chroma_persist_dir or Path("./chroma_db")
        
        # Ensure chroma directory exists
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize models and database connections."""
        try:
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
            # Initialize ChromaDB
            logger.info(f"Initializing ChromaDB at: {self.chroma_persist_dir}")
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise HTTPException(status_code=500, detail=f"Service initialization failed: {e}")
    
    def generate_embedding(self, text: str, model_name: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            model_name: Optional model override (not supported in single implementation)
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            HTTPException: If model not loaded or embedding generation fails
        """
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if model_name and model_name != self.model_name:
            logger.warning(f"Model override requested ({model_name}) but single implementation uses: {self.model_name}")
        
        try:
            embedding = self.model.encode([text])[0]
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed for text: '{text[:50]}...': {e}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")
    
    def generate_batch_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts to embed
            model_name: Optional model override (not supported in single implementation)
            
        Returns:
            Embedding matrix as numpy array
            
        Raises:
            HTTPException: If model not loaded or embedding generation fails
        """
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if model_name and model_name != self.model_name:
            logger.warning(f"Model override requested ({model_name}) but single implementation uses: {self.model_name}")
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding generation failed for {len(texts)} texts: {e}")
            raise HTTPException(status_code=500, detail=f"Batch embedding generation failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get service health status.
        
        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy" if self.model is not None and self.chroma_client is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "chroma_connected": self.chroma_client is not None,
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension() if self.model else None
        }

# Initialize service
service = EmbeddingService()

# Create FastAPI app
app = FastAPI(
    title="RSE Embedding Service",
    description="Local sentence-transformers embedding service for RSE system",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    await service.initialize()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    status = service.get_health_status()
    return HealthResponse(**status)

@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate embedding for a single text.
    
    Args:
        request: Embedding request containing text and optional model name
        
    Returns:
        Embedding response with vector, model used, and dimension
    """
    embedding = service.generate_embedding(request.text, request.model_name)
    
    return EmbeddingResponse(
        embedding=embedding.tolist(),
        model_used=service.model_name,
        dimension=len(embedding)
    )

@app.post("/embed/batch", response_model=List[EmbeddingResponse])
async def generate_batch_embeddings(request: BatchEmbeddingRequest):
    """
    Generate embeddings for multiple texts efficiently.
    
    Args:
        request: Batch embedding request containing texts and optional model name
        
    Returns:
        List of embedding responses
    """
    embeddings = service.generate_batch_embeddings(request.texts, request.model_name)
    
    responses = []
    for i, embedding in enumerate(embeddings):
        responses.append(EmbeddingResponse(
            embedding=embedding.tolist(),
            model_used=service.model_name,
            dimension=len(embedding)
        ))
    
    return responses

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "RSE Embedding Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "embed": "/embed",
            "batch_embed": "/embed/batch",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "embedding_service:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )
