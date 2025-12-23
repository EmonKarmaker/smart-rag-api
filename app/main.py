from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import upload, query
from app.config import settings

# Create FastAPI app
app = FastAPI(
    title="Smart RAG API",
    description="A free, local RAG API that answers questions from any document type",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, tags=["Documents"])
app.include_router(query.router, tags=["Query"])


@app.get("/")
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "Welcome to Smart RAG API",
        "description": "Upload documents and ask questions about them",
        "endpoints": {
            "POST /upload": "Upload a document (PDF, DOCX, TXT, images, CSV, SQLite)",
            "GET /files": "List uploaded files",
            "GET /stats": "Get vector store statistics",
            "DELETE /clear": "Clear vector store",
            "POST /query": "Ask a question about your documents",
            "GET /health": "Check API health status"
        },
        "supported_formats": settings.SUPPORTED_EXTENSIONS,
        "llm_model": settings.OLLAMA_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)