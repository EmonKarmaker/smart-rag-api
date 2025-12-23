import base64
from io import BytesIO
from PIL import Image
import pytesseract
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vector_store import vector_store
from app.services.llm import llm_service

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    image_base64: str | None = None
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    context_used: str
    model: str
    image_text: str | None = None


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question about the uploaded documents.
    
    Optionally include an image (base64 encoded) for OCR processing.
    """
    # Check if we have any documents
    stats = vector_store.get_stats()
    if stats["total_documents"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents in knowledge base. Please upload documents first using /upload endpoint."
        )
    
    image_text = None
    
    # Process image if provided
    if request.image_base64:
        try:
            image_text = _extract_text_from_base64(request.image_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing image: {str(e)}"
            )
    
    # Build search query (combine question with image text if available)
    search_query = request.question
    if image_text:
        search_query = f"{request.question} {image_text[:200]}"
    
    # Search vector store
    results = vector_store.search(search_query, top_k=request.top_k)
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found for your question."
        )
    
    # Build context from search results
    context = _build_context(results)
    
    # Generate answer using LLM
    if image_text:
        llm_response = llm_service.process_image_question(
            question=request.question,
            image_text=image_text,
            context=context
        )
    else:
        llm_response = llm_service.generate_answer(
            question=request.question,
            context=context
        )
    
    # Prepare sources
    sources = [
        {
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "score": r["score"],
            "preview": r["content"][:150] + "..." if len(r["content"]) > 150 else r["content"]
        }
        for r in results
    ]
    
    return QueryResponse(
        answer=llm_response["answer"],
        sources=sources,
        context_used=context[:500] + "..." if len(context) > 500 else context,
        model=llm_response["model"],
        image_text=image_text
    )


@router.get("/health")
async def health_check():
    """Check if all services are running."""
    from app.config import settings
    
    llm_ok = llm_service.health_check()
    vector_stats = vector_store.get_stats()
    
    return {
        "status": "healthy" if llm_ok else "degraded",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": llm_service.model,
        "llm_status": "connected" if llm_ok else "disconnected",
        "vector_store": vector_stats
    }


def _extract_text_from_base64(base64_string: str) -> str:
    """Extract text from base64 encoded image using OCR."""
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    # OCR
    text = pytesseract.image_to_string(image)
    return text.strip()


def _build_context(results: list[dict]) -> str:
    """Build context string from search results."""
    context_parts = []
    
    for i, result in enumerate(results, 1):
        source = result.get("source", "Unknown")
        content = result["content"]
        context_parts.append(f"[Source {i}: {source}]\n{content}")
    
    return "\n\n".join(context_parts)