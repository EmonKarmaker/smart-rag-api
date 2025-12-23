import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.services.document_parser import document_parser
from app.services.vector_store import vector_store

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and process it into the vector store.
    
    Supported formats: .pdf, .docx, .txt, .jpg, .jpeg, .png, .csv, .db
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {settings.SUPPORTED_EXTENSIONS}"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())[:8]
    
    # Create upload directory if not exists
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file with unique name
    safe_filename = f"{file_id}_{file.filename}"
    file_path = upload_dir / safe_filename
    
    try:
        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Parse document
        parsed = document_parser.parse(str(file_path))
        
        # Add to vector store
        chunks_added = vector_store.add_documents(parsed["chunks"])
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "file_type": file_ext,
            "chunks_created": chunks_added,
            "message": f"Document processed successfully. {chunks_added} chunks added to knowledge base."
        }
    
    except Exception as e:
        # Clean up file if processing failed
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.get("/files")
async def list_files():
    """List all uploaded files."""
    upload_dir = Path(settings.UPLOAD_DIR)
    
    if not upload_dir.exists():
        return {"files": []}
    
    files = []
    for file_path in upload_dir.iterdir():
        if file_path.is_file():
            files.append({
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "file_type": file_path.suffix.lower()
            })
    
    return {"files": files, "total": len(files)}


@router.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    return vector_store.get_stats()


@router.delete("/clear")
async def clear_vector_store():
    """Clear all documents from vector store."""
    vector_store.clear()
    return {"success": True, "message": "Vector store cleared."}