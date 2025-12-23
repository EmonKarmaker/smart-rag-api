from app.config import settings


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[dict]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk (default from settings)
        overlap: Overlap between chunks (default from settings)
    
    Returns:
        List of dicts with 'content' and 'chunk_index'
    """
    if not text or not text.strip():
        return []
    
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    
    # Clean the text
    text = text.strip()
    text = " ".join(text.split())  # Normalize whitespace
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end]
        
        # Try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence boundary
            last_period = chunk_content.rfind(". ")
            last_newline = chunk_content.rfind("\n")
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:
                chunk_content = chunk_content[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append({
            "content": chunk_content.strip(),
            "chunk_index": chunk_index
        })
        
        chunk_index += 1
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text) - overlap:
            break
    
    return chunks