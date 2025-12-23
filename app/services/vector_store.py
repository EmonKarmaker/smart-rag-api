import os
import json
import faiss
import numpy as np
from pathlib import Path

from app.config import settings
from app.services.embeddings import embedding_service


class VectorStore:
    """Handles vector storage and similarity search using FAISS."""
    
    def __init__(self):
        self.index = None
        self.documents = []  # Store chunk metadata
        self.store_dir = Path(settings.VECTOR_STORE_DIR)
        self.index_path = self.store_dir / "faiss.index"
        self.docs_path = self.store_dir / "documents.json"
        
        # Ensure directory exists
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and documents from disk."""
        if self.index_path.exists() and self.docs_path.exists():
            print("Loading existing vector store...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            print(f"Loaded {len(self.documents)} documents from store.")
        else:
            print("No existing vector store found. Starting fresh.")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        dimension = embedding_service.get_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
    
    def _save_index(self):
        """Save FAISS index and documents to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def add_documents(self, chunks: list[dict]) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of dicts with 'content', 'chunk_index', 'source', 'file_type'
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Extract text content
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = embedding_service.embed_texts(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype("float32")
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store document metadata
        for chunk in chunks:
            self.documents.append(chunk)
        
        # Save to disk
        self._save_index()
        
        print(f"Added {len(chunks)} chunks to vector store.")
        return len(chunks)
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = embedding_service.embed_text(query)
        query_array = np.array([query_embedding]).astype("float32")
        
        # Search FAISS
        distances, indices = self.index.search(query_array, top_k)
        
        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx].copy()
                doc["score"] = float(distances[0][i])
                results.append(doc)
        
        return results
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0
        }
    
    def clear(self):
        """Clear all documents from the store."""
        self._create_new_index()
        self._save_index()
        print("Vector store cleared.")


# Create singleton instance
vector_store = VectorStore()