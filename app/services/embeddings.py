from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingService:
    """Handles text embedding generation using SentenceTransformers."""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.EMBEDDING_MODEL
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded!")
        return self.model
    
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        model = self._load_model()
        embedding = model.encode(text)
        return embedding.tolist()
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        model = self._load_model()
        embeddings = model.encode(texts)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


# Create singleton instance
embedding_service = EmbeddingService()