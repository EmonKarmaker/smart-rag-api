import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Embedding model (runs locally, free)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # LLM Provider: "ollama" (local) or "huggingface" (web)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "huggingface")
    
    # Ollama settings (local, free)
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # HuggingFace settings (web, free)
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    
    # Directories
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    
    # Chunking settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Supported file types
    SUPPORTED_EXTENSIONS: list = [
        ".pdf", ".docx", ".txt", 
        ".jpg", ".jpeg", ".png",
        ".csv", ".db"
    ]

settings = Settings()