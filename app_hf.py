import streamlit as st
import os
import uuid
import base64
import tempfile
from pathlib import Path
from io import BytesIO

# Set up environment
os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
os.environ["LLM_PROVIDER"] = "huggingface"
os.environ["HUGGINGFACE_MODEL"] = "HuggingFaceH4/zephyr-7b-beta"

# Import after setting env
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document
import pandas as pd
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ============== CONFIG ==============
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png", ".csv", ".db"]

# ============== TEXT PROCESSING ==============
def chunk_text(text: str) -> list[dict]:
    if not text or not text.strip():
        return []
    
    text = " ".join(text.strip().split())
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_content = text[start:end]
        
        if end < len(text):
            last_period = chunk_content.rfind(". ")
            if last_period > CHUNK_SIZE * 0.5:
                chunk_content = chunk_content[:last_period + 1]
                end = start + last_period + 1
        
        chunks.append({"content": chunk_content.strip(), "chunk_index": chunk_index})
        chunk_index += 1
        start = end - CHUNK_OVERLAP
        
        if start >= len(text) - CHUNK_OVERLAP:
            break
    
    return chunks

# ============== DOCUMENT PARSER ==============
def parse_pdf(file_bytes) -> str:
    text_parts = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if not page_text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            try:
                page_text = pytesseract.image_to_string(img)
            except:
                page_text = ""
        if page_text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
    doc.close()
    return "\n\n".join(text_parts)

def parse_docx(file_bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)

def parse_txt(file_bytes) -> str:
    return file_bytes.decode("utf-8")

def parse_image(file_bytes) -> str:
    img = Image.open(BytesIO(file_bytes))
    try:
        text = pytesseract.image_to_string(img)
    except:
        text = "[OCR not available]"
    return text

def parse_csv(file_bytes) -> str:
    df = pd.read_csv(BytesIO(file_bytes))
    lines = [f"Columns: {', '.join(df.columns.tolist())}", f"Total rows: {len(df)}", "\nData:"]
    for idx, row in df.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
        lines.append(row_text)
    return "\n".join(lines)

def parse_document(file_bytes, filename) -> dict:
    ext = Path(filename).suffix.lower()
    
    if ext == ".pdf":
        text = parse_pdf(file_bytes)
    elif ext == ".docx":
        text = parse_docx(file_bytes)
    elif ext == ".txt":
        text = parse_txt(file_bytes)
    elif ext in [".jpg", ".jpeg", ".png"]:
        text = parse_image(file_bytes)
    elif ext == ".csv":
        text = parse_csv(file_bytes)
    else:
        text = ""
    
    chunks = chunk_text(text)
    for chunk in chunks:
        chunk["source"] = filename
        chunk["file_type"] = ext
    
    return {"text": text, "chunks": chunks, "metadata": {"filename": filename, "file_type": ext, "total_chunks": len(chunks)}}

# ============== EMBEDDING SERVICE ==============
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> np.ndarray:
    model = load_embedding_model()
    return model.encode(texts)

# ============== VECTOR STORE ==============
class SimpleVectorStore:
    def __init__(self):
        self.index = None
        self.documents = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
    
    def add_documents(self, chunks: list[dict]):
        if not chunks:
            return 0
        
        texts = [c["content"] for c in chunks]
        embeddings = embed_texts(texts).astype("float32")
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings)
        self.documents.extend(chunks)
        return len(chunks)
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = embed_texts([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(distances[0][i])
                results.append(doc)
        return results
    
    def clear(self):
        self.index = None
        self.documents = []

# ============== LLM SERVICE ==============
@st.cache_resource
def get_llm_client():
    return InferenceClient(
        model="HuggingFaceH4/zephyr-7b-beta",
        token=os.getenv("HUGGINGFACE_API_KEY", st.secrets.get("HUGGINGFACE_API_KEY", ""))
    )

def generate_answer(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question based ONLY on the context provided above.
- If the context doesn't contain enough information, say "I don't have enough information."
- Be concise and direct.

QUESTION: {question}

ANSWER:"""
    
    try:
        client = get_llm_client()
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ============== STREAMLIT APP ==============
st.set_page_config(page_title="Smart RAG API", page_icon="🔍", layout="wide")

st.title("🔍 Smart RAG API")
st.markdown("Upload documents and ask questions about them - Powered by LangChain & HuggingFace")

# Initialize vector store in session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = SimpleVectorStore()

# Sidebar
with st.sidebar:
    st.header("📊 Status")
    st.success("✅ App Running")
    st.metric("Documents", len(st.session_state.vector_store.documents))
    
    st.divider()
    
    if st.button("🗑️ Clear All Documents"):
        st.session_state.vector_store.clear()
        st.success("Cleared!")
        st.rerun()
    
    st.divider()
    st.markdown("### ℹ️ Supported Files")
    st.markdown("📄 PDF, 📝 DOCX, TXT, 🖼️ JPG, PNG, 📊 CSV")

# Main content
col1, col2 = st.columns(2)

# Upload section
with col1:
    st.header("📁 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "csv"],
        help="Supported: PDF, DOCX, TXT, Images, CSV"
    )
    
    if uploaded_file:
        if st.button("📤 Upload & Process", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    parsed = parse_document(file_bytes, uploaded_file.name)
                    chunks_added = st.session_state.vector_store.add_documents(parsed["chunks"])
                    st.success(f"✅ Added {chunks_added} chunks from {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Query section
with col2:
    st.header("💬 Ask Questions")
    
    question = st.text_area("Your question:", placeholder="What is this document about?", height=100)
    
    with st.expander("📷 Add Image for OCR (Optional)"):
        image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img")
        if image_file:
            st.image(image_file, width=200)
    
    top_k = st.slider("Number of sources", 1, 10, 3)
    
    if st.button("🔍 Search & Answer", type="primary"):
        if not question:
            st.warning("Please enter a question")
        elif len(st.session_state.vector_store.documents) == 0:
            st.warning("Please upload documents first")
        else:
            with st.spinner("Searching and generating answer..."):
                # Handle image OCR if provided
                image_text = ""
                if image_file:
                    try:
                        img_bytes = image_file.getvalue()
                        image_text = parse_image(img_bytes)
                    except:
                        pass
                
                # Search
                search_query = f"{question} {image_text[:200]}" if image_text else question
                results = st.session_state.vector_store.search(search_query, top_k)
                
                if results:
                    # Build context
                    context = "\n\n".join([f"[Source: {r['source']}]\n{r['content']}" for r in results])
                    
                    # Generate answer
                    answer = generate_answer(question, context)
                    
                    st.subheader("📝 Answer")
                    st.markdown(answer)
                    
                    if image_text:
                        st.subheader("🖼️ Text from Image")
                        st.text(image_text[:500])
                    
                    st.subheader("📚 Sources")
                    for i, r in enumerate(results, 1):
                        with st.expander(f"Source {i}: {r['source']}"):
                            st.write(r["content"][:300] + "...")
                else:
                    st.warning("No relevant documents found")

st.divider()
st.caption("Built with FastAPI, FAISS, LangChain, SentenceTransformers & HuggingFace | 100% Free")