import streamlit as st
import requests
import base64

# API Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Smart RAG API",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Smart RAG API")
st.markdown("Upload documents and ask questions about them")

# Sidebar
with st.sidebar:
    st.header("📊 Status")
    
    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
        if health["status"] == "healthy":
            st.success("✅ API Connected")
            st.success(f"✅ LLM: {health['llm_provider']}/{health['llm_model']}")
        else:
            st.warning(f"⚠️ Status: {health['status']}")
        
        st.metric("Documents", health["vector_store"]["total_documents"])
    except Exception as e:
        st.error("❌ API not running")
        st.info("Start API with: `uvicorn app.main:app --reload`")
    
    st.divider()
    
    # Clear button
    if st.button("🗑️ Clear All Documents"):
        try:
            response = requests.delete(f"{API_URL}/clear", timeout=10)
            if response.status_code == 200:
                st.success("Vector store cleared!")
                st.rerun()
        except:
            st.error("Failed to clear")
    
    st.divider()
    st.markdown("### ℹ️ Supported Files")
    st.markdown("""
    - 📄 PDF
    - 📝 DOCX, TXT
    - 🖼️ JPG, PNG
    - 📊 CSV
    - 🗃️ SQLite DB
    """)

# Main content - Two columns
col1, col2 = st.columns(2)

# Left column - Upload
with col1:
    st.header("📁 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "csv", "db"],
        help="Supported: PDF, DOCX, TXT, Images, CSV, SQLite"
    )
    
    if uploaded_file:
        # Show file preview
        st.text(f"📄 {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        if st.button("📤 Upload & Process", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.json(result)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to upload: {e}")
    
    # Show uploaded files
    st.divider()
    st.subheader("📂 Uploaded Files")
    try:
        files_response = requests.get(f"{API_URL}/files", timeout=10).json()
        if files_response["files"]:
            for f in files_response["files"]:
                file_icon = "📄"
                if f["file_type"] in [".jpg", ".jpeg", ".png"]:
                    file_icon = "🖼️"
                elif f["file_type"] == ".pdf":
                    file_icon = "📕"
                elif f["file_type"] == ".csv":
                    file_icon = "📊"
                elif f["file_type"] == ".db":
                    file_icon = "🗃️"
                
                size_kb = f["size_bytes"] / 1024
                st.text(f"{file_icon} {f['filename']} ({size_kb:.1f} KB)")
        else:
            st.info("No files uploaded yet")
    except:
        st.warning("Could not fetch files")

# Right column - Query
with col2:
    st.header("💬 Ask Questions")
    
    question = st.text_area(
        "Your question:",
        placeholder="What is this document about?",
        height=100
    )
    
    # Optional image upload for OCR
    with st.expander("📷 Add Image for OCR (Optional)"):
        image_file = st.file_uploader(
            "Upload image to extract text and ask questions",
            type=["jpg", "jpeg", "png"],
            key="image_query"
        )
        if image_file:
            st.image(image_file, caption="Uploaded Image", width=200)
    
    top_k = st.slider("Number of sources to retrieve", 1, 10, 3)
    
    if st.button("🔍 Search & Answer", type="primary"):
        if not question:
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching and generating answer..."):
                try:
                    # Prepare request
                    payload = {
                        "question": question,
                        "top_k": top_k
                    }
                    
                    # Add image if provided
                    if image_file:
                        image_base64 = base64.b64encode(image_file.getvalue()).decode()
                        payload["image_base64"] = image_base64
                    
                    response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display answer
                        st.subheader("📝 Answer")
                        st.markdown(result["answer"])
                        
                        # Display image text if any
                        if result.get("image_text"):
                            st.subheader("🖼️ Text Extracted from Image")
                            st.text_area("OCR Result", result["image_text"], height=100)
                        
                        # Display sources
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result["sources"], 1):
                            score = source["score"]
                            with st.expander(f"Source {i}: {source['source']} (relevance: {1/(1+score):.1%})"):
                                st.write(source["preview"])
                        
                        # Model info
                        st.caption(f"🤖 Model: {result['model']}")
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"Error: {error_detail}")
                        
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The LLM might be slow to respond.")
                except Exception as e:
                    st.error(f"Failed to query: {e}")

# Footer
st.divider()
st.caption("Built with FastAPI, FAISS, SentenceTransformers & Streamlit |  HuggingFace (web)")