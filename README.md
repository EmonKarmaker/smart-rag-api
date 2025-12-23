\# Smart RAG API



A free, local Retrieval-Augmented Generation (RAG) API that answers questions based on information extracted from any document type.



\## Features



\- \*\*Multi-format Support\*\*: PDF, DOCX, TXT, Images (JPG, PNG), CSV, SQLite

\- \*\*OCR\*\*: Extracts text from images and scanned PDFs

\- \*\*100% Free \& Local\*\*: Uses Ollama (LLM) + SentenceTransformers (embeddings)

\- \*\*Vector Search\*\*: FAISS for fast similarity search

\- \*\*Image Questions\*\*: Ask questions with image attachments (base64)



\## Tech Stack



| Component | Technology |

|-----------|------------|

| API | FastAPI |

| LLM | Ollama (llama3.2) |

| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |

| Vector Store | FAISS |

| OCR | Tesseract (pytesseract) |



\## Setup



\### Prerequisites



1\. Python 3.10+

2\. Ollama installed with a model:

```bash

&nbsp;  ollama pull llama3.2

```

3\. Tesseract OCR installed



\### Installation

```bash

\# Clone the repo

git clone https://github.com/yourusername/smart-rag-api.git

cd smart-rag-api



\# Create virtual environment

python -m venv venv



\# Activate (Windows)

.\\venv\\Scripts\\Activate



\# Activate (Linux/Mac)

source venv/bin/activate



\# Install dependencies

pip install -r requirements.txt

```



\### Configuration



Create a `.env` file:

```env

EMBEDDING\_MODEL=all-MiniLM-L6-v2

OLLAMA\_MODEL=llama3.2

OLLAMA\_BASE\_URL=http://localhost:11434

UPLOAD\_DIR=uploads

VECTOR\_STORE\_DIR=data/vector\_store

```



\### Run the API

```bash

uvicorn app.main:app --reload

```



API available at: http://127.0.0.1:8000



\### Run Streamlit UI (Optional)

```bash

streamlit run streamlit\_app.py

```



UI available at: http://localhost:8501



\## API Endpoints



\### Upload Document

```bash

POST /upload

Content-Type: multipart/form-data



curl -X POST "http://127.0.0.1:8000/upload" \\

&nbsp; -F "file=@document.pdf"

```



Response:

```json

{

&nbsp; "success": true,

&nbsp; "file\_id": "abc123",

&nbsp; "filename": "document.pdf",

&nbsp; "chunks\_created": 8

}

```



\### Query Documents

```bash

POST /query

Content-Type: application/json



curl -X POST "http://127.0.0.1:8000/query" \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"question": "What is this document about?", "top\_k": 5}'

```



Response:

```json

{

&nbsp; "answer": "The document is about...",

&nbsp; "sources": \[...],

&nbsp; "context\_used": "...",

&nbsp; "model": "llama3.2"

}

```



\### Query with Image

```bash

POST /query



{

&nbsp; "question": "What is written in this image?",

&nbsp; "image\_base64": "base64\_encoded\_image\_string",

&nbsp; "top\_k": 5

}

```



\### Other Endpoints

| Endpoint | Method | Description |

|----------|--------|-------------|

| `/` | GET | API info |

| `/health` | GET | Health check |

| `/files` | GET | List uploaded files |

| `/stats` | GET | Vector store statistics |

| `/clear` | DELETE | Clear vector store |



\## Interactive Docs



Visit http://127.0.0.1:8000/docs for Swagger UI.



\## Project Structure

```

smart-rag-api/

├── app/

│   ├── main.py              # FastAPI app

│   ├── config.py            # Settings

│   ├── routers/

│   │   ├── upload.py        # Upload endpoints

│   │   └── query.py         # Query endpoints

│   ├── services/

│   │   ├── document\_parser.py  # Parse documents

│   │   ├── embeddings.py       # Generate embeddings

│   │   ├── vector\_store.py     # FAISS operations

│   │   └── llm.py              # Ollama integration

│   └── utils/

│       └── text\_processing.py  # Text chunking

├── data/vector\_store/       # FAISS index storage

├── uploads/                 # Uploaded files

├── streamlit\_app.py         # Web UI

├── Dockerfile               # Docker config

├── requirements.txt

├── .env

└── README.md

```



\## Docker (Optional)

```bash

\# Build

docker build -t smart-rag-api .



\# Run

docker run -p 8000:8000 smart-rag-api

```



Note: Docker setup requires Ollama running on host machine.



\## License



MIT

