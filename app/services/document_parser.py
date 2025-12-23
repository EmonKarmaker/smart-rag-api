import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document
import pandas as pd
import sqlite3
from pathlib import Path

from app.config import settings
from app.utils.text_processing import chunk_text

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


class DocumentParser:
    """Handles parsing of various document types."""
    
    def __init__(self):
        self.supported_extensions = settings.SUPPORTED_EXTENSIONS
    
    def parse(self, file_path: str) -> dict:
        """
        Parse a document and return extracted text with metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with 'text', 'chunks', 'metadata'
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        # Route to appropriate parser
        if extension == ".pdf":
            text = self._parse_pdf(file_path)
        elif extension == ".docx":
            text = self._parse_docx(file_path)
        elif extension == ".txt":
            text = self._parse_txt(file_path)
        elif extension in [".jpg", ".jpeg", ".png"]:
            text = self._parse_image(file_path)
        elif extension == ".csv":
            text = self._parse_csv(file_path)
        elif extension == ".db":
            text = self._parse_sqlite(file_path)
        else:
            raise ValueError(f"No parser for: {extension}")
        
        # Create chunks
        chunks = chunk_text(text)
        
        # Add metadata to each chunk
        for chunk in chunks:
            chunk["source"] = file_path.name
            chunk["file_type"] = extension
        
        return {
            "text": text,
            "chunks": chunks,
            "metadata": {
                "filename": file_path.name,
                "file_type": extension,
                "total_chunks": len(chunks)
            }
        }
    
    def _parse_pdf(self, file_path: Path) -> str:
        """Extract text from PDF, with OCR fallback for scanned pages."""
        text_parts = []
        
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            # Try normal text extraction
            page_text = page.get_text()
            
            # If no text, try OCR
            if not page_text.strip():
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
            
            if page_text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        doc.close()
        return "\n\n".join(text_parts)
    
    def _parse_docx(self, file_path: Path) -> str:
        """Extract text from Word document."""
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs)
    
    def _parse_txt(self, file_path: Path) -> str:
        """Read plain text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _parse_image(self, file_path: Path) -> str:
        """Extract text from image using OCR."""
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
    
    def _parse_csv(self, file_path: Path) -> str:
        """Convert CSV to readable text."""
        df = pd.read_csv(file_path)
        
        # Convert to readable format
        lines = []
        lines.append(f"Columns: {', '.join(df.columns.tolist())}")
        lines.append(f"Total rows: {len(df)}")
        lines.append("\nData:")
        
        # Add each row as text
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            lines.append(row_text)
        
        return "\n".join(lines)
    
    def _parse_sqlite(self, file_path: Path) -> str:
        """Extract data from SQLite database."""
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        lines = []
        for table in tables:
            table_name = table[0]
            lines.append(f"\n=== Table: {table_name} ===")
            
            # Get table data
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            lines.append(f"Columns: {', '.join(df.columns.tolist())}")
            lines.append(f"Rows: {len(df)}")
            
            # Add data
            for idx, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                lines.append(row_text)
        
        conn.close()
        return "\n".join(lines)


# Create singleton instance
document_parser = DocumentParser()