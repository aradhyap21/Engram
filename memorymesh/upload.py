"""
MemoryMesh - Document upload and text extraction functionality.
Handles PDF and Word document uploads, extracts text content, and formats
text for further processing by the main application.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from io import BytesIO
import fitz  # PyMuPDF
import docx
import os
from pathlib import Path
from typing import Union
def extract_text_from_document(content: bytes, filename: str) -> str:
    """
    Extract text from PDF, Word, or extracted binary files.
    PDFs are opened with PyMuPDF (fitz), should handle 800+ pages.
    """
    # Determine file type based on extension
    filename_lower = filename.lower()

    if filename_lower.endswith('.pdf'):
        # Extract text from PDF using PyMuPDF
        try:
            doc = fitz.open(stream=BytesIO(content), filetype="pdf")
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting text from PDF: {str(e)}"
            )

    elif filename_lower.endswith('.docx') or filename_lower.endswith('.doc'):
        # Extract text from Word document
        try:
            doc = docx.Document(BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting text from Word document: {str(e)}"
            )

    else:
        # Try to extract text from unknown format
        try:
            # Decode as UTF-8 and return if successful
            return content.decode('utf-8', errors='ignore')
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload PDF or Word documents."
            )
class DocumentRequest(BaseModel):
    text: str
    filename: str
    file_type: str
class UploadResponse(BaseModel):
    success: bool
    text_length: int
    text_preview: str
    filename: str
    file_type: str
def format_extracted_text(text: str, filename: str) -> str:
    """
    Format extracted text for better processing.
    Clean up, normalize newlines, and remove excessive whitespace.
    """
    # Remove excessive whitespace and normalize newlines
    text = ' '.join(text.split())

    # Add file identifier if text is substantial
    if len(text) > 100:
        text = f"Document: {filename}\n\n{text}"

    return text
# Helper function to convert extracted text to MemoryRequest format
async def process_uploaded_document(file: UploadFile) -> dict:
    """
    Process an uploaded document file and return formatted text.
    Compatible with FastAPI's UploadFile.
    """
    # Get file content
    content = await file.read()

    # Validate file size (max 50MB)
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds 50MB limit"
        )

    # Determine file type
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        file_type = 'pdf'
    elif filename.endswith('.docx') or filename.endswith('.doc'):
        file_type = 'docx'
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload PDF or Word documents."
        )

    # Extract text
    text = extract_text_from_document(content, file.filename)

    # Format extracted text
    formatted_text = format_extracted_text(text, file.filename)

    return {
        "text": formatted_text,
        "filename": file.filename,
        "file_type": file_type,
        "text_length": len(formatted_text),
        "success": True
    }