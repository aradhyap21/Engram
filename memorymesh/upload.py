"""
Engram - Document upload and text extraction functionality.
Handles PDF and Word document uploads, extracts text content, and formats
text for further processing by the main application.
"""

from fastapi import HTTPException, UploadFile
from io import BytesIO
import fitz  # PyMuPDF
import docx


def extract_text_from_document(content: bytes, filename: str) -> str:
    """
    Extract text from PDF or Word documents.
    PDFs are opened with PyMuPDF (fitz).

    Args:
        content: Raw bytes of the document.
        filename: Original filename (used for error messages only).

    Returns:
        Extracted plain text.

    Raises:
        HTTPException: On extraction failure or unsupported format.
    """
    filename_lower = filename.lower()

    if filename_lower.endswith('.pdf'):
        try:
            doc = fitz.open(stream=BytesIO(content), filetype="pdf")
            text_parts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting text from PDF: {e}"
            ) from e

    elif filename_lower.endswith('.docx') or filename_lower.endswith('.doc'):
        try:
            doc = docx.Document(BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error extracting text from Word document: {e}"
            ) from e

    raise HTTPException(
        status_code=400,
        detail="Unsupported file format. Please upload PDF or Word documents."
    )


def format_extracted_text(text: str, filename: str) -> str:
    """
    Format extracted text: normalize whitespace and prepend file identifier.
    """
    cleaned = ' '.join(text.split())
    if len(cleaned) > 100:
        cleaned = f"Document: {filename}\n\n{cleaned}"
    return cleaned


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


async def process_uploaded_document(file: UploadFile) -> dict:
    """
    Process an uploaded document: validate, extract text, format, and return.

    Args:
        file: FastAPI UploadFile object.

    Returns:
        Dict with text, filename, file_type, text_length, success.

    Raises:
        HTTPException: 413 for oversized files, 400 for unsupported format or extraction errors.
    """
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds 50MB limit"
        )

    # file.filename is guaranteed non-None by FastAPI when a file is uploaded
    original_filename = file.filename
    if original_filename is None:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    filename_lower = original_filename.lower()

    if not (filename_lower.endswith('.pdf')
            or filename_lower.endswith('.docx')
            or filename_lower.endswith('.doc')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload PDF or Word documents."
        )

    text = extract_text_from_document(content, original_filename)
    formatted_text = format_extracted_text(text, original_filename)

    file_type = 'pdf' if filename_lower.endswith('.pdf') else 'docx'

    return {
        "text": formatted_text,
        "filename": original_filename,
        "file_type": file_type,
        "text_length": len(formatted_text),
        "success": True
    }