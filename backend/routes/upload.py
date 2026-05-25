from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil
import uuid

from services.text import extract_pdf_text
from services.chunking import chunk_documents
from services.embedding import store_in_chroma
from services.text import extract_pdf_text



router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Unique document ID
    document_id = str(uuid.uuid4())

    file_path = UPLOAD_DIR / f"{document_id}.pdf"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Extract text
    docs = extract_pdf_text(
        str(file_path)
    )

    text = extract_pdf_text(
    file_path
    )   

    # Chunk
    chunks = chunk_documents(
        docs
    )

    # Store embeddings
    store_in_chroma(
        chunks,
        document_id
    )

    return {
        "success": True,
        "document_id": document_id,
        "filename": file.filename
    }