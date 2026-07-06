import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.ingest_documents import ingest_document
from vectorstore.faiss_store import FAISSVectorStore

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    upload_dir = Path("data/raw_pdfs")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISSVectorStore(
        index_path="data/faiss_index"
    )

    ingest_document(
        pdf_path=str(file_path),
        embeddings=embeddings,
        vector_store=vector_store
    )

    return {
        "message": "Document uploaded and indexed successfully.",
        "file_name": file.filename
    }
