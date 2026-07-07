import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.ingest_documents import ingest_document
from vectorstore.faiss_store import FAISSVectorStore

router = APIRouter()

UPLOAD_DIR = Path("data/raw_pdfs")
INDEX_PATH = "data/faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISSVectorStore(index_path=INDEX_PATH)


def index_document(file_path: str):
    ingest_document(
        pdf_path=file_path,
        embeddings=embeddings,
        vector_store=vector_store
    )


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    background_tasks.add_task(index_document, str(file_path))

    return {
        "message": "Document uploaded successfully. Indexing started in background.",
        "file_name": safe_name
    }
