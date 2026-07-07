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


def process_document(file_path: str):
    """
    Background task:
    1. Index uploaded PDF into FAISS
    2. Later you can add metrics extraction + DB save here
    """

    try:
        print(f"Indexing started: {file_path}")

        ingest_document(
            pdf_path=file_path,
            embeddings=embeddings,
            vector_store=vector_store
        )

        print(f"Indexing completed: {file_path}")

        # Next step:
        # metrics = extract_financial_metrics(file_path)
        # save_metrics(metrics)

    except Exception as e:
        print(f"Background processing failed for {file_path}: {e}")


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

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

    finally:
        await file.close()

    background_tasks.add_task(process_document, str(file_path))

    return {
        "message": "Document uploaded successfully. Processing started in background.",
        "file_name": safe_name
    }
