import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from vectorstore.faiss_store import FAISSVectorStore, Retriever
from rag.kpi_extractor_rag import extract_financial_metrics
from database.save_metrics import save_metrics

load_dotenc()

def parse_company_year(pdf_file:Path)->tuple[str,str]:
    """parse comapaany and year from a pdf filename.
    supports name like'2024_apple.pdf'

    """
    stem=pdf_file.stem
    parts=stem.splits("_")
    if parts and parts[0].isdidgit():
        year=parts[0]
        company=parts[1]
    elif len(parts)>=2:
        comapany=parts[0]
        year=parts[1]
    else:
        compaany=stem
        year=""
    return comapany,year

def ingest_document(pdf_path:str,embeddings,vector_store)->None:
    pdf_file=Path(pdf_path)
    company,year=parse_company_year(pdf_file)
    print(f"Ingesting{pdf_file.name}for{company},year={year!r}")
    converter=PDFToMarkdownConverter()
    markdown_content=converter.convert(pdf_path,output_dir="data/markdown")
    chunks=chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )
    print(f"generated{len(chunks)}chunks for {pdf_file.name}")
    vector_store.upload_chunks(
        chunks=chunks,
        embeddings=embeddings,
        company=company,
        year=year,
        source_file=pdf_file.name
    )
    metrics = extract_financial_metrics(
        retriever=Retriever(vector_store, embeddings),
        company=company,
        year=int(year) if year.isdigit() else None
    )

    if metrics:
        save_metrics(
            company=company,
            year=int(year) if year.isdigit() else None,
            metrics=metrics
        )
def ingest_directory(input_dir: str) -> None:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISSVectorStore(
        index_path="data/faiss_index"
    )

    pdf_files = list(Path(input_dir).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF(s)")

    for pdf_file in pdf_files:
        ingest_document(
            pdf_path=str(pdf_file),
            embeddings=embeddings,
            vector_store=vector_store
        )
    

if __name__=="__main__":
    ingest_directory("data/raw_pdfs")
    
