from pathlib import Path
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
load_dotenv()


def read_markdown(markdown_file:str)->str:
    """
    read markdown file and return its content as a string.
    Args:
        markdown_file:path to the markdown file.
    Returns:
        content of the markdown file.
    """
    return Path(markdown_file).read_text(encoding="utf-8")

def chunk_markdown(markdown_text:str,embeddings)->list[Document]:
    """
    chunk markdown text using semantic chunker.
    Args:
        markdown_text:markdown content to chunk.
        embeddings:embedding model to use for chunking.
    Returns:
        list of document chunks.
    """
    markdown_content =read_markdown(markdown_file)
    splitter=SemanticChunker(embeddings=embeddings,breakpoint_threshold_type="percentile")
    return splitter.create_documents([markdown_content])

if __name__ == "__main__":
    import os
    from langchain_groq import ChatGroq 
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm=ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment variables")
    markdown_file="data\markdown\2024_Apple.md"
    chunks=chunk_markdown(markdown_file,embeddings)
    print(f"Generated {len(chunks)} chunks.")
    for index,chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()
    
    
    
