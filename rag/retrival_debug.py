import sys

from langchain_huggingface import HuggingFaceEmbeddings
from vectorstore.faiss_store import FAISSVectorStore, Retriever


def search_vectorstore(query: str, top: int = 5):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISSVectorStore(
        index_path="data/faiss_index"
    )

    retriever = Retriever(
        vector_store=vector_store,
        embeddings=embeddings
    )

    results = retriever.invoke(
        query=query,
        top_k=top
    )

    print(f"Query: {query!r}")
    print(f"Top: {top}")
    print(f"Results: {len(results)}\n")

    for idx, doc in enumerate(results, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")

        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."

        print(f"Result {idx}")
        print(f"  content snippet: {snippet}")
        print("  " + "-" * 60)

    if not results:
        print("No results returned. Check if FAISS index exists at data/faiss_index.")

    return results


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m rag.retrieval_debug "your query here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_vectorstore(query)


if __name__ == "__main__":
    main()
