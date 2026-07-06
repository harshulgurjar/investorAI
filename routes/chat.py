import os

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings

from vectorstore.faiss_store import FAISSVectorStore, Retriever

load_dotenv()

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
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

        docs = retriever.invoke(
            query=request.question,
            company=request.company,
            year=request.year,
            top_k=5
        )

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are an expert financial analyst.

Use only the following context from corporate reports to answer the user's question.
If the context does not contain enough information, say you do not have enough data.

Context:
{context}

User Question:
{request.question}

Answer:
"""

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources_found": len(docs)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
