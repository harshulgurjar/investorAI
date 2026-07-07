# InvestorAI 📊

AI-powered investor intelligence platform that analyzes company annual reports using RAG, extracts key financial metrics, and enables natural language Q&A over uploaded PDFs.

## Features

- Upload annual report PDFs
- Convert PDF reports into markdown
- Chunk financial documents for retrieval
- Store document embeddings using FAISS
- Ask questions from uploaded reports
- Extract financial KPIs automatically:
  - Revenue
  - Net Income
  - Operating Income
  - Cash Flow
  - Total Assets
  - Total Liabilities
  - Risk Factors
  - Growth Drivers
- Save extracted metrics in PostgreSQL
- Streamlit frontend with FastAPI backend

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Database:** PostgreSQL
- **Vector Store:** FAISS
- **LLM:** Groq
- **Embeddings:** HuggingFace Sentence Transformers
- **PDF Processing:** PyMuPDF4LLM
- **Frameworks:** LangChain, SQLAlchemy

## Project Architecture

```text
User
 │
 ▼
Streamlit Frontend
 │
 ▼
FastAPI Backend
 │
 ├── Upload PDF
 ├── Convert PDF to Markdown
 ├── Chunk Document
 ├── Generate Embeddings
 ├── Store in FAISS
 ├── Extract Financial Metrics using LLM
 └── Save Metrics to PostgreSQL



investorAI/
│
├── app.py
├── main.py
├── requirements.txt
│
├── routes/
│   ├── upload.py
│   ├── chat.py
│   ├── metrics.py
│   └── health.py
│
├── ingestion/
│   ├── ingest_documents.py
│   ├── pdf_to_markdown.py
│   └── semantic_chunking.py
│
├── rag/
│   └── kpi_extractor_rag.py
│
├── vectorstore/
│   └── faiss_store.py
│
├── database/
│   ├── postgres_sql.py
│   ├── create_table.py
│   └── save_metrics.py
