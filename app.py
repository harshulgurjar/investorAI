import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.ingest_documents import parse_company_year
from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunking import chunk_markdown
from vectorstore.faiss_store import FAISSVectorStore, Retriever
from rag.kpi_extractor_rag import extract_financial_metrics
from rag.rag_chain import answer_question

load_dotenv()

st.set_page_config(
    page_title="InvestorAI",
    page_icon="📊",
    layout="wide"
)

UPLOAD_DIR = Path("data/raw_pdfs")
MARKDOWN_DIR = "data/markdown"
INDEX_PATH = "data/faiss_index"
DATA_FILE = Path("saved_metrics.csv")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.title("📊 InvestorAI")
st.caption("Streamlit-only AI-powered investor intelligence platform")


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vector_store():
    return FAISSVectorStore(index_path=INDEX_PATH)


def save_metrics_csv(company, year, metrics):
    row = {
        "company": company,
        "year": year,
        "revenue": metrics.get("Revenue") or metrics.get("revenue"),
        "net_income": metrics.get("Net Income") or metrics.get("net_income"),
        "operating_income": metrics.get("Operating Income") or metrics.get("operating_income"),
        "cash_flow": metrics.get("Cash Flow from Operating Activities") or metrics.get("cash_flow"),
        "total_assets": metrics.get("Total Assets") or metrics.get("total_assets"),
        "total_liabilities": metrics.get("Total Liabilities") or metrics.get("total_liabilities"),
        "risk_factors": metrics.get("Top Risk Factors") or metrics.get("risk_factors"),
        "growth_drivers": metrics.get("Top Growth Drivers") or metrics.get("growth_drivers"),
    }

    for key in ["risk_factors", "growth_drivers"]:
        if isinstance(row[key], list):
            row[key] = "\n".join(row[key])

    df_new = pd.DataFrame([row])

    if DATA_FILE.exists():
        df_old = pd.read_csv(DATA_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(DATA_FILE, index=False)


def load_metrics_csv():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()


embeddings = load_embeddings()
vector_store = load_vector_store()

tab1, tab2, tab3 = st.tabs(["📄 Upload Report", "💬 Chat", "📈 Saved Metrics"])

with tab1:
    st.header("Upload Annual Report PDF")

    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"])

    if uploaded_file and st.button("Upload, Index & Extract Metrics"):
        file_path = UPLOAD_DIR / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        company, year = parse_company_year(file_path)

        st.info(f"Detected company: {company}, year: {year}")

        try:
            with st.spinner("Converting PDF to markdown..."):
                converter = PDFToMarkdownConverter()
                markdown_file = converter.convert_pdf(
                    str(file_path),
                    output_dir=MARKDOWN_DIR
                )

            with st.spinner("Creating semantic chunks..."):
                chunks = chunk_markdown(
                    markdown_file=markdown_file,
                    embeddings=embeddings
                )

            with st.spinner("Saving to FAISS vector store..."):
                vector_store.upload_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    company=company,
                    year=year,
                    source_file=file_path.name
                )

            with st.spinner("Extracting financial metrics..."):
                metrics = extract_financial_metrics(
                    retriever=Retriever(vector_store, embeddings),
                    company=company,
                    year=year
                )

            if metrics:
                save_metrics_csv(company, year, metrics)
                st.success("Report indexed and metrics saved successfully.")
                st.json(metrics)
            else:
                st.warning("Report indexed, but no metrics were extracted.")

        except Exception as e:
            st.error(f"Processing failed: {e}")

with tab2:
    st.header("Chat with Annual Reports")

    company = st.text_input("Company", placeholder="apple")
    year = st.text_input("Year", placeholder="2024")
    question = st.text_area("Question", placeholder="What was the company's revenue?")

    if st.button("Ask AI"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Thinking..."):
                    retriever = Retriever(vector_store, embeddings)

                    docs = retriever.invoke(
                        query=question,
                        company=company if company else None,
                        year=year if year else None,
                        top_k=5
                    )

                    context = "\n\n".join([doc.page_content for doc in docs])

                    answer = answer_question(
                        question=question,
                        context=context
                    )

                st.subheader("Answer")
                st.write(answer)
                st.caption(f"Sources found: {len(docs)}")

            except Exception as e:
                st.error(f"Chat failed: {e}")

with tab3:
    st.header("Saved Financial Metrics")

    df = load_metrics_csv()

    if df.empty:
        st.info("No metrics saved yet.")
    else:
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv,
            file_name="financial_metrics.csv",
            mime="text/csv"
        )
