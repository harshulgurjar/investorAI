from pydantic import BaseModel,Field
from dotenv import load_dotenv
from llm.groq_llm import get_structured_completion
from vectorstore.faiss_store import FAISSVectorStore, Retriever

load_dotenv()


class FinancialMetrics(BaseModel):
    revenue: str | int | None = Field(None, alias="Revenue")
    net_income: str | int | None = Field(None, alias="Net Income")
    operating_income: str | int | None = Field(None, alias="Operating Income")
    cash_flow: str | int | None = Field(None, alias="Cash Flow from Operating Activities")
    total_assets: str | int | None = Field(None, alias="Total Assets")
    total_liabilities: str | int | None = Field(None, alias="Total Liabilities")
    risk_factors: str | list | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list | None = Field(None, alias="Top Growth Drivers")

def retrieve_context(retriever: Retriever, company: str, year: int) -> str:
    query = f"""
    Annual report financial statements,
    income statement,
    balance sheet,
    cash flow statement,
    risks,
    growth drivers,
    financial performance
    for {company} fiscal year {year}
    """

    documents = retriever.invoke(
        query=query,
        company=company,
        year=year,
        top_k=20
    )
    return "\n\n".join([d.page_content for d in documents])


def build_extraction_prompt(company:str,year:int,context:str)->str:
    return f"""
    You are an expert financial analyst.

    Extract KPIs from the context below for {company}, fiscal year {year}.

    Context:
    {context}

    Extract the following information:

1. Revenue
2. Net Income
3. Operating Income
4. Cash Flow from Operating Activities
5. Total Assets
6. Total Liabilities
7. Top Risk Factors
8. Top Growth Drivers

Instructions:
- Use only the provided context.
- Return null if unavailable.
- Financial values must match the report exactly.
- Risk factors should be concise.
- Growth drivers should be concise.
- Return valid JSON only.
    """


def extract_financial_metrics(retriever: Retriever, company: str, year: int) -> dict:
    context = retrieve_context(retriever, company, year)
    prompt=build_extraction_prompt(company,year,context)
    metrics=get_structured_completion(prompt,FinancialMetrics)
    return metrics.model_dump(exclude_none=True)

def main()-> None:
    from langchain_huggingface import HuggingFaceEmbeddings
    from database.save_metrics import save_metrics

    company="Apple"
    year = 2024
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

    results = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=year
    )
    print(f"\nExtracted KPIs for {company} {year}\n")

    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print("-" * 80)

    save_metrics(
        company=company,
        year=year,
        metrics=results
    )

if __name__=="__main__":
    main()
