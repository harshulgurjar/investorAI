from sqlalchemy import text
from database.postgres_sql import get_engine


def to_text(value):
    if value is None:
        return None
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def save_metrics(company: str, year, metrics: dict) -> None:
    engine = get_engine()

    query = """
    INSERT INTO financial_metrics (
        company,
        year,
        revenue,
        net_income,
        operating_income,
        cash_flow,
        total_assets,
        total_liabilities,
        risk_factors,
        growth_drivers
    )
    VALUES (
        :company,
        :year,
        :revenue,
        :net_income,
        :operating_income,
        :cash_flow,
        :total_assets,
        :total_liabilities,
        :risk_factors,
        :growth_drivers
    )
    """

    params = {
        "company": company,
        "year": str(year) if year else "",
        "revenue": to_text(metrics.get("Revenue") or metrics.get("revenue")),
        "net_income": to_text(metrics.get("Net Income") or metrics.get("net_income")),
        "operating_income": to_text(metrics.get("Operating Income") or metrics.get("operating_income")),
        "cash_flow": to_text(
            metrics.get("Cash Flow from Operating Activities")
            or metrics.get("cash_flow")
        ),
        "total_assets": to_text(metrics.get("Total Assets") or metrics.get("total_assets")),
        "total_liabilities": to_text(metrics.get("Total Liabilities") or metrics.get("total_liabilities")),
        "risk_factors": to_text(metrics.get("Top Risk Factors") or metrics.get("risk_factors")),
        "growth_drivers": to_text(metrics.get("Top Growth Drivers") or metrics.get("growth_drivers")),
    }

    print("Saving metrics params:", params)

    with engine.begin() as connection:
        connection.execute(text(query), params)

    print(f"Successfully saved metrics for {company} {year}")
