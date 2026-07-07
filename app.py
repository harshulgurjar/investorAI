import requests
import streamlit as st

API_BASE = st.secrets["API_BASE"]
API = f"{API_BASE}/api"

st.set_page_config(
    page_title="Investor Intelligence",
    layout="wide"
)

st.title("AI-Powered Investor Intelligence Platform")
st.caption("Streamlit Frontend + FastAPI Backend + FAISS + Groq + PostgreSQL")

# Health check
try:
    health = requests.get(f"{API_BASE}/health", timeout=5)
    if health.status_code == 200:
        st.success("Backend connected")
    else:
        st.warning("Backend running but health check failed")
except Exception:
    st.error("FastAPI backend not connected. Run: uvicorn main:app --reload --port 8080")

tab1, tab2, tab3 = st.tabs(["Upload Report", "Chat", "Saved Metrics"])


with tab1:
    st.header("Upload Annual Report PDF")

    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload and Ingest"):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            with st.spinner("Uploading and processing PDF..."):
                try:
                    response = requests.post(
                        f"{API}/upload",
                        files=files,
                        timeout=300
                    )

                    if response.status_code == 200:
                        st.success(response.json().get("message", "Uploaded successfully"))
                        st.json(response.json())
                    else:
                        st.error(response.text)

                except Exception as e:
                    st.error(f"Upload failed: {e}")


with tab2:
    st.header("Chat with Annual Reports")

    company = st.text_input("Company", placeholder="Apple")
    year = st.number_input("Year", min_value=2000, max_value=2035, value=2024)
    question = st.text_area("Question", placeholder="What was the company's revenue?")

    if st.button("Ask AI"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            payload = {
                "question": question,
                "company": company if company else None,
                "year": int(year) if year else None
            }

            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API}/chat",
                        json=payload,
                        timeout=120
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.subheader("Answer")
                        st.write(data.get("answer"))

                        if "sources_found" in data:
                            st.caption(f"Sources found: {data['sources_found']}")
                    else:
                        st.error(response.text)

                except Exception as e:
                    st.error(f"Chat failed: {e}")


with tab3:
    st.header("Saved Financial Metrics")

    if st.button("Refresh Metrics"):
        st.rerun()

    try:
        response = requests.get(f"{API}/metrics", timeout=30)

        if response.status_code == 200:
            metrics = response.json()

            if metrics:
                st.dataframe(metrics, use_container_width=True)
            else:
                st.info("No metrics saved yet.")
        else:
            st.error(response.text)

    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
