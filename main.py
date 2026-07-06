from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.health import router as health_router
from routes.upload import router as upload_router
from routes.chat import router as chat_router
from routes.metrics import router as metrics_router

app = FastAPI(
    title="AI-Powered Investor Intelligence API",
    description="Backend API for Investor Intelligence Platform",
    version="1.0.0"
)

# Allow Streamlit frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(
    health_router,
    tags=["Health"]
)

app.include_router(
    upload_router,
    prefix="/api",
    tags=["Upload"]
)

app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"]
)

app.include_router(
    metrics_router,
    prefix="/api",
    tags=["Metrics"]
)


@app.get("/")
def root():
    return {
        "message": "AI-Powered Investor Intelligence API",
        "status": "Running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
