from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.knowledge_base import router as kb_router
from app.api.document import router as doc_router
from app.api.chat import router as chat_router
from app.db.database import init_db
from app.services.storage import get_client, init_milvus


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    try:
        client = get_client()
        init_milvus(client)
    except Exception:
        print("Warning: Milvus not available — vector search will fail at runtime.")
    yield
    # Shutdown


app = FastAPI(
    title="Multi-Modal RAG System",
    description="RAG system supporting text + image retrieval and Qwen-VL QA",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(kb_router)
app.include_router(doc_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
