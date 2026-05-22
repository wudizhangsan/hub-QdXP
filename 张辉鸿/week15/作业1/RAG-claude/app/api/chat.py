"""Multi-modal chat/QA endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import get_db
from app.models.db_models import KnowledgeBase, Document, TextChunk, Image as ImageModel
from app.models.schemas import ChatRequest, ChatResponse, SourceInfo
from app.services.embedding import embed_text, embed_multimodal_query
from app.services.retrieval import hybrid_search
from app.services.qa import answer_question
from app.services.storage import get_client

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    # Validate KB exists
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == body.kb_id))
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Encode query: text embedding for text search, multimodal for image search
    try:
        text_emb = embed_text(body.question)
        img_query_emb = embed_multimodal_query(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # Multi-modal retrieval from Milvus
    milvus_client = get_client()
    text_hits, image_hits = hybrid_search(
        milvus_client, body.kb_id, text_emb, img_query_emb,
    )

    # Enrich results with filename info from SQLite
    doc_cache: dict[str, str] = {}
    async def get_filename(doc_id: str) -> str:
        if doc_id not in doc_cache:
            res = await db.execute(select(Document).where(Document.id == doc_id))
            doc = res.scalar_one_or_none()
            doc_cache[doc_id] = doc.filename if doc else "unknown"
        return doc_cache[doc_id]

    for hit in text_hits:
        hit["filename"] = await get_filename(hit["doc_id"])
    for hit in image_hits:
        hit["filename"] = await get_filename(hit["doc_id"])

    # Generate answer via Qwen-VL
    try:
        answer = answer_question(body.question, text_hits, image_hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA generation failed: {e}")

    # Build source list
    sources = _build_sources(text_hits, image_hits)

    return ChatResponse(
        answer=answer,
        sources=sources,
        kb_id=body.kb_id,
        question=body.question,
    )


def _build_sources(text_hits: list[dict], image_hits: list[dict]) -> list[SourceInfo]:
    sources = []
    for hit in text_hits:
        snippet = hit.get("text", "")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        sources.append(SourceInfo(
            doc_filename=hit.get("filename", "unknown"),
            doc_id=hit["doc_id"],
            page_num=hit.get("page_start"),
            content_type="text",
            snippet=snippet,
        ))
    for hit in image_hits:
        caption = hit.get("caption", "")
        if not caption:
            caption = f"Image from page {hit.get('page_num', '?')}"
        sources.append(SourceInfo(
            doc_filename=hit.get("filename", "unknown"),
            doc_id=hit["doc_id"],
            page_num=hit.get("page_num"),
            content_type="image",
            snippet=caption,
        ))
    return sources
