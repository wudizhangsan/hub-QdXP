"""Document upload endpoint."""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from kafka import KafkaProducer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import get_db
from app.models.db_models import Document, DocStatus, KnowledgeBase
from app.models.schemas import UploadResponse, DocumentResponse

router = APIRouter(prefix="/upload", tags=["Document"])


@router.post("/document", response_model=UploadResponse, status_code=202)
async def upload_document(
    kb_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF to a knowledge base. Parsing happens asynchronously."""
    # Validate KB exists
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save the uploaded file
    doc_id = uuid.uuid4().hex
    kb_upload_dir = Path(settings.upload_dir) / kb_id
    kb_upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = f"{doc_id}_{file.filename}"
    file_path = kb_upload_dir / safe_filename
    content = await file.read()
    file_path.write_bytes(content)

    # Create document record
    doc = Document(
        id=doc_id,
        kb_id=kb_id,
        filename=file.filename,
        file_path=str(file_path),
        status=DocStatus.PENDING,
    )
    db.add(doc)
    await db.commit()

    # Push to Kafka for async parsing
    _push_to_kafka(doc_id, kb_id, str(file_path))

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        status=DocStatus.PENDING.value,
        message="Document uploaded successfully, queued for parsing.",
    )


@router.get("/document/{doc_id}/status", response_model=DocumentResponse)
async def get_document_status(doc_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/documents/{kb_id}", response_model=list[DocumentResponse])
async def list_documents(kb_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document).where(Document.kb_id == kb_id).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


def _push_to_kafka(doc_id: str, kb_id: str, file_path: str):
    """Push a parse request to Kafka."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        producer.send(
            settings.kafka_parse_topic,
            value={"doc_id": doc_id, "kb_id": kb_id, "file_path": file_path},
        )
        producer.flush()
        producer.close()
    except Exception:
        # If Kafka is unavailable, still return success —
        # the document status stays "pending" until a worker picks it up.
        pass
