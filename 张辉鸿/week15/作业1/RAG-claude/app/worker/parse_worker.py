"""Kafka consumer worker: parse PDF → chunk → embed → store in Milvus → update status."""

import json
import logging

from kafka import KafkaConsumer
from sqlalchemy import select

from app.config import settings
from app.db.database import get_sync_db
from app.models.db_models import Document, DocStatus, TextChunk, Image
from app.services.parser import parse_pdf
from app.services.chunker import chunk_markdown
from app.services.embedding import embed_texts, embed_image
from app.services.storage import get_client, TEXT_COLLECTION, IMAGE_COLLECTION

logger = logging.getLogger(__name__)


def run_worker():
    """Main worker loop: consume from Kafka and process documents."""
    consumer = KafkaConsumer(
        settings.kafka_parse_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_parse_group,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        max_poll_interval_ms=600000,
        session_timeout_ms=30000,
    )
    milvus_client = get_client()
    logger.info("Parse worker started, waiting for messages...")

    for message in consumer:
        try:
            data = message.value
            doc_id = data["doc_id"]
            pdf_path = data["file_path"]
            kb_id = data["kb_id"]
            logger.info(f"Processing document {doc_id}: {pdf_path}")
            _process_document(doc_id, kb_id, pdf_path, milvus_client)
        except Exception:
            logger.exception(f"Failed to process: {message.value}")

    consumer.close()


def _process_document(doc_id: str, kb_id: str, pdf_path: str, milvus_client):
    db = get_sync_db()
    try:
        # 1. Update status → parsing
        _update_status(db, doc_id, DocStatus.PARSING)

        # 2. Parse with MinerU
        result = parse_pdf(doc_id, pdf_path)
        logger.info(f"Parsed {doc_id}: {len(result.markdown_files)} md files, "
                    f"{len(result.image_files)} images")

        # 3. Update status → chunking
        _update_status(db, doc_id, DocStatus.CHUNKING)

        # 4. Chunk the markdown
        chunks = chunk_markdown(result.full_markdown)

        # 5. Update status → embedding
        _update_status(db, doc_id, DocStatus.EMBEDDING)

        # 6. Embed text chunks → Milvus + SQLite
        if chunks:
            texts = [c["text"] for c in chunks]
            text_embeddings = embed_texts(texts)
            _store_text_chunks(db, doc_id, kb_id, chunks, text_embeddings, milvus_client)

        # 7. Embed images → Milvus + SQLite
        for img_info in result.image_files:
            try:
                img_embedding = embed_image(img_info["path"])
                _store_image(db, doc_id, kb_id, img_info, img_embedding, milvus_client)
            except Exception:
                logger.exception(f"Failed to embed image: {img_info['path']}")

        # 8. Update status → completed
        _update_status(db, doc_id, DocStatus.COMPLETED)
        logger.info(f"Document {doc_id} processing completed.")

    except Exception as e:
        doc = db.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
        if doc:
            doc.status = DocStatus.FAILED
            doc.error_message = str(e)
            db.commit()
        raise
    finally:
        db.close()


def _update_status(db, doc_id: str, status: DocStatus):
    doc = db.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
    if doc:
        doc.status = status
        db.commit()


def _store_text_chunks(db, doc_id: str, kb_id: str, chunks: list[dict],
                       embeddings: list[list[float]], milvus_client):
    # Insert into Milvus
    rows = []
    for ch, emb in zip(chunks, embeddings):
        rows.append({
            "kb_id": kb_id,
            "doc_id": doc_id,
            "chunk_text": ch["text"],
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "embedding": emb,
        })
    result = milvus_client.insert(collection_name=TEXT_COLLECTION, data=rows)
    milvus_ids = result["ids"]

    # Insert into SQLite
    for ch, mid in zip(chunks, milvus_ids):
        tc = TextChunk(
            doc_id=doc_id, kb_id=kb_id,
            chunk_text=ch["text"], chunk_index=ch["chunk_index"],
            page_start=ch["page_start"], page_end=ch["page_end"],
            milvus_id=mid,
        )
        db.add(tc)
    db.commit()
    logger.info(f"Stored {len(rows)} text chunks for doc {doc_id}")


def _store_image(db, doc_id: str, kb_id: str, img_info: dict, embedding: list[float], milvus_client):
    result = milvus_client.insert(collection_name=IMAGE_COLLECTION, data=[{
        "kb_id": kb_id,
        "doc_id": doc_id,
        "image_path": img_info["path"],
        "page_num": img_info["page_num"],
        "caption": "",
        "embedding": embedding,
    }])
    milvus_id = result["ids"][0]

    img = Image(
        doc_id=doc_id, kb_id=kb_id,
        image_path=img_info["path"], page_num=img_info["page_num"],
        caption="", milvus_id=milvus_id,
    )
    db.add(img)
    db.commit()
