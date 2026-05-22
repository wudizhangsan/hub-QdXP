"""Multi-modal retrieval from Milvus."""

from pymilvus import MilvusClient

from app.config import settings
from app.services.storage import TEXT_COLLECTION, IMAGE_COLLECTION


def search_text(
    client: MilvusClient,
    kb_id: str,
    query_embedding: list[float],
    top_k: int | None = None,
) -> list[dict]:
    """Search text_chunks collection by BGE embedding."""
    top_k = top_k or settings.top_k_text
    results = client.search(
        collection_name=TEXT_COLLECTION,
        data=[query_embedding],
        filter=f'kb_id == "{kb_id}"',
        limit=top_k,
        output_fields=["id", "doc_id", "chunk_text", "page_start", "page_end"],
    )
    return _format_hits(results, "text")


def search_images(
    client: MilvusClient,
    kb_id: str,
    query_embedding: list[float],
    top_k: int | None = None,
) -> list[dict]:
    """Search images collection by CLIP embedding."""
    top_k = top_k or settings.top_k_images
    results = client.search(
        collection_name=IMAGE_COLLECTION,
        data=[query_embedding],
        filter=f'kb_id == "{kb_id}"',
        limit=top_k,
        output_fields=["id", "doc_id", "image_path", "page_num", "caption"],
    )
    return _format_hits(results, "image")


def hybrid_search(
    client: MilvusClient,
    kb_id: str,
    text_query_embedding: list[float],
    image_query_embedding: list[float],
) -> tuple[list[dict], list[dict]]:
    """Search both collections and return combined results."""
    texts = search_text(client, kb_id, text_query_embedding)
    images = search_images(client, kb_id, image_query_embedding)
    return texts, images


def _format_hits(results: list, content_type: str) -> list[dict]:
    """Transform Milvus search results into uniform dicts."""
    if not results:
        return []
    hits = []
    for row in results[0]:  # results is [[SearchResult...]]
        hit = {
            "milvus_id": row["id"],
            "doc_id": row["entity"].get("doc_id", ""),
            "score": row.get("distance", 0.0),
            "content_type": content_type,
        }
        if content_type == "text":
            hit["text"] = row["entity"].get("chunk_text", "")
            hit["page_start"] = row["entity"].get("page_start", 0)
            hit["page_end"] = row["entity"].get("page_end", 0)
        else:
            hit["image_path"] = row["entity"].get("image_path", "")
            hit["page_num"] = row["entity"].get("page_num", 0)
            hit["caption"] = row["entity"].get("caption", "")
        hits.append(hit)
    return hits
