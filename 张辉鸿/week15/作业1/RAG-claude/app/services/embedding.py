"""Embedding service: BGE text embeddings + CLIP image embeddings via DashScope API."""

import base64
from pathlib import Path

import dashscope
from dashscope import MultiModalEmbedding

from app.config import settings

dashscope.api_key = settings.dashscope_api_key

# DashScope text embedding model (BGE-like, 1024d)
TEXT_MODEL = "text-embedding-v3"
# DashScope multimodal embedding model (CLIP-like, can encode both text and images)
MULTIMODAL_MODEL = "multimodal-embedding-v1"


def embed_text(text: str) -> list[float]:
    """Encode a text string into a 1024-dim BGE embedding."""
    from dashscope import TextEmbedding
    resp = TextEmbedding.call(model=TEXT_MODEL, input=text)
    if resp.status_code != 200:
        raise RuntimeError(f"Text embedding failed: {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch encode multiple texts."""
    from dashscope import TextEmbedding
    resp = TextEmbedding.call(model=TEXT_MODEL, input=texts)
    if resp.status_code != 200:
        raise RuntimeError(f"Text embedding failed: {resp.message}")
    return [e["embedding"] for e in resp.output["embeddings"]]


def embed_image(image_path: str) -> list[float]:
    """Encode an image file into a 512-dim CLIP embedding."""
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    resp = MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=[{"image": image_base64}],
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Image embedding failed: {resp.message}")
    return resp.output["embeddings"][0]


def embed_multimodal_query(text: str) -> list[float]:
    """Encode a user query for cross-modal retrieval (text→text, text→image).

    Returns a 512-dim CLIP text embedding that can search against
    both text chunks (BGE-embedded) and images (CLIP-embedded).
    For best results we encode the query in both modalities.
    """
    resp = MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=[{"text": text}],
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Multimodal query embedding failed: {resp.message}")
    return resp.output["embeddings"][0]
