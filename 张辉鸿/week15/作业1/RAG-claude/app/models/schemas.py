from datetime import datetime
from pydantic import BaseModel, Field


# ─── Knowledge Base ───────────────────────────────────────────────

class KBCreate(BaseModel):
    name: str = Field(..., max_length=255)
    description: str | None = None


class KBResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ─── Document ─────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    kb_id: str
    filename: str
    status: str
    page_count: int | None = None
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


# ─── Chat ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    kb_id: str = Field(..., description="Knowledge base ID to search within")
    question: str = Field(..., description="User question")


class SourceInfo(BaseModel):
    doc_filename: str
    doc_id: str
    page_num: int | None = None
    content_type: str  # "text" or "image"
    snippet: str | None = None  # text snippet or image description


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    kb_id: str
    question: str


# ─── Error ────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
