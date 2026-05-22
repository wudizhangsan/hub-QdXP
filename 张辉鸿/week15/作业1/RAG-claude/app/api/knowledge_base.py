"""Knowledge Base CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.db_models import KnowledgeBase
from app.models.schemas import KBCreate, KBResponse

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])


@router.post("", response_model=KBResponse, status_code=201)
async def create_kb(body: KBCreate, db: AsyncSession = Depends(get_db)):
    kb = KnowledgeBase(name=body.name, description=body.description)
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


@router.get("", response_model=list[KBResponse])
async def list_kbs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()))
    return result.scalars().all()


@router.get("/{kb_id}", response_model=KBResponse)
async def get_kb(kb_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@router.delete("/{kb_id}", status_code=204)
async def delete_kb(kb_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    await db.delete(kb)
    await db.commit()
