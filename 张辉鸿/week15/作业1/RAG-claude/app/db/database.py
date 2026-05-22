from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Session

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Synchronous engine for workers/scripts
_sync_url = settings.database_url.replace("+aiosqlite", "").replace("sqlite+aiosqlite:///", "sqlite:///")
sync_engine = create_sync_engine(_sync_url, echo=False)
SyncSession = Session(bind=sync_engine)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


def get_sync_db() -> Session:
    """Synchronous session for worker/script usage."""
    return SyncSession()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
