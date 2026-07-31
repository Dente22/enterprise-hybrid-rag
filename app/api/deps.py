"""Shared FastAPI dependencies."""

from app.core.config import Settings, get_settings
from app.db.session import get_db_session

get_session = get_db_session


async def get_settings_dep() -> Settings:
    """Provide application settings."""
    return get_settings()
