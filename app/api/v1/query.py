"""Hybrid Q&A query endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, get_settings_dep
from app.core.config import Settings
from app.core.security import require_api_key
from app.schemas.documents import QueryRequest, QueryResponse
from app.services.embedding_service import EmbeddingError
from app.services.llm_service import LLMServiceError
from app.services.query_service import QueryService
from app.services.sanitizer import SanitizationError

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Hybrid search + grounded structured answer",
)
async def query_documents(
    body: QueryRequest,
    _: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> QueryResponse:
    """
    Run vector + FTS hybrid retrieval with RRF, rerank top chunks,
    and return a Pydantic-validated grounded answer.
    """
    service = QueryService(session=session, settings=settings)
    try:
        return await service.ask(body)
    except SanitizationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (EmbeddingError, LLMServiceError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
