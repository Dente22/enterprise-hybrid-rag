"""Document ingest endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, get_settings_dep
from app.core.config import Settings
from app.core.security import require_api_key
from app.schemas.documents import IngestResponse, TextIngestRequest
from app.services.embedding_service import EmbeddingError
from app.services.ingest_service import IngestService
from app.services.parsers import extract_text_from_upload
from app.services.sanitizer import SanitizationError

router = APIRouter(prefix="/documents")


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest raw text into the hybrid index",
)
async def ingest_text(
    body: TextIngestRequest,
    _: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> IngestResponse:
    """Accept raw text, chunk it, embed it, and index for hybrid search."""
    if len(body.text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"text exceeds MAX_TEXT_LENGTH ({settings.max_text_length})",
        )
    service = IngestService(session=session, settings=settings)
    try:
        return await service.ingest_text(
            raw_text=body.text,
            source=body.source,
            content_type="text/plain",
            metadata=body.metadata,
        )
    except SanitizationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post(
    "/ingest-file",
    response_model=IngestResponse,
    summary="Ingest a PDF or Markdown file into the hybrid index",
)
async def ingest_file(
    file: UploadFile = File(...),
    source: str | None = Form(default=None),
    _: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> IngestResponse:
    """Accept PDF / Markdown / text uploads, extract text, and index."""
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file exceeds MAX_UPLOAD_BYTES ({settings.max_upload_bytes})",
        )
    filename = file.filename or "upload.txt"
    try:
        text, content_type = extract_text_from_upload(filename=filename, data=data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse upload: {exc}",
        ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file produced no extractable text",
        )

    service = IngestService(session=session, settings=settings)
    try:
        return await service.ingest_text(
            raw_text=text,
            source=source or filename,
            content_type=content_type,
            metadata={"filename": filename},
        )
    except SanitizationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
