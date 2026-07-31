"""Aggregate v1 routers."""

from fastapi import APIRouter

from app.api.v1 import documents, health, query

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(query.router, tags=["query"])
