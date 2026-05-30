"""Pydantic API schemas."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RecommendImageRequest(BaseModel):
    user_id: str
    image: str = Field(description="Base64-encoded image")
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendTextRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendHybridRequest(BaseModel):
    user_id: str
    image: Optional[str] = None
    query: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendationItem(BaseModel):
    product_id: str
    title: str
    score: float
    image_url: str


class InteractionRequest(BaseModel):
    user_id: str
    product_id: str
    event_type: Literal["click", "purchase", "view"]


class StatusResponse(BaseModel):
    status: str
    index_size: int
    model: str
    phase_mode: str
    ranker_loaded: bool


class DebugRetrieveRequest(BaseModel):
    query: Optional[str] = None
    image: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)


class RawRetrievalItem(BaseModel):
    product_id: str
    score: float


class IndexStats(BaseModel):
    index_size: int
    embeddings_path_exists: bool
    faiss_index_exists: bool
    embeddings_shape: tuple[int, int] | None = None
    faiss_path: str | None = None


class CacheInspectItem(BaseModel):
    key: str
    created_at: float | None = None
    ttl_seconds: int | None = None


class CacheInspectResponse(BaseModel):
    embedding_keys: list[CacheInspectItem]
    retrieval_keys: list[CacheInspectItem]


class RankBatchRequest(BaseModel):
    user_id: str
    product_ids: list[str] = Field(min_length=1, max_length=100)


class RankedProduct(BaseModel):
    product_id: str
    score: float


class RankBatchResponse(BaseModel):
    user_id: str
    scored_products: list[RankedProduct]
