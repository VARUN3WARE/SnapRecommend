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
