"""FastAPI application for multimodal recommendations."""

from __future__ import annotations

import base64
import hashlib
import time
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from sqlalchemy import select

from api.db import Interaction, Product, User, init_db, session_scope
from api.schemas import (
    InteractionRequest,
    RecommendHybridRequest,
    RecommendImageRequest,
    RecommendTextRequest,
    RecommendationItem,
    StatusResponse,
)
from config import (
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    FINAL_TOP_N,
    ITEM_IDS_PATH,
    MAX_INPUT_IMAGE_PX,
    PHASE_MODE,
    RANKER_CHECKPOINT_PATH,
    USE_RANKER,
    USER_ENCODER_MODE,
)
from models.clip_encoder import ClipEncoder
from models.fusion import fuse
from models.ranker import Ranker, build_ranking_features
from models.user_encoder import encode_user
from retrieval.faiss_index import load_index
from retrieval.search import retrieve_item_ids


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.encoder = ClipEncoder(device="cuda")
    app.state.phase_mode = PHASE_MODE
    app.state.use_ranker = USE_RANKER
    app.state.user_encoder_mode = USER_ENCODER_MODE
    app.state.item_embeddings = np.zeros((0, 512), dtype=np.float32)
    app.state.item_ids = np.array([], dtype=str)
    app.state.item_id_to_index = {}
    app.state.index = None
    app.state.ranker = None

    if EMBEDDINGS_PATH.exists() and ITEM_IDS_PATH.exists():
        app.state.item_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
        app.state.item_ids = np.load(ITEM_IDS_PATH)
        app.state.item_id_to_index = {str(pid): i for i, pid in enumerate(app.state.item_ids)}

    if FAISS_INDEX_PATH.exists() or (FAISS_INDEX_PATH.parent / f"{FAISS_INDEX_PATH.name}.npy").exists():
        try:
            app.state.index = load_index(str(FAISS_INDEX_PATH), use_gpu=True)
        except Exception:
            app.state.index = None

    if app.state.phase_mode == "phase2" and app.state.use_ranker and torch is not None:
        if RANKER_CHECKPOINT_PATH.exists():
            try:
                ranker = Ranker()
                payload = torch.load(RANKER_CHECKPOINT_PATH, map_location="cpu")
                ranker.load_state_dict(payload["model_state_dict"])
                ranker.eval()
                app.state.ranker = ranker
            except Exception:
                app.state.ranker = None

    yield


app = FastAPI(title="Multimodal Recommender", version="0.1.0", lifespan=lifespan)


def _decode_image(b64_image: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64_image)
        image = Image.open(BytesIO(raw)).convert("RGB")
        if max(image.size) > MAX_INPUT_IMAGE_PX:
            image.thumbnail((MAX_INPUT_IMAGE_PX, MAX_INPUT_IMAGE_PX))
        return image
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid base64 image payload") from exc


def _category_to_id(category: str) -> float:
    digest = hashlib.sha256(category.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False) / 2**32


def _score_candidates_with_ranker(user_vec: np.ndarray, products: list[Product]) -> list[float] | None:
    if app.state.ranker is None or torch is None:
        return None

    features = []
    for product in products:
        idx = app.state.item_id_to_index.get(product.product_id)
        if idx is None:
            continue
        item_vec = app.state.item_embeddings[idx]
        features.append(
            build_ranking_features(
                user_vec=user_vec,
                item_vec=item_vec,
                price=float(product.price),
                category_id=_category_to_id(product.category),
            )
        )

    if not features:
        return None

    with torch.no_grad():
        feature_tensor = torch.tensor(np.stack(features, axis=0), dtype=torch.float32)
        scores = app.state.ranker.score(feature_tensor)
        return scores.detach().cpu().numpy().astype(float).tolist()


def _recommend_from_query(user_id: str, query_vec: np.ndarray, top_k: int) -> list[RecommendationItem]:
    if app.state.index is None or len(app.state.item_ids) == 0:
        raise HTTPException(status_code=503, detail="Index not ready. Run pipeline first.")

    with session_scope() as session:
        user_vec = encode_user(
            user_id=user_id,
            db_session=session,
            item_id_to_index=app.state.item_id_to_index,
            item_embeddings=app.state.item_embeddings,
            mode=app.state.user_encoder_mode,
        )
        fused_vec = fuse(user_vec=user_vec, image_vec=query_vec)

        retrieved = retrieve_item_ids(
            index=app.state.index,
            item_ids=app.state.item_ids,
            query_vec=fused_vec,
            k=min(max(top_k, FINAL_TOP_N), 100),
        )

        products: list[Product] = []
        retrieval_scores: list[float] = []
        for product_id, score in retrieved[:top_k]:
            product = session.get(Product, product_id)
            if product is None:
                continue
            products.append(product)
            retrieval_scores.append(float(score))

        ranked_scores = None
        if app.state.phase_mode == "phase2" and app.state.use_ranker:
            ranked_scores = _score_candidates_with_ranker(user_vec=user_vec, products=products)

        if ranked_scores is not None and len(ranked_scores) == len(products):
            scored_products = list(zip(products, ranked_scores))
        else:
            scored_products = list(zip(products, retrieval_scores))

        scored_products.sort(key=lambda item: item[1], reverse=True)
        return [
            RecommendationItem(
                product_id=product.product_id,
                title=product.title,
                score=float(score),
                image_url=product.image_path,
            )
            for product, score in scored_products[:top_k]
        ]


@app.post("/recommend/image", response_model=list[RecommendationItem])
def recommend_image(payload: RecommendImageRequest):
    image = _decode_image(payload.image)
    query_vec = app.state.encoder.encode_pil(image)
    return _recommend_from_query(payload.user_id, query_vec, payload.top_k)


@app.post("/recommend/text", response_model=list[RecommendationItem])
def recommend_text(payload: RecommendTextRequest):
    query_vec = app.state.encoder.encode_text(payload.query)
    return _recommend_from_query(payload.user_id, query_vec, payload.top_k)


@app.post("/recommend/hybrid", response_model=list[RecommendationItem])
def recommend_hybrid(payload: RecommendHybridRequest):
    if not payload.image and not payload.query:
        raise HTTPException(status_code=422, detail="Provide at least one of image or query")

    vectors = []
    if payload.image:
        image = _decode_image(payload.image)
        vectors.append(app.state.encoder.encode_pil(image))
    if payload.query:
        vectors.append(app.state.encoder.encode_text(payload.query))

    query_vec = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    return _recommend_from_query(payload.user_id, query_vec, payload.top_k)


@app.get("/product/{product_id}")
def get_product(product_id: str):
    with session_scope() as session:
        product = session.get(Product, product_id)
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found")
        return {
            "product_id": product.product_id,
            "title": product.title,
            "category": product.category,
            "price": product.price,
            "image_path": product.image_path,
            "description": product.description,
        }


@app.post("/interaction")
def log_interaction(payload: InteractionRequest):
    weight_map = {"purchase": 1.0, "click": 0.3, "view": 0.1}
    now_ts = int(time.time())

    with session_scope() as session:
        if session.get(User, payload.user_id) is None:
            session.add(User(user_id=payload.user_id, created_at=now_ts))

        if session.get(Product, payload.product_id) is None:
            raise HTTPException(status_code=404, detail="Product not found")

        interaction = Interaction(
            user_id=payload.user_id,
            product_id=payload.product_id,
            event_type=payload.event_type,
            timestamp=now_ts,
            weight=weight_map[payload.event_type],
        )
        session.add(interaction)

    return {"ok": True}


@app.get("/health", response_model=StatusResponse)
def health():
    model_name = "clip" if app.state.encoder.model is not None else "fallback-encoder"
    return StatusResponse(
        status="ok",
        index_size=int(len(app.state.item_ids)),
        model=model_name,
        phase_mode=str(app.state.phase_mode),
        ranker_loaded=bool(app.state.ranker is not None),
    )
