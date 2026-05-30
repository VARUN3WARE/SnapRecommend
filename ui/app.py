"""Streamlit frontend for image/text recommendation queries."""

from __future__ import annotations

import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="SnapRecommend", layout="wide")
st.title("SnapRecommend MVP")
st.caption("Image + text product recommendations")

# Sidebar: API and phase controls
with st.sidebar:
    st.header("Configuration")
    API_BASE = st.text_input("API Base URL", value="http://localhost:8000")
    
    st.divider()
    st.subheader("Phase 2 Settings")
    phase_mode = st.selectbox(
        "Phase Mode",
        options=["phase1", "phase2"],
        help="phase1: Legacy weighted fusion. phase2: Two-tower + ranker model."
    )
    use_ranker = st.checkbox(
        "Use Ranker",
        value=False,
        help="Enable post-retrieval reranking (requires phase2 mode)."
    )
    
    st.divider()
    st.subheader("Comparison Mode")
    enable_comparison = st.checkbox(
        "Show Phase1 vs Phase2",
        value=False,
        help="Display side-by-side comparison of phase1 and phase2 results"
    )
    
    st.divider()
    cache_stats = st.checkbox(
        "Show Cache Stats",
        value=False,
        help="Display cache hit rate and performance metrics"
    )
    
    if cache_stats:
        try:
            cache_resp = requests.get(f"{API_BASE}/cache/stats", timeout=5)
            if cache_resp.status_code == 200:
                stats = cache_resp.json()
                st.metric("Cache Hit Rate", f"{stats['hit_rate']:.1%}")
                st.metric("Cache Hits", stats['hits'])
                st.metric("Retrieval Cache Size", stats['retrieval_cache_size'])
        except Exception:
            pass

    st.divider()
    st.subheader("Debug / Admin")
    show_debug = st.checkbox("Show debug controls", value=False)
    if show_debug:
        if st.button("Index Stats"):
            try:
                resp = requests.get(f"{API_BASE}/debug/index_stats", timeout=5)
                if resp.status_code == 200:
                    st.json(resp.json())
                else:
                    st.error(f"Index stats failed: {resp.status_code} {resp.text}")
            except Exception as exc:
                st.error(f"Index stats request error: {exc}")

        st.markdown("**Raw Retrieval**")
        dbg_mode = st.radio("Query type", ["text", "image"], horizontal=True)
        dbg_text = st.text_input("Debug text query") if dbg_mode == "text" else ""
        dbg_image = st.file_uploader("Debug image", type=["jpg", "jpeg", "png"]) if dbg_mode == "image" else None
        dbg_topk = st.number_input("Top K", min_value=1, max_value=100, value=10)
        if st.button("Run Raw Retrieval"):
            payload = {"top_k": dbg_topk}
            if dbg_mode == "text":
                payload["query"] = dbg_text
            else:
                if dbg_image is None:
                    st.error("Upload an image for raw retrieval")
                else:
                    img = Image.open(dbg_image).convert("RGB")
                    payload["image"] = _img_to_base64(img)

            try:
                resp = requests.post(f"{API_BASE}/debug/retrieve", json=payload, timeout=10)
                if resp.status_code == 200:
                    st.table(resp.json())
                else:
                    st.error(f"Retrieve failed: {resp.status_code} {resp.text}")
            except Exception as exc:
                st.error(f"Retrieve request error: {exc}")

        st.markdown("**Batch Ranker Score**")
        rank_user = st.text_input("User ID for ranking", value=user_id)
        rank_pids = st.text_area("Product IDs (comma separated)")
        if st.button("Run Ranker Score"):
            pids = [p.strip() for p in rank_pids.split(",") if p.strip()]
            if not pids:
                st.error("Provide at least one product id")
            else:
                try:
                    resp = requests.post(f"{API_BASE}/debug/ranker/score", json={"user_id": rank_user, "product_ids": pids}, timeout=10)
                    if resp.status_code == 200:
                        st.table(resp.json().get("scored_products", []))
                    else:
                        st.error(f"Ranker failed: {resp.status_code} {resp.text}")
                except Exception as exc:
                    st.error(f"Ranker request error: {exc}")

        if st.button("Inspect Cache"):
            try:
                resp = requests.get(f"{API_BASE}/debug/cache/inspect", timeout=5)
                if resp.status_code == 200:
                    st.json(resp.json())
                else:
                    st.error(f"Cache inspect failed: {resp.status_code} {resp.text}")
            except Exception as exc:
                st.error(f"Cache inspect request error: {exc}")
    
    st.divider()
    st.info(
        f"**Current Status:** Phase={phase_mode}, Ranker={'enabled' if use_ranker else 'disabled'}, Comparison={'enabled' if enable_comparison else 'disabled'}"
    )

user_id = st.text_input("User ID", value="u00000")
top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
mode = st.radio("Mode", ["Image", "Text", "Hybrid"], horizontal=True)

query_text = st.text_input("Text Query", value="") if mode in {"Text", "Hybrid"} else ""
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"]) if mode in {"Image", "Hybrid"} else None


def _img_to_base64(img: Image.Image) -> str:
    buff = BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def _fetch_recommendations(api_base: str, endpoint: str, payload: dict, phase: str, use_ranker: bool):
    """Fetch recommendations for a specific phase."""
    params = {"phase_mode": phase, "use_ranker": str(use_ranker).lower()}
    try:
        resp = requests.post(f"{api_base}{endpoint}", json=payload, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _display_results(results, title, columns=2):
    """Display recommendation results in a grid."""
    if not results:
        st.warning("No recommendations returned.")
        return
    
    st.subheader(title)
    cols = st.columns(columns)
    for i, item in enumerate(results):
        with cols[i % columns]:
            st.write(f"**{item['title']}**")
            st.caption(f"Product ID: {item['product_id']}")
            st.metric("Score", f"{item['score']:.4f}")
            if item.get("image_url"):
                st.image(item["image_url"], use_column_width=True)


if st.button("Recommend"):
    payload = {"user_id": user_id, "top_k": top_k}
    endpoint = ""

    if mode == "Image":
        if uploaded is None:
            st.error("Please upload an image.")
            st.stop()
        image = Image.open(uploaded).convert("RGB")
        payload["image"] = _img_to_base64(image)
        endpoint = "/recommend/image"

    elif mode == "Text":
        if not query_text.strip():
            st.error("Please enter a text query.")
            st.stop()
        payload["query"] = query_text
        endpoint = "/recommend/text"

    else:
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            payload["image"] = _img_to_base64(image)
        if query_text.strip():
            payload["query"] = query_text
        endpoint = "/recommend/hybrid"

    if enable_comparison:
        # Fetch both phase1 and phase2 results
        phase1_results = _fetch_recommendations(API_BASE, endpoint, payload, "phase1", False)
        phase2_results = _fetch_recommendations(API_BASE, endpoint, payload, "phase2", use_ranker)
        
        if phase1_results is not None or phase2_results is not None:
            col1, col2 = st.columns(2)
            
            if phase1_results is not None:
                with col1:
                    _display_results(phase1_results, "🔵 Phase 1 Results (Weighted Fusion)")
            
            if phase2_results is not None:
                with col2:
                    _display_results(phase2_results, "🟢 Phase 2 Results (Two-Tower + Ranker)")
    else:
        # Fetch results with current phase settings
        params = {}
        if phase_mode:
            params["phase_mode"] = phase_mode
        if use_ranker:
            params["use_ranker"] = str(use_ranker).lower()

        try:
            url = API_BASE + endpoint
            resp = requests.post(url, json=payload, params=params, timeout=30)
            if resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.text}")
                st.stop()

            results = resp.json()
            if results:
                title = f"{mode} Recommendations ({phase_mode})"
                _display_results(results, title)
