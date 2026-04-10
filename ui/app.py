"""Streamlit frontend for image/text recommendation queries."""

from __future__ import annotations

import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_BASE = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

st.set_page_config(page_title="SnapRecommend", layout="wide")
st.title("SnapRecommend MVP")
st.caption("Image + text product recommendations")

user_id = st.text_input("User ID", value="u00000")
top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
mode = st.radio("Mode", ["Image", "Text", "Hybrid"], horizontal=True)

query_text = st.text_input("Text Query", value="") if mode in {"Text", "Hybrid"} else ""
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"]) if mode in {"Image", "Hybrid"} else None


def _img_to_base64(img: Image.Image) -> str:
    buff = BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


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

    try:
        resp = requests.post(API_BASE + endpoint, json=payload, timeout=30)
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
            st.stop()

        results = resp.json()
        if not results:
            st.warning("No recommendations returned.")
        else:
            cols = st.columns(2)
            for i, item in enumerate(results):
                with cols[i % 2]:
                    st.subheader(item["title"])
                    st.write(f"Product ID: {item['product_id']}")
                    st.write(f"Score: {item['score']:.4f}")
                    st.write(item["image_url"])

    except Exception as exc:
        st.error(f"Request failed: {exc}")
