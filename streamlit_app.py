from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


ARTIFACTS = Path(__file__).parent / "artifacts"
TARGETS = ["ProductType", "ProductGroup", "ProductFamily", "ProductLine", "ProductKey"]


@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def predict(target, gpc, unspsc, description, k):
    prefix = "product_" + target.removeprefix("Product").lower()
    model = joblib.load(ARTIFACTS / f"{prefix}_rf_3feat_model.joblib")
    encoder = joblib.load(ARTIFACTS / f"{prefix}_rf_3feat_ohe.joblib")

    unspsc = str(int(float(unspsc))) if unspsc else "-1"

    categorical = encoder.transform(pd.DataFrame(
        [[gpc or "Missing", unspsc]],
        columns=["GPCBrickCode", "UNSPSCNumber"],
    ))

    text = description.strip()
    embedding = (
        get_embedder().encode([text], convert_to_numpy=True)
        if text
        else np.zeros((1, 384), dtype=np.float32)
    )

    probabilities = model.predict_proba(np.hstack([categorical, embedding]))[0]
    order = np.argsort(probabilities)[::-1][:k]
    return pd.DataFrame({
        "Candidate": model.classes_[order],
        "Confidence": [f"{probabilities[index]:.1%}" for index in order],
    })


st.set_page_config(page_title="Product classification", layout="wide")
st.title("Product classification")

gpc = st.text_input("GPC brick code")
unspsc = st.text_input("UNSPSC number")
description = st.text_area("Product description")
target = st.selectbox("Output label", ["All targets", *TARGETS])
k = st.selectbox("Candidates", [1, 3, 5], index=1)

if st.button("Generate suggestions", type="primary", width="stretch"):
    for name in TARGETS if target == "All targets" else [target]:
        st.subheader(name)
        st.dataframe(predict(name, gpc, unspsc, description, k), width="stretch")