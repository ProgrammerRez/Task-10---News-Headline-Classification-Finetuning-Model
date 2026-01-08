import os
import torch
import streamlit as st
from transformers import pipeline

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = os.getenv("news_classifer_model")  # HF repo id or local path
device = 0 if torch.cuda.is_available() else -1

# -------------------------
# LOAD MODEL ONCE
# -------------------------
@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=device,
        return_all_scores=True
    )

clf = load_classifier()

# -------------------------
# CLASSIFICATION LOGIC
# -------------------------
def classify_headline(text: str):
    if not text or not text.strip():
        return None

    outputs = clf(text)[0]
    probs = {
        o["label"]: float(o["score"])
        for o in sorted(outputs, key=lambda x: x["score"], reverse=True)
    }
    return probs

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="News Headline Classifier", layout="centered")

st.title("📰 News Headline Classifier")

headline = st.text_area(
    "Headline",
    placeholder="Enter a news headline...",
    height=80
)

if st.button("Classify"):
    result = classify_headline(headline)

    if result is None:
        st.warning("Please enter a valid headline.")
    else:
        # -------------------------
        # TOP PREDICTIONS
        # -------------------------
        st.subheader("Predicted (Top 5)")
        top5 = dict(list(result.items())[:5])
        st.json(top5)

        # -------------------------
        # ALL PROBABILITIES
        # -------------------------
        st.subheader("All Class Probabilities")
        st.json(result)
