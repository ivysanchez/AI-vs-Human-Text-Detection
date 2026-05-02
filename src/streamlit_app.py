"""
Streamlit App — AI vs. Human Text Detector
Loads the saved model bundle and serves predictions with LIME word-level explanations.

Usage:
    streamlit run src/streamlit_app.py

Requirements:
    - models/avh_best_bundle.pkl must exist (run notebook 04 first)
    - All packages in requirements.txt must be installed
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import sys

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI vs. Human Text Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Load model bundle ─────────────────────────────────────────────────────────
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "avh_best_bundle.pkl")

@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        return None
    return joblib.load(BUNDLE_PATH)

bundle = load_bundle()

# ── Helper: build feature matrix from raw text ───────────────────────────────
def build_features(text: str, bundle: dict) -> np.ndarray:
    """Reproduces the inference pipeline: BERT CLS + stylometric features."""
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    from textblob import TextBlob

    # BERT embedding
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        cls_emb = model(**inputs).last_hidden_state[:, 0, :].squeeze().numpy()

    # Stylometric features (must match training order)
    blob = TextBlob(text)
    words = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    stylo = {
        "word_count": len(words),
        "character_count": len(text),
        "sentence_count": len(sentences),
        "lexical_diversity": len(set(words)) / max(len(words), 1),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "punctuation_ratio": sum(1 for c in text if c in ".,!?;:") / max(len(text), 1),
        "sentiment_polarity": blob.sentiment.polarity,
        "sentiment_subjectivity": blob.sentiment.subjectivity,
    }

    stylo_arr = np.array(list(stylo.values()), dtype=float)
    top_idx = bundle["top_stylometric_indices"]
    stylo_selected = stylo_arr[top_idx] if len(stylo_arr) > max(top_idx) else stylo_arr

    features = np.hstack([cls_emb, stylo_selected]).reshape(1, -1)

    # Pad/truncate to expected feature count
    expected = bundle["expected_features"]
    if features.shape[1] < expected:
        features = np.pad(features, ((0, 0), (0, expected - features.shape[1])))
    elif features.shape[1] > expected:
        features = features[:, :expected]

    return features


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 AI vs. Human Text Detector")
st.markdown(
    "Paste any text sample below. The model will classify it as **Human-written** or "
    "**AI-generated** and show a confidence score."
)

if bundle is None:
    st.error(
        "Model bundle not found at `models/avh_best_bundle.pkl`. "
        "Please run `notebooks/04_train_model_clean.ipynb` first to generate the model."
    )
    st.stop()

text_input = st.text_area(
    "Text to classify",
    placeholder="Paste your text here (minimum ~50 words recommended for stable results)...",
    height=200,
)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("Classify", type="primary")

if run and text_input.strip():
    with st.spinner("Running inference..."):
        try:
            X = build_features(text_input, bundle)
            clf = bundle["model"]
            pred = clf.predict(X)[0]
            proba = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None

            label = "🤖 AI-Generated" if pred == 1 else "✍️ Human-Written"
            conf = float(proba.max()) if proba is not None else None

            st.markdown("---")
            st.subheader("Prediction")
            st.markdown(f"### {label}")
            if conf is not None:
                st.metric("Confidence", f"{conf:.1%}")
                if conf < 0.65:
                    st.info(
                        "⚠️ Low confidence — the text sits near the decision boundary. "
                        "Results are less reliable for very short or ambiguous samples."
                    )

            st.markdown("---")
            st.caption(
                f"Model: {bundle.get('model_name', 'Unknown')} | "
                f"Bundle saved: {bundle.get('saved_at', 'N/A')[:10]}"
            )

        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.info("Make sure all requirements are installed and the model bundle is valid.")

elif run and not text_input.strip():
    st.warning("Please enter some text before classifying.")

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About")
    st.markdown(
        "This tool uses a **Hybrid DistilBERT + Stylometric** classifier trained to distinguish "
        "human-written text from AI-generated text (GPT-4o-mini evaluation).\n\n"
        "**In-distribution F1:** ~0.84 (synthetic benchmark)\n\n"
        "**Real LLM F1:** ~0.70 (GPT-4o-mini)\n\n"
        "**Label convention:** Human = 0, AI = 1"
    )
    st.markdown("---")
    st.markdown("**Dataset:** [Kaggle — AI vs Human Content Detection](https://www.kaggle.com/datasets/pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025)")
    st.markdown("**Course:** DATA 4382 — Capstone 2, UT Arlington")
