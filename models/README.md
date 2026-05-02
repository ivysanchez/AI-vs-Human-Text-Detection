# Saved Models

Model artifacts are excluded from version control due to file size.

To regenerate all models, run the notebooks in order:

1. `notebooks/capstone1/phase4_baseline_models.ipynb` — trains baseline LR and SVM models
2. `notebooks/train_model.ipynb` — trains the full pipeline through to the hybrid BERT + LightGBM deployment bundle

Running notebook 04 end-to-end will produce:
- `avh_bert.pt` — frozen DistilBERT weights (loaded from HuggingFace, saved locally for offline use)
- `avh_best_bundle.pkl` — full deployment bundle: best classifier + TF-IDF vectorizer + char n-gram vectorizer + SelectKBest selector + stylometric indices + expected feature count

The Streamlit app (`src/streamlit_app.py`) loads `avh_best_bundle.pkl` at startup.

**Note:** Slight numeric variations (±0.001 F1) are expected across reruns. See the reproducibility note in notebook 04, Cell 1.
