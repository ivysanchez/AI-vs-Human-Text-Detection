# Notebooks

The notebooks are split by phase across two folders that mirror the two capstone courses. Each notebook is self-contained — it loads its inputs from `outputs/` at the top and saves its outputs to `outputs/` at the bottom, so any phase can be re-run independently without restarting the whole pipeline.

---

## Execution Order

```
capstone1/
  phase1 → phase2 → phase3 → phase4 → phase5
                                              ↓
capstone2/
  phase6 → phase7 → phase8 → phase9 → phase10 → phase11
```

---

## `capstone1/` — Phases 1–5
*Capstone 1 end-of-semester deliverable*

| Notebook | Content | Key Output |
|---|---|---|
| `phase1_data_understanding.ipynb` | Data loading, class balance, missing value audit, mean imputation | `outputs/df_cleaned.pkl` |
| `phase2_eda_visualization.ipynb` | Class distributions, correlation heatmap, boxplots by label, word clouds | *(visual only)* |
| `phase3_feature_engineering.ipynb` | 14 intrinsic + 16 linguistic features + TF-IDF + char n-grams → SelectKBest(800) → StandardScaler | `outputs/features_baseline.pkl` |
| `phase4_baseline_models.ipynb` | LR F1 = 0.645 · SVM F1 = 0.609 — establishes performance floor | *(results only)* |
| `phase5_augmentation_sr.ipynb` | SR vs. BT comparison. SR wins (+19 pp). **Capstone 1 best: LR F1 = 0.839** | `outputs/sr_augmented.pkl` |

---

## `capstone2/` — Phases 6–11
*Capstone 2 final deliverable*

| Notebook | Content | Key Output |
|---|---|---|
| `phase6_model_optimization.ipynb` | Grid search across LR, SVM, GB, RF. **RF(n=250) wins: F1 = 0.930** | `outputs/best_rf_bundle.pkl` |
| `phase7_error_analysis.ipynb` | 6-pass error analysis. Discovers SR duplicate contamination → fuzzy dedup → validated **93.0%** | `outputs/error_analysis.pkl` |
| `phase8_shap_explainability.ipynb` | SHAP TreeExplainer. **Top features are Faker bigrams — central finding** | `outputs/shap_results.pkl` |
| `phase9_lime_explainability.ipynb` | LIME 4-pass word-level analysis. Confirms SHAP finding. Documents bigram/perturbation limitation | *(results only)* |
| `phase10_transformer_embeddings.ipynb` | Frozen DistilBERT [CLS] + top-20 stylometric → 788-dim hybrid. Trains **both LightGBM and Hybrid RF**; best val F1 becomes `deploy_model` | `outputs/hybrid_model_bundle.pkl` |
| `phase11_real_llm_evaluation.ipynb` | Real LLM test: 80 GPT-4o-mini samples. Evaluates all 3 models. Original RF AI F1 = **0.000**. Hybrid BERT/LGBM macro F1 = **~0.99** | `models/avh_best_bundle.pkl` |


## 'Other files:'
| Notebook | Content | Key Output |
|---|---|---|
| `ai_vs_human_final.ipynb` | Complete full ran pipeline with comments/interpretations |
|`train_model.ipynb` | Clean code no comments/interpretations, used for deployment \


---

## Handoff Map — `outputs/` directory

Each notebook writes intermediate artifacts to `outputs/` for the next to load. This folder is excluded from git (see `.gitignore`) — regenerate by running notebooks in order.

| File | Written by | Read by |
|---|---|---|
| `outputs/df_cleaned.pkl` | Phase 1 | Phases 2, 3, 5 |
| `outputs/features_baseline.pkl` | Phase 3 | Phase 4 |
| `outputs/sr_augmented.pkl` | Phase 5 | Phase 6 |
| `outputs/best_rf_bundle.pkl` | Phase 6 | Phases 7, 8, 9, 10, 11 |
| `outputs/error_analysis.pkl` | Phase 7 | *(reference)* |
| `outputs/shap_results.pkl` | Phase 8 | Phases 9, 10 |
| `outputs/hybrid_model_bundle.pkl` | Phase 10 | Phase 11 |
| `models/avh_best_bundle.pkl` | Phase 11 | Streamlit app |

---

## Reproducibility

Always run the **seed cell (Cell 1) first** in every notebook before any other cell. All `train_test_split` calls use `random_state=42, stratify=y`. See the [main README](../README.md#reproducibility) for full details.

> ⚠️ Slight numeric variation (±0.001 F1) is expected across reruns due to LightGBM histogram binning. This is not a bug.
