# Images

Store exported plot images here for display in the README.

## Recommended exports from the notebooks

After running `notebooks/04_train_model_clean.ipynb`, save the following figures to this directory:

| Suggested filename | Description | Source cell |
|---|---|---|
| `01_class_distribution.png` | Bar chart of Human vs AI label counts | Part 2 |
| `02_feature_correlation_heatmap.png` | 14-feature correlation matrix | Part 2 |
| `03_wordcloud_ai.png` | Word cloud for AI class | Part 2 |
| `04_wordcloud_human.png` | Word cloud for Human class | Part 2 |
| `05_model_progression_bar.png` | F1 progression: Baseline → SR → Optimized RF | Part 6 |
| `06_confusion_matrices.png` | confusion matrices on deduplicated SR test set | Part 6 |
| `07_shap_bar.png` | SHAP global bar chart — top 20 features | Part 8 |
| `08_shap_force_fn1.png` | SHAP force plot — False Negative #1 | Part 8 |
| `09_lime_pass3.png` | LIME word-weight chart — False Negative (Pass 3) | Part 9 |
| `10_confusion_matrices.png` | confusion matrices on real LLM eval | Part 11 |
| `11_f1_by_source.png` | F1 comparison: GPT-4o-mini vs Human by model | Part 11 |

## How to save plots from notebooks

Add this to any matplotlib cell before `plt.show()`:
```python
plt.savefig("images/FILENAME.png", dpi=150, bbox_inches="tight")
```
