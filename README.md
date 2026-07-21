# Product Master-Data Field Classifier (lean)

Auto-suggests values for five product master-data fields from a product's text and a few categorical
attributes — entirely in-tenant with open-source models (no external LLM or API calls).

## What it does

- Fuses seven short text columns and three categorical columns into features
  (word + character TF-IDF + one-hot encoding), fitting on the training fold only (no leakage).
- Benchmarks three models with leakage-safe 4-fold cross-validation:
  **Logistic Regression** (recommended), **LightGBM**, **XGBoost**.
- Reports macro-F1 (plus top-1 / top-3) per field, and per-class confusion matrices for the GPC Brick field.

## Headline result

Logistic Regression wins on **macro-F1** across all five fields — the fair, imbalance-aware metric.
On raw top-1/top-3 the tree models are close. See the rendered chart and tables in the notebook.

## Viewing the notebook

`product_classifier_lean.ipynb` renders directly on GitHub — the charts and tables are embedded, so no
setup is needed just to read it.

For the fully styled tables (GitHub can strip some HTML/CSS), open **`product_classifier_lean.html`** in a
browser, or view the notebook through [nbviewer](https://nbviewer.org/).

## Running it

> **The dataset is not included.** The outputs in the notebook are pre-rendered from a private 523-row sample,
> so re-running requires supplying your own CSV.

1. `python -m venv .venv && source .venv/bin/activate`  (Python 3.12 recommended)
2. `pip install -r requirements.txt`
3. Place your CSV at `data/product_master_sample_523.csv` — schema in [`data/README.md`](data/README.md).
4. `jupyter notebook product_classifier_lean.ipynb`, then **Run All**.

## Notes

- Fully offline and open-source; no data leaves your environment.
- **Macro-F1** is the headline metric because the data is imbalanced — plain accuracy can be inflated by simply
  predicting the majority class, whereas macro-F1 weights every category equally.
- Some visualization cells open collapsed (input hidden) in JupyterLab / Notebook 7; click the bar on the left
  of a cell to expand its code.
