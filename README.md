# Product Master-Data Classification

This repository contains open-source, in-tenant workflows for suggesting five product master-data labels without external LLM or API calls.

## Random Forest Application

The Random Forest workflow predicts labels from three inputs:

- `GPCBrickCode`
- `UNSPSCNumber`
- `ProductDescription`

The solution trains one multiclass Random Forest per output label: `ProductType`, `ProductGroup`, `ProductFamily`, `ProductLine`, and `ProductKey`.

Each model combines train-fitted one-hot encodings of the two categorical codes with a 384-dimensional `all-MiniLM-L6-v2` description embedding. The Streamlit app returns ranked Top-1, Top-3, or Top-5 suggestions.

## Results

Held-out performance from the final Random Forest notebook:

| Output | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Product Group | 77.8% | 92.0% | 95.5% |
| Product Family | 62.7% | 82.0% | 87.8% |
| Product Line | 52.6% | 66.1% | 72.9% |
| Product Type | 64.2% | 88.9% | 96.5% |
| Product Key | 57.3% | 84.3% | 89.5% |

A controlled paired benchmark compares Random Forest with the local MiniLM cosine 1-NN approach from the comparison notebook. Both methods use the same cleaned records, rare-label policy, and held-out rows.

## Repository Structure

```text
.
├── artifacts/                              # Models, encoders, embedding metadata
├── datasets/                               # Full source workbook and sample CSV
├── evaluation/                             # Paired benchmark, results, and report
├── test_sets/                              # Exact held-out rows for each target
├── product_classification_rf_3features_proba.ipynb
├── product_type_classification_rf_3features_proba.ipynb
├── semarchy_example_product 1.ipynb
├── streamlit_app.py
├── requirements.txt
├── .gitattributes
└── .gitignore
```

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first embedding operation downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face and then uses the local cache.

## Run the Streamlit App

The repository must contain the 15 Joblib files under `artifacts/`.

```bash
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

The app accepts the three input fields, supports one output or all five outputs, and returns 1, 3, or 5 ranked candidates with model probability values.

## Reproduce Training

Open and run `product_classification_rf_3features_proba.ipynb`.

The notebook:

1. Loads `datasets/product_classified_Full.xlsx`, sheet `in`.
2. Removes rows where all three input fields are missing.
3. Folds target classes with fewer than 10 rows into `Other`.
4. Creates an independent target-stratified 80/20 split for each output.
5. Fits each categorical encoder on training rows only.
6. Embeds unique nonblank descriptions with MiniLM.
7. Trains one 600-tree Random Forest per target.
8. Evaluates Top-1, Top-3, and Top-5 performance.
9. Writes held-out CSVs to `test_sets/`.
10. Writes models, encoders, and metadata to `artifacts/`.

## Reproduce the Paired Comparison

After installing the requirements and ensuring the RF artifacts exist:

```bash
python evaluation/paired_rf_vs_1nn.py
```
