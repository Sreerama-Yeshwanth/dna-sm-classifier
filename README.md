# Product Master-Data Field Classifier

Auto-suggests values for five product master-data fields from a product's text and a few categorical
attributes, Entirely in-tenant with open-source models (no external LLM or API calls).

## What it does

- Fuses seven short text columns and three categorical columns into features
  (word + character TF-IDF + one-hot encoding), fitting on the training fold only.
- Benchmarks three models with leakage-safe 4-fold cross-validation:
  **Logistic Regression**, **LightGBM**, **XGBoost**.
- Reports macro-F1 (plus top-1 / top-3) per field, and per-class confusion matrices for the GPC Brick field.
