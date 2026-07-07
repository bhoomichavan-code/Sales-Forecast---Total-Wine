# Total Wine — Cabernet Sauvignon Sales Prediction

Predicting store-item annual sales for Cabernet Sauvignon across Total Wine &
More's national footprint, to support **inventory planning** and **targeted
promotion**. A tuned **XGBoost** model explains roughly **two-thirds of the
variance** in normalized sales (test **R² = 0.64** on the dollar scale, **0.62**
on the log scale) with a mean absolute error of about **$1,310** per store-item —
a large improvement over the interpretable baselines.

This is a clean, interview-explainable rework of a graduate capstone
(BUDT758W, R.H. Smith School of Business).

## What's in this folder

| Folder | What it is |
|--------|-----------|
| **`cabernet-sales-forecast/`** | The revamped, GitHub-ready project — reproducible Python pipeline, models, figures, docs, and Power BI extracts. Start here. |
| **`OG Project/`** | The original capstone files (business framing document, raw data, source workbooks) kept for reference. |

## The revamped project at a glance

Test set = 13,395 store-items (20% hold-out).

| Model | R² (log) | R² ($) | MAE |
|-------|:--------:|:------:|:---:|
| Linear Regression | 0.51 | 0.31 | $1,660 |
| Decision Tree | 0.56 | 0.48 | $1,579 |
| Random Forest | 0.61 | 0.63 | $1,375 |
| **XGBoost (tuned)** | **0.62** | **0.64** | **$1,310** |

**Key findings.** Which wine it is (a leakage-safe target encoding of item
identity) is the single strongest predictor, followed by the store's sales tier
and store size; local affluence adds signal at the margin. Full write-up and
run instructions are in `cabernet-sales-forecast/README.md`.

## How to run

```bash
cd cabernet-sales-forecast
pip install -r requirements.txt
python analysis/04_modeling.py      # reproduces every number above
```

## Note on the earlier version

The original capstone predicted the same target with an XGBoost model. This
rework makes it reproducible and honest: all preprocessing is fit on the training
split only (no leakage), identity fields use out-of-fold target encoding, models
are compared on one shared split, and the headline is reported as **R²** (the
correct metric for a continuous target) rather than "accuracy." See
`cabernet-sales-forecast/docs/methodology.md`.
