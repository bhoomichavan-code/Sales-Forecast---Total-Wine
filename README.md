# Total Wine Cabernet Sauvignon — How Much Will Each Store Sell?

## Overview

This repository predicts **annual Cabernet Sauvignon sales for every store-item
combination** across Total Wine & More's national footprint. It pairs internal
sales and stocking data with store attributes and 5-mile-radius trade-area
demographics to estimate a store-item's **normalized annual sales dollars**, so
sellers can plan inventory depth and target promotions. A tuned **XGBoost** model
explains about **two-thirds of the variance** in normalized sales (test **R² =
0.64** on the dollar scale, **0.62** on log) with a mean absolute error of about
**$1,310** per store-item. It reworks a graduate capstone (BUDT758W, R.H. Smith
School of Business) into a reproducible, interview-explainable pipeline.

## What's in this folder

| Folder | What it is |
|--------|-----------|
| **[`cabernet-sales-forecast/`](cabernet-sales-forecast/)** | The revamped and reproducible Python pipeline, models, figures, docs, and Power BI extracts. **Start here**; full write-up and run instructions are in its [README](cabernet-sales-forecast/README.md). |
| **[`OG Project/`](OG%20Project/)** | The original capstone files (business framing document, raw data, source workbooks) kept for reference. |

## Business Problem

Total Wine carries an enormous Cabernet assortment — **1,451 distinct wines**
across **269 stores in 28 states** — and sellers must decide **how much of each
wine each store will sell** to stock the right depth and aim promotions at stores
with untapped demand. The task: given information available before an item is
stocked (price, package, store attributes, local demographics), predict a
store-item's annual sales dollars.

Two realistic wrinkles:

- **The target must be engineered.** We use **normalized sales** = `(last-52-week
  sales ÷ weeks in stock) × 52`, which isolates true demand from availability.
- **"Accuracy" doesn't apply.** This is regression (a dollar amount), so the
  honest headline is **R²** (share of variance explained), not accuracy.

## Results

Held-out test set = **13,395 store-items** (20% split). R² is share of variance
explained; MAE/RMSE are in dollars.

| Model | R² (log) | R² ($) | MAE | RMSE |
|-------|:--------:|:------:|:---:|:----:|
| Naive (mean) | 0.00 | −0.06 | $2,395 | $7,289 |
| Linear Regression | 0.51 | 0.31 | $1,660 | $5,863 |
| Decision Tree | 0.56 | 0.48 | $1,579 | $5,113 |
| Random Forest | 0.61 | 0.63 | $1,375 | $4,304 |
| **XGBoost (tuned)** | **0.62** | **0.64** | **$1,310** | **$4,213** |

## Key Findings

1. **XGBoost is best**, just ahead of Random Forest; both clearly beat the linear
   baseline, so non-linear interactions between price, store and demographics matter.
2. **Which wine it is dominates** — a leakage-safe out-of-fold target encoding of
   item identity is the single strongest predictor.
3. **Store tier and size track sales strongly** — mean normalized sales rise from
   **$1,896** (lowest tier) to **$4,006** (highest); Extra-Large stores average
   **$7,642**, about **2.6×** the overall mean of **$2,963**.
4. **Local affluence matters at the margin** — income, net worth and the >$100K
   income share are positively associated with sales after controlling for store
   and item.

## How to Run

```bash
cd cabernet-sales-forecast
pip install -r requirements.txt
python analysis/04_modeling.py      # reproduces every number above
```

See [`cabernet-sales-forecast/README.md`](cabernet-sales-forecast/README.md) for
the full project structure, deliverables, and step-by-step instructions.

## Dashboard

**Power BI** — flat extracts and a beginner build guide live in the project's
`dashboard/` and `docs/`; the published link will go here once posted.

---

*Portfolio project. Each step is documented so the analysis and decisions can be
followed end to end. See
[`cabernet-sales-forecast/docs/methodology.md`](cabernet-sales-forecast/docs/methodology.md)
for the target definition, leakage controls, and modeling choices.*
