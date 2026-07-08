# Total Wine Cabernet Sauvignon — How Much Will Each Store Sell?

## Overview

This is a data-analysis project that predicts **annual Cabernet
Sauvignon sales for every store-item combination** across Total Wine & More's
national footprint. It pairs internal sales and stocking data with store
attributes and 5-mile-radius trade-area demographics to estimate a store-item's
**normalized annual sales dollars**, then works through profiling, exploratory
analysis, a baseline-to-challenger modeling sequence, and a set of BI-ready
extracts. The emphasis is on **rigorous, honest methodology** — turning a deep,
messy assortment into a demand estimate a buyer can act on for inventory depth
and promotion targeting. It reworks a graduate capstone (BUDT758W, R.H. Smith
School of Business) into a reproducible pipeline.

## Business Problem

Total Wine carries an enormous Cabernet assortment — **1,451 distinct wines**
across **269 stores in 28 states** — and sellers must decide **how much of each
wine each store will sell** so they can stock the right depth and aim promotions
at stores with untapped demand. This project answers that directly: **given the
information available before an item is stocked (price, package, store
attributes, local demographics), how many sales dollars will a store-item
generate in a year?** The output is the kind of evidence an assortment or
inventory team uses to plan depth and target promotions.

Two realistic wrinkles:

- **The target must be engineered.** Raw sales punish items that were only
  stocked for part of the year. We use **normalized sales** = `(last-52-week
  sales ÷ weeks in stock) × 52`, which isolates true demand from availability —
  a documented analytical step in its own right.
- **"Accuracy" doesn't apply.** This is a regression problem (a dollar amount),
  so the honest headline metric is **R²** (share of variance explained), not
  accuracy — reported alongside error in real dollars.

## Dataset

One row per **store × item** (key = `Store_Number` + `Item_Code`):

- **`TW_Master_Internal_Data.csv`** — ~66,974 store-items: retail price, package
  type, weeks in stock, store age/size/tier, and 5-mile-radius demographics
  (household counts, income, net worth, education, ethnicity, age mix).
- **`TW_Final Test Data.csv`** — ~7,320 unlabeled store-items (sales blank),
  scored as a deployment simulation.

Full data dictionary:
[`data/raw/data_dictionary.md`](data/raw/data_dictionary.md).

**Target — `Normalized_Sales_$_L52W`:** annualized sales dollars, corrected for
weeks in stock. It is extremely right-skewed (median **$1,203**, mean **$2,963**,
max **$454K**, skew **15.5**), so it is modeled on a **log scale** and
back-transformed to dollars for reporting.

## Project Structure

```
cabernet-sales-forecast/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/               <- Original, immutable data + data dictionary
│   └── processed/         <- Cleaned model dataset & predictions (generated)
├── analysis/
│   ├── 01_data_profiling.py
│   ├── 02_clean_features.py
│   ├── 03_eda.py
│   ├── 04_modeling.py
│   └── 05_export_bi.py
├── src/                   <- Shared cleaning + feature-engineering module
├── dashboard/
│   └── extracts/          <- Flat CSVs exported for Power BI (generated)
├── visualizations/        <- Exported charts/figures
├── models/                <- Saved model bundle (.pkl)
└── docs/                  <- Findings, methodology, metrics
```

## How to Run

1. **Clone and enter the repo**

   ```bash
   git clone <your-repo-url>
   cd cabernet-sales-forecast
   ```

2. **Set up a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add the raw data** — place `TW_Master_Internal_Data.csv` and
   `TW_Final Test Data.csv` in `data/raw/` (large CSVs are git-ignored).

4. **Run the scripts in order**

   ```bash
   python analysis/01_data_profiling.py     # data profile
   python analysis/02_clean_features.py     # cleaned model dataset
   python analysis/03_eda.py                # EDA figures
   python analysis/04_modeling.py           # train, evaluate, save model + metrics
   python analysis/05_export_bi.py          # Power BI extracts
   ```

   `python analysis/04_modeling.py` reproduces every number below; set `SEARCH=1`
   to re-run the XGBoost hyper-parameter search.

## Deliverables

- Five documented, numbered analysis scripts (profiling → cleaning → EDA →
  modeling → BI extracts)
- A saved model bundle and hold-out predictions
- Two **Power BI extracts** plus a click-by-click dashboard build guide
- A one-page findings summary
- Exported visualizations and a cleaned analysis dataset

## Dashboard

**Power BI** — flat extracts (`dashboard/extracts/`) and a beginner build guide
(`docs/dashboard_guide.md`) are ready; the published link will go here once the
dashboard is posted.

## Results

Across **13,395 held-out store-items** (20% test split), the tuned model explains
about **64% of the variance** in normalized sales, roughly halving the naive
baseline's error. Metrics are R² (share of variance explained) plus MAE/RMSE in
dollars:

| Model | R² (log) | R² ($) | MAE | RMSE |
|-------|:--------:|:------:|:---:|:----:|
| Naive (mean) | 0.00 | −0.06 | $2,395 | $7,289 |
| Linear Regression | 0.51 | 0.31 | $1,660 | $5,863 |
| Decision Tree | 0.56 | 0.48 | $1,579 | $5,113 |
| Random Forest | 0.61 | 0.63 | $1,375 | $4,304 |
| **XGBoost (tuned)** | **0.62** | **0.64** | **$1,310** | **$4,213** |

The work produced an **engineered, documented target** (normalized sales), a
**leakage-safe pipeline** with out-of-fold target encoding of item/store
identity, a **baseline-to-challenger model comparison** on one shared split, and
two **dashboard-ready extracts** with a one-page
[findings summary](docs/findings_summary.md).

A deliberate, stated limitation: this is a cross-sectional snapshot (one 52-week
window), so no seasonality is modeled; the model captures **association, not
causation**; and genuinely new wines or stores revert to the global average until
they have a track record.

## Key Findings

1. **XGBoost is best**, just ahead of Random Forest (R² 0.64 vs 0.63 on the
   dollar scale); both clearly beat the linear baseline (0.31), so non-linear
   interactions between price, store and demographics matter.
2. **Which wine it is dominates.** A leakage-safe out-of-fold target encoding of
   item identity is the single strongest predictor — price and demographics add
   signal but explain far less on their own.
3. **Store tier and size track sales strongly.** Mean normalized sales rise
   monotonically from **$1,896** (lowest tier) to **$4,006** (highest); Extra-Large
   stores average **$7,642**, about **2.6×** the overall mean of **$2,963**.
4. **Local affluence matters at the margin.** Median household income, net worth
   and the share of households earning >$100K are positively associated with
   sales after controlling for store and item.
5. **Honest error, not a vanity metric.** The winner's ~$1,310 MAE and R² ≈ 0.64
   are reported instead of an "accuracy" figure, because the target is a
   continuous dollar amount.

See [`docs/findings_summary.md`](docs/findings_summary.md) for the one-page
write-up.

## Roadmap

- [x] 01 — Data profiling & honest target assessment
- [x] 02 — Cleaning & reproducible feature engineering
- [x] 03 — EDA with segment lift analysis
- [x] 04 — Baselines (Linear, Decision Tree) + challengers (Random Forest, XGBoost)
- [x] 04 — Leakage-safe out-of-fold target encoding + hyper-parameter tuning
- [x] 05 — Power BI extracts and hold-out (deployment) scoring
- [ ] Power BI dashboard published

---

*Portfolio project. Each step is documented so the analysis and decisions can be
followed end to end. See [`docs/methodology.md`](docs/methodology.md) for the
target definition, leakage controls, and modeling choices.*
