# Total Wine — Cabernet Sauvignon Sales Prediction

Predicting store-item annual sales for Cabernet Sauvignon across Total Wine &
More's national footprint, to support **inventory planning** and **targeted
promotion** decisions. A gradient-boosted model (XGBoost) explains roughly
**two-thirds of the variance** in normalized sales (test **R² = 0.64** on the
dollar scale, **0.62** on the log scale) with a mean absolute error of about
**$1,310** per store-item — a large improvement over the interpretable baselines.

## Business problem

Total Wine carries a very deep Cabernet assortment (1,451 distinct products in
this dataset) across 269 stores in 28 states. Buyers must decide **how much of
each wine each store will sell** so they can stock the right depth and aim
promotions at stores with untapped demand. This project builds a model that
predicts a store-item's **normalized annual sales dollars** from price, store
attributes, and local trade-area demographics.

## Data

| File | Grain | Rows | Notes |
|------|-------|------|-------|
| `TW_Master_Internal_Data.csv` | store × item | 66,974 | Internal sales + store attributes + 5-mile-radius demographics (raw, immutable) |
| `TW_Final Test Data.csv` | store × item | 7,320 | Unlabeled hold-out (sales blank) — scored as a deployment simulation |
| `Data Field Definitions.xlsx` | — | — | Field definitions |
| `Store Information.xlsx` | store | — | Store metadata |

**Target — `Normalized_Sales_$_L52W`:** last-52-weeks sales dollars, scaled by
the number of weeks the item was actually in stock, i.e.
`(L52W sales $ / weeks in stock) × 52`. This removes the penalty on items that
were only stocked part of the year, giving a fair measure of underlying demand.
The target is extremely right-skewed (median $1,203, mean $2,963, max $454K,
skew 15.5), so it is modeled on a **log scale** and back-transformed to dollars
for reporting.

## Results

Test set = 13,395 store-items (20% hold-out). Headline metric is R² (share of
variance explained); MAE and RMSE are in real dollars.

| Model | R² (log) | R² ($) | MAE | RMSE |
|-------|:--------:|:------:|:---:|:----:|
| Naive (mean) | 0.00 | −0.06 | $2,395 | $7,289 |
| Linear Regression | 0.51 | 0.31 | $1,660 | $5,863 |
| Decision Tree | 0.56 | 0.48 | $1,579 | $5,113 |
| Random Forest | 0.61 | 0.63 | $1,375 | $4,304 |
| **XGBoost (tuned)** | **0.62** | **0.64** | **$1,310** | **$4,213** |

### Key findings

- **The winning XGBoost model explains ~64% of the variance in normalized sales
  ($ scale) and cuts the naive mean's error nearly in half** ($2,395 → $1,310
  MAE).
- **Which wine it is dominates.** Item identity (a leakage-safe out-of-fold
  target encoding of `Item_Name`) is the single strongest predictor, followed by
  the store's price-band sales tier and overall sales tier. Price and demographics
  add signal but explain much less on their own.
- **Store tier and store size track sales strongly.** Mean normalized sales rise
  monotonically from **$1,896** in the lowest-tier stores to **$4,006** in the
  highest; Extra-Large stores average **$7,642**, about **2.6×** the overall mean
  of **$2,963**.
- **Local affluence matters at the margin.** Median household income, net worth,
  and the share of households earning >$100K are positively associated with
  Cabernet sales after controlling for store and item.

## Project structure

```
Total Wine Sales Forecast/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                     <- immutable source data + data dictionary
│   └── processed/               <- generated model dataset & predictions
├── analysis/                    <- numbered, run-in-order scripts
│   ├── 01_data_profiling.py
│   ├── 02_clean_features.py
│   ├── 03_eda.py
│   ├── 04_modeling.py
│   └── 05_export_bi.py
├── src/
│   └── data_prep.py             <- shared cleaning + feature engineering
├── visualizations/              <- exported figures
├── models/                      <- saved model bundle (.pkl)
├── dashboard/extracts/          <- flat CSVs for Power BI
└── docs/                        <- profile, methodology, findings
```

## How to run

```bash
pip install -r requirements.txt
python analysis/01_data_profiling.py     # data profile
python analysis/02_clean_features.py     # cleaned model dataset
python analysis/03_eda.py                # EDA figures
python analysis/04_modeling.py           # train, evaluate, save model + metrics
python analysis/05_export_bi.py          # Power BI extracts
```

`python analysis/04_modeling.py` reproduces every number above. Set `SEARCH=1`
to re-run the XGBoost hyper-parameter search instead of using the stored values.

## Roadmap / status

- [x] Data profiling and honest target assessment
- [x] Cleaning + reproducible feature engineering (shared module)
- [x] EDA with segment lift analysis
- [x] Baseline models (Linear, Decision Tree) + challengers (Random Forest, XGBoost)
- [x] Leakage-safe out-of-fold target encoding of identity fields
- [x] Hyper-parameter tuning and model comparison
- [x] Hold-out (deployment) scoring
- [x] Power BI extracts
- [ ] Publish Power BI dashboard and link it here

## Limitations

- The model predicts an **association**, not a causal effect; a demographic
  coefficient does not mean changing that demographic would change sales.
- Sales are cross-sectional (one 52-week window), so no seasonality or trend is
  modeled.
- Target encoding of item/store identity is powerful but assumes future items and
  stores resemble past ones; genuinely new products fall back to the global mean.
