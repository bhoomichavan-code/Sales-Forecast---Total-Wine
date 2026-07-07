# Findings summary — one page

**Goal.** Predict a store-item's normalized annual Cabernet sales to guide
inventory depth and promotion targeting.

**What we built.** A reproducible pipeline over 66,972 store-item records (269
stores, 1,451 wines, 28 states) that cleans the raw data, engineers store/price
features, applies leakage-safe out-of-fold target encoding of item and store
identity, and compares four models against a naive baseline.

**Headline result.** A tuned **XGBoost** model explains about **64% of the
variance** in normalized sales (test R² = 0.64 on $, 0.62 on log) with a mean
absolute error of **~$1,310** per store-item — roughly half the naive baseline's
$2,395 error, and clearly ahead of Linear Regression (R² 0.31) and the Decision
Tree (0.48).

**What drives sales.**
1. **The wine itself** — item identity is the strongest single predictor.
2. **Store sales tier** (both overall and within the item's price band).
3. **Store size** — Extra-Large stores sell ~2.6× the average.
4. **Local affluence** — income, net worth and the >$100K-income share add signal.

**How it's used.** The model scores the 7,320-row hold-out file to simulate
deployment, producing a predicted sales figure per new store-item that buyers can
compare against actual stocking to find over- and under-served combinations.

**Honest limitations.** Associations, not causation; a single 52-week snapshot
with no seasonality; and brand-new items/stores revert to the global average
until they have a track record.
