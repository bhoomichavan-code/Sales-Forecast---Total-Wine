# Methodology notes

Framework: **CRISP-DM**. This file records the judgment calls and their
trade-offs so each choice can be defended.

## Target definition
The target is `Normalized_Sales_$_L52W` — sales dollars annualised by weeks in
stock. We chose the normalized figure over raw `Actual_Sales_$` because raw
sales punish items that were only stocked for part of the year, which would
confound "low demand" with "low availability". Because the distribution is
extremely right-skewed (skew ≈ 15.5), we model `log1p(sales)` and back-transform
predictions to dollars; this keeps the loss from being dominated by a handful of
very-high-volume store-items.

## Metric choice
This is a **regression** problem, so the headline metric is **R²** (share of
variance explained), reported both on the log scale (what the model optimises)
and on the dollar scale (business-readable), alongside **MAE** and **RMSE** in
dollars and a **naive mean baseline** as the R²=0 reference. R² is the honest
regression analogue of an "accuracy" figure: our winning model explains about
**64% of the variance in normalized sales**.

## Leakage control
`Actual_Sales_$_L52W` is the raw target and is **excluded** from predictors.
High-cardinality identity fields (`Item_Name` = 1,451 levels, `Store_Number` =
269) are encoded with **out-of-fold target encoding**: each training row's
encoding is computed from *other* folds only, test rows use train-only means, and
category means are **smoothed** (weight 20) toward the global mean so rare
items/stores are not overconfident. This captures the dominant "which wine /
which store" signal without leaking a row's own target into its features.

## Store tier as a feature
`Store_Tier` and `Store_Tier_Matched` are store-level sales-volume ranks. They
are legitimately known before predicting a *new* item's sales (they describe the
store, not the item's own sales), so they are kept — and they are among the top
predictors. We flag that they are coarse, sales-derived store attributes, not
independent demographics.

## Modeling approach
Per the baseline-first standard: an interpretable **Linear Regression** and
**Decision Tree** are fit first, then **Random Forest** and **XGBoost**
challengers built to beat them, all through **one shared, leakage-safe
pipeline** (median-impute + scale numerics, most-frequent-impute + one-hot the
low-cardinality categoricals) on an identical 80/20 split. XGBoost hyper-
parameters were chosen with a randomized search over a documented grid; the final
values are stored so results are exactly reproducible (`SEARCH=1` re-runs the
search). Final: `n_estimators=400, max_depth=8, learning_rate=0.05,
subsample=0.9, colsample_bytree=0.8, min_child_weight=1, reg_lambda=2.0`.

## Interpretation
Feature importances are gain-based and describe **association, not causation**.
The dominance of the item-identity encoding means the model mostly learns "this
wine sells at this level, adjusted up or down by this store's tier, size, price
band and local affluence."
