"""
04_modeling.py
--------------
WHAT: Trains and compares models through one shared, leakage-safe pipeline and
      reports honest regression metrics for predicting store-item Cabernet sales.
WHY : The business goal is to predict a store-item's normalised annual sales so
      Total Wine can plan inventory and target promotions. We follow the
      standard: an interpretable BASELINE first (Linear Regression, Decision
      Tree), then CHALLENGERS built to beat it (Random Forest, XGBoost).

Modelling choices (justified in docs/methodology.md):
  * Target = log1p(Normalized_Sales_$). Sales are extremely right-skewed, so we
    fit on the log scale and BACK-TRANSFORM to dollars for reporting.
  * Identity columns (Item_Name, Store_Number, Store_State) use OUT-OF-FOLD
    target encoding (leakage-safe) -- which wine and which store dominate sales.
    Low-card fields (Package_Type, Price_Band) are one-hot encoded.
  * Headline metric = R^2 (share of variance explained), reported on the log
    scale AND in real dollars, alongside MAE, RMSE and a naive mean baseline.

The XGBoost hyper-parameters below were selected with a randomized search over
the grid in PARAM_GRID (run `SEARCH=1 python analysis/04_modeling.py` to repeat
the search); the final values are hard-set so results are exactly reproducible.

Run:  python analysis/04_modeling.py
Outputs: docs/metrics.json, docs/model_comparison.csv,
         visualizations/{model_comparison,xgb_feature_importance,actual_vs_predicted}.png,
         models/xgboost_sales_model.pkl, data/processed/holdout_predictions.csv
"""
import os
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_prep import (build_model_frame, oof_target_encode, apply_target_maps,
                           MODEL_NUMERIC, FEATURES_CATEGORICAL, ALL_FEATURES,
                           RAW_TEST, ROOT)

RANDOM_STATE = 42
DOCS, VIZ, MODELS = ROOT / "docs", ROOT / "visualizations", ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
for d in (DOCS, VIZ, MODELS, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)

# Final XGBoost hyper-parameters (selected via randomized search; see docstring).
XGB_PARAMS = dict(n_estimators=400, max_depth=8, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.8, min_child_weight=1,
                  reg_lambda=2.0)
PARAM_GRID = {
    "n_estimators": [400, 600, 800], "max_depth": [4, 5, 6, 8],
    "learning_rate": [0.03, 0.05, 0.08], "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0], "min_child_weight": [1, 3, 5],
    "reg_lambda": [1.0, 2.0, 5.0]}


def make_preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, MODEL_NUMERIC),
                              ("cat", cat, FEATURES_CATEGORICAL)])


def _metrics(name, pred_log, y_te_log):
    pu, tu = np.expm1(pred_log), np.expm1(y_te_log)
    return {"model": name,
            "r2_log": float(r2_score(y_te_log, pred_log)),
            "r2_usd": float(r2_score(tu, pu)),
            "mae_usd": float(mean_absolute_error(tu, pu)),
            "rmse_usd": float(np.sqrt(mean_squared_error(tu, pu)))}


def main():
    df, y = build_model_frame()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)

    # Leakage-safe target encoding fit on TRAIN only; keep maps for the holdout.
    df, te_maps = oof_target_encode(df, y, tr, te)
    Xtr, Xte = df.iloc[tr][ALL_FEATURES], df.iloc[te][ALL_FEATURES]
    ytr, yte = y[tr], y[te]
    print(f"Train {len(tr):,} | Test {len(te):,} | features {len(ALL_FEATURES)}")

    pre = make_preprocessor()
    Atr = pre.fit_transform(Xtr).astype("float32")
    Ate = pre.transform(Xte).astype("float32")

    results = [_metrics("Naive (mean)", np.full(len(yte), ytr.mean()), yte)]
    results[0]["r2_log"] = 0.0

    base = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=50,
                                               random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_leaf=15,
            n_jobs=2, random_state=RANDOM_STATE),
    }
    for name, est in base.items():
        est.fit(Atr, ytr)
        results.append(_metrics(name, est.predict(Ate), yte))
        print(f"{name:20s} R2log={results[-1]['r2_log']:.3f} "
              f"R2$={results[-1]['r2_usd']:.3f} MAE=${results[-1]['mae_usd']:,.0f}")

    # XGBoost challenger. Default uses the selected params; SEARCH=1 re-tunes.
    if os.environ.get("SEARCH") == "1":
        search = RandomizedSearchCV(
            XGBRegressor(objective="reg:squarederror", tree_method="hist",
                         random_state=RANDOM_STATE, n_jobs=2),
            PARAM_GRID, n_iter=20, cv=3, scoring="r2",
            random_state=RANDOM_STATE, n_jobs=1)
        search.fit(Atr, ytr)
        best, params = search.best_estimator_, search.best_params_
    else:
        best = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                            random_state=RANDOM_STATE, n_jobs=2, **XGB_PARAMS)
        best.fit(Atr, ytr)
        params = XGB_PARAMS
    results.append(_metrics("XGBoost (tuned)", best.predict(Ate), yte))
    print(f"{'XGBoost (tuned)':20s} R2log={results[-1]['r2_log']:.3f} "
          f"R2$={results[-1]['r2_usd']:.3f} MAE=${results[-1]['mae_usd']:,.0f}")

    comp = pd.DataFrame(results)
    comp.to_csv(DOCS / "model_comparison.csv", index=False)
    winner = max((r for r in results if r["model"] != "Naive (mean)"),
                 key=lambda r: r["r2_log"])["model"]
    json.dump({"n_train": len(tr), "n_test": len(te), "n_features": len(ALL_FEATURES),
               "best_params": params, "winner": winner, "results": results},
              open(DOCS / "metrics.json", "w"), indent=2)

    _plot_comparison(comp)
    _plot_importance(best, pre)
    _plot_actual_vs_pred(best, Ate, yte)

    with open(MODELS / "xgboost_sales_model.pkl", "wb") as f:
        pickle.dump({"model": best, "preprocessor": pre, "te_maps": te_maps,
                     "features": ALL_FEATURES}, f)

    _score_holdout(best, pre, te_maps)
    hs = [r for r in results if r["model"] == winner][0]
    print(f"\nWINNER {winner}: R2(log)={hs['r2_log']:.3f}  R2($)={hs['r2_usd']:.3f}")


def _plot_comparison(comp):
    d = comp[comp.model != "Naive (mean)"].sort_values("r2_log")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(d.model, d.r2_log, color="#8b1a1a")
    for i, v in enumerate(d.r2_log):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center")
    ax.set_xlabel("Test R$^2$ (log sales — variance explained)")
    ax.set_title("Model comparison — higher is better")
    ax.set_xlim(0, min(1.0, float(d.r2_log.max()) + 0.12))
    fig.tight_layout(); fig.savefig(VIZ / "model_comparison.png", dpi=130); plt.close(fig)


def _plot_importance(model, pre, top=15):
    names = pre.get_feature_names_out()
    s = (pd.Series(model.feature_importances_, index=names)
         .sort_values(ascending=False).head(top)[::-1])
    labels = [n.split("__", 1)[-1] for n in s.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, s.values, color="#c99700")
    ax.set_title(f"XGBoost — top {top} feature importances")
    ax.set_xlabel("Gain-based importance")
    fig.tight_layout(); fig.savefig(VIZ / "xgb_feature_importance.png", dpi=130); plt.close(fig)


def _plot_actual_vs_pred(model, Ate, yte):
    pred, true = np.expm1(model.predict(Ate)), np.expm1(yte)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, s=6, alpha=0.25, color="#8b1a1a")
    lim = float(np.percentile(true, 99))
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual normalized sales ($)"); ax.set_ylabel("Predicted ($)")
    ax.set_title("XGBoost — actual vs predicted (<=99th pct)")
    fig.tight_layout(); fig.savefig(VIZ / "actual_vs_predicted.png", dpi=130); plt.close(fig)


def _score_holdout(model, pre, te_maps):
    """Score the provided unlabeled test file (simulates deployment)."""
    try:
        dfh, _ = build_model_frame(RAW_TEST, is_labeled=False)
        dfh = apply_target_maps(dfh, te_maps)
        Ah = pre.transform(dfh[ALL_FEATURES]).astype("float32")
        out = dfh[["PK", "Store_Number", "Item_Name", "Retail"]].copy()
        out["Predicted_Normalized_Sales_$"] = np.expm1(model.predict(Ah)).round(0)
        out.to_csv(PROCESSED / "holdout_predictions.csv", index=False)
        print(f"[holdout] scored {len(out):,} rows -> data/processed/holdout_predictions.csv")
    except Exception as e:
        print(f"[holdout] skipped: {e}")


if __name__ == "__main__":
    main()
