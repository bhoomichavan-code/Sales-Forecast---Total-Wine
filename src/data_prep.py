"""
data_prep.py
------------
Shared data-loading, cleaning and feature-engineering helpers for the
Cabernet Sauvignon sales-prediction project.

WHY a shared module: every numbered analysis script starts from the SAME raw
file and the SAME cleaning logic, so results are reproducible and never drift
between the profiling, EDA and modelling steps. The raw CSV is treated as
immutable; every transformation lives here in code.

Target:  Normalized_Sales_$_L52W
    = annualised sales dollars for a store-item, corrected for how many of the
      last 52 weeks the item was actually in stock (a fair measure of demand).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW_MASTER = RAW / "TW_Master_Internal_Data.csv"
RAW_TEST = RAW / "TW_Final Test Data.csv"

TARGET = "Normalized_Sales_$_L52W"
STORE_SIZE_ORDER = {"Small": 0, "Medium": 1, "Large": 2, "Extra Large": 3}
LEAKAGE_COLS = ["Actual_Sales_$_L52W", "Normalized_Sales_$_L52W"]
ID_COLS = ["PK", "Store_Number", "Item_Code", "Item_Name"]


def _money(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip(),
        errors="coerce")


def _pct(s):
    """'77.30%' and 0.773 both -> 77.3 (common 0-100 scale)."""
    raw = s.astype(str).str.replace("%", "", regex=False).str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    had_sign = s.astype(str).str.contains("%")
    frac = (~had_sign) & (num <= 1)
    return num.where(~frac, num * 100)


def _price_band(retail):
    return pd.cut(retail, bins=[-np.inf, 20, 50, np.inf],
                  labels=["Under_$20", "$20-50", "Over_$50"])


def load_clean(path=RAW_MASTER, is_labeled=True):
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    money_cols = ["Retail", "Median_HH_Income", "Average_Net_Worth"]
    if is_labeled:
        money_cols += ["Actual_Sales_$_L52W", TARGET]
    for c in money_cols:
        if c in df.columns:
            df[c] = _money(df[c])
    for c in ["%_HH_Income_>_$100K", "%_Population_w/_Bachelor's_Degree_+",
              "%_Hispanic", "%_Asian", "%_African_American",
              "%_Population_Age_50-70"]:
        df[c] = _pct(df[c])
    for c in ["L52W_in_Stock", "Age_of_the_Store_(years)", "Households_(HH)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Store_Size"] = df["Store_Size"].astype(str).str.strip()
    df["Store_Size_ord"] = df["Store_Size"].map(STORE_SIZE_ORDER)
    for c in ["Store_Tier_(Under_$20)", "Store_Tier_($20-50)",
              "Store_Tier_(Over_$50)", "Store_Tier"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def engineer(df):
    df = df.copy()
    df["Price_Band"] = _price_band(df["Retail"]).astype("object")
    band_to_col = {"Under_$20": "Store_Tier_(Under_$20)",
                   "$20-50": "Store_Tier_($20-50)",
                   "Over_$50": "Store_Tier_(Over_$50)"}
    df["Store_Tier_Matched"] = np.nan
    for band, col in band_to_col.items():
        m = df["Price_Band"] == band
        df.loc[m, "Store_Tier_Matched"] = df.loc[m, col]
    df["Income_Age_Interaction"] = (
        df["Median_HH_Income"] * df["%_Population_Age_50-70"] / 100.0)
    df["log_Retail"] = np.log1p(df["Retail"])
    return df


FEATURES_NUMERIC = [
    "Retail", "log_Retail", "L52W_in_Stock", "Age_of_the_Store_(years)",
    "Store_Size_ord", "Households_(HH)", "%_HH_Income_>_$100K",
    "Median_HH_Income", "Average_Net_Worth", "%_Population_w/_Bachelor's_Degree_+",
    "%_Hispanic", "%_Asian", "%_African_American", "%_Population_Age_50-70",
    "Income_Age_Interaction", "Store_Tier", "Store_Tier_Matched",
]
TE_COLS = ["Item_Name", "Store_Number", "Store_State"]
TE_FEATURES = [c + "_te" for c in TE_COLS]
FEATURES_CATEGORICAL = ["Package_Type", "Price_Band"]
MODEL_NUMERIC = FEATURES_NUMERIC + TE_FEATURES
ALL_FEATURES = MODEL_NUMERIC + FEATURES_CATEGORICAL


def oof_target_encode(df, y, train_idx, test_idx, cols=TE_COLS, n_splits=5,
                      smoothing=20.0, random_state=1):
    """Out-of-fold, smoothed mean-target encoding for identity columns.

    TRAIN rows encoded out-of-fold (from other folds only); TEST rows encoded
    from full-train means; smoothing shrinks rare categories to the global mean.
    Returns (df_with_te_cols, maps) where maps score brand-new rows later.
    """
    df = df.copy()
    y = np.asarray(y, dtype=float)
    global_mean = float(y[train_idx].mean())
    maps = {}
    for col in cols:
        enc = np.full(len(df), global_mean, dtype=float)
        tmp = pd.DataFrame({col: df[col].values, "_y": y})
        agg = tmp.iloc[train_idx].groupby(col)["_y"].agg(["mean", "count"])
        smooth = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
            agg["count"] + smoothing)
        maps[col] = {"map": smooth.to_dict(), "global": global_mean}
        enc[test_idx] = df.iloc[test_idx][col].map(smooth).fillna(global_mean).values
        tr = np.asarray(train_idx)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for inner_tr, inner_val in kf.split(tr):
            a, b = tr[inner_tr], tr[inner_val]
            sub = tmp.iloc[a].groupby(col)["_y"].agg(["mean", "count"])
            sm = (sub["mean"] * sub["count"] + global_mean * smoothing) / (
                sub["count"] + smoothing)
            enc[b] = df.iloc[b][col].map(sm).fillna(global_mean).values
        df[col + "_te"] = enc
    return df, maps


def apply_target_maps(df, maps):
    df = df.copy()
    for col, m in maps.items():
        df[col + "_te"] = df[col].map(m["map"]).fillna(m["global"])
    return df


def build_model_frame(path=RAW_MASTER, is_labeled=True):
    df = engineer(load_clean(path, is_labeled=is_labeled))
    if is_labeled:
        df = df[df[TARGET].notna()].reset_index(drop=True)
        y = np.log1p(df[TARGET].clip(lower=0).to_numpy())
        return df, y
    return df.reset_index(drop=True), None


if __name__ == "__main__":
    df, y = build_model_frame()
    print("frame:", df.shape, "target rows:", None if y is None else y.shape)
    print("model features:", ALL_FEATURES)
