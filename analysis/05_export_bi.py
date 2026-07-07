"""
05_export_bi.py
---------------
WHAT: Exports purpose-built flat extracts for the Power BI dashboard.
WHY : Power BI visualises flat tables, not Python. We ship two grains:
      * store_item_predictions.csv -- one row per store-item with actual and
        predicted sales, absolute error, price band and store attributes
        (drives KPI cards, scatter, and item/segment analysis);
      * store_summary.csv -- one row per store (aggregates) for the store-tier
        and geography visuals.

Run:  python analysis/05_export_bi.py   (after 04_modeling.py)
Outputs: dashboard/extracts/{store_item_predictions,store_summary}.csv
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_prep import (build_model_frame, oof_target_encode, ALL_FEATURES,
                           TARGET, ROOT)
from sklearn.model_selection import train_test_split

EXTRACTS = ROOT / "dashboard" / "extracts"
EXTRACTS.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"


def main():
    df, y = build_model_frame()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    df, _ = oof_target_encode(df, y, tr, te)

    with open(MODELS / "xgboost_sales_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    pre, model = bundle["preprocessor"], bundle["model"]

    A = pre.transform(df[ALL_FEATURES]).astype("float32")
    df["Predicted_Sales"] = np.expm1(model.predict(A)).round(0)
    df["Actual_Sales"] = df[TARGET].round(0)
    df["Abs_Error"] = (df["Predicted_Sales"] - df["Actual_Sales"]).abs()
    df["Split"] = np.where(np.isin(df.index, te), "Test", "Train")

    wide = df[[
        "PK", "Store_Number", "Store_State", "Item_Name", "Package_Type",
        "Price_Band", "Retail", "Store_Size", "Store_Tier",
        "Age_of_the_Store_(years)", "Median_HH_Income", "%_HH_Income_>_$100K",
        "Average_Net_Worth", "Actual_Sales", "Predicted_Sales", "Abs_Error", "Split",
    ]].rename(columns={"Age_of_the_Store_(years)": "Store_Age_Years",
                       "Actual_Sales": "Actual_Sales_$",
                       "Predicted_Sales": "Predicted_Sales_$",
                       "Abs_Error": "Abs_Error_$"})
    wide.to_csv(EXTRACTS / "store_item_predictions.csv", index=False)

    store = (df.groupby(["Store_Number", "Store_State", "Store_Size", "Store_Tier"])
             .agg(**{"Items": ("Item_Name", "nunique"),
                     "Actual_Sales_Total": ("Actual_Sales", "sum"),
                     "Predicted_Sales_Total": ("Predicted_Sales", "sum"),
                     "Median_HH_Income": ("Median_HH_Income", "first")})
             .reset_index())
    store.to_csv(EXTRACTS / "store_summary.csv", index=False)

    print(f"[written] dashboard/extracts/store_item_predictions.csv  {wide.shape}")
    print(f"[written] dashboard/extracts/store_summary.csv  {store.shape}")


if __name__ == "__main__":
    main()
