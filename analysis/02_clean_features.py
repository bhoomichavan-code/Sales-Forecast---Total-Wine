"""
02_clean_features.py
--------------------
WHAT: Applies the shared cleaning + feature engineering and writes a single,
      consistent processed dataset used by EDA and modelling.
WHY : Reproducibility. Downstream steps load THIS artifact so numbers never
      drift. Raw data stays immutable; all transforms happen in code.

Feature-engineering decisions (each justified in src/data_prep.py):
  * Store_Size_ord      -- ordinal (Small<Medium<Large<Extra Large): order = signal.
  * Price_Band          -- Total Wine's own <$20 / $20-50 / >$50 cut points.
  * Store_Tier_Matched  -- store's sales-volume tier within the item's price band.
  * Income_Age_Interaction -- high income x older (50-70) buyer base overlap.
  * log_Retail          -- tames price skew for linear models.

Target encoding of the identity columns (Item_Name, Store_Number, Store_State)
is done at MODELLING time, per train/test split, so it stays leakage-safe.

Run:  python analysis/02_clean_features.py
Output: data/processed/model_data.csv
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_prep import (engineer, load_clean, TARGET, FEATURES_NUMERIC,
                           FEATURES_CATEGORICAL, TE_COLS, ID_COLS, PROCESSED, ROOT)


def main():
    df = engineer(load_clean())
    base = FEATURES_NUMERIC + FEATURES_CATEGORICAL
    keep = ID_COLS + [c for c in TE_COLS if c not in ID_COLS] + base + [TARGET]
    keep = list(dict.fromkeys(keep))  # de-dup, preserve order
    out = df[keep].copy()
    out = out[out[TARGET].notna()].reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / "model_data.csv"
    out.to_csv(path, index=False)
    print(f"[written] {path.relative_to(ROOT)}  shape={out.shape}")
    print(f"Base predictors: {len(base)} + target-encoded {len(TE_COLS)} | target: {TARGET}")
    n = out[base].isna().sum()
    print("Null check on base predictors:", n[n > 0].to_dict() or "none")


if __name__ == "__main__":
    main()
