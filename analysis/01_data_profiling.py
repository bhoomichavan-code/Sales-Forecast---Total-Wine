"""
01_data_profiling.py
--------------------
WHAT: Profiles the raw internal dataset -- shape, dtypes, missingness, target
      distribution, and cardinality of key categoricals.
WHY : Before any cleaning or modelling we need an honest picture of what the
      data can and cannot support. This step documents data quality issues
      (skew, missing values, rare categories) that drive later decisions.

Run:  python analysis/01_data_profiling.py
Output: prints a profile report and writes docs/data_profile.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_prep import load_clean, engineer, TARGET, ROOT  # noqa: E402

OUT = ROOT / "docs" / "data_profile.md"


def main():
    df = engineer(load_clean())
    lines = ["# Data Profile -- Cabernet Sauvignon Internal Sales", ""]
    lines.append(f"- Rows: **{len(df):,}**  |  Columns (raw): 25")
    lines.append(f"- Distinct stores: **{df['Store_Number'].nunique()}**  |  "
                 f"Distinct items: **{df['Item_Name'].nunique():,}**  |  "
                 f"States: **{df['Store_State'].nunique()}**")
    lines.append("")

    # Missingness (count and %) -- report both, per profiling standard.
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    lines.append("## Missing values")
    if len(miss):
        for c, n in miss.items():
            lines.append(f"- `{c}`: {n} ({n/len(df)*100:.2f}%)")
    else:
        lines.append("- None.")
    lines.append("")

    # Target distribution -- the headline honesty check.
    t = df[TARGET].dropna()
    lines.append("## Target: Normalized_Sales_$_L52W")
    lines.append(f"- min ${t.min():,.0f} | p10 ${t.quantile(.1):,.0f} | "
                 f"median ${t.median():,.0f} | mean ${t.mean():,.0f} | "
                 f"p90 ${t.quantile(.9):,.0f} | max ${t.max():,.0f}")
    lines.append(f"- Skew: **{t.skew():.1f}** (extreme right skew -> model log scale)")
    lines.append(f"- Exact-zero sales rows: {int((t == 0).sum())} "
                 f"({(t == 0).mean()*100:.1f}%)")
    lines.append("")

    lines.append("## Key categoricals")
    for c in ["Store_Size", "Package_Type", "Price_Band", "Store_Tier"]:
        vc = df[c].value_counts(dropna=False)
        top = ", ".join(f"{k}={v:,}" for k, v in vc.head(6).items())
        lines.append(f"- `{c}` ({df[c].nunique()} levels): {top}")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[written] {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
