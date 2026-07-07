"""
03_eda.py
---------
WHAT: Exploratory data analysis on the cleaned dataset -- target distribution,
      sales by store size / price band / store tier, and a numeric correlation
      heatmap. Saves publication-quality figures to visualizations/.
WHY : EDA surfaces the relationships the model will exploit and the honesty
      caveats (skew, small groups). Segment "lift" vs the overall average shows
      effect sizes a stakeholder can act on.

Run:  python analysis/03_eda.py
Outputs: visualizations/eda_*.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_prep import (engineer, load_clean, TARGET, FEATURES_NUMERIC, ROOT)  # noqa: E402

VIZ = ROOT / "visualizations"
VIZ.mkdir(parents=True, exist_ok=True)
RED, GOLD = "#8b1a1a", "#c99700"


def main():
    df = engineer(load_clean())
    df = df[df[TARGET].notna()].copy()
    overall = df[TARGET].mean()

    # 1) Target distribution (raw vs log) -- justifies the log transform.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df[TARGET].clip(upper=df[TARGET].quantile(.99)), bins=50, color=RED)
    ax[0].set_title("Normalized sales ($, ≤99th pct)"); ax[0].set_xlabel("$")
    ax[1].hist(np.log1p(df[TARGET]), bins=50, color=GOLD)
    ax[1].set_title("log(1 + normalized sales)"); ax[1].set_xlabel("log $")
    fig.suptitle("Target is extremely right-skewed → model on the log scale")
    fig.tight_layout(); fig.savefig(VIZ / "eda_target_distribution.png", dpi=130); plt.close(fig)

    # 2) Mean sales by segment, with lift vs overall.
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, col, order in [
        (ax[0], "Store_Size", ["Small", "Medium", "Large", "Extra Large"]),
        (ax[1], "Price_Band", ["Under_$20", "$20-50", "Over_$50"]),
        (ax[2], "Store_Tier", [0, 1, 2, 3, 4])]:
        g = df.groupby(col)[TARGET].agg(["mean", "size"]).reindex(order)
        a.bar([str(i) for i in g.index], g["mean"], color=RED)
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            a.text(i, m, f"{m/overall:.1f}x\nn={int(n):,}", ha="center", va="bottom", fontsize=8)
        a.set_title(f"Mean sales by {col}"); a.set_ylabel("$")
        a.set_ylim(0, g["mean"].max() * 1.25)
    fig.suptitle(f"Segment means vs overall average (${overall:,.0f}) — lift shown as ×")
    fig.tight_layout(); fig.savefig(VIZ / "eda_segment_sales.png", dpi=130); plt.close(fig)

    # 3) Correlation heatmap of numeric predictors + log target.
    num = [c for c in FEATURES_NUMERIC if df[c].notna().any()]
    corr = df[num + [TARGET]].assign(**{TARGET: np.log1p(df[TARGET])}).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Correlation (numeric features + log target)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(VIZ / "eda_correlation_heatmap.png", dpi=130); plt.close(fig)

    print("EDA figures written to visualizations/:")
    print(" eda_target_distribution.png, eda_segment_sales.png, eda_correlation_heatmap.png")
    print(f"Overall mean normalized sales: ${overall:,.0f}")
    for col in ["Store_Size", "Price_Band", "Store_Tier"]:
        g = df.groupby(col)[TARGET].mean()
        print(f"  {col}: " + ", ".join(f"{k}=${v:,.0f}" for k, v in g.items()))


if __name__ == "__main__":
    main()
