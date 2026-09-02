"""
IEEE-CIS Fraud Detection — EDA Analysis
Loads data exactly as the notebook does (identity-left-join → 144,233 rows x 434 cols).
Saves all plots to OUTPUT_DIR for embedding in the report.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Notebook")
DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_DIR   = "/Users/shriyanshraj/.gemini/antigravity-ide/brain/98590e36-f8e7-4b2b-8f32-e55098f07b90"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {"Non-Fraud": "#4C9BE8", "Fraud": "#E8524C"}
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "savefig.bbox": "tight",
                     "savefig.facecolor": "#1a1a2e"})
FIG_BG = "#1a1a2e"
AX_BG  = "#16213e"
TEXT   = "#e0e0e0"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    if title:   ax.set_title(title, color=TEXT)
    if xlabel:  ax.set_xlabel(xlabel, color=TEXT)
    if ylabel:  ax.set_ylabel(ylabel, color=TEXT)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, facecolor=FIG_BG)
    plt.close(fig)
    print(f"  [saved] {name}")
    return path

# ── Load data (notebook-style: identity left-join) ────────────────────────────
print("=" * 60)
print("Loading data (notebook-style identity left-join) …")
os.chdir(NOTEBOOK_DIR)
train_identity    = pd.read_csv("../data/train_identity.csv")
train_transaction = pd.read_csv("../data/train_transaction.csv")
df = pd.merge(train_identity, train_transaction, on="TransactionID", how="left")
print(f"  df.shape = {df.shape}")
assert df.shape == (144233, 434), f"Unexpected shape: {df.shape}"
print("  Shape confirmed: 144,233 × 434 ✓")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLASS IMBALANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. CLASS IMBALANCE")
print("=" * 60)

vc = df["isFraud"].value_counts()
n_total   = len(df)
n_fraud   = int(vc[1])
n_legit   = int(vc[0])
pct_fraud = n_fraud / n_total * 100
pct_legit = n_legit / n_total * 100

print(f"  Total rows  : {n_total:,}")
print(f"  Fraud  (1)  : {n_fraud:,}   ({pct_fraud:.4f}%)")
print(f"  Legit  (0)  : {n_legit:,} ({pct_legit:.4f}%)")
print(f"  Imbalance ratio (legit:fraud) = {n_legit/n_fraud:.1f}:1")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=FIG_BG)

# Bar
bars = axes[0].bar(["Non-Fraud", "Fraud"], [n_legit, n_fraud],
                   color=[PALETTE["Non-Fraud"], PALETTE["Fraud"]], width=0.5,
                   edgecolor="#333", linewidth=1.2)
for bar, val, pct in zip(bars, [n_legit, n_fraud], [pct_legit, pct_fraud]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.015,
                 f"{val:,}\n({pct:.2f}%)", ha="center", va="bottom",
                 color=TEXT, fontsize=10, fontweight="bold")
axes[0].set_ylim(0, n_legit * 1.18)
style_ax(axes[0], "Class Distribution (Count)", "", "Count")

# Pie
wedges, texts, autotexts = axes[1].pie(
    [n_legit, n_fraud],
    labels=["Non-Fraud", "Fraud"],
    colors=[PALETTE["Non-Fraud"], PALETTE["Fraud"]],
    autopct="%1.2f%%", startangle=90,
    wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
    textprops={"color": TEXT})
for at in autotexts: at.set_color(TEXT); at.set_fontweight("bold")
axes[1].set_facecolor(AX_BG)
style_ax(axes[1], "Class Distribution (Share)")
fig.suptitle("IEEE-CIS Fraud Detection — Class Imbalance", color=TEXT,
             fontsize=14, fontweight="bold", y=1.01)
save(fig, "eda_01_class_imbalance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MISSINGNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. MISSINGNESS ANALYSIS")
print("=" * 60)

miss_pct = df.isnull().mean().sort_values(ascending=False) * 100

n_over50 = (miss_pct > 50).sum()
n_over75 = (miss_pct > 75).sum()
n_zero   = (miss_pct == 0).sum()
print(f"  Columns >50% missing : {n_over50} / {len(miss_pct)}  ({n_over50/len(miss_pct)*100:.1f}%)")
print(f"  Columns >75% missing : {n_over75} / {len(miss_pct)}  ({n_over75/len(miss_pct)*100:.1f}%)")
print(f"  Columns 0%  missing  : {n_zero}")

print("\n  Most COMPLETE columns (lowest missingness):")
print(miss_pct.tail(20).to_string())

print("\n  Most EMPTY columns (highest missingness, top 20):")
print(miss_pct.head(20).to_string())

# 2a — overall missingness heatmap (sorted)
fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=FIG_BG)
ax.set_facecolor(AX_BG)
bins = [0, 10, 25, 50, 75, 90, 100]
hist_vals, _ = np.histogram(miss_pct.values, bins=bins)
bar_labels = ["0–10%", "10–25%", "25–50%", "50–75%", "75–90%", "90–100%"]
bar_colors = ["#4C9BE8", "#6BC9A0", "#F5D76E", "#F0A500", "#E8524C", "#8B0000"]
bars = ax.bar(bar_labels, hist_vals, color=bar_colors, edgecolor="#333", linewidth=1)
for bar, val in zip(bars, hist_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            str(val), ha="center", va="bottom", color=TEXT, fontweight="bold")
style_ax(ax, "Missingness Distribution Across 434 Columns",
         "Missing % Bucket", "Number of Columns")
save(fig, "eda_02a_missingness_hist.png")

# 2b — missingness diff fraud vs non-fraud (top 30 by absolute diff)
fraud_miss = df[df["isFraud"] == 1].isnull().mean() * 100
legit_miss = df[df["isFraud"] == 0].isnull().mean() * 100
miss_diff  = (fraud_miss - legit_miss).abs().sort_values(ascending=False)
top_diff   = miss_diff.head(30)

print("\n  Top 15 columns where missingness differs most between fraud/non-fraud:")
for col in top_diff.head(15).index:
    print(f"    {col:<20}  fraud miss={fraud_miss[col]:.1f}%  legit miss={legit_miss[col]:.1f}%  diff={miss_diff[col]:.1f}pp")

fig, ax = plt.subplots(figsize=(13, 7), facecolor=FIG_BG)
ax.set_facecolor(AX_BG)
y_pos = range(len(top_diff))
ax.barh(y_pos, top_diff.values, color="#E8524C", edgecolor="#333", linewidth=0.8)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(top_diff.index, color=TEXT, fontsize=9)
style_ax(ax, "Top 30 Columns: |Missingness Diff| Fraud vs Non-Fraud (pp)",
         "Absolute Difference (percentage points)", "")
save(fig, "eda_02b_missingness_fraud_diff.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUTION CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. DISTRIBUTION CHECKS")
print("=" * 60)

fraud_df = df[df["isFraud"] == 1]
legit_df = df[df["isFraud"] == 0]

# ── 3a TransactionAmt ─────────────────────────────────────────────────────────
print("\n  TransactionAmt stats by class:")
for label, sub in [("Non-Fraud", legit_df), ("Fraud", fraud_df)]:
    a = sub["TransactionAmt"]
    print(f"    {label}: median=${a.median():.2f}  mean=${a.mean():.2f}  "
          f"p95=${a.quantile(.95):.2f}  max=${a.max():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=FIG_BG)
cap = df["TransactionAmt"].quantile(0.99)
for label, sub, col in [("Non-Fraud", legit_df, PALETTE["Non-Fraud"]),
                         ("Fraud",     fraud_df, PALETTE["Fraud"])]:
    data = sub["TransactionAmt"].clip(upper=cap)
    axes[0].hist(data, bins=80, alpha=0.6, color=col, label=label, density=True,
                 edgecolor="none")
    axes[1].hist(np.log1p(sub["TransactionAmt"]), bins=80, alpha=0.6,
                 color=col, label=label, density=True, edgecolor="none")
style_ax(axes[0], f"TransactionAmt (capped at ${cap:.0f})", "Amount ($)", "Density")
style_ax(axes[1], "log(1 + TransactionAmt)", "log(1+$)", "Density")
for ax in axes:
    ax.legend(facecolor="#222", labelcolor=TEXT, framealpha=0.7)
save(fig, "eda_03a_transaction_amt.png")

# ── 3b Categorical: card4, card6, ProductCD ────────────────────────────────────
for col in ["card4", "card6", "ProductCD"]:
    if col not in df.columns:
        print(f"  {col} not found, skipping")
        continue
    print(f"\n  {col} value counts + fraud rate:")
    vc2 = df.groupby(col)["isFraud"].agg(["count", "mean"]).rename(
        columns={"count": "n", "mean": "fraud_rate"}).sort_values("n", ascending=False)
    vc2["fraud_rate_pct"] = vc2["fraud_rate"] * 100
    print(vc2.to_string())

    cats = vc2.index.tolist()
    x    = np.arange(len(cats))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(cats)*1.4), 4.5), facecolor=FIG_BG)

    # counts stacked
    legit_cnts = [legit_df[col].value_counts().get(c, 0) for c in cats]
    fraud_cnts = [fraud_df[col].value_counts().get(c, 0) for c in cats]
    axes[0].bar(x, legit_cnts, label="Non-Fraud", color=PALETTE["Non-Fraud"],
                edgecolor="#333")
    axes[0].bar(x, fraud_cnts, bottom=legit_cnts, label="Fraud",
                color=PALETTE["Fraud"], edgecolor="#333")
    axes[0].set_xticks(x); axes[0].set_xticklabels(cats, rotation=30, ha="right",
                                                     color=TEXT)
    axes[0].legend(facecolor="#222", labelcolor=TEXT)
    style_ax(axes[0], f"{col} — Count by Class", col, "Count")

    # fraud rate
    colors_bar = [PALETTE["Fraud"] if r > pct_fraud/100 else "#6BC9A0"
                  for r in vc2["fraud_rate"]]
    axes[1].bar(x, vc2["fraud_rate_pct"].values, color=colors_bar, edgecolor="#333")
    axes[1].axhline(pct_fraud, color="white", linestyle="--", linewidth=1.2,
                    label=f"Overall avg ({pct_fraud:.2f}%)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cats, rotation=30, ha="right",
                                                     color=TEXT)
    axes[1].legend(facecolor="#222", labelcolor=TEXT, framealpha=0.7)
    style_ax(axes[1], f"{col} — Fraud Rate (%)", col, "Fraud %")
    save(fig, f"eda_03b_{col}.png")

# ── 3c Device/browser columns ──────────────────────────────────────────────────
device_cols = [c for c in ["DeviceType", "id_30", "id_31", "DeviceInfo"]
               if c in df.columns]
print(f"\n  Device/browser columns present: {device_cols}")

for col in device_cols:
    n_unique = df[col].nunique()
    print(f"\n  {col}: {n_unique} unique values, {df[col].isnull().mean()*100:.1f}% missing")
    top_cats = df[col].value_counts().head(12).index.tolist()
    sub_df   = df[df[col].isin(top_cats)]

    fraud_rate_by_cat = sub_df.groupby(col)["isFraud"].mean().sort_values(ascending=False)
    count_by_cat      = sub_df.groupby(col)["isFraud"].count().reindex(fraud_rate_by_cat.index)
    cats_plot = fraud_rate_by_cat.index.tolist()
    x = np.arange(len(cats_plot))

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(cats_plot)*1.2), 5), facecolor=FIG_BG)
    axes[0].barh(x, count_by_cat.values, color=PALETTE["Non-Fraud"], edgecolor="#333")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels([str(c)[:30] for c in cats_plot], color=TEXT, fontsize=8)
    style_ax(axes[0], f"{col} — Count (top {len(cats_plot)})", "Count", "")

    bar_colors2 = [PALETTE["Fraud"] if r > pct_fraud/100 else "#6BC9A0"
                   for r in fraud_rate_by_cat.values]
    axes[1].barh(x, fraud_rate_by_cat.values * 100, color=bar_colors2, edgecolor="#333")
    axes[1].axvline(pct_fraud, color="white", linestyle="--", linewidth=1.2,
                    label=f"Overall avg ({pct_fraud:.2f}%)")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([str(c)[:30] for c in cats_plot], color=TEXT, fontsize=8)
    axes[1].legend(facecolor="#222", labelcolor=TEXT, framealpha=0.7)
    style_ax(axes[1], f"{col} — Fraud Rate (%)", "Fraud %", "")
    save(fig, f"eda_03c_{col}.png")
    print(f"    Top by fraud rate: {fraud_rate_by_cat.head(5).to_dict()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TransactionDT — FRAUD RATE OVER TIME
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. TransactionDT — FRAUD RATE OVER TIME")
print("=" * 60)

dt_min = df["TransactionDT"].min()
dt_max = df["TransactionDT"].max()
span_days = (dt_max - dt_min) / 86400
print(f"  TransactionDT range: {dt_min:,} — {dt_max:,}  ({span_days:.1f} days)")

# Convert to day offset
df["day_offset"] = (df["TransactionDT"] - dt_min) / 86400

# Bucket into 7-day windows
df["week_bucket"] = (df["day_offset"] // 7).astype(int)
weekly = df.groupby("week_bucket").agg(
    total=("isFraud", "count"),
    frauds=("isFraud", "sum")
).reset_index()
weekly["fraud_rate"] = weekly["frauds"] / weekly["total"] * 100
weekly["week_start"] = weekly["week_bucket"] * 7

print("\n  Weekly fraud rate summary:")
print(weekly[["week_start", "total", "frauds", "fraud_rate"]].to_string(index=False))

fig, axes = plt.subplots(2, 1, figsize=(13, 8), facecolor=FIG_BG, sharex=True)
# Volume
axes[0].fill_between(weekly["week_start"], weekly["total"],
                      color=PALETTE["Non-Fraud"], alpha=0.6, label="Total txns")
axes[0].fill_between(weekly["week_start"], weekly["frauds"],
                      color=PALETTE["Fraud"], alpha=0.85, label="Fraud txns")
style_ax(axes[0], "Transaction Volume by Week", "", "Count")
axes[0].legend(facecolor="#222", labelcolor=TEXT)
# Fraud rate
axes[1].plot(weekly["week_start"], weekly["fraud_rate"],
             color="#F5D76E", linewidth=2.5, marker="o", markersize=5)
axes[1].axhline(pct_fraud, color="white", linestyle="--", linewidth=1,
                label=f"Overall avg ({pct_fraud:.2f}%)")
axes[1].fill_between(weekly["week_start"], weekly["fraud_rate"],
                      pct_fraud, alpha=0.2,
                      where=weekly["fraud_rate"] > pct_fraud, color="#E8524C",
                      interpolate=True)
axes[1].fill_between(weekly["week_start"], weekly["fraud_rate"],
                      pct_fraud, alpha=0.2,
                      where=weekly["fraud_rate"] <= pct_fraud, color="#4C9BE8",
                      interpolate=True)
style_ax(axes[1], "Weekly Fraud Rate (%)", "Day Offset (from dataset start)", "Fraud Rate %")
axes[1].legend(facecolor="#222", labelcolor=TEXT)
fig.suptitle("TransactionDT — Fraud Distribution Over Time", color=TEXT,
             fontsize=14, fontweight="bold")
save(fig, "eda_04_transactiondt.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. V-COLUMNS SIGNAL CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. V-COLUMNS — SIGNAL CHECK (point-biserial correlation with isFraud)")
print("=" * 60)

v_cols = [c for c in df.columns if c.startswith("V")]
print(f"  Total V-columns: {len(v_cols)}")

pb_corrs = {}
for col in v_cols:
    valid = df[[col, "isFraud"]].dropna()
    if len(valid) < 500: continue
    r, p = stats.pointbiserialr(valid["isFraud"], valid[col])
    pb_corrs[col] = r

pb_series = pd.Series(pb_corrs).sort_values(key=abs, ascending=False)
print(f"\n  Top 20 V-columns by |point-biserial r| with isFraud:")
print(pb_series.head(20).to_string())
print(f"\n  Bottom 5 (near-zero signal):")
print(pb_series.tail(5).to_string())

# Plot top 30
top_v = pb_series.head(30)
fig, ax = plt.subplots(figsize=(13, 7), facecolor=FIG_BG)
ax.set_facecolor(AX_BG)
colors_v = [PALETTE["Fraud"] if v > 0 else PALETTE["Non-Fraud"] for v in top_v.values]
ax.barh(range(len(top_v)), top_v.values[::-1], color=colors_v[::-1], edgecolor="#333")
ax.set_yticks(range(len(top_v)))
ax.set_yticklabels(top_v.index[::-1].tolist(), color=TEXT, fontsize=9)
ax.axvline(0, color="white", linewidth=0.8)
style_ax(ax, "Top 30 V-Columns — Point-Biserial Correlation with isFraud",
         "Point-Biserial r", "")
save(fig, "eda_05a_vcols_correlation.png")

# Distribution plot for top 6 V-columns
top6_v = pb_series.head(6).index.tolist()
print(f"\n  Top 6 V-columns for distribution plot: {top6_v}")
fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=FIG_BG)
axes = axes.flatten()
for i, col in enumerate(top6_v):
    ax = axes[i]
    ax.set_facecolor(AX_BG)
    cap_lo = df[col].quantile(0.01)
    cap_hi = df[col].quantile(0.99)
    for label, sub, color in [("Non-Fraud", legit_df, PALETTE["Non-Fraud"]),
                               ("Fraud",     fraud_df, PALETTE["Fraud"])]:
        data = sub[col].dropna().clip(cap_lo, cap_hi)
        ax.hist(data, bins=60, alpha=0.55, color=color, label=label,
                density=True, edgecolor="none")
    style_ax(ax, f"{col}  (r={pb_series[col]:.3f})", col, "Density")
    ax.legend(facecolor="#222", labelcolor=TEXT, fontsize=8, framealpha=0.7)
fig.suptitle("Top 6 V-Columns — Distribution: Fraud vs Non-Fraud",
             color=TEXT, fontsize=14, fontweight="bold")
save(fig, "eda_05b_top_vcols_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXTRA: mean difference table for top V-cols (for the interview summary)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Mean values for top 10 V-columns, by class:")
top10_v = pb_series.head(10).index.tolist()
mean_tbl = df.groupby("isFraud")[top10_v].mean().T
mean_tbl.columns = ["Non-Fraud mean", "Fraud mean"]
mean_tbl["diff"] = mean_tbl["Fraud mean"] - mean_tbl["Non-Fraud mean"]
mean_tbl["pb_r"] = pb_series[mean_tbl.index]
print(mean_tbl.to_string())

print("\n" + "=" * 60)
print("EDA script complete. All plots saved.")
print("=" * 60)
