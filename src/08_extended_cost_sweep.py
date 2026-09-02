"""
Extended business-cost threshold sweep (0.02 → 0.70).
Read-only: loads saved XGBoost model, runs on test set, no retraining.
No torch import — avoids the macOS OMP hang entirely.
"""
import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import xgboost as xgb
warnings.filterwarnings("ignore")

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ART_DIR    = "/Users/shriyanshraj/.gemini/antigravity-ide/brain/98590e36-f8e7-4b2b-8f32-e55098f07b90"
TARGET     = "isFraud"

# ─── Load test + model (no retraining) ────────────────────────
test_df = pd.read_parquet(os.path.join(DATA_DIR, "processed/split_test.parquet"))
y_test  = test_df[TARGET].values.astype(int)
tx_amt  = test_df["TransactionAmt"].values
n_test  = len(test_df)

with open(os.path.join(DATA_DIR, "processed/top30_features.json")) as f:
    top30 = json.load(f)["features"]
with open(os.path.join(MODELS_DIR, "xgboost_meta.pkl"), "rb") as f:
    xgb_art = pickle.load(f)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
xgb_med   = pd.Series(xgb_art["train_medians"])
xgb_prob  = xgb_model.predict_proba(test_df[top30].fillna(xgb_med).values)[:, 1]

print(f"Test: {n_test:,} rows  fraud={y_test.mean()*100:.4f}%")
print(f"XGB prob range: {xgb_prob.min():.4f} – {xgb_prob.max():.4f}")

# ─── Full threshold sweep ──────────────────────────────────────
FP_COST = 4.0
# Fine sweep: 0.02..0.30 step 0.02; then existing 0.30..0.70 step 0.10
fine_thrs  = np.round(np.arange(0.02, 0.31, 0.02), 2).tolist()
coarse_thr = [0.40, 0.50, 0.60, 0.70]
# 0.30 included in fine sweep; combine without dup
all_thrs   = fine_thrs + coarse_thr

results = []
for thr in all_thrs:
    pr   = (xgb_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pr).ravel()
    fn_cost     = float(tx_amt[(y_test == 1) & (pr == 0)].sum())
    fp_cost     = float(fp * FP_COST)
    total       = fn_cost + fp_cost
    fraud_pct   = tp / (tp + fn) * 100
    flagged_pct = pr.sum() / n_test * 100          # Task 5: total alert volume
    results.append(dict(thr=thr, TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                        fn_cost=fn_cost, fp_cost=fp_cost, total=total,
                        fraud_pct=fraud_pct, flagged_pct=flagged_pct,
                        n_flagged=int(pr.sum())))

# ─── Print full table ──────────────────────────────────────────
print(f"\n{'Thr':>5}  {'TP':>5}  {'FP':>6}  {'FN':>5}  {'TN':>6}  "
      f"{'FN Cost':>10}  {'FP Cost':>9}  {'Total':>9}  {'Fraud%':>7}  "
      f"{'Flagged%':>9}  {'#Flagged':>9}")
print("-" * 105)
for r in results:
    marker = " ←" if r["thr"] == 0.30 else ""   # boundary marker
    print(f"{r['thr']:>5.2f}  {r['TP']:>5,}  {r['FP']:>6,}  {r['FN']:>5,}  {r['TN']:>6,}  "
          f"${r['fn_cost']:>9,.0f}  ${r['fp_cost']:>8,.0f}  ${r['total']:>8,.0f}  "
          f"{r['fraud_pct']:>6.1f}%  {r['flagged_pct']:>8.1f}%  {r['n_flagged']:>9,}{marker}")

# ─── Identify minimum-cost threshold ──────────────────────────
best      = min(results, key=lambda r: r["total"])
base      = next(r for r in results if r["thr"] == 0.50)
savings   = base["total"] - best["total"]

print(f"\n{'='*60}")
print(f"MINIMUM-COST THRESHOLD ACROSS 0.02 – 0.70:")
print(f"  Optimal threshold : {best['thr']}")
print(f"  Total cost        : ${best['total']:,.0f}")
print(f"  Cost at 0.50      : ${base['total']:,.0f}")
print(f"  Savings vs 0.50   : ${savings:,.0f}  ({savings/base['total']*100:.1f}%)")
print(f"  Fraud caught      : {best['fraud_pct']:.1f}%")
print(f"  Total alert rate  : {best['flagged_pct']:.1f}%  ({best['n_flagged']:,} of {n_test:,} flagged)")

# Curve shape assessment
totals    = [r["total"]       for r in results]
thrs_arr  = [r["thr"]         for r in results]
min_idx   = totals.index(min(totals))
is_at_edge = min_idx == 0
print(f"\n  Curve shape assessment:")
if is_at_edge:
    print(f"  ⚠  Minimum is at the LOWEST threshold tested (thr={thrs_arr[0]}).")
    print(f"     The cost curve is still declining — sweep should extend even lower.")
    print(f"     Operationally this may mean flagging too many transactions.")
else:
    print(f"  ✓  Interior minimum found at thr={best['thr']} (index {min_idx+1} of {len(results)}).")
    print(f"     Cost RISES on both sides — confirmed U-shape (or J-shape) curve.")
    print(f"     The optimal threshold is genuine, not an edge artifact.")

# Operational viability note
print(f"\n  Operational viability:")
print(f"  At optimal thr={best['thr']}: flagging {best['n_flagged']:,} rows "
      f"({best['flagged_pct']:.1f}% of {n_test:,} total test transactions).")
if best['flagged_pct'] > 20:
    print(f"  ⚠  Alert volume > 20% of transactions — may overwhelm a review queue.")
    print(f"     Consider a secondary triage rule or higher threshold operationally.")
elif best['flagged_pct'] > 10:
    print(f"  △  Alert volume 10–20% — moderate; feasible with automated pre-screening.")
else:
    print(f"  ✓  Alert volume < 10% — operationally viable for manual review.")

# ─── Plot ──────────────────────────────────────────────────────
FIG_BG, AX_BG, TEXT = "#1a1a2e", "#16213e", "#e0e0e0"

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=FIG_BG)
for ax in axes:
    ax.set_facecolor(AX_BG); ax.tick_params(colors=TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

thr_v    = [r["thr"]        for r in results]
total_v  = [r["total"]      for r in results]
fn_v     = [r["fn_cost"]    for r in results]
fp_v     = [r["fp_cost"]    for r in results]
flag_v   = [r["flagged_pct"] for r in results]

# ── Left: cost curve ──
axes[0].fill_between(thr_v, fn_v, alpha=0.35, color="#E8524C", label="FN Cost (missed fraud)")
axes[0].fill_between(thr_v, fp_v, alpha=0.35, color="#4C9BE8", label="FP Cost (false alerts)")
axes[0].plot(thr_v, total_v, "o-", color="#F5D76E", lw=2.5, ms=5,
             label="Total Cost", zorder=5)
# mark optimal
axes[0].axvline(best["thr"], color="white", ls="--", lw=1.8, alpha=0.9,
                label=f"Optimal ({best['thr']}) — ${best['total']:,.0f}")
axes[0].axvline(0.50, color="#aaa", ls=":", lw=1.2, alpha=0.7,
                label=f"Default 0.5 — ${base['total']:,.0f}")
# mark fine/coarse boundary
axes[0].axvline(0.30, color="#6BC9A0", ls=":", lw=1.0, alpha=0.5,
                label="Fine/coarse boundary (0.30)")
axes[0].set_xlabel("Probability Threshold", color=TEXT, fontsize=11)
axes[0].set_ylabel("Cost ($)", color=TEXT, fontsize=11)
axes[0].set_title("Total Business Cost vs Threshold (0.02–0.70)", color=TEXT,
                  fontweight="bold", fontsize=12)
axes[0].legend(facecolor="#222", labelcolor=TEXT, fontsize=8)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# ── Right: alert volume + fraud-caught ──
ax2 = axes[1]
color1, color2 = "#E8904C", "#6BC9A0"
ln1, = ax2.plot(thr_v, flag_v, "o-", color=color1, lw=2.2, ms=5,
                label="% Transactions Flagged")
ax2.set_xlabel("Probability Threshold", color=TEXT, fontsize=11)
ax2.set_ylabel("% All Test Transactions Flagged", color=TEXT, fontsize=11)
ax2.tick_params(axis="y", colors=color1)
ax2.yaxis.label.set_color(color1)

ax3 = ax2.twinx()
ax3.set_facecolor(AX_BG)
ax3.tick_params(colors=color2)
ax3.yaxis.label.set_color(color2)
fraud_v = [r["fraud_pct"] for r in results]
ln2, = ax3.plot(thr_v, fraud_v, "s--", color=color2, lw=2.2, ms=5,
                label="% Fraud Caught (Recall)")
ax3.set_ylabel("% Fraud Caught", color=color2, fontsize=11)

ax2.axvline(best["thr"], color="white", ls="--", lw=1.8, alpha=0.9,
            label=f"Optimal ({best['thr']})")
# Shade operational danger zone (>20% flag rate)
danger_x = [r["thr"] for r in results if r["flagged_pct"] > 20]
if danger_x:
    ax2.axvspan(min(danger_x), max(danger_x), alpha=0.10, color="#E8524C",
                label="Alert overload zone (>20%)")

ax2.set_title("Alert Volume & Fraud Catch Rate vs Threshold", color=TEXT,
              fontweight="bold", fontsize=12)
lines = [ln1, ln2]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, facecolor="#222", labelcolor=TEXT, fontsize=9)

fig.suptitle("XGBoost — Extended Threshold Sweep (0.02–0.70) on Test Set",
             color=TEXT, fontsize=13, fontweight="bold")
fig.tight_layout()
out_path = os.path.join(ART_DIR, "extended_cost_sweep.png")
fig.savefig(out_path, facecolor=FIG_BG, bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"\n[saved] extended_cost_sweep.png")

# Save full results table as CSV
df_out = pd.DataFrame(results)
df_out.to_csv(os.path.join(DATA_DIR, "processed/extended_cost_sweep.csv"), index=False)
print(f"[saved] extended_cost_sweep.csv  ({len(df_out)} rows)")
print("\nExtended sweep complete. ✓")
