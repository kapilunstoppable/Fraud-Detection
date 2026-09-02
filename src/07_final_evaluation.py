"""Final evaluation — isolated single process, no DataLoader workers."""
import os, json, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ART_DIR    = "/Users/shriyanshraj/.gemini/antigravity-ide/brain/98590e36-f8e7-4b2b-8f32-e55098f07b90"
TARGET     = "isFraud"

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score, confusion_matrix)
import xgboost as xgb
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Test data ──────────────────────────────────────────────────
test_df = pd.read_parquet(os.path.join(DATA_DIR, "processed/split_test.parquet"))
y_test  = test_df[TARGET].values.astype(int)
with open(os.path.join(DATA_DIR, "processed/top30_features.json")) as f:
    top30 = json.load(f)["features"]
print(f"Test: {test_df.shape}  fraud_rate={y_test.mean()*100:.4f}%")

def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {"AUC-ROC": roc_auc_score(y, prob),
            "PR-AUC":  average_precision_score(y, prob),
            "F1":      f1_score(y, pred, zero_division=0),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall":  recall_score(y, pred, zero_division=0)}

# ── Logistic Regression ────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "rb") as f:
    lr_art = pickle.load(f)
lr_med  = pd.Series(lr_art["train_medians"])
lr_prob = lr_art["pipeline"].predict_proba(test_df[top30].fillna(lr_med).values)[:, 1]
lr_t    = metrics(y_test, lr_prob)
lr_v    = lr_art["val_metrics"]
print(f"LR  test: {lr_t}")

# ── XGBoost ────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
with open(os.path.join(MODELS_DIR, "xgboost_meta.pkl"), "rb") as f:
    xgb_art = pickle.load(f)
xgb_med  = pd.Series(xgb_art["train_medians"])
xgb_prob = xgb_model.predict_proba(test_df[top30].fillna(xgb_med).values)[:, 1]
xgb_t    = metrics(y_test, xgb_prob)
xgb_v    = xgb_art["val_metrics"]
print(f"XGB test: {xgb_t}")

# ── Autoencoder ────────────────────────────────────────────────
class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Linear(32,16),nn.BatchNorm1d(16),nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(16,32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Linear(32,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,d))
    def forward(self, x): return self.decoder(self.encoder(x))

ck  = torch.load(os.path.join(MODELS_DIR, "autoencoder_state.pt"),
                 map_location="cpu", weights_only=False)
ae  = AE(ck["input_dim"]); ae.load_state_dict(ck["state_dict"]); ae.eval()
with open(os.path.join(MODELS_DIR, "ae_scaler_and_medians.pkl"), "rb") as f:
    ae_sc = pickle.load(f)
ae_med = pd.Series(ae_sc["train_medians"])
Xae    = ae_sc["scaler"].transform(test_df[ck["features"]].fillna(ae_med).values)
with torch.no_grad():
    xt     = torch.FloatTensor(Xae)
    ae_err = ((xt - ae(xt)) ** 2).mean(dim=1).numpy()
thr99 = ck["threshold_p99"]
ae_t  = metrics(y_test, ae_err, thr=thr99)
ae_v  = ck["val_metrics"]
print(f"AE  test (thr={thr99:.4f}): {ae_t}")

# ── Val vs Test comparison table ───────────────────────────────
M = ["AUC-ROC","PR-AUC","F1","Precision","Recall"]
print("\n── VALIDATION SET (fraud 9.671%) ──")
print(f"{'Metric':<12}  {'LR':>8}  {'XGB':>8}  {'AE':>8}")
for m in M:
    print(f"{m:<12}  {lr_v[m]:>8.4f}  {xgb_v[m]:>8.4f}  {ae_v[m]:>8.4f}")

print("\n── TEST SET (fraud 9.059%) ──")
print(f"{'Metric':<12}  {'LR':>8}  {'XGB':>8}  {'AE':>8}  {'Best':>6}")
for m in M:
    vals = {"LR": lr_t[m], "XGB": xgb_t[m], "AE": ae_t[m]}
    best = max(vals, key=vals.get)
    print(f"{m:<12}  {lr_t[m]:>8.4f}  {xgb_t[m]:>8.4f}  {ae_t[m]:>8.4f}  {best:>6}")

print("\n── SHIFT (Test − Val) ──")
print(f"{'Metric':<12}  {'LR':>8}  {'XGB':>8}  {'AE':>8}")
for m in M:
    print(f"{m:<12}  {lr_t[m]-lr_v[m]:>+8.4f}  {xgb_t[m]-xgb_v[m]:>+8.4f}  {ae_t[m]-ae_v[m]:>+8.4f}")

# ── Business cost analysis ─────────────────────────────────────
print("\n── BUSINESS COST ANALYSIS ──")
print("FP cost: $3 review + 2%×$50 churn = $4.00 per FP")
print("FN cost: full TransactionAmt of missed fraud")
FP_COST = 4.0
tx_amt  = test_df["TransactionAmt"].values
thresholds_biz = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []
print(f"\n{'Thr':>4}  {'TP':>5}  {'FP':>6}  {'FN':>5}  {'TN':>6}  {'FN_Cost':>11}  {'FP_Cost':>10}  {'Total':>10}  {'Fraud%':>7}")
print("-"*80)
for thr in thresholds_biz:
    pr  = (xgb_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pr).ravel()
    fn_cost = float(tx_amt[(y_test==1) & (pr==0)].sum())
    fp_cost = float(fp * FP_COST)
    total   = fn_cost + fp_cost
    frac    = tp / (tp + fn) * 100
    results.append(dict(thr=thr, TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                        fn_cost=fn_cost, fp_cost=fp_cost, total=total, frac=frac))
    print(f"{thr:>4.1f}  {tp:>5,}  {fp:>6,}  {fn:>5,}  {tn:>6,}  "
          f"${fn_cost:>10,.0f}  ${fp_cost:>9,.0f}  ${total:>9,.0f}  {frac:>6.1f}%")

best = min(results, key=lambda r: r["total"])
base = next(r for r in results if r["thr"] == 0.5)
savings = base["total"] - best["total"]
print(f"\nOptimal thr={best['thr']}  total=${best['total']:,.0f}  "
      f"savings vs 0.5=${savings:,.0f} ({savings/base['total']*100:.1f}%)")

# ── Plots ──────────────────────────────────────────────────────
FIG_BG, AX_BG, TEXT = "#1a1a2e", "#16213e", "#e0e0e0"

# Cost analysis plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=FIG_BG)
for ax in axes:
    ax.set_facecolor(AX_BG); ax.tick_params(colors=TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

thr_v  = [r["thr"]     for r in results]
tot_v  = [r["total"]   for r in results]
fn_v   = [r["fn_cost"] for r in results]
fp_v   = [r["fp_cost"] for r in results]
rec_v  = [r["frac"]    for r in results]
x      = np.array(thr_v); w = 0.06

axes[0].bar(x-w/2, fn_v, w, label="FN Cost (missed fraud)", color="#E8524C", edgecolor="#333")
axes[0].bar(x+w/2, fp_v, w, label="FP Cost (false alerts)", color="#4C9BE8", edgecolor="#333")
axes[0].plot(x, tot_v, "o-", color="#F5D76E", lw=2.5, ms=8, label="Total Cost", zorder=5)
axes[0].axvline(best["thr"], color="white", ls="--", lw=1.5, alpha=0.8,
                label=f"Optimal ({best['thr']})")
axes[0].set_xlabel("Threshold", color=TEXT); axes[0].set_ylabel("Cost ($)", color=TEXT)
axes[0].set_title("Business Cost Breakdown", color=TEXT, fontweight="bold")
axes[0].legend(facecolor="#222", labelcolor=TEXT, fontsize=9)

axes[1].plot(thr_v, rec_v, "o-", color="#6BC9A0", lw=2.5, ms=8)
axes[1].axvline(best["thr"], color="white", ls="--", lw=1.5, alpha=0.8,
                label=f"Optimal ({best['thr']})")
axes[1].axvline(0.5, color="#F5D76E", ls=":", lw=1.2, alpha=0.7, label="Default (0.5)")
axes[1].set_xlabel("Threshold", color=TEXT); axes[1].set_ylabel("% Fraud Caught", color=TEXT)
axes[1].set_title("Fraud Catch Rate", color=TEXT, fontweight="bold")
axes[1].legend(facecolor="#222", labelcolor=TEXT, fontsize=9)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
fig.suptitle("XGBoost — Business Cost Analysis on Test Set", color=TEXT, fontsize=13, fontweight="bold")
fig.savefig(os.path.join(ART_DIR, "final_cost_analysis.png"), facecolor=FIG_BG, bbox_inches="tight", dpi=130)
plt.close(fig)
print("\n[saved] final_cost_analysis.png")

# Val vs Test grouped bar chart
fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=FIG_BG)
ax.set_facecolor(AX_BG); ax.tick_params(colors=TEXT)
for sp in ax.spines.values(): sp.set_edgecolor("#444")
mlabels = ["LR", "XGBoost", "AE"]
mcols   = ["#4C9BE8", "#E8524C", "#6BC9A0"]
t_all   = [lr_t, xgb_t, ae_t]
v_all   = [lr_v,  xgb_v,  ae_v]
xp = np.arange(len(M)); w2 = 0.12
for i,(lab,tm,vm,col) in enumerate(zip(mlabels,t_all,v_all,mcols)):
    off = (i-1)*w2*2.3
    ax.bar(xp+off,       [tm[m] for m in M], w2, color=col, alpha=0.9,
           label=f"{lab} (test)", edgecolor="#333", lw=0.8)
    ax.bar(xp+off+w2,    [vm[m] for m in M], w2, color=col, alpha=0.4,
           label=f"{lab} (val)", edgecolor="#333", lw=0.8, hatch="///")
ax.set_xticks(xp+w2/2); ax.set_xticklabels(M, color=TEXT)
ax.set_ylabel("Score", color=TEXT)
ax.set_title("All Models — Val (hatched) vs Test (solid)", color=TEXT, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.legend(facecolor="#222", labelcolor=TEXT, fontsize=8, ncol=3,
          bbox_to_anchor=(0.5,-0.18), loc="upper center")
fig.tight_layout()
fig.savefig(os.path.join(ART_DIR, "final_model_comparison.png"), facecolor=FIG_BG, bbox_inches="tight", dpi=130)
plt.close(fig)
print("[saved] final_model_comparison.png")

# Save JSON for report
out = dict(lr_test=lr_t, xgb_test=xgb_t, ae_test=ae_t,
           lr_val=lr_v,  xgb_val=xgb_v,  ae_val=ae_v,
           cost=results, best=best, base=base, savings=savings)
with open(os.path.join(DATA_DIR, "processed/final_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("[saved] final_results.json")
print("Done ✓")
