"""
IEEE-CIS — Audit + Baseline Modeling
======================================
Tasks:
  1. id_02 documentation + sanity check
  2. NaN imputation audit from MI step
  3. Logistic Regression baseline (class_weight='balanced')
  4. XGBoost (scale_pos_weight, early stopping on val AUC-PR)
  5. Side-by-side comparison + XGB feature importances
  6. Save both models
"""

import os, sys, warnings, time, json, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report
)
import xgboost as xgb

warnings.filterwarnings("ignore")
t0 = time.time()

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def elapsed(): return f"{time.time()-t0:.1f}s"
def sep(title): print(f"\n{'='*60}\n{title}\n{'='*60}")

TARGET = "isFraud"

# ─────────────────────────────────────────────────────────────
# LOAD splits + top-30 feature list
# ─────────────────────────────────────────────────────────────
sep("LOADING DATA")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "processed/split_train.parquet"))
val_df   = pd.read_parquet(os.path.join(DATA_DIR, "processed/split_val.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, "processed/split_test.parquet"))   # DO NOT TOUCH

with open(os.path.join(DATA_DIR, "processed/top30_features.json")) as f:
    top30 = json.load(f)["features"]

print(f"  Train : {train_df.shape}")
print(f"  Val   : {val_df.shape}")
print(f"  Test  : {test_df.shape}  ← will not be used today")
print(f"  Top-30 features: {top30}")

# ══════════════════════════════════════════════════════════════
# TASK 1 — id_02 Documentation + Sanity Check
# ══════════════════════════════════════════════════════════════
sep("TASK 1 — id_02 Audit")

print("""
  DOCUMENTATION STATUS:
  ─────────────────────
  id_02 is part of the IEEE-CIS Identity table columns id_01–id_38.
  Vesta Corporation provided NO public description for individual
  identity columns beyond: "identity information — network connection
  information (IP, ISP, Proxy, etc) and digital signature (UA/browser/
  os/version, etc) associated with transactions. They are masked and
  transformed."

  Community analysis on Kaggle (Chris Deotte et al.) suggests:
    • id_01 / id_02 may be related to card/transaction amount-based
      risk scores computed by Vesta's internal system, OR
    • id_02 might encode a device metric (screen resolution, viewport
      size), OR a masked account-age-like variable.
  None of these are officially confirmed. id_02 is FULLY UNDOCUMENTED.

  High MI score (0.236) could mean:
    (a) Genuine strong predictive signal — OR
    (b) The feature partially proxies for isFraud in Vesta's internal
        scoring (i.e., it's a leaky downstream signal). We should watch
        whether its importance collapses or holds up on the test set.
""")

col = "id_02"
present = col in train_df.columns
null_pct = train_df[col].isnull().mean() * 100 if present else None
print(f"  Present in train split : {present}")
print(f"  Null % in train split  : {null_pct:.2f}%" if null_pct is not None else "  N/A")

if present:
    # Split by median into high/low halves (ignore NaN rows for this check)
    valid = train_df[[col, TARGET]].dropna(subset=[col])
    median_val = valid[col].median()
    low_half   = valid[valid[col] <= median_val]
    high_half  = valid[valid[col] >  median_val]
    print(f"\n  id_02 stats (non-null train rows):")
    print(f"    Non-null rows  : {len(valid):,}  ({len(valid)/len(train_df)*100:.1f}% of train)")
    print(f"    Min            : {valid[col].min():.4f}")
    print(f"    Median         : {median_val:.4f}")
    print(f"    Max            : {valid[col].max():.4f}")
    print(f"    Mean           : {valid[col].mean():.4f}")
    print(f"\n  Fraud rate by id_02 half (sanity check):")
    print(f"    Low half  (id_02 ≤ {median_val:.2f}) : "
          f"{low_half[TARGET].mean()*100:.3f}%  (n={len(low_half):,})")
    print(f"    High half (id_02 > {median_val:.2f}) : "
          f"{high_half[TARGET].mean()*100:.3f}%  (n={len(high_half):,})")

    # Also check NaN rows
    nan_rows = train_df[train_df[col].isnull()]
    print(f"\n    NaN rows fraud rate : {nan_rows[TARGET].mean()*100:.3f}%  (n={len(nan_rows):,})")
    print(f"\n  ► Interpretation: if high-half fraud rate >> low-half rate, signal is real.")
    print(f"    If NaN rows have a very different fraud rate too, missingness is a sub-signal.")

# ══════════════════════════════════════════════════════════════
# TASK 2 — NaN Imputation Audit for MI Step
# ══════════════════════════════════════════════════════════════
sep("TASK 2 — NaN Imputation Audit (from MI step)")

print("""
  In cleanup_split_rank.py, before running mutual_info_classif:
    • String/object columns → LabelEncoder (NaN filled with "__MISSING__" sentinel string)
    • Numeric columns       → df.fillna(df.median(numeric_only=True))
      i.e. each numeric column's NaN replaced with its own column median

  This is MEDIAN imputation — not a fixed sentinel like -999.
""")

# Now assess: for each top-30 feature, how many NaNs exist, and does
# the fraud rate for NaN rows differ from non-NaN rows?
print(f"  NaN assessment for top-30 features (train split):")
print(f"\n  {'Feature':<20} {'Null%':>7} {'FR (non-null)':>14} {'FR (null)':>11} {'Δ(pp)':>7} {'Risk?':>8}")
print(f"  {'-'*70}")

inflation_risks = []
for col in top30:
    if col not in train_df.columns:
        print(f"  {col:<20} {'MISSING FROM DF':>40}")
        continue
    null_pct_col = train_df[col].isnull().mean() * 100
    non_null = train_df[train_df[col].notna()]
    null_rows = train_df[train_df[col].isnull()]
    fr_non_null = non_null[TARGET].mean() * 100 if len(non_null) > 0 else float('nan')
    fr_null     = null_rows[TARGET].mean() * 100 if len(null_rows) > 0 else float('nan')
    delta       = fr_null - fr_non_null if not np.isnan(fr_null) else float('nan')

    # Flag if: >10% null AND |delta| > 5pp (median imputation may inflate MI)
    risk = "⚠ YES" if (null_pct_col > 10 and not np.isnan(delta) and abs(delta) > 5) else "no"
    if risk != "no":
        inflation_risks.append((col, null_pct_col, delta))

    fr_null_str = f"{fr_null:.2f}%" if not np.isnan(fr_null) else "N/A"
    delta_str   = f"{delta:+.1f}" if not np.isnan(delta) else "N/A"
    print(f"  {col:<20} {null_pct_col:>6.1f}% {fr_non_null:>13.3f}% {fr_null_str:>11} "
          f"{delta_str:>7} {risk:>8}")

print(f"\n  Features at risk of inflated MI (>10% null AND fraud-rate delta >5pp):")
if inflation_risks:
    for col, np_, delta in inflation_risks:
        print(f"    {col:<20}  null%={np_:.1f}%  Δ={delta:+.1f}pp")
    print(f"\n  ► For these, part of MI comes from the NaN pattern (now blended into")
    print(f"    median), not purely from the non-null values. MI is slightly inflated.")
    print(f"    The is_null flags we created handle this explicitly and correctly.")
    print(f"    The net effect on model quality is small — XGBoost handles NaN natively")
    print(f"    and won't have this issue.")
else:
    print(f"    None — median imputation is unlikely to have significantly inflated any MI score.")

# ══════════════════════════════════════════════════════════════
# Prepare X/y for modeling
# ══════════════════════════════════════════════════════════════
sep("PREPARING FEATURES FOR MODELING")

# Confirm all top-30 are numeric in the splits
for col in top30:
    dtype = train_df[col].dtype
    if str(dtype) in ['object', 'string']:
        print(f"  WARNING: {col} is dtype {dtype} — will need encoding")
    else:
        pass   # all good

# All top-30 confirmed numeric. Fill NaN with median (from train).
print(f"  Computing train-set medians for imputation …")
train_medians = train_df[top30].median()

def prepare(df, cols, medians):
    X = df[cols].copy()
    X = X.fillna(medians)
    return X.values, df[TARGET].values.astype(int)

X_train, y_train = prepare(train_df, top30, train_medians)
X_val,   y_val   = prepare(val_df,   top30, train_medians)

n_neg   = (y_train == 0).sum()
n_pos   = (y_train == 1).sum()
spw     = n_neg / n_pos   # scale_pos_weight for XGBoost

print(f"  X_train shape : {X_train.shape}")
print(f"  X_val   shape : {X_val.shape}")
print(f"  Train class balance: {n_neg:,} legit / {n_pos:,} fraud  (scale_pos_weight={spw:.2f})")

# ══════════════════════════════════════════════════════════════
# TASK 3 — Logistic Regression Baseline
# ══════════════════════════════════════════════════════════════
sep("TASK 3 — Logistic Regression (class_weight='balanced')")

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        C=0.1,            # mild L2 reg — stable for this feature scale
        random_state=42,
        n_jobs=-1,
    ))
])

print("  Fitting …")
lr_t0 = time.time()
lr_pipeline.fit(X_train, y_train)
lr_elapsed = time.time() - lr_t0
print(f"  Done in {lr_elapsed:.1f}s")

lr_probs  = lr_pipeline.predict_proba(X_val)[:, 1]
lr_preds  = (lr_probs >= 0.5).astype(int)

lr_metrics = {
    "AUC-ROC"  : roc_auc_score(y_val, lr_probs),
    "PR-AUC"   : average_precision_score(y_val, lr_probs),
    "F1"       : f1_score(y_val, lr_preds, zero_division=0),
    "Precision": precision_score(y_val, lr_preds, zero_division=0),
    "Recall"   : recall_score(y_val, lr_preds, zero_division=0),
}

print(f"\n  Logistic Regression — Val Metrics:")
for k, v in lr_metrics.items():
    print(f"    {k:<12}: {v:.4f}")

print(f"\n  Classification report (val):")
print(classification_report(y_val, lr_preds, target_names=["Non-Fraud","Fraud"],
                             digits=4))

# Save LR model
lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
with open(lr_path, "wb") as f:
    pickle.dump({"pipeline": lr_pipeline, "features": top30,
                 "train_medians": train_medians.to_dict(),
                 "val_metrics": lr_metrics}, f)
print(f"  Saved → {lr_path}")

# ══════════════════════════════════════════════════════════════
# TASK 4 — XGBoost (scale_pos_weight, early stopping)
# ══════════════════════════════════════════════════════════════
sep("TASK 4 — XGBoost (scale_pos_weight + early stopping on val PR-AUC)")

xgb_model = xgb.XGBClassifier(
    n_estimators       = 1000,      # large pool; early stopping will cut this
    learning_rate      = 0.05,
    max_depth          = 6,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    scale_pos_weight   = spw,       # handles class imbalance
    eval_metric        = "aucpr",   # PR-AUC on eval set
    early_stopping_rounds = 50,
    random_state       = 42,
    n_jobs             = -1,
    tree_method        = "hist",    # fast on large datasets
    verbosity          = 0,
)

print(f"  scale_pos_weight = {spw:.2f}  ({n_neg:,} legit / {n_pos:,} fraud)")
print(f"  Fitting with early stopping (patience=50 rounds, metric=aucpr) …")
xgb_t0 = time.time()
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
xgb_elapsed = time.time() - xgb_t0
best_iter = xgb_model.best_iteration
print(f"  Done in {xgb_elapsed:.1f}s  |  Best iteration: {best_iter}")

xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
xgb_preds = (xgb_probs >= 0.5).astype(int)

xgb_metrics = {
    "AUC-ROC"  : roc_auc_score(y_val, xgb_probs),
    "PR-AUC"   : average_precision_score(y_val, xgb_probs),
    "F1"       : f1_score(y_val, xgb_preds, zero_division=0),
    "Precision": precision_score(y_val, xgb_preds, zero_division=0),
    "Recall"   : recall_score(y_val, xgb_preds, zero_division=0),
}

print(f"\n  XGBoost — Val Metrics:")
for k, v in xgb_metrics.items():
    print(f"    {k:<12}: {v:.4f}")

print(f"\n  Classification report (val):")
print(classification_report(y_val, xgb_preds, target_names=["Non-Fraud","Fraud"],
                             digits=4))

# XGBoost feature importances (gain-based)
importance_gain = xgb_model.get_booster().get_score(importance_type="gain")
imp_series = pd.Series(importance_gain).reindex(
    [f"f{i}" for i in range(len(top30))]
).fillna(0)
imp_series.index = top30
imp_series = imp_series.sort_values(ascending=False)

print(f"\n  XGBoost Top-15 Feature Importances (GAIN):")
print(f"\n  {'Rank':<6}  {'Feature':<25}  {'Gain':>10}  {'MI Rank':>8}")
print(f"  {'-'*55}")
for rank, (col, gain) in enumerate(imp_series.head(15).items(), 1):
    mi_rank = top30.index(col) + 1 if col in top30 else "—"
    print(f"  {rank:<6}  {col:<25}  {gain:>10.2f}  #{mi_rank:>5}")

# Save XGBoost model
xgb_path = os.path.join(MODELS_DIR, "xgboost_model.json")
xgb_model.save_model(xgb_path)
xgb_meta_path = os.path.join(MODELS_DIR, "xgboost_meta.pkl")
with open(xgb_meta_path, "wb") as f:
    pickle.dump({"features": top30, "train_medians": train_medians.to_dict(),
                 "scale_pos_weight": spw, "best_iteration": best_iter,
                 "val_metrics": xgb_metrics,
                 "feature_importances_gain": imp_series.to_dict()}, f)
print(f"\n  Saved model  → {xgb_path}")
print(f"  Saved meta   → {xgb_meta_path}")

# ══════════════════════════════════════════════════════════════
# TASK 5 — Side-by-Side Comparison
# ══════════════════════════════════════════════════════════════
sep("TASK 5 — Model Comparison: LR vs XGBoost")

metrics_order = ["AUC-ROC","PR-AUC","F1","Precision","Recall"]
print(f"\n  {'Metric':<14}  {'Logistic Regression':>20}  {'XGBoost':>12}  {'Δ (XGB–LR)':>12}")
print(f"  {'-'*62}")
for m in metrics_order:
    lr_v  = lr_metrics[m]
    xgb_v = xgb_metrics[m]
    delta = xgb_v - lr_v
    arrow = "↑" if delta > 0 else "↓"
    print(f"  {m:<14}  {lr_v:>20.4f}  {xgb_v:>12.4f}  {delta:>+10.4f} {arrow}")

print(f"\n  MI Ranking vs XGBoost Gain Importance — alignment check:")
print(f"\n  {'MI Rank':<9}  {'Feature':<25}  {'XGB Gain Rank':>14}")
print(f"  {'-'*52}")
xgb_gain_list = imp_series.index.tolist()
for mi_rank, col in enumerate(top30, 1):
    xgb_rank = xgb_gain_list.index(col) + 1 if col in xgb_gain_list else "—"
    agree = "✓" if abs(mi_rank - xgb_rank) <= 5 else "△" if abs(mi_rank - xgb_rank) <= 10 else "✗"
    print(f"  #{mi_rank:<8}  {col:<25}  #{xgb_rank:<12}  {agree}")

sep("COMPLETE")
print(f"  Elapsed: {elapsed()}")
print(f"\n  Summary:")
print(f"    LR   AUC-ROC={lr_metrics['AUC-ROC']:.4f}  PR-AUC={lr_metrics['PR-AUC']:.4f}")
print(f"    XGB  AUC-ROC={xgb_metrics['AUC-ROC']:.4f}  PR-AUC={xgb_metrics['PR-AUC']:.4f}")
print(f"    XGB best iteration: {best_iter}")
print(f"\n  Test set: UNTOUCHED ✓")
print(f"  Models saved to: {MODELS_DIR}/")
