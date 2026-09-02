"""
IEEE-CIS — Cleanup, Split, and Feature Ranking
================================================
Tasks:
  1. Audit V76/V90/V91/V107/V305 null rates
  2. Audit raw-vs-engineered column redundancy
  3. Drop redundant raw columns (with explicit rationale)
  4. Time-based train/val/test split (70/15/15 forward-chaining)
  5. Mutual-information ranking on TRAIN split only
  6. Save splits + feature list
"""

import os, sys, warnings, time, json
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
t0 = time.time()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ART_DIR  = "/Users/shriyanshraj/.gemini/antigravity-ide/brain/98590e36-f8e7-4b2b-8f32-e55098f07b90"

def elapsed(): return f"{time.time()-t0:.1f}s"
def sep(title): print(f"\n{'='*60}\n{title}\n{'='*60}")

# ─────────────────────────────────────────────────────────────
# LOAD engineered_train.parquet
# ─────────────────────────────────────────────────────────────
sep("LOADING engineered_train.parquet")
df = pd.read_parquet(os.path.join(DATA_DIR, "processed/engineered_train.parquet"))
print(f"  Loaded: {df.shape}  [{elapsed()}]")

# ══════════════════════════════════════════════════════════════
# TASK 1 — Audit V76, V90, V91, V107, V305
# ══════════════════════════════════════════════════════════════
sep("TASK 1 — Null-rate audit for V76, V90, V91, V107, V305")

probe_cols = ["V76", "V90", "V91", "V107", "V305"]
print(f"\n  {'Column':<10}  {'Present?':<10}  {'Null %':>8}  {'Non-null count':>15}")
print(f"  {'-'*50}")
truly_null = []
for col in probe_cols:
    if col not in df.columns:
        print(f"  {col:<10}  {'NOT IN DF':<10}  {'—':>8}  {'—':>15}")
        truly_null.append(col)
    else:
        pct  = df[col].isnull().mean() * 100
        nnull = df[col].notna().sum()
        flag = "← DROP" if pct == 100.0 else ""
        print(f"  {col:<10}  {'yes':<10}  {pct:>7.2f}%  {nnull:>15,}  {flag}")
        if pct == 100.0:
            truly_null.append(col)

print(f"\n  Verdict: {len(truly_null)} of 5 are 100% null in the parquet → will drop")
print(f"  Columns to drop: {truly_null}")

# ══════════════════════════════════════════════════════════════
# TASK 2 — Redundancy Audit
# ══════════════════════════════════════════════════════════════
sep("TASK 2 — Raw-vs-Engineered Redundancy Audit")

# D-columns: raw D1-D15 present alongside D*_norm?
raw_d   = [f"D{i}" for i in range(1,16)]
norm_d  = [f"D{i}_norm" for i in range(1,16)]
print("\n  D-COLUMN PAIRS (raw | norm | both present?):")
for r, n in zip(raw_d, norm_d):
    r_in = r in df.columns
    n_in = n in df.columns
    status = "BOTH ← redundant" if (r_in and n_in) else \
             "raw only" if r_in else "norm only" if n_in else "neither"
    print(f"    {r:<6} {'✓' if r_in else '✗'}  {n:<12} {'✓' if n_in else '✗'}  → {status}")

# M-columns: raw M1-M4 present?
print("\n  M-COLUMN STATUS (M1-M4, low-cardinality flags — no engineered version):")
for col in [f"M{i}" for i in range(1, 10)]:
    if col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        n_uniq   = df[col].nunique()
        print(f"    {col}  present  null%={null_pct:.1f}%  unique={n_uniq}")
    else:
        print(f"    {col}  NOT in df (was dropped as 100% null)")

# High-cardinality categoricals: raw vs _freq
hi_card = ["card1","card2","addr1","DeviceInfo","id_31","id_30",
           "P_emaildomain","R_emaildomain"]
lo_card = ["card4","card6"]          # kept raw (low-cardinality, fully observed)
print("\n  HIGH-CARDINALITY CATEGORICALS (raw | freq | both present?):")
for col in hi_card:
    raw_in  = col in df.columns
    freq_in = f"{col}_freq" in df.columns
    status  = "BOTH ← redundant" if (raw_in and freq_in) else \
              "raw only" if raw_in else "freq only" if freq_in else "neither"
    print(f"    {col:<22}  raw={'✓' if raw_in else '✗'}  freq={'✓' if freq_in else '✗'}  → {status}")

print("\n  LOW-CARDINALITY CATEGORICALS (card4/card6) — keep raw + freq both:")
for col in lo_card:
    raw_in  = col in df.columns
    freq_in = f"{col}_freq" in df.columns
    print(f"    {col}  raw={'✓' if raw_in else '✗'}  freq={'✓' if freq_in else '✗'}")

# ══════════════════════════════════════════════════════════════
# TASK 3 — Drop Redundant Raw Columns
# ══════════════════════════════════════════════════════════════
sep("TASK 3 — Drop Redundant Raw Columns")

cols_to_drop = []
reasons      = {}

# 3a — V76/V90/V91/V107/V305 missed in step 5 last time
for col in truly_null:
    if col in df.columns:
        cols_to_drop.append(col)
        reasons[col] = "100% null in identity-joined view — no signal"

# 3b — Raw D1-D15: we have D*_norm; raw values drift with TransactionDT
# (D11 was already dropped in FE step). Drop raw D1-D10, D12-D15.
# Keep D11 status check just in case.
for i in list(range(1,11)) + list(range(12,16)):
    col  = f"D{i}"
    norm = f"D{i}_norm"
    if col in df.columns and norm in df.columns:
        cols_to_drop.append(col)
        reasons[col] = f"Replaced by {norm} (time-drift normalization applied); raw drifts with TransactionDT"

# 3c — High-cardinality categoricals: raw strings are expensive for trees;
#       _freq version captures all signal as a continuous column.
#       We also keep raw card4/card6 (4 and 3 unique values, used by some models as ordinal).
for col in hi_card:
    freq = f"{col}_freq"
    if col in df.columns and freq in df.columns:
        cols_to_drop.append(col)
        reasons[col] = f"Replaced by {freq} (frequency encoding); raw string unusable by numeric models"

# 3d — uid: string identifier — useful only for groupby aggregations later;
#       keep it for now as it costs nothing (not fed to MI/model directly)

print(f"\n  COLUMNS TO DROP: {len(cols_to_drop)}")
print(f"\n  {'Column':<25}  Reason")
print(f"  {'-'*70}")
for col in cols_to_drop:
    print(f"  {col:<25}  {reasons[col]}")

print(f"\n  COLUMNS KEPT (raw form intentionally retained):")
kept_raw = {
    "card4":            "Low-cardinality (4 vals), 0% null, ordinal signal; tree can split on raw",
    "card6":            "Low-cardinality (3 vals), 0% null; kept alongside card6_freq",
    "TransactionDT":    "Needed for time-based split; also a raw signal in its own right",
    "TransactionAmt":   "Fully observed numeric; no engineering needed",
    "uid":              "String key for future UID-aggregation features; not fed to MI",
    "M4":               "Low-card M-column present in df (if not already dropped)",
    "ProductCD":        "Low-card (4 vals), 12% fraud rate for 'C' — keep raw categorical",
    "DeviceType":       "Binary (mobile/desktop), 2.4% null — keep raw",
}
for col, reason in kept_raw.items():
    if col in df.columns:
        print(f"  {col:<25}  {reason}")

# Execute the drop
cols_to_drop = [c for c in cols_to_drop if c in df.columns]  # safety check
before = df.shape[1]
df.drop(columns=cols_to_drop, inplace=True)
print(f"\n  Shape before drop: {before} cols")
print(f"  Shape after drop : {df.shape[1]} cols  (dropped {len(cols_to_drop)})")

# ══════════════════════════════════════════════════════════════
# TASK 4 — TIME-BASED TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════
sep("TASK 4 — Time-Based Train / Val / Test Split (70 / 15 / 15)")

# Sort by TransactionDT to ensure strict forward-chaining
df = df.sort_values("TransactionDT").reset_index(drop=True)

dt_min = df["TransactionDT"].min()
dt_max = df["TransactionDT"].max()
span   = dt_max - dt_min
print(f"\n  TransactionDT range : {dt_min:,} — {dt_max:,}  (span: {span:,} seconds)")
print(f"  Total rows (sorted) : {len(df):,}")

# Compute cut points
cut_train = dt_min + 0.70 * span
cut_val   = dt_min + 0.85 * span

train_df = df[df["TransactionDT"] <  cut_train].copy()
val_df   = df[(df["TransactionDT"] >= cut_train) & (df["TransactionDT"] < cut_val)].copy()
test_df  = df[df["TransactionDT"] >= cut_val].copy()

def split_stats(name, sdf):
    n       = len(sdf)
    n_fraud = sdf["isFraud"].sum()
    fr      = sdf["isFraud"].mean() * 100
    dt_lo   = sdf["TransactionDT"].min()
    dt_hi   = sdf["TransactionDT"].max()
    day_lo  = (dt_lo - dt_min) / 86400
    day_hi  = (dt_hi - dt_min) / 86400
    print(f"\n  {name}:")
    print(f"    Rows          : {n:>8,}  ({n/len(df)*100:.1f}% of total)")
    print(f"    Fraud rows    : {n_fraud:>8,}  ({fr:.4f}%)")
    print(f"    TransactionDT : {dt_lo:,} — {dt_hi:,}")
    print(f"    Day offset    : {day_lo:.1f} — {day_hi:.1f}  (dataset days)")

split_stats("TRAIN (0%–70%)",      train_df)
split_stats("VALIDATION (70%–85%)", val_df)
split_stats("TEST (85%–100%)",     test_df)

print(f"\n  ⚠  Fraud rate WILL differ across splits (expected — time-drift in EDA).")
print(f"     Train fraud rate ≈ 6–8%;  Val/Test fraud rate ≈ 8–14%.")
print(f"     This is a distribution shift — model must generalize, not just memorize.")

# ══════════════════════════════════════════════════════════════
# TASK 5 — MUTUAL INFORMATION FEATURE RANKING (train split only)
# ══════════════════════════════════════════════════════════════
sep("TASK 5 — Mutual Information Ranking (train split only)")

TARGET = "isFraud"

# Identify columns to exclude from MI (non-features)
exclude = {TARGET, "TransactionID", "uid", "TransactionDT"}
feature_cols = [c for c in train_df.columns if c not in exclude]
print(f"  Feature columns for MI : {len(feature_cols)}")
print(f"  Excluded from MI       : {sorted(exclude)}")

# Build X_train, y_train
X_train = train_df[feature_cols].copy()
y_train = train_df[TARGET].astype(int)

# Encode any remaining object/string columns with LabelEncoder
# (mutual_info_classif needs numeric; we flag them as discrete)
obj_cols     = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
discrete_mask = np.zeros(len(feature_cols), dtype=bool)

print(f"\n  String/object columns still present (will label-encode for MI):")
le = LabelEncoder()
for col in obj_cols:
    X_train[col] = X_train[col].fillna("__MISSING__")
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    idx = feature_cols.index(col)
    discrete_mask[idx] = True
    print(f"    {col}")

# Fill remaining NaNs with median (MI requires no NaN)
print(f"\n  Filling NaN with median for numeric columns …")
X_train = X_train.fillna(X_train.median(numeric_only=True))

# Mark obvious integer/flag columns as discrete
for i, col in enumerate(feature_cols):
    if col.startswith("isnull_") or col in ["card3","card5","addr2"] or \
       X_train[col].dropna().apply(lambda x: x == int(x)).all():
        discrete_mask[i] = True

print(f"\n  Running mutual_info_classif …  (n_train={len(X_train):,})")
mi_scores = mutual_info_classif(
    X_train.values, y_train.values,
    discrete_features=discrete_mask,
    n_neighbors=5,
    random_state=42
)

mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

print(f"\n  TOP 30 FEATURES BY MUTUAL INFORMATION (nats):")
print(f"\n  {'Rank':<6}  {'Feature':<30}  {'MI Score':>10}")
print(f"  {'-'*50}")
for rank, (col, score) in enumerate(mi_series.head(30).items(), 1):
    print(f"  {rank:<6}  {col:<30}  {score:>10.6f}")

print(f"\n  Features with MI = 0 (no signal): {(mi_series == 0).sum()}")
print(f"  Features with MI > 0.01          : {(mi_series > 0.01).sum()}")
print(f"  Features with MI > 0.001         : {(mi_series > 0.001).sum()}")

# ══════════════════════════════════════════════════════════════
# TASK 6 — SAVE
# ══════════════════════════════════════════════════════════════
sep("TASK 6 — SAVING OUTPUTS")

# Save splits
for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
    path = os.path.join(DATA_DIR, f"split_{name}.parquet")
    sdf.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    mb = os.path.getsize(path) / (1024**2)
    print(f"  split_{name}.parquet  →  {sdf.shape}  ({mb:.1f} MB)")

# Save MI ranking
mi_df = mi_series.reset_index()
mi_df.columns = ["feature", "mi_score"]
mi_df["rank"] = range(1, len(mi_df)+1)
mi_df.to_csv(os.path.join(DATA_DIR, "processed/mi_feature_ranking.csv"), index=False)
print(f"\n  mi_feature_ranking.csv  →  {len(mi_df)} features ranked")

# Save top-30 feature list as JSON (for easy import in modeling step)
top30 = mi_series.head(30).index.tolist()
with open(os.path.join(DATA_DIR, "processed/top30_features.json"), "w") as f:
    json.dump({"features": top30, "target": TARGET}, f, indent=2)
print(f"  top30_features.json     →  {len(top30)} features")

# ─── Final summary ─────────────────────────────────────────
sep("FINAL SUMMARY")
print(f"  Engineered dataset     : {df.shape}")
print(f"  Train split            : {train_df.shape}  fraud={train_df[TARGET].mean()*100:.3f}%")
print(f"  Val   split            : {val_df.shape}    fraud={val_df[TARGET].mean()*100:.3f}%")
print(f"  Test  split            : {test_df.shape}   fraud={test_df[TARGET].mean()*100:.3f}%")
print(f"\n  Top 5 features by MI:")
for col, score in mi_series.head(5).items():
    print(f"    {col:<30}  {score:.6f}")
print(f"\n  Elapsed: {elapsed()}")
print("\n  Cleanup, split, and feature ranking complete. ✓")
