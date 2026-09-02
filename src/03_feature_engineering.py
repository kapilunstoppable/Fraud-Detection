"""
IEEE-CIS Fraud Detection — Feature Engineering
===============================================
Loads the identity-joined dataset (144,233 × 434) and applies:
  1. UID construction       (card1 + card2 + addr1 + D1)
  2. D-column normalization (D_n - TransactionDay)
  3. Frequency encoding     (high-cardinality categoricals)
  4. Missingness flags      (D5, V138-V157 block)
  5. Drop fully-null cols   (identified in EDA)
  6. Save to Parquet

No modeling, no feature selection, no train/val split.
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

t0 = time.time()
NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Notebook")
DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def elapsed(): return f"{time.time()-t0:.1f}s"
def sep(title): print(f"\n{'='*60}\n{title}\n{'='*60}")

# ─────────────────────────────────────────────────────────────
# LOAD  (notebook-style identity left-join)
# ─────────────────────────────────────────────────────────────
sep("LOADING DATA")
os.chdir(NOTEBOOK_DIR)
identity    = pd.read_csv("../data/train_identity.csv")
transaction = pd.read_csv("../data/train_transaction.csv")
df = pd.merge(identity, transaction, on="TransactionID", how="left")
print(f"  Loaded: {df.shape}  [{elapsed()}]")
assert df.shape == (144233, 434)

# Keep track of new features we add
new_features = {}

# ══════════════════════════════════════════════════════════════
# STEP 1 — UID CONSTRUCTION
# ══════════════════════════════════════════════════════════════
sep("STEP 1 — UID CONSTRUCTION")

# --- Why card1 + card2 + addr1 + D1? ---
# Top Kaggle solutions for IEEE-CIS observed that there is no explicit customer ID.
# However, card1 (masked card number prefix), card2 (another card attribute),
# addr1 (billing zip/postal code), and D1 (days-since-last-seen, rounded to int)
# together form a quasi-unique fingerprint for a specific physical card.
# Combining them lets us group transactions by "likely same cardholder",
# enabling aggregation features (frequency, velocity, etc.) without leaking
# test-set information.
#
# D1 is included as an extra disambiguation key because card1+card2+addr1
# occasionally collides across different cards (shared zip, similar card prefix).
# Using round(D1) adds the approximate account age as a tiebreaker without
# introducing float noise.

# Confirm columns present
for col in ["card1", "card2", "addr1", "D1"]:
    null_pct = df[col].isnull().mean()*100
    print(f"  {col:<8} — {df[col].nunique():>6} unique vals, {null_pct:.1f}% null")

# Build UID: stringify each part (NaN → 'nan', acceptable for grouping)
df["uid"] = (
    df["card1"].astype(str) + "_" +
    df["card2"].astype(str) + "_" +
    df["addr1"].astype(str) + "_" +
    df["D1"].round().astype(str)   # round to int to avoid float drift
)

n_uid = df["uid"].nunique()
print(f"\n  Unique UIDs: {n_uid:,}  (out of {len(df):,} rows → avg {len(df)/n_uid:.1f} txns/UID)")

uid_counts = df.groupby("uid").size()
print("\n  Transactions-per-UID distribution:")
print(uid_counts.describe().to_string())
print(f"\n  UIDs with only 1 transaction : {(uid_counts==1).sum():,} ({(uid_counts==1).mean()*100:.1f}%)")
print(f"  UIDs with 2-5 transactions   : {((uid_counts>=2)&(uid_counts<=5)).sum():,}")
print(f"  UIDs with >5 transactions    : {(uid_counts>5).sum():,}")
print(f"  Max transactions for 1 UID   : {uid_counts.max():,}")

new_features["UID"] = ["uid"]

# ══════════════════════════════════════════════════════════════
# STEP 2 — D-COLUMN NORMALIZATION
# ══════════════════════════════════════════════════════════════
sep("STEP 2 — D-COLUMN NORMALIZATION")

# --- Why normalize D-columns? ---
# D1–D15 are described by the data provider as "timedelta" columns — they
# represent intervals in days relative to some internal reference date.
# Problem: because TransactionDT keeps increasing over the 182-day dataset,
# the raw D-values drift upward over time even for the SAME type of event.
# For example, D1 = 10 early in the dataset might mean the same thing as
# D1 = 182 late in the dataset if both represent "account created 10 days
# before the reference date".
#
# Fix: subtract the "transaction day offset" (TransactionDT ÷ 86400) from
# each D-column. This anchors every D_n to a stable timeline and removes
# the spurious time-drift correlation with fraud rate that we saw in EDA.
#
# Reference: top public notebooks by Chris Deotte and others on Kaggle
# all apply this exact transformation.

transaction_day = df["TransactionDT"] / 86400   # fractional day since dataset epoch

import re
# Only match D1, D2, … D15 — NOT DeviceType or DeviceInfo
d_cols = [c for c in df.columns if re.fullmatch(r"D\d+", c)]
d_cols_available = [c for c in d_cols if df[c].isnull().mean() < 1.0]
print(f"  D-columns found: {d_cols}")
print(f"  D-columns non-null: {d_cols_available}")

norm_d_cols = []
for col in d_cols_available:
    new_col = f"{col}_norm"
    # Cast to numeric first — pandas 3.x may infer Arrow-backed dtypes from CSV
    col_num = pd.to_numeric(df[col], errors="coerce")
    df[new_col] = col_num - transaction_day
    null_before = col_num.isnull().mean()*100
    null_after  = df[new_col].isnull().mean()*100
    mean_before = col_num.mean()
    mean_after  = df[new_col].mean()
    print(f"  {col:<5} → {new_col:<12}  "
          f"null%: {null_before:.1f}%   "
          f"mean before: {mean_before:.1f}  "
          f"mean after: {mean_after:.1f}")
    norm_d_cols.append(new_col)

new_features["Normalized D-columns"] = norm_d_cols
print(f"\n  Created {len(norm_d_cols)} normalized D-columns.")

# ══════════════════════════════════════════════════════════════
# STEP 3 — FREQUENCY ENCODING
# ══════════════════════════════════════════════════════════════
sep("STEP 3 — FREQUENCY ENCODING")

# --- Why frequency encoding? ---
# Tree models can use label-encoded categoricals, but high-cardinality
# columns like DeviceInfo (1,786 unique values) or id_31 (130 values)
# are hard to split on efficiently by a tree. Frequency encoding replaces
# each category with its proportion in the training data — a continuous
# signal that captures "how common is this value". Rare/unusual values
# (low frequency) often correspond to fraud.
#
# We use proportion (0–1) rather than raw count so the scale is stable
# if the dataset grows.

freq_cols = [
    "card1", "card2", "addr1",
    "P_emaildomain", "R_emaildomain",
    "DeviceInfo", "id_31", "id_30",
    "card4", "card6",          # low-card but still useful as continuous signal
]
freq_cols = [c for c in freq_cols if c in df.columns]

freq_encoded = []
print(f"  Columns to frequency-encode: {freq_cols}\n")
for col in freq_cols:
    freq_map  = df[col].value_counts(normalize=True)   # proportion in training data
    new_col   = f"{col}_freq"
    df[new_col] = df[col].map(freq_map).astype(float)
    # NaN rows (missing original) → 0 (unknown/unseen category)
    df[new_col] = df[new_col].fillna(0.0)
    n_unique  = df[col].nunique()
    null_orig = df[col].isnull().mean()*100
    print(f"  {col:<20} {n_unique:>6} unique  {null_orig:5.1f}% null  → {new_col}")
    freq_encoded.append(new_col)

new_features["Frequency-encoded columns"] = freq_encoded
print(f"\n  Created {len(freq_encoded)} frequency-encoded columns.")

# ══════════════════════════════════════════════════════════════
# STEP 4 — MISSINGNESS-AS-SIGNAL FLAGS
# ══════════════════════════════════════════════════════════════
sep("STEP 4 — MISSINGNESS-AS-SIGNAL FLAGS")

# From EDA:
#  • D5:       missing in 51.5% of fraud vs 79.2% of legit  → D5 being PRESENT ≈ fraud signal
#  • V138–V157 block: missing in 67.8% of fraud vs 41.3% of legit → V-block ABSENT ≈ fraud signal
#
# We create binary columns: 1 = original column is null, 0 = original column has a value.
# These are fed as features directly; the model learns the direction.

is_null_targets = ["D5"] + [f"V{i}" for i in range(138, 158)]   # V138..V157
is_null_targets = [c for c in is_null_targets if c in df.columns]

null_flags = []
print(f"  Creating is_null flags for {len(is_null_targets)} columns:\n")
print(f"  {'Column':<10} {'% null fraud':>13} {'% null legit':>13} {'Δ (pp)':>8}  → flag name")
print(f"  {'-'*60}")

fraud_mask = df["isFraud"] == 1
for col in is_null_targets:
    flag_col = f"isnull_{col}"
    df[flag_col] = df[col].isnull().astype(np.int8)
    pf = df.loc[fraud_mask,  col].isnull().mean()*100
    pl = df.loc[~fraud_mask, col].isnull().mean()*100
    print(f"  {col:<10} {pf:>12.1f}% {pl:>12.1f}%  {pf-pl:>+7.1f}pp  → {flag_col}")
    null_flags.append(flag_col)

new_features["Missingness flags (is_null)"] = null_flags
print(f"\n  Created {len(null_flags)} is_null flag columns.")

# ══════════════════════════════════════════════════════════════
# STEP 5 — DROP FULLY-NULL COLUMNS
# ══════════════════════════════════════════════════════════════
sep("STEP 5 — DROP FULLY-NULL COLUMNS")

# Detect columns that are 100% null in this identity-joined view
null_100 = [c for c in df.columns if df[c].isnull().all()]
print(f"  Columns that are 100% null: {len(null_100)}")
print(f"  {sorted(null_100)}\n")

# Verify these are the ones we saw in EDA (V1-V10, M1-M9, D11, dist1, etc.)
df.drop(columns=null_100, inplace=True)
print(f"  Dropped {len(null_100)} columns.")
print(f"  Shape after drop: {df.shape}")

# ══════════════════════════════════════════════════════════════
# STEP 6 — SAVE ENGINEERED DATASET
# ══════════════════════════════════════════════════════════════
sep("STEP 6 — SAVING ENGINEERED DATASET")

out_path = os.path.join(DATA_DIR, "processed/engineered_train.parquet")
df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
size_mb = os.path.getsize(out_path) / (1024**2)
print(f"  Saved to : {out_path}")
print(f"  File size : {size_mb:.1f} MB")
print(f"  Final shape: {df.shape}")

# ══════════════════════════════════════════════════════════════
# STEP 7 — SUMMARY
# ══════════════════════════════════════════════════════════════
sep("STEP 7 — SUMMARY")

print(f"  Original shape  : 144,233 × 434")
print(f"  Final shape     : {df.shape[0]:,} × {df.shape[1]}")
print(f"  Rows unchanged  : ✓ (no rows dropped)")
print(f"  Columns dropped : {len(null_100)} (100%-null in identity join)")
added = sum(len(v) for v in new_features.values())
print(f"  Columns added   : {added}")
print()
for category, cols in new_features.items():
    print(f"  [{category}]  ({len(cols)} feature(s))")
    for c in cols:
        print(f"    {c}")

print(f"\n  Elapsed: {elapsed()}")
print("\n  Feature engineering complete. ✓")
