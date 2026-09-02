"""
IEEE-CIS — Anomaly Autoencoder
================================
Architecture : 30 → 64 → 32 → 16 → 32 → 64 → 30
               BatchNorm + ReLU, MSE reconstruction loss

Training philosophy:
  - Scaler/medians fit ONLY on non-fraud train rows
  - Model trained ONLY on non-fraud train rows
  - Early stopping monitored on non-fraud val rows
  - Reconstruction error used as anomaly score at inference time
"""

import os, sys, json, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
t0 = time.time()

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ART_DIR    = "/Users/shriyanshraj/.gemini/antigravity-ide/brain/98590e36-f8e7-4b2b-8f32-e55098f07b90"
os.makedirs(MODELS_DIR, exist_ok=True)

def elapsed(): return f"{time.time()-t0:.1f}s"
def sep(title): print(f"\n{'='*60}\n{title}\n{'='*60}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = "isFraud"

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
sep("LOADING DATA")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "processed/split_train.parquet"))
val_df   = pd.read_parquet(os.path.join(DATA_DIR, "processed/split_val.parquet"))
# test_df deliberately NOT loaded

with open(os.path.join(DATA_DIR, "processed/top30_features.json")) as f:
    top30 = json.load(f)["features"]

print(f"  Train : {train_df.shape}")
print(f"  Val   : {val_df.shape}")
print(f"  Features (top-30): {top30}")
print(f"  Device : {DEVICE}")

# ══════════════════════════════════════════════════════════════
# TASK 1 — Feature Prep (fit ONLY on non-fraud train rows)
# ══════════════════════════════════════════════════════════════
sep("TASK 1 — Feature Preparation (non-fraud train rows only)")

# Isolate non-fraud train rows
train_legit = train_df[train_df[TARGET] == 0].copy()
train_fraud  = train_df[train_df[TARGET] == 1].copy()
val_legit    = val_df[val_df[TARGET] == 0].copy()
val_fraud    = val_df[val_df[TARGET] == 1].copy()

print(f"  Train legit rows : {len(train_legit):,}")
print(f"  Train fraud rows : {len(train_fraud):,}  ← NEVER seen by AE during training")
print(f"  Val   legit rows : {len(val_legit):,}  ← early-stopping signal")
print(f"  Val   fraud rows : {len(val_fraud):,}  ← only used at final eval")

# Medians computed from NON-FRAUD TRAIN rows only
train_medians = train_legit[top30].median()
print(f"\n  Medians computed from {len(train_legit):,} non-fraud train rows only ✓")

def impute(df, medians, cols):
    X = df[cols].copy()
    return X.fillna(medians)

# Impute
X_train_legit_raw = impute(train_legit, train_medians, top30)
X_val_legit_raw   = impute(val_legit,   train_medians, top30)
X_val_fraud_raw   = impute(val_fraud,   train_medians, top30)
X_val_all_raw     = impute(val_df,      train_medians, top30)   # for full val eval

# StandardScaler fit on NON-FRAUD TRAIN rows only
scaler = StandardScaler()
X_train_legit = scaler.fit_transform(X_train_legit_raw.values)
print(f"  StandardScaler fit on {len(X_train_legit_raw):,} non-fraud train rows only ✓")

# Apply to val splits
X_val_legit = scaler.transform(X_val_legit_raw.values)
X_val_fraud = scaler.transform(X_val_fraud_raw.values)
X_val_all   = scaler.transform(X_val_all_raw.values)

# Labels for the full val set (in sorted order)
y_val_all   = val_df[TARGET].values.astype(int)

INPUT_DIM = X_train_legit.shape[1]
print(f"  Input dimension: {INPUT_DIM}")

# ══════════════════════════════════════════════════════════════
# TASK 2 — Autoencoder Architecture + Training
# ══════════════════════════════════════════════════════════════
sep("TASK 2 — Autoencoder: Build + Train")

class FraudAutoencoder(nn.Module):
    """
    Undercomplete autoencoder: input_dim → 64 → 32 → 16 → 32 → 64 → input_dim
    Each encoder/decoder layer: Linear → BatchNorm → ReLU
    Final decoder output: Linear only (no activation — we need real-valued output)

    WHY undercomplete:
      The bottleneck (16 dims from 30 inputs) forces the network to learn a
      compressed representation of "normal" transaction structure. If the
      latent space is as large as the input, the network can learn the identity
      function and reconstruction error carries no signal. The bottleneck makes
      it *hard* to reconstruct anything — the network only "learns" what it was
      repeatedly shown during training (non-fraud patterns). Fraud rows, which
      have different statistical structure, can't be compressed and reconstructed
      as accurately → higher MSE → anomaly flagged.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # Encoder: input_dim → 64 → 32 → 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        # Decoder: 16 → 32 → 64 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim),   # linear output — reconstruct raw scaled values
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruct_error(self, x):
        """Per-row MSE between input and reconstruction."""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=1)   # shape: (batch,)


# DataLoaders — train on legit only
def to_tensor(arr):
    return torch.FloatTensor(arr).to(DEVICE)

train_ds = TensorDataset(to_tensor(X_train_legit))
val_ds   = TensorDataset(to_tensor(X_val_legit))

BATCH_SIZE = 512
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

model     = FraudAutoencoder(INPUT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
)
criterion = nn.MSELoss()

print(f"\n  Model architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Trainable parameters: {total_params:,}")

MAX_EPOCHS      = 150
PATIENCE        = 15   # early stopping patience on non-fraud val loss
best_val_loss   = float("inf")
patience_counter = 0
best_state_dict  = None
history          = {"train_loss": [], "val_loss": []}

print(f"\n  Training on {len(X_train_legit):,} non-fraud rows "
      f"({len(train_loader)} batches/epoch) …")
print(f"  Early stopping patience: {PATIENCE} epochs on non-fraud val MSE\n")
print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>8}  {'Status'}")
print(f"  {'-'*55}")

for epoch in range(1, MAX_EPOCHS + 1):
    # ── Train
    model.train()
    tr_losses = []
    for (xb,) in train_loader:
        optimizer.zero_grad()
        recon = model(xb)
        loss  = criterion(recon, xb)
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())
    tr_loss = np.mean(tr_losses)

    # ── Val (non-fraud only — no label leakage)
    model.eval()
    vl_losses = []
    with torch.no_grad():
        for (xb,) in val_loader:
            recon = model(xb)
            vl_losses.append(criterion(recon, xb).item())
    vl_loss = np.mean(vl_losses)

    scheduler.step(vl_loss)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)

    status = ""
    if vl_loss < best_val_loss:
        best_val_loss    = vl_loss
        patience_counter = 0
        best_state_dict  = {k: v.clone() for k, v in model.state_dict().items()}
        status = "✓ best"
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  {epoch:>6}  {tr_loss:>10.6f}  {vl_loss:>10.6f}  "
                  f"{optimizer.param_groups[0]['lr']:>8.2e}  STOP")
            print(f"\n  Early stopping triggered at epoch {epoch}. "
                  f"Best val MSE: {best_val_loss:.6f}")
            break

    if epoch % 10 == 0 or epoch == 1 or status:
        print(f"  {epoch:>6}  {tr_loss:>10.6f}  {vl_loss:>10.6f}  "
              f"{optimizer.param_groups[0]['lr']:>8.2e}  {status}")

# Restore best weights
model.load_state_dict(best_state_dict)
model.eval()
print(f"\n  Best model restored (val MSE = {best_val_loss:.6f})  [{elapsed()}]")

# ══════════════════════════════════════════════════════════════
# TASK 3 — Reconstruction Error Distribution + Plot
# ══════════════════════════════════════════════════════════════
sep("TASK 3 — Reconstruction Error Distribution")

with torch.no_grad():
    err_legit = model.reconstruct_error(to_tensor(X_val_legit)).cpu().numpy()
    err_fraud = model.reconstruct_error(to_tensor(X_val_fraud)).cpu().numpy()
    err_all   = model.reconstruct_error(to_tensor(X_val_all)).cpu().numpy()

print(f"\n  Non-Fraud reconstruction error (val):")
print(f"    mean={err_legit.mean():.4f}  median={np.median(err_legit):.4f}  "
      f"p95={np.percentile(err_legit,95):.4f}  p99={np.percentile(err_legit,99):.4f}  "
      f"max={err_legit.max():.4f}")
print(f"\n  Fraud reconstruction error (val):")
print(f"    mean={err_fraud.mean():.4f}  median={np.median(err_fraud):.4f}  "
      f"p95={np.percentile(err_fraud,95):.4f}  p99={np.percentile(err_fraud,99):.4f}  "
      f"max={err_fraud.max():.4f}")

# Plot
FIG_BG, AX_BG, TEXT = "#1a1a2e", "#16213e", "#e0e0e0"
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=FIG_BG)

clip = np.percentile(np.concatenate([err_legit, err_fraud]), 99.5)
bins = np.linspace(0, clip, 100)

for ax in axes:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

# Left: linear y
axes[0].hist(err_legit, bins=bins, alpha=0.65, color="#4C9BE8",
             label=f"Non-Fraud (n={len(err_legit):,})", density=True)
axes[0].hist(err_fraud, bins=bins, alpha=0.65, color="#E8524C",
             label=f"Fraud (n={len(err_fraud):,})", density=True)
axes[0].set_xlabel("Reconstruction MSE", color=TEXT)
axes[0].set_ylabel("Density", color=TEXT)
axes[0].set_title("Reconstruction Error Distribution (linear)", color=TEXT, fontweight="bold")
axes[0].legend(facecolor="#222", labelcolor=TEXT)

# Right: log y
axes[1].hist(err_legit, bins=bins, alpha=0.65, color="#4C9BE8",
             label=f"Non-Fraud (n={len(err_legit):,})", density=True)
axes[1].hist(err_fraud, bins=bins, alpha=0.65, color="#E8524C",
             label=f"Fraud (n={len(err_fraud):,})", density=True)
axes[1].set_yscale("log")
axes[1].set_xlabel("Reconstruction MSE", color=TEXT)
axes[1].set_ylabel("Density (log scale)", color=TEXT)
axes[1].set_title("Reconstruction Error Distribution (log-scale)", color=TEXT, fontweight="bold")
axes[1].legend(facecolor="#222", labelcolor=TEXT)

fig.suptitle("Autoencoder: Reconstruction Error — Fraud vs Non-Fraud (Val Set)",
             color=TEXT, fontsize=13, fontweight="bold")
plot_path = os.path.join(ART_DIR, "ae_reconstruction_error.png")
fig.savefig(plot_path, facecolor=FIG_BG, bbox_inches="tight", dpi=130)
plt.close(fig)
print(f"\n  [saved] ae_reconstruction_error.png")

# ══════════════════════════════════════════════════════════════
# TASK 4 — Threshold + Metrics
# ══════════════════════════════════════════════════════════════
sep("TASK 4 — Threshold Selection + Metrics")

# Threshold: percentile of NON-FRAUD val errors only (no fraud rows used)
# We test both p95 and p99 and report both
thresholds = {
    "p95": float(np.percentile(err_legit, 95)),
    "p99": float(np.percentile(err_legit, 99)),
}

print(f"\n  Threshold candidates (from non-fraud val rows only):")
for name, thr in thresholds.items():
    print(f"    {name}: {thr:.4f}")

print(f"\n  {'Threshold':<8}  {'AUC-ROC':>8}  {'PR-AUC':>8}  {'F1':>8}  "
      f"{'Precision':>10}  {'Recall':>8}  {'Flagged':>8}")
print(f"  {'-'*70}")

ae_metrics_by_thr = {}
for name, thr in thresholds.items():
    y_pred = (err_all >= thr).astype(int)
    m = {
        "AUC-ROC"  : roc_auc_score(y_val_all, err_all),
        "PR-AUC"   : average_precision_score(y_val_all, err_all),
        "F1"       : f1_score(y_val_all, y_pred, zero_division=0),
        "Precision": precision_score(y_val_all, y_pred, zero_division=0),
        "Recall"   : recall_score(y_val_all, y_pred, zero_division=0),
        "Flagged"  : int(y_pred.sum()),
    }
    ae_metrics_by_thr[name] = m
    print(f"  {name:<8}  {m['AUC-ROC']:>8.4f}  {m['PR-AUC']:>8.4f}  {m['F1']:>8.4f}  "
          f"{m['Precision']:>10.4f}  {m['Recall']:>8.4f}  {m['Flagged']:>8,}")

# Select p99 as primary threshold (fewer false positives, more defensible)
chosen_thr_name = "p99"
chosen_thr      = thresholds[chosen_thr_name]
ae_metrics      = ae_metrics_by_thr[chosen_thr_name]

print(f"\n  ► Selected threshold: {chosen_thr_name} = {chosen_thr:.4f}")
print(f"    Rationale: at p99, 99% of normal transactions pass through; only the top")
print(f"    1% of 'unusual-looking' normal traffic is flagged alongside fraud. This")
print(f"    gives a false positive rate among non-fraud rows of ~1%, which is operationally")
print(f"    reasonable for a fraud alert queue. p95 catches more fraud but produces")
print(f"    5× more false positives — worth trying in cost-analysis step.")

# ══════════════════════════════════════════════════════════════
# TASK 5 — Three-Way Comparison Table
# ══════════════════════════════════════════════════════════════
sep("TASK 5 — Three-Way Model Comparison (Val Set)")

# Load LR and XGB val metrics from saved metadata
with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "rb") as f:
    lr_data = pickle.load(f)
with open(os.path.join(MODELS_DIR, "xgboost_meta.pkl"), "rb") as f:
    xgb_data = pickle.load(f)

lr_m  = lr_data["val_metrics"]
xgb_m = xgb_data["val_metrics"]

metrics_order = ["AUC-ROC", "PR-AUC", "F1", "Precision", "Recall"]
print(f"\n  {'Metric':<12}  {'Log. Reg.':>12}  {'XGBoost':>12}  {'Autoencoder':>14}  {'Best':>8}")
print(f"  {'-'*65}")
for m in metrics_order:
    lrv  = lr_m[m]
    xgbv = xgb_m[m]
    aev  = ae_metrics[m]
    vals = {"LR": lrv, "XGB": xgbv, "AE": aev}
    best = max(vals, key=vals.get)
    print(f"  {m:<12}  {lrv:>12.4f}  {xgbv:>12.4f}  {aev:>14.4f}  {best:>8}")

print(f"\n  Notes:")
print(f"  • AUC-ROC/PR-AUC are threshold-independent (use raw reconstruction error as score)")
print(f"  • F1/Precision/Recall for AE use {chosen_thr_name} threshold = {chosen_thr:.4f}")
print(f"  • XGBoost and LR use default 0.5 probability threshold")
print(f"  • All metrics computed on split_val; test set untouched ✓")

# ══════════════════════════════════════════════════════════════
# TASK 6 — Plain Language Explanation (printed for record)
# ══════════════════════════════════════════════════════════════
sep("TASK 6 — Plain Language Explanation")
print("""
  WHY TRAIN ONLY ON NON-FRAUD ROWS?
  ─────────────────────────────────
  The autoencoder is an unsupervised anomaly detector. It learns to
  compress and reconstruct "normal" transactions. If we train on fraud
  rows too, the network learns to reconstruct fraud patterns equally
  well, destroying the anomaly signal — a fraudulent transaction would
  reconstruct just as accurately as a legitimate one, and reconstruction
  error would carry no discriminatory power.

  By training only on the 105,660 non-fraud rows, we create a model
  that knows "what normal looks like" but has never seen fraud during
  training. When a fraud transaction is fed to this model at inference,
  its unusual statistical structure can't be faithfully compressed and
  re-expanded — the bottleneck doesn't have enough capacity to encode
  patterns it was never trained on — hence higher MSE.

  WHY AN UNDERCOMPLETE BOTTLENECK?
  ─────────────────────────────────
  Our architecture: 30 → 64 → 32 → [16] → 32 → 64 → 30
  The bottleneck (16 dimensions < 30 input dimensions) is key.
  An overcomplete autoencoder (bottleneck ≥ input dim) can learn the
  identity function — it just copies the input, achieving zero
  reconstruction loss for EVERYTHING, fraud included. The bottleneck
  forces the encoder to find a lossy but maximally informative
  compression. The encoder learns which 16 "directions" capture the
  most variance of normal transactions. Fraud transactions — which lie
  in different regions of the 30-D feature space — project poorly onto
  this normal-transaction subspace, producing higher reconstruction error.

  WHY DOES HIGH RECONSTRUCTION ERROR = ANOMALY?
  ──────────────────────────────────────────────
  Think of it as a "shape template" learned by the network. After training
  on 100k normal transactions, the network has an internal model of what
  a normal transaction's 30-feature profile looks like in compressed form.
  When a new transaction arrives, the network tries to reconstruct it
  through this normal template. Legitimate transactions — even ones the
  network hasn't seen before — follow similar distributional patterns to
  the training data, so they reconstruct well (low MSE). Fraud
  transactions often have unusual combinations of feature values (e.g.
  very high V-columns, unusual D-norm patterns, specific device fingerprints)
  that the normal template poorly represents → high MSE → anomaly flag.

  HOW THE THRESHOLD WAS CHOSEN AND WHY IT'S DEFENSIBLE:
  ───────────────────────────────────────────────────────
  Threshold = 99th percentile of reconstruction errors of NON-FRAUD
  validation rows = {:.4f}.

  Why this approach:
    1. It is computed ONLY from non-fraud rows — no fraud labels used.
       The autoencoder remains fully unsupervised in the traditional sense.
    2. The p99 choice means: "flag a transaction if its reconstruction
       error is in the top 1% of what normal transactions produce."
       This gives a ~1% false positive rate on non-fraud rows, which
       is operationally acceptable (1 in 100 legitimate transactions
       incorrectly flagged for review).
    3. It's interpretable in an interview: "our threshold is the 99th
       percentile of the anomaly score distribution on clean data,
       evaluated on a held-out validation set that the model never
       trained on."
    4. We did NOT use fraud rows to pick the threshold — doing so would
       introduce label leakage into the threshold-selection process.
""".format(chosen_thr))

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
sep("SAVING MODEL")

ae_path = os.path.join(MODELS_DIR, "autoencoder_state.pt")
torch.save({
    "state_dict"      : best_state_dict,
    "input_dim"       : INPUT_DIM,
    "architecture"    : "30→64→32→16→32→64→30 (BN+ReLU, MSE loss)",
    "best_val_loss"   : best_val_loss,
    "features"        : top30,
    "threshold_p95"   : thresholds["p95"],
    "threshold_p99"   : thresholds["p99"],
    "chosen_threshold": chosen_thr,
    "chosen_thr_name" : chosen_thr_name,
    "val_metrics"     : ae_metrics,
}, ae_path)

scaler_path = os.path.join(MODELS_DIR, "ae_scaler_and_medians.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({
        "scaler"        : scaler,
        "train_medians" : train_medians.to_dict(),
        "features"      : top30,
    }, f)

print(f"  autoencoder_state.pt         → {ae_path}")
print(f"  ae_scaler_and_medians.pkl    → {scaler_path}")
print(f"\n  Test set: UNTOUCHED ✓")
print(f"  Elapsed : {elapsed()}")
print("\n  Autoencoder training and evaluation complete. ✓")
