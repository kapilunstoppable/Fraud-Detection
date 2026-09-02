"""
Step 4 helper: merge train_transaction.csv + train_identity.csv on TransactionID
and save the merged result to data/merged_train.csv.

Run AFTER the Kaggle download + unzip is complete:
    python merge_data.py
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

transaction_path = os.path.join(DATA_DIR, "train_transaction.csv")
identity_path    = os.path.join(DATA_DIR, "train_identity.csv")
merged_path      = os.path.join(DATA_DIR, "merged_train.csv")

print("Loading train_transaction.csv …")
train_transaction = pd.read_csv(transaction_path)
print(f"  shape: {train_transaction.shape}")

print("Loading train_identity.csv …")
train_identity = pd.read_csv(identity_path)
print(f"  shape: {train_identity.shape}")

print("Merging on TransactionID (left join) …")
merged = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
print(f"  merged shape: {merged.shape}")

print(f"Saving to {merged_path} …")
merged.to_csv(merged_path, index=False)
print("Done ✓")
