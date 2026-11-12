
import pandas as pd
import numpy as np

df = pd.read_csv("data/egfr_raw.csv")
print(f"Started with {len(df)} rows")

# YOUR JOB — write these two lines:
# 1. Keep only rows where standard_relation equals "="
#    (Why: ">" or "<" means they never pinned down the real IC50 --

#   "greater than 10000" isn't a real measurement, it's a non-result.
#     Training a model on those would teach it garbage.)
#
# 2. From what's left, keep only rows where standard_units is "nM"
#    (Why: you can't compare 50 nM to 3 µM without converting. Easiest
#     first pass: just keep the nM rows so everything's on one scale.)
df = df[df["standard_relation"] == "="]

df = df[df["standard_units"] == "nM"]
print(f"After cleaning: {len(df)} rows")
df = df.dropna(subset=["canonical_smiles", "standard_value"])
df = df[df["standard_value"] > 0]
df["pIC50"] = 9 - np.log10(df["standard_value"])
df = df.groupby("molecule_chembl_id").agg(
    canonical_smiles=("canonical_smiles", "first"),
    pIC50=("pIC50", "median")
).reset_index()
df = df[(df["pIC50"] >= 6) | (df["pIC50"] <= 5)]
df["active"] = (df["pIC50"] >= 6).astype(int)
print(df["active"].value_counts())
df.to_csv("data/egfr_clean.csv", index=False)