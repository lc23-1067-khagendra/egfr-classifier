import pandas as pd
import numpy as np

df = pd.read_csv("data/egfr_raw.csv")
print(f"Started with {len(df)} rows")

# -----------------------------------------------------------------------------
# RELATION FILTER
#
# An "=" row is a real measurement: the lab pinned down the exact IC50.
#
# A ">" or "<" row is CENSORED — the lab stopped before pinning it down.
# Most censored rows are useless ("> 100 nM" could mean anything above 100).
# But two kinds are definitive, because of where our activity cutoffs sit:
#
#   "> 10000 nM"  -> true IC50 is above 10,000, and our inactive rule is
#                    IC50 >= 10,000. So this is a GUARANTEED inactive.
#                    The chemist tested it and it definitively failed.
#
#   "< 1000 nM"   -> true IC50 is below 1,000, and our active rule is
#                    IC50 <= 1,000. So this is a GUARANTEED active.
#
# Rescuing these matters: the original "=" only filter threw away thousands
# of confirmed duds, which is a major cause of the 90/10 class imbalance.
#
# Both directions are rescued deliberately. Keeping only the censored
# inactives would be selectively including data in the direction that
# improves our numbers.
#
# Note pIC50 = 9 - log10(10000) = 5.0 exactly, and 9 - log10(1000) = 6.0
# exactly, so the rescued rows land cleanly on the correct side of the
# activity cliff below without special handling.
# -----------------------------------------------------------------------------
is_exact = df["standard_relation"] == "="
is_censored_inactive = df["standard_relation"].isin([">", ">="]) & (df["standard_value"] >= 10000)
is_censored_active = df["standard_relation"].isin(["<", "<="]) & (df["standard_value"] <= 1000)

print(f"Rescued {is_censored_inactive.sum()} censored inactives, "
      f"{is_censored_active.sum()} censored actives")

df = df[is_exact | is_censored_inactive | is_censored_active].copy()

# Keep one unit scale. You can't compare 50 nM to 3 uM as raw numbers.
df = df[df["standard_units"] == "nM"]
print(f"After relation + unit filters: {len(df)} rows")

# No structure means no fingerprint. log10(0) is undefined, and a zero
# IC50 is physically impossible anyway.
df = df.dropna(subset=["canonical_smiles", "standard_value"])
df = df[df["standard_value"] > 0]

# Raw IC50 spans ~6 orders of magnitude bunched at the low end. The log
# spreads it out; the "9 -" flips it so higher = more potent, range ~4-10.
df["pIC50"] = 9 - np.log10(df["standard_value"])

# One row per molecule. The same compound is tested by many labs with
# varying results; median resists assay outliers better than mean.
df = df.groupby("molecule_chembl_id").agg(
    canonical_smiles=("canonical_smiles", "first"),
    pIC50=("pIC50", "median")
).reset_index()

# Activity cliff: keep only the two extremes, discard the ambiguous middle.
# A compound at 5,000 nM is genuinely unclear; forcing a label teaches noise.
df = df[(df["pIC50"] >= 6) | (df["pIC50"] <= 5)]
df["active"] = (df["pIC50"] >= 6).astype(int)

print(f"\nFinal: {len(df)} molecules")
print(df["active"].value_counts())
print(f"Percent active: {100 * df['active'].mean():.1f}%")

df.to_csv("data/egfr_clean.csv", index=False)
