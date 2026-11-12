"""
Day 1: Pull real EGFR bioactivity data from ChEMBL.

Biology context:
EGFR (Epidermal Growth Factor Receptor) is one of the most important
cancer drug targets in existence. When it's mutated or overactive, cells
grow uncontrollably -- it drives many lung and breast cancers. Drugs like
gefitinib and erlotinib work by blocking it. Because it's so heavily
studied, ChEMBL has thousands of measured compounds against it, which is
exactly why it's a great target for a first ML project: lots of clean data.
"""

import pandas as pd
from chembl_webresource_client.new_client import new_client


def find_egfr_target():
    """
    Search ChEMBL for the human EGFR target and show what we find.

    We PRINT the candidates before trusting any single result -- the same
    'inspect before you trust it' habit that saved us in the NCBI lesson.
    """
    target = new_client.target
    results = target.search("EGFR")

    print("Candidate targets matching 'EGFR':")
    for r in results[:5]:
        print(f"  {r['target_chembl_id']:12}  "
              f"{r['organism']:20}  {r['pref_name']}")
    return "CHEMBL203"  # human EGFR, confirmed from the printout above


def fetch_ic50_activities(target_id):
    """
    Pull all IC50 measurements for our target.

    IC50 = concentration of compound needed to inhibit the target by 50%.
    Lower IC50 = more potent.
    """
    activity = new_client.activity
    query = activity.filter(
        target_chembl_id=target_id,
        standard_type="IC50",
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "standard_relation",
    ])

    print("\nDownloading from ChEMBL... this can take a few minutes.")
    df = pd.DataFrame(query)
    return df


def main():
    target_id = find_egfr_target()
    df = fetch_ic50_activities(target_id)

    print(f"\nPulled {len(df):,} rows.")
    print("Columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head())

    out_path = "data/egfr_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()