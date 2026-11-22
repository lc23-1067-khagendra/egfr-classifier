import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

df = pd.read_csv("data/egfr_clean.csv")
print(f"Loaded {len(df)} molecules")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

fingerprints = []
labels = []
failed = 0

for smiles, active in zip(df["canonical_smiles"], df["active"]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        failed += 1
        continue
    fp = gen.GetFingerprintAsNumPy(mol)
    fingerprints.append(fp)
    labels.append(active)

X = np.array(fingerprints)
y = np.array(labels)

print(f"Failed to parse: {failed}")
print(f"Feature matrix: {X.shape}")
print(f"Labels: {y.shape}")

np.save("data/X.npy", X)
np.save("data/y.npy", y)