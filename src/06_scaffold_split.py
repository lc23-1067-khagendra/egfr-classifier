import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

df = pd.read_csv("data/egfr_clean.csv")
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Group molecule indices by their Murcko scaffold
scaffolds = defaultdict(list)
for i, smiles in enumerate(df["canonical_smiles"]):
    mol = Chem.MolFromSmiles(smiles)
    core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    scaffolds[core].append(i)

print(f"{len(df)} molecules -> {len(scaffolds)} distinct scaffolds")

# Biggest scaffold groups go to train, smallest to test.
# This is the standard convention: it puts the rare, unusual chemistry
# in the test set, which is the harder and more honest exam.
groups = sorted(scaffolds.values(), key=len, reverse=True)

train_idx, test_idx = [], []
cutoff = len(df) * 0.8
for g in groups:
    if len(train_idx) + len(g) <= cutoff:
        train_idx.extend(g)
    else:
        test_idx.extend(g)

print(f"Train: {len(train_idx)}  Test: {len(test_idx)}")

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train actives: {y_train.sum()}/{len(y_train)}")
print(f"Test actives:  {y_test.sum()}/{len(y_test)}")

# Exact same model as 04_train.py — only the split changed
model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nSCAFFOLD SPLIT ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print("(random split was 0.973)")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["inactive", "active"]))