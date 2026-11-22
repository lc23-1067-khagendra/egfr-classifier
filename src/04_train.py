# =============================================================================
# 04_train.py — Train a random forest to predict EGFR activity
# =============================================================================

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

# -----------------------------------------------------------------------------
# LOAD
# X: (8981, 2048) — one row per molecule, 2048 columns of 1/0 saying which
#    chemical substructures that molecule contains (from 03_features.py).
# y: (8981,)      — the answer key. 1 = active (IC50 <= 1000 nM), 0 = inactive.
# -----------------------------------------------------------------------------
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# -----------------------------------------------------------------------------
# SPLIT  <-- THIS IS WHERE MEMORIZATION GETS IN
#
# Holding out test data is correct: you must score the model on molecules it
# never trained on, or you are just asking it to recite.
#
# But HOW we hold them out matters enormously, and this line gets it wrong
# for chemistry data.
#
# train_test_split shuffles all 8,981 molecules and deals 80/20 at random.
# That assumes every molecule is independent of every other one.
#
# They are not. ChEMBL is built from published papers, and a med-chem paper
# reports a SERIES: one core scaffold with 30 small variations around it.
# Compound 12 and compound 13 may differ by a single methyl group.
#
# A random deal scatters those 30 siblings across train and test. So the model
# trains on #12 and is tested on #13 — a molecule it has effectively already
# seen. It scores well by recognizing the family, not by understanding what
# makes a molecule bind EGFR.
#
# That is why ROC-AUC 0.973 is optimistic. It measures "can you recognize a
# scaffold you were trained on", not "will this work on new chemistry" —
# and the second question is the one drug discovery actually asks.
#
# The fix is a SCAFFOLD SPLIT: group molecules by their core ring structure
# and send whole scaffolds to train OR test, never both. (Next script.)
#
# stratify=y keeps the 90/10 active:inactive ratio in both halves. Without it,
# random chance could leave the test set with almost no inactives, making the
# inactive scores meaningless.
#
# random_state=42 fixes the shuffle so the split is identical every run —
# reproducibility, so your reported numbers can be checked by someone else.
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# -----------------------------------------------------------------------------
# MODEL
#
# n_estimators=200 — build 200 decision trees. Each tree trains on a random
#   sample of molecules and considers a random subset of the 2048 bits at
#   each split, so no two trees are alike. Each is mediocre and wrong in its
#   own way; averaging 200 of them cancels the individual errors and leaves
#   the shared signal. That averaging is the main defense against any single
#   tree memorizing.
#
#   NOTE: max_depth is unset, so each tree grows until its leaves are pure —
#   it can isolate individual molecules. This is a second, smaller route to
#   memorization. The forest averaging mostly absorbs it. It does NOT fix
#   the split problem above, which is the real issue.
#
# class_weight="balanced" — your data is 8051 active vs 930 inactive. Without
#   this, a model that blindly answers "active" every time scores 90% accuracy
#   while learning nothing. This makes each inactive count ~8.7x as much in
#   the training loss, so ignoring them becomes expensive.
#
# n_jobs=-1 — use all CPU cores. The 200 trees are independent, so this
#   parallelizes almost perfectly.
# -----------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# -----------------------------------------------------------------------------
# THE ACTUAL LEARNING — this one line is the machine learning.
#
# For each tree: look at every one of the 2048 bits, ask "which single bit
# best separates actives from inactives in this group of molecules?", put that
# question at the top, split the data on it, then repeat on each half.
# Recurse until the groups are pure or too small to split.
#
# Nobody told it any chemistry. It discovers from the data alone that
# certain substructures co-occur with activity.
# -----------------------------------------------------------------------------
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# PREDICT ON HELD-OUT DATA
#
# .predict()       — the 200 trees vote; majority wins; returns 0 or 1.
# .predict_proba() — returns the vote SHARE as a probability. [:, 1] takes
#                    the column for class 1 (active). Gefitinib scored 0.995
#                    because 199 of 200 trees voted active.
#
# The probability is more useful than the hard 0/1: in real screening you
# rank 100,000 candidates and test the top 500, so you need an ordering,
# not a yes/no.
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# -----------------------------------------------------------------------------
# EVALUATE
#
# ROC-AUC: hand the model one random active and one random inactive — how
#   often does it rank the active higher? 0.5 = coin flip, 1.0 = perfect.
#   Chosen because it is unaffected by class imbalance, unlike accuracy.
#
# Confusion matrix layout:
#                      predicted inactive   predicted active
#   actual inactive          138                  48
#   actual active             23                1588
#
#   The number that matters is inactive recall (138/186 = 0.74). The model
#   catches 74% of inactives but 99% of actives — exactly the gap you would
#   predict from 8051 vs 930 training examples.
#
# Accuracy is printed in the report but should NOT be reported as the
#   headline. 0.96 accuracy on 90%-active data hides the entire weakness.
# -----------------------------------------------------------------------------
print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["inactive", "active"]))

# -----------------------------------------------------------------------------
# SAVE
# Without this the trained model vanishes when the script ends. Saving it
# lets 05_predict.py score new molecules without retraining.
# -----------------------------------------------------------------------------
joblib.dump(model, "data/model.pkl")
print("\nSaved model to data/model.pkl")