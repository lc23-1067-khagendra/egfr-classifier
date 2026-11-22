# EGFR Bioactivity Classifier

Predicting whether a drug-like molecule inhibits **EGFR** — a receptor that drives many lung and breast cancers — from its chemical structure alone.

Built on 26,600 published IC50 measurements from ChEMBL. The interesting part of this project isn't the model; it's what happened when I validated it properly.

---

## The headline result

The same model scores very differently depending on how you split the data:

| Metric | Random split | **Scaffold split** |
|---|---|---|
| ROC-AUC | 0.973 | **0.917** |
| Inactive recall | 0.81 | **0.56** |
| Inactive precision | 0.89 | 0.75 |
| Active recall | 0.97 | 0.94 |
| Accuracy | 0.94 | 0.85 |

A **random split** shuffles molecules and deals 80/20. That assumes every molecule is independent — but medicinal chemistry papers publish *series*: one core scaffold with 30 near-identical analogues around it. A random split scatters those siblings across train and test, so the model trains on compound #12 and is tested on #13. It scores well by recognizing a family it was shown, not by predicting new chemistry.

A **scaffold split** groups molecules by their Bemis-Murcko core and assigns whole scaffolds to train *or* test, never both. That asks the question drug discovery actually cares about: does this work on chemistry the model has never seen?

The 0.056 gap between those two numbers is the part of the random-split score that was scaffold recognition rather than generalization.

**Most ChEMBL classifier repos report the random-split number and stop.**

---

## Finding the real bottleneck

The first scaffold split scored 0.881, with **inactive recall at 0.40** — on novel chemistry, the model missed 60% of the duds. Active recall was 0.98. It could recognize hits; it couldn't reject misses.

The cause was in the data. ChEMBL is built from published papers, and nobody publishes a paper full of compounds that did nothing. The initial labels came out **8,051 active vs 930 inactive — 90/10**.

### Recovering discarded data

My cleaning script dropped every row where `standard_relation != "="`, reasoning that `>10000 nM` is a non-measurement.

That was wrong. The inactive threshold *is* IC50 ≥ 10,000 nM, so a row reading `>10000` satisfies it with certainty — the exact value is unnecessary. A chemist writing `>10000 nM` isn't saying "I don't know," they're saying "I tested this and it definitively failed." Those rows were exactly the data the model was starving for.

Rescuing them in both directions (`>10000` → inactive, `<1000` → active, so as not to selectively keep only data that flatters the result):

| | Before | After |
|---|---|---|
| Molecules | 8,981 | **11,260** |
| Active | 8,051 | 8,936 |
| **Inactive** | **930** | **2,324** |
| % active | 89.6% | 79.4% |

**2.5x more inactive training data**, and the effect was exactly where predicted:

| | Before rescue | After rescue |
|---|---|---|
| Scaffold ROC-AUC | 0.881 | **0.917** |
| Scaffold inactive recall | 0.40 | **0.56** |
| Scaffold inactive precision | 0.62 | 0.75 |
| **Random − scaffold gap** | **0.092** | **0.056** |

The generalization gap narrowed by ~40%: the model now leans less on family recognition and more on structure-activity relationships.

### A wrinkle worth noting

The rescue pulled near-equal raw rows — 2,219 censored inactives, 2,149 censored actives — but after deduplication the actives gained only 885 unique molecules while inactives gained 1,394.

That asymmetry *is* publication bias, visible structurally. A potent compound gets measured exactly, by many labs, so its censored readings were redundant with data already present. A dud gets one `>10000` reading and nobody revisits it.

Two examples of what came back:
- **L-DOPA** — a real non-kinase drug, the kind of true negative the set entirely lacked
- **SB-203580** — a p38 MAP kinase inhibitor that does *not* hit EGFR. A hard negative, teaching the model that "looks like a kinase inhibitor" isn't sufficient.

---

## Sanity check against known compounds

| Compound | Before rescue | After rescue | Expected |
|---|---|---|---|
| Gefitinib (approved EGFR inhibitor) | 0.995 | 0.995 | High |
| Caffeine (no EGFR activity) | 0.390 | **0.295** | Low |

Gefitinib at 0.995 means 199 of 200 trees voted active, on a drug never seen in training.

Caffeine was the more informative case. At 0.390 it was ranked correctly but far too high — caffeine is a purine, and many kinase inhibitors use nitrogen heterocycles that mimic ATP, so it genuinely shares substructure with the actives. The deeper problem was that every "inactive" in training was a *failed drug candidate*, not a true negative, so the model had never seen an ordinary non-drug molecule and had no basis for rejecting one. Adding compounds like L-DOPA pushed it to 0.295 without costing any confidence on gefitinib.

---

## Method

**Ligand-based**, not structure-based. The model never sees the EGFR protein — no docking, no 3D coordinates, no binding simulation. It learns from molecular structure alone, on the assumption that similar structures have similar activity.

**Pipeline:**

| Script | Does |
|---|---|
| `01_fetch_data.py` | Confirms CHEMBL203 = human EGFR, pulls all IC50 records |
| `02_clean_data.py` | Relation/unit filters incl. censored rescue, pIC50 conversion, dedupe, label |
| `03_features.py` | Morgan fingerprints (radius 2, 2048 bits ≈ ECFP4) |
| `04_train.py` | Random forest, 200 trees, `class_weight="balanced"`, random split |
| `05_predict.py` | Score a new SMILES string |
| `06_scaffold_split.py` | Same model, Bemis-Murcko scaffold split |

**Key cleaning decisions:**
- **pIC50 = 9 − log10(IC50 nM)** — raw IC50 spans ~6 orders of magnitude bunched at the low end; the log spreads it to ~4–10 and the flip makes higher = more potent
- **Median** across duplicate measurements — the same compound is tested by many labs with varying results, and median resists assay outliers better than mean
- **Activity cliff** — active at pIC50 ≥ 6, inactive at ≤ 5, **ambiguous middle discarded**. A compound at 5,000 nM is genuinely unclear, and forcing a label teaches noise.

**Why not accuracy.** On a 90/10 dataset, answering "active" every time scores 90%. Even now it would score 79%. Accuracy also *fell* (0.93 → 0.85) when the model got better, because the rescue made the test set harder — 526 inactives instead of 154. Anyone tracking accuracy as the headline would have concluded the improvement made things worse.

**Dataset:** 11,260 molecules → 4,018 distinct Bemis-Murcko scaffolds (~2.8 compounds per core).

---

## Limitations

- **44% of inactives on novel scaffolds are still waved through as active** (recall 0.56). Better than 60%, not good.
- **Narrow applicability domain.** Improved but not solved — caffeine's 0.295 is still higher than a compound with zero EGFR activity deserves. An out-of-domain check is the next planned addition.
- **Active performance fell slightly** (recall 0.98 → 0.94, precision 0.95 → 0.88). With more inactives the model is willing to say no, so it says no to some real actives. A real tradeoff.
- **No training signal between 1,000 and 10,000 nM** — the discarded middle.
- **Fingerprint hashing collisions.** There are far more than 2,048 possible substructures, so unrelated ones can share a bit.
- **Not a controlled A/B.** The before/after comparison changed the dataset, so both train and test differ between conditions. The direction is unambiguous and the new test set is strictly more demanding, but it isn't a clean controlled experiment.
- **No cross-validation.** All numbers come from single splits, so their stability is unmeasured.

## Considered and rejected

**3D descriptors and docking.** A flexible molecule has hundreds of conformers and the bioactive one is unknown without a crystal structure, so generated 3D coordinates often add more noise than signal. ECFP4 is also an unusually strong baseline — 2D fingerprints match or beat 3D descriptors for ligand-based activity prediction more often than not, since connectivity already constrains geometry. Docking is reasonable at pose prediction but poor at ranking affinity.

---

## Reproducing

```bash
git clone https://github.com/lc23-1067-khagendra/egfr-classifier
cd egfr-classifier
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy rdkit scikit-learn chembl_webresource_client

python src/02_clean_data.py      # raw CSV is committed; skip 01 unless refetching
python src/03_features.py
python src/04_train.py
python src/06_scaffold_split.py
```

`data/egfr_raw.csv` is committed deliberately — ChEMBL updates over time, so the exact pull is what lets these numbers be reproduced. Derived artifacts (`.npy`, `.pkl`) are gitignored and regenerate in about two minutes.

**Data source:** [ChEMBL](https://www.ebi.ac.uk/chembl/) target CHEMBL203 (human EGFR), IC50 endpoint.
