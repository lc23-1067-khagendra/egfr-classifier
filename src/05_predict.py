import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

model = joblib.load("data/model.pkl")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def predict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = gen.GetFingerprintAsNumPy(mol).reshape(1, -1)
    prob = model.predict_proba(fp)[0, 1]
    return prob

gefitinib = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
caffeine = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

for name, smi in [("Gefitinib", gefitinib), ("Caffeine", caffeine)]:
    print(f"{name}: {predict(smi):.3f}")