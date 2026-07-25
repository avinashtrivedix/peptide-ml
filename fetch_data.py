import os
import itertools
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

print("genrating 525- smaple peptide for pipeline integration test")

# 20 standard amino acid codes
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


# Generate Combinations: 400 dipeptides + 125 tripeptides
dipeptides = [''.join(p) for p in itertools.product(AMINO_ACIDS, repeat = 2)]
tripeptides = [''.join(p) for p in itertools.product(AMINO_ACIDS[:5], repeat = 3)]
all_peptides = dipeptides + tripeptides

dataset = []
for seq in all_peptides:
    mol = Chem.MolFromFASTA(seq)
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        h_donors = Descriptors.NumHDonors(mol)

        # SYnthetic Proxy target for testing the code flow
        raw_score = 1.0 - (mw / 1000.0) + (logp * 0.15) - (tpsa / 500.0)
        normalized_score = max(0.05, min(0.98, round(raw_score, 4)))

        dataset.append({
            "sequence" : seq,
            "molecular_weight" : round(mw, 2),
            "logp" : round(logp, 2),
            "tpsa" : round(tpsa, 2),
            "h_donors": h_donors,
            "absorption_score" : normalized_score
        })


df = pd.DataFrame(dataset)

# save cleanly to data/peptides.csv
os.makedirs("data", exist_ok=True)
df.to_csv("data/peptides.csv", index = False)

print(f"success: generated {len(df)} samples and saved to data/peptides.csv")