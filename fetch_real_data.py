import os
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

print("📥 Downloading Real Experimental Bioactive Peptide Dataset...")

# Benchmark FASTA repositories for bioactive peptides
URL_POS = "https://raw.githubusercontent.com/dataprofessor/AMP/main/train_po.fasta"
URL_NEG = "https://raw.githubusercontent.com/dataprofessor/AMP/main/train_ne.fasta"

def parse_fasta_from_url(url):
    response = requests.get(url)
    lines = response.text.splitlines()
    sequences = []
    current_seq = ""
    for line in lines:
        if line.startswith(">"):
            if current_seq:
                sequences.append(current_seq)
                current_seq = ""
        else:
            current_seq += line.strip()
    if current_seq:
        sequences.append(current_seq)
    return sequences

# Download active (1) and inactive (0) lab-tested peptide sequences
pos_seqs = parse_fasta_from_url(URL_POS)
neg_seqs = parse_fasta_from_url(URL_NEG)

print(f"Downloaded {len(pos_seqs)} Active Peptides and {len(neg_seqs)} Inactive Peptides.")

records = []

# Process Active Peptides (Label = 1)
for seq in pos_seqs:
    mol = Chem.MolFromFASTA(seq)
    if mol:
        records.append({
            "sequence": seq,
            "length": len(seq),
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "h_donors": Descriptors.NumHDonors(mol),
            "h_acceptors": Descriptors.NumHAcceptors(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "is_bioactive": 1
        })

# Process Inactive Peptides (Label = 0)
for seq in neg_seqs:
    mol = Chem.MolFromFASTA(seq)
    if mol:
        records.append({
            "sequence": seq,
            "length": len(seq),
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "h_donors": Descriptors.NumHDonors(mol),
            "h_acceptors": Descriptors.NumHAcceptors(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "is_bioactive": 0
        })

df = pd.DataFrame(records)

# Save to data/real_peptides.csv
os.makedirs("data", exist_ok=True)
df.to_csv("data/real_peptides.csv", index=False)

print(f"✅ SUCCESS: Saved {len(df)} real experimental samples to data/real_peptides.csv!")
print("\nDataset Class Distribution:")
print(df['is_bioactive'].value_counts())