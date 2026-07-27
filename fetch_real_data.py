import os
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

print("📥 Downloading Real Experimental Bioactive Peptide Dataset...")

# Benchmark FASTA repositories for bioactive peptides
URL_POS = "https://raw.githubusercontent.com/dataprofessor/AMP/main/train_po.fasta"
URL_NEG = "https://raw.githubusercontent.com/dataprofessor/AMP/main/train_ne.fasta"


AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


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

def process_sequences(seq_list, label):
    for seq in seq_list:
        mol = Chem.MolFromFASTA(seq)
        if mol:
            seq_len = len(seq)
            # base RDkit features
            record = {
                "sequence": seq,
                "length": seq_len,
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "h_donors": Descriptors.NumHDonors(mol),
                "h_acceptors": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "is_bioactive": label
            }
        
            # Extract 20 Amino Acid Composition (AAC) Features
            for aa in AMINO_ACIDS:
                record[f"aac_{aa}"] = round(seq.count(aa)/ seq_len,4)
                #at it's core amino acid composition is just basic arithmetic | AAC = how many times a specificamino acid appears / Total length of the petide sequence

            records.append(record)

process_sequences(pos_seqs, 1)
process_sequences(neg_seqs, 0)



df = pd.DataFrame(records)

# Save to data/real_peptides.csv
os.makedirs("data", exist_ok=True)
df.to_csv("data/real_peptides.csv", index=False)

print(f"✅ SUCCESS: Saved {len(df)} real experimental samples to data/real_peptides.csv!")
print("\nDataset Class Distribution:")
print(df['is_bioactive'].value_counts())