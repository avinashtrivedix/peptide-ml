import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import requests
import shap
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import AutoTokenizer, EsmModel
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

print("🚀 Running Tier 3: SHAP Interpretability Analysis on Hybrid Model...\n")

# 1. Fetch Data
URL_POS = "https://raw.githubusercontent.com/dataprofessor/AMP/main/train_po.fasta"
URL_NEG = "https://raw.githubusercontent.com/dataprofessor/AMP/main/train_ne.fasta"

def parse_fasta(url):
    res = requests.get(url)
    lines = res.text.splitlines()
    seqs, current = [], ""
    for line in lines:
        if line.startswith(">"):
            if current: seqs.append(current); current = ""
        else: current += line.strip()
    if current: seqs.append(current)
    return seqs

pos_seqs, neg_seqs = parse_fasta(URL_POS), parse_fasta(URL_NEG)
sequences = pos_seqs + neg_seqs
labels = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))

# 2. Extract 27-D Biophysical Descriptors with Explicit Feature Names
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
feature_names = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "NetCharge"]
feature_names += [f"AAC_{aa}" for aa in AMINO_ACIDS]
feature_names += [f"ESM2_Dim_{i}" for i in range(320)]

def get_biophysical_features(seq):
    mol = Chem.MolFromFASTA(seq)
    if mol is None: return None
    rdkit_feats = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Chem.GetFormalCharge(mol)
    ]
    seq_len = len(seq)
    aac_feats = [seq.count(aa) / seq_len for aa in AMINO_ACIDS]
    return rdkit_feats + aac_feats

X_bio_list, valid_indices = [], []
for idx, seq in enumerate(sequences):
    feats = get_biophysical_features(seq)
    if feats:
        X_bio_list.append(feats)
        valid_indices.append(idx)

X_bio = np.array(X_bio_list)
sequences = [sequences[i] for i in valid_indices]
labels = labels[valid_indices]

# 3. Extract 320-D ESM-2 Embeddings
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME)
model.eval()

batch_size = 64
embeddings_list = []
with torch.no_grad():
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        inputs = tokenizer(batch_seqs, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        seq_embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        embeddings_list.append(seq_embeddings.numpy())

X_esm = np.vstack(embeddings_list)
X_hybrid = np.hstack([X_bio, X_esm])

# 4. Train XGBoost Model on Train Split
X_train, X_test, y_train, y_test = train_test_split(X_hybrid, labels, test_size=0.2, random_state=42, stratify=labels)

clf = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
clf.fit(X_train, y_train)

# 5. Compute SHAP Values
print("Computing SHAP values using TreeExplainer...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test)

# Assign readable feature names
shap_values.feature_names = feature_names

# 6. Save SHAP Summary Plot
print("Generating and saving SHAP summary plot as 'shap_summary.png'...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=20)
plt.title("Top 20 Most Important Features in Hybrid AMP Classifier", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
print("✅ Successfully saved 'shap_summary.png' in project directory!\n")