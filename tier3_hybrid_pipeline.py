import os
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent C++ thread clashing

import torch
import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import AutoTokenizer, EsmModel
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

print("🚀 Running Tier 3 (Hybrid): ESM-2 + Biophysical Feature Fusion...\n")

# 1. Fetch Raw Sequences
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

pos_seqs = parse_fasta(URL_POS)
neg_seqs = parse_fasta(URL_NEG)

sequences = pos_seqs + neg_seqs
labels = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))

print(f"Loaded {len(sequences)} sequence strings.")

# 2. Extract 27-D Biophysical Features (Tier 1 Logic)
print("Extracting 27-D Biophysical Descriptors (RDKit + AAC)...")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def get_biophysical_features(seq):
    mol = Chem.MolFromFASTA(seq)
    if mol is None:
        return None
    
    # RDKit Global Descriptors (7 Features)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    charge = Chem.GetFormalCharge(mol)
    
    rdkit_feats = [mw, logp, tpsa, hbd, hba, rotb, charge]
    
    # Amino Acid Composition (20 Features)
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

print(f"Biophysical Matrix Shape: {X_bio.shape}")

# 3. Extract 320-D ESM-2 Embeddings (Tier 3 Logic)
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
print(f"Loading pre-trained transformer: {MODEL_NAME}...")

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
        
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        seq_embeddings = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        embeddings_list.append(seq_embeddings.numpy())

X_esm = np.vstack(embeddings_list)
print(f"ESM-2 Matrix Shape: {X_esm.shape}")

# 4. Feature Concatenation (Hybrid Matrix: 27-D + 320-D = 347-D)
X_hybrid = np.hstack([X_bio, X_esm])
print(f"✅ Extracted Hybrid Feature Matrix: Shape {X_hybrid.shape}\n")

# 5. 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_accs, val_aucs = [], []

print("Training XGBoost Classifier on 347-D Hybrid Feature Vectors:")
print("-" * 60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_hybrid, labels), 1):
    X_train, X_val = X_hybrid[train_idx], X_hybrid[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_train)

    y_probs = clf.predict_proba(X_val)[:, 1]
    y_preds = clf.predict(X_val)

    acc = accuracy_score(y_val, y_preds)
    auc = roc_auc_score(y_val, y_probs)
    val_accs.append(acc)
    val_aucs.append(auc)

    print(f"Fold {fold} | Val Acc: {acc*100:.2f}% | Val ROC-AUC: {auc:.4f}")

print("-" * 60)
print(f"Mean Hybrid Val Accuracy: {np.mean(val_accs)*100:.2f}% ± {np.std(val_accs)*100:.2f}%")
print(f"Mean Hybrid Val ROC-AUC:  {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")