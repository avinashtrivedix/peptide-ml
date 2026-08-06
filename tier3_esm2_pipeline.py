import torch
import requests
import numpy as np
from transformers import AutoTokenizer, EsmModel
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

print("🚀 Running Tier 3: Meta ESM-2 Transformer Pipeline...\n")

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

# 2. Load Meta's Pre-trained ESM-2 Model & Tokenizer
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
print(f"Loading pre-trained transformer: {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME)
model.eval()  # Freeze weights (using strictly as a feature extractor)

# 3. Batch Extraction of 320-Dimensional Sequence Embeddings
print("Extracting 320-D ESM-2 embeddings (Mean-Pooled over residues)...")

batch_size = 64
embeddings_list = []

with torch.no_grad():
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        
        # Tokenize sequence strings
        inputs = tokenizer(batch_seqs, padding=True, truncation=True, return_tensors="pt")
        
        # Forward pass through 6-layer 8M Transformer
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # Shape: [batch, seq_len, 320]
        
        # Mean pooling over amino acid tokens (excluding special start/end tokens)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        seq_embeddings = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        embeddings_list.append(seq_embeddings.numpy())

X = np.vstack(embeddings_list)
print(f"✅ Extracted ESM-2 Feature Matrix: Shape {X.shape}\n")

# 4. 5-Fold Stratified Cross-Validation using XGBoost on ESM-2 Vectors
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_accs, val_aucs = [], []

print("Training XGBoost Classifier on 320-D ESM-2 Embeddings:")
print("-" * 55)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, labels), 1):
    X_train, X_val = X[train_idx], X[val_idx]
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

print("-" * 55)
print(f"Mean ESM-2 Val Accuracy: {np.mean(val_accs)*100:.2f}% ± {np.std(val_accs)*100:.2f}%")
print(f"Mean ESM-2 Val ROC-AUC:  {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")