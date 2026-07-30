import torch
import torch.nn as nn
import torch.nn.functional as F
# 1. Fixed PyG DataLoader Import
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
import requests
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

print("🚀 Running Tier 2A (V2): Enriched Graph Convolutional Network (GCN)...\n")

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

def sequence_to_graph(seq, label):
    mol = Chem.MolFromFASTA(seq)
    if mol is None:
        return None
    
    # 2. Fixed Atom Featurizer (Using updated Valence API)
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            int(atom.GetHybridization()),
            atom.GetValence(Chem.ValenceType.IMPLICIT), # 👈 Fixed deprecation
            atom.GetMass() * 0.01
        ])
    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    
    if len(edge_indices) == 0:
        return None
        
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

print("Constructing Enriched Molecular Graphs...")
dataset = []
for seq in pos_seqs:
    g = sequence_to_graph(seq, 1)
    if g: dataset.append(g)
for seq in neg_seqs:
    g = sequence_to_graph(seq, 0)
    if g: dataset.append(g)

print(f"Constructed {len(dataset)} molecular graphs!\n")

class PeptideGCNEnriched(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(PeptideGCNEnriched, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels // 2, 2)
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

labels = np.array([data.y.item() for data in dataset])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

val_accs, val_aucs = [], []

print("Training Enriched GCN across 5 Folds:")
print("-" * 45)

for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels), 1):
    train_sub = [dataset[i] for i in train_idx]
    val_sub = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

    model = PeptideGCNEnriched(in_channels=8, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(20):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    model.eval()
    y_true, y_probs, y_preds = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)
            
            y_true.extend(batch.y.numpy())
            y_probs.extend(probs.numpy())
            y_preds.extend(preds.numpy())

    acc = accuracy_score(y_true, y_preds)
    auc = roc_auc_score(y_true, y_probs)
    val_accs.append(acc)
    val_aucs.append(auc)

    print(f"Fold {fold} | Val Acc: {acc*100:.2f}% | Val AUC: {auc:.4f}")

print("-" * 45)
print(f"Mean Enriched GCN Val Accuracy: {np.mean(val_accs)*100:.2f}% ± {np.std(val_accs)*100:.2f}%")
print(f"Mean Enriched GCN Val ROC-AUC:  {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")