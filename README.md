# 🧬 Bioactive Peptide Classification & Feature Engineering Pipeline
> **Applied ML Research: In Silico Screening of Bioactive Peptides using Biophysical Descriptors & XGBoost**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Classification-green)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-74.51%25-brightgreen)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.8149-success)

An applied machine learning system designed to screen bioactive antimicrobial peptides (AMPs) *in silico*. This pipeline transforms raw peptide sequence strings into a 27-dimensional biophysical feature vector—combining 2D molecular graph descriptors from `RDKit` with 20-dimensional **Amino Acid Composition (AAC)**—and models non-linear bioactivity using `XGBoost`.

---

## 📊 Experimental Results & Benchmark Progress

We evaluated our feature engineering pipeline across **3,058 laboratory-tested experimental peptide sequences** (1,529 active, 1,529 inactive) pulled from open-source biomedical repositories.

| Iteration | Feature Set | Dimensions | Accuracy | ROC-AUC | Key Improvement |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline** | RDKit Descriptors Only | 7 | 70.26% | 0.7354 | Initial pipeline verification on real lab data |
| **Iteration 1** | **RDKit + AAC Features** | **27** | **74.51%** | **0.8149** | **+4.25% Acc / +0.0795 AUC (Broken 0.80 AUC threshold)** |

---

## 🛠️ Feature Engineering Architecture

To resolve "permutation blindness" and represent biological matter as mathematical vectors, our pipeline extracts two distinct classes of features:

### 1. Global Physicochemical Descriptors (`RDKit`) — 7 Features
- **Molecular Mass & Size:** Molecular Weight ($MolWt$), Sequence Length, Rotatable Bond count.
- **Solubility & Charge:** $LogP$ (lipophilicity/fat solubility), Topological Polar Surface Area ($TPSA$).
- **Hydrogen Bonding:** Number of Hydrogen Bond Donors and Acceptors.

### 2. Sequence Composition Features (AAC) — 20 Features
Calculates the relative percentage frequency of each standard amino acid residue ($0.0$ to $1.0$):
$$\text{AAC}_i = \frac{\text{Count of Amino Acid}_i}{\text{Total Sequence Length}}$$

---

## 🧪 Biochemical Insights (Feature Importance)

The top features driving decision tree splits in `XGBoost` closely match established cell biology mechanisms:

1. **`aac_D` (14.42%) & `aac_E` (8.62%):** Aspartic Acid and Glutamic Acid carry negatively charged side chains. Because bacterial cell membranes are negatively charged, high acidic content causes electrostatic repulsion, serving as the model's strongest indicator of **inactivity (`0`)**.
2. **`aac_K` (8.14%):** Lysine carries a positive charge that acts as an electrostatic magnet toward bacterial membranes, serving as a strong signal for **bioactivity (`1`)**.
3. **`aac_C` (4.59%):** Cysteine forms disulfide bonds crucial for maintaining stable 3D hairpin secondary structures.

---

## 📁 Repository Architecture

```text
peptide-ml/
├── data/
│   ├── peptides.csv           # 525-sample synthetic dataset (Pipeline integration test)
│   └── real_peptides.csv      # 3,058 experimental samples (27 feature dimensions)
├── .gitignore                 # Environment and cache exclusion rules
├── README.md                  # Project research documentation
├── baseline.py                # Main ML pipeline (XGBClassifier + evaluation)
├── fetch_data.py              # Synthetic proxy generator (Integration test)
├── fetch_real_data.py         # Real lab data puller & AAC/RDKit feature extractor
└── requirements.txt           # Environment dependencies