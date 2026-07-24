# Peptide Bioavailability & Gut Absorption Predictor
> **Applied ML Research & Feature Engineering Pipeline**

An applied machine learning system designed to predict the intestinal absorption (bioavailability) of bioactive peptides. This pipeline transforms raw peptide text representations (FASTA/SMILES) into high-dimensional biochemical descriptor vectors using `RDKit` and models non-linear absorption dynamics using gradient-boosted ensembles (`XGBoost`).

---

## 📌 Problem & Motivation
Understanding how efficiently human intestinal transporters (such as PepT1) absorb specific protein and peptide sequences is critical for computational nutrition and targeted peptide therapeutics. Traditional wet-lab testing is slow and expensive. This project establishes an automated, end-to-end Machine Learning pipeline to screen candidate peptides computationally.

---

## 🛠️ Technical Architecture & Pipeline

1. **Chemical Feature Extraction (`RDKit`):**
   - Parses amino acid sequences into molecular graph representations.
   - Extracts key physical-chemical descriptors: Molecular Weight ($MolWt$), Lipophilicity ($LogP$), Topological Polar Surface Area ($TPSA$), and Hydrogen Bond Donors.
2. **Data Preprocessing & Feature Scaling:**
   - Standardizes high-dimensional numeric arrays and handles structural validation.
3. **Predictive Modeling (`XGBoost`):**
   - Trains gradient-boosted decision trees to map molecular feature vectors to quantitative absorption scores.

---

## 📁 Repository Structure
```text
peptide-ml/
├── baseline.py          # End-to-end data extraction & XGBoost baseline script
├── requirements.txt     # Environment dependencies
├── .gitignore           # Ignored system & virtual environment files
└── README.md            # Applied research overview & documentation