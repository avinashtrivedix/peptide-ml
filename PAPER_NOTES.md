# 📝 Research Technical Ledger & Experimental Log

> **Project Title:** *In Silico* Screening of Antimicrobial Bioactive Peptides Using Biophysical Descriptors, Amino Acid Composition, and Gradient-Boosted Decision Trees  
> **Repository:** `peptide-ml`  
> **Current Pipeline Status:** Tier 1 Baseline Validated & Benchmark Reached (0.8149 ROC-AUC)

---

## 1. Research Scope & Problem Formulation

### 1.1 The Machine Learning & Biological Objective
The primary bottleneck in therapeutic peptide design and oral bioavailability research is the cost and latency of wet-lab assays. 
* **Core Task:** Develop an end-to-end computational pipeline that converts raw unstructured amino acid sequence strings ($X$) into high-dimensional biophysical vectors to predict bioactive function ($y$).
* **Target Transferability:** While currently benchmarked on binary antimicrobial bioactivity ($y \in \{0, 1\}$), the engineered feature extraction engine ($X$) is mathematically identical for continuous human intestinal absorption ($P_{app}$ Caco-2 permeability regression).

### 1.2 Biological Justification for Short Peptide Focus
* Intact, long-chain proteins (>50–100+ amino acids) cannot cross the human intestinal brush border intact due to tight junctions and size constraints.
* Digestion reduces proteins into **dipeptides**, **tripeptides**, and **oligopeptides** (10–35 amino acids), which are actively transported via intestinal transporters like **PepT1**. Thus, modeling short peptide sequences is biologically accurate for both bioactivity and gut absorption.

---

## 2. Dataset Specification

* **Source:** Crated from benchmark FASTA repositories of lab-tested experimental antimicrobial peptides (AMP Data Professor).
* **Total Sample Count ($N$):** $3,058$ laboratory-tested sequences.
* **Class Distribution:** Perfectly balanced 50/50 split ($1,529$ Active [`1`] / $1,529$ Inactive [`0`]).
* **Sequence Length Range:** $10$ to $35$ amino acid residues.

---

## 3. Mathematical Feature Engineering Formulations

### 3.1 Global Physicochemical Descriptors (`RDKit`) — 7 Features
Converts 2D chemical graph structures parsed from FASTA sequences into physical descriptors:
1. **Molecular Weight ($MolWt$):** Total atomic mass.
2. **$LogP$ (Lipophilicity):** Octanol-water partition coefficient measuring fat vs. water solubility.
3. **Topological Polar Surface Area ($TPSA$):** Sum of polar atom surfaces, critical for membrane permeability.
4. **Hydrogen Bond Donors ($H_{donors}$):** Count of $N-H$ and $O-H$ bonds.
5. **Hydrogen Bond Acceptors ($H_{acceptors}$):** Count of electronegative $N$ and $O$ atoms.
6. **Rotatable Bonds:** Measure of molecular conformational flexibility.
7. **Sequence Length:** Total amino acid residue count.

### 3.2 Sequence Composition Features (AAC) — 20 Features
Calculates the relative proportion of each standard amino acid residue ($i \in \{A, C, D, \dots, Y\}$) across the sequence length ($L$):

$$\text{AAC}_i = \frac{\text{Count of Amino Acid}_i}{L}$$

---

## 4. Experimental Iterations & Benchmark Comparison

| Iteration | Feature Vector | Dimensions ($D$) | Mean Val Accuracy | Mean Val ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Iteration 1** | RDKit Descriptors Only | 7 | 70.26% | 0.7354 | Baseline set |
| **Iteration 2** | **RDKit + AAC** | **27** | **74.51% ($\pm 1.1\%$)** | **0.8149 ($\pm 0.008$)** | **OPTIMAL TIER 1 BENCHMARK** |
| **Iteration 3** | RDKit + AAC + DPC | 427 | 74.59% | 0.8083 | **REJECTED (High-dimensional noise)** |

---

## 5. Detailed Experimental Breakdowns

### Iteration 1: Physicochemical Baseline
* **Features:** 7 RDKit Descriptors.
* **Model:** `XGBClassifier` ($N_{est}=100$, $\text{max\_depth}=4$, $\eta=0.05$).
* **Results:** $70.26\%$ Accuracy | $0.7354$ ROC-AUC.
* **Finding:** Global stats set a strong foundation but suffer from "permutation blindness" (cannot distinguish sequence composition or ordering).

### Iteration 2: RDKit + Amino Acid Composition (The Breakthrough)
* **Features:** 27 Dimensions (7 RDKit + 20 AAC).
* **Validation Protocol:** 5-Fold Stratified Cross-Validation (`StratifiedKFold`, $K=5$, `random_state=42`).
* **Results:** 
  * **Mean Accuracy:** $74.51\% \pm 1.1\%$ (Jump of $+4.25\%$)
  * **Mean ROC-AUC:** $\mathbf{0.8149 \pm 0.008}$ (Jump of $+0.0795$, breaking the critical $0.80$ AUC threshold)
* **Artifact Generated:** Publication-grade figure exported as `roc_curve.png` (300 DPI) displaying individual fold trajectories, bold mean curve, and standard deviation envelope.

### Iteration 3: Dipeptide Composition (DPC) Expansion [REJECTED]
* **Features:** 427 Dimensions (7 RDKit + 20 AAC + 400 DPC adjacent pair frequencies).
* **Validation:** Regularized XGBoost (`colsample_bytree=0.7`, `reg_alpha=0.1`, 5-Fold CV).
* **Results:** Train Accuracy = $82.57\%$ | Validation Accuracy = $74.59\%$ | Validation ROC-AUC = $0.8083$.
* **Overfitting Gap:** $7.98\%$ gap between training and validation scores.
* **Scientific Verdict:** **Rejected.** Adding 400 sparse dipeptide features increased feature space by 15x, inducing high-dimensional noise (Curse of Dimensionality) and degrading validation ROC-AUC. Occam's Razor confirms the 27-dimensional feature matrix is more parsimonious and effective.

---

## 6. Biochemical Mechanism Analysis (Feature Importance)

Tree split decisions in the optimal 27-feature model match established biophysical membrane interaction mechanisms:

```text
Feature      Importance    Biophysical Mechanism
----------------------------------------------------------------------------------------------------
aac_D        14.42%        Aspartic Acid (Acidic, Negatively Charged)  ──> Electrostatic repulsion from 
aac_E         8.62%        Glutamic Acid (Acidic, Negatively Charged)  ──> bacterial wall (Predicts 0)
aac_K         8.14%        Lysine (Basic, Positively Charged)         ──> Magnet for bacterial wall (Predicts 1)
aac_C         4.59%        Cysteine (Thiol side group)                ──> Disulfide bonds (3D hairpin stability)
aac_Q         4.56%        Glutamine (Polar Uncharged)                ──> Binding and solubility
aac_M         4.49%        Methionine (Hydrophobic)                   ──> Anchors into lipid bilayer


### Tier 2A: Geometric Deep Learning (GCN)
- **Basic GCN (5-D Features):** 65.47% Mean Val Acc | 0.6869 Mean Val ROC-AUC
- **Enriched GCN (8-D Features):** 65.66% Mean Val Acc | 0.6978 Mean Val ROC-AUC
- **Finding:** Adding rich atomic features (valence, hybridization, mass) slightly improved GCN performance (+0.0109 AUC), but isotropic message passing still trails tabular tree ensembles due to data scarcity.