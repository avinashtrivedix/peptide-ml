# Peptide-ML: Benchmarking Representation Paradigms & Feature Fusion for Biological Sequence Classification

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-2DBA4E?style=flat)](https://xgboost.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/Transformers-ESM--2-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/facebook/esm2_t6_8M_UR50D)

An empirical machine learning benchmark evaluating three biological sequence representation paradigms on N = 3,058 validated peptide sequences: **Tabular Biophysical Descriptors**, **2D Geometric Graph Neural Networks (GCN/GAT)**, and **Pre-trained Protein Language Transformers (Meta ESM-2)**.

---

## Executive Summary

Predicting functional activity from 1D amino acid sequence text accelerates early-stage therapeutic discovery by digitally filtering non-functional candidates before lab synthesis. This repository evaluates representation trade-offs across model capacity and sample scarcity (N = 3,058).

### Key Findings
* **Graph Neural Network Collapse:** Training GCN and Multi-Head GAT models from scratch on small datasets (N = 3,058) yields severe over-fitting (60.44%–65.66% accuracy) due to sample scarcity.
* **Hybrid Feature Fusion Superiority:** Fusing explicit global biophysical invariants (LogP, formal charge, amino acid ratios) with mean-pooled ESM-2 transformer embeddings (347-D) yields our top-performing model (**75.31% Val Acc / 0.8109 ROC-AUC**), outperforming pure graph deep learning by **+14.87%**.
* **TreeSHAP Interpretability:** Game-theoretic feature attribution validates that deep transformer representations dominate decision boundaries while explicit negative charge penalties (`AAC_D`) drive structural activity constraints.

## System Architecture
```text
Raw Sequence ("KKLFKKILKY...")
    │
    ├──► Tier 1: Tabular Descriptors (27-D) ──┐
    │                                         │
    ├──► Tier 2: 2D Atom Graphs (8-D) ────────┼──► XGBoost Classifier (5-Fold CV)
    │                                         │
    └──► Tier 3: ESM-2 Embeddings (320-D) ────┘
```

## Benchmark Results

All architectures were evaluated using **5-Fold Stratified Cross-Validation**.

| Tier | Paradigm / Architecture | Input Space (D) | Mean Val Acc (%) | Mean Val ROC-AUC | Peak Fold ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Tier 1** | XGBoost Baseline (RDKit + AAC) | 27 | 74.51% ± 1.98% | 0.8149 ± 0.0182 | 0.8350 |
| **Tier 2A** | Basic GCN (Atom Graph) | 5 | 65.47% ± 2.11% | 0.6869 ± 0.0210 | 0.7012 |
| **Tier 2A** | Enriched GCN (Node-Featured) | 8 | 65.66% ± 1.89% | 0.6978 ± 0.0195 | 0.7210 |
| **Tier 2B** | Multi-Head GAT (Attention) | 8 | 60.44% ± 3.12% | 0.6375 ± 0.0241 | 0.6810 |
| **Tier 3** | ESM-2 (8M) Alone | 320 | 74.52% ± 1.65% | 0.8075 ± 0.0191 | 0.8308 |
| **Tier 3** | **Hybrid Fusion (8M ESM-2 + Bio)** | **347** | **75.31% ± 2.16%** | **0.8109 ± 0.0201** | **0.8322** |
| **Tier 3** | Hybrid Fusion (35M ESM-2 + Bio) | 507 | 74.23% ± 2.37% | 0.8122 ± 0.0229 | **0.8419** |


## Interpretability & Infrastructure

* **TreeSHAP Attribution:** Deep transformer dimensions (`ESM2_Dim_75`) serve as primary split nodes in decision trees, while Aspartic Acid proportion (`AAC_D`) penalizes activity due to electrostatic repulsion.
* **Apple Silicon Concurrency Lock:** Resolved macOS OpenMP/PyTorch thread collisions (`Segmentation fault: 11`) by setting `OMP_NUM_THREADS=1` prior to model execution.

---

## Quick Start & Execution

```bash
# Clone repository
git clone [https://github.com/your-username/peptide-ml.git](https://github.com/your-username/peptide-ml.git)
cd peptide-ml

# Install dependencies
pip install -r requirements.txt

# Run Tier 3 Hybrid Pipeline
python tier3_hybrid_pipeline.py

# Run SHAP Interpretability Analysis
python tier3_shap_analysis.py
```

For complete theoretical formulations and full empirical benchmarks, see TECHNICAL_REPORT.md.