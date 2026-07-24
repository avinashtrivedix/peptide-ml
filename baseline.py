import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Starting peptide Machine Learning Pipeline...")

# 1. Dataset : Sample peptide sequence with mpck gut-absorbtion scores

data = [
    {"seq": "LL", "absorption": 0.85},
    {"seq": "LAG", "absorption": 0.92},
    {"seq": "VIL", "absorption": 0.78},
    {"seq": "GDF", "absorption": 0.45},
    {"seq": "FLL", "absorption": 0.88},
    {"seq": "AAA", "absorption": 0.30},
    {"seq": "PHE", "absorption": 0.65},
    {"seq": "VAL", "absorption": 0.72}
]

df = pd.DataFrame(data)


# Feature Extraction engine using RDKit

def extract_chemical_feature(seq):
    mol = Chem.MolFromFASTA(seq)
    if not mol:
        return [0,0,0,0]
    return [
        Descriptors.MolWt(mol),         # Molecular Weight
        Descriptors.MolLogP(mol),       # Lipophilicity(fat solubility)
        Descriptors.TPSA(mol),          # Surface Area
        Descriptors.NumHDonors(mol)     # Hydrogen Bond Donor
    ]


# convert fast test sting -> 4d numerical vectors
features = df['seq'].apply(extract_chemical_feature).tolist()
X = pd.DataFrame(features, columns= ['MolWt', 'LogP', 'TPSA', 'H_Donors'])
y = df['absorption']

#3 train/test split & XGboost Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state= 42)
model = XGBRegressor(n_estimators =10,max_depth = 2, random_state = 42 )
model.fit(X_train, y_train)

#model_evaluation
prediction  = model.predict(X_test)
mse = mean_squared_error(y_test, prediction)

print("=== PIPELINE EXECUTION SUCCESSFUL ===")
print(f"Extracted Feature Matrix Shape: {X.shape}")
print(f"Baseline XGBoost Mean Squared Error: {mse:.4f}")
print("\nSample Prediction vs Actual:")
print(f"Predicted: {prediction[0]:.2f} | Actual: {y_test.iloc[0]:.2f}")

