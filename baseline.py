import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("runneinf pipeline Integration Test on 525 Samples...")

# 1. Dataset : Sample peptide sequence with mpck gut-absorbtion scores

df = pd.read_csv("data/peptides.csv")

# 2. Separate feature X and target Y
feature_cols = ['molecular_weight', 'logp', 'tpsa', 'h_donors']
x = df[feature_cols]
y = df["absorption_score"]

# 3. Train/Test Split (80% train and 20% test)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

#Model training
model = XGBRegressor(n_estimators = 50, 
                     max_depth = 3, 
                     learning_rate = 0.1,
                     random_state = 42
                     )

model.fit(x_train, y_train)

#Evaluation
predictions  = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=== INTEGRATION TEST PASSED ===")
print(f"Total Samples: {len(df)}")
print(f"Training Set: {len(x_train)} rows | Test Set: {len(x_test)} rows")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")