import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

print("runneinf pipeline Integration Test on 525 Samples...")

# 1. Dataset : Sample peptide sequence with mpck gut-absorbtion scores

df = pd.read_csv("data/real_peptides.csv")

# 2. Separate feature X and target Y
feature_cols = ['length', 'molecular_weight', 'logp', 'tpsa', 'h_donors', 'h_acceptors', 'rotatable_bonds']
x = df[feature_cols]
y = df['is_bioactive']

# 3. Train/Test Split (80% train and 20% test)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

#Model training
model = XGBClassifier(n_estimators = 50, 
                     max_depth = 3, 
                     learning_rate = 0.1,
                     random_state = 42
                     )

model.fit(X_train, y_train)

#Evaluation
predictions  = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=== REAL EXPERIMENTAL MODEL PERFORMANCE ===")
print(f"Total Samples: {len(df)} | Test Set: {len(X_test)}")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"ROC-AUC Score: {auc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, predictions))

# 6. Feature Importance Ranking
print("=== FEATURE IMPORTANCE RANKING ===")
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance.to_string(index=False))