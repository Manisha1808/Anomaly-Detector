# train_iforest.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Step 1: Load cleaned data
df = pd.read_csv("data/cleaned_kdd.csv")

# ✅ Drop rows with NaN in 'target' column
df = df.dropna(subset=["target"])

# ✅ If still unsure, force target to int
df["target"] = df["target"].astype(int)

# Step 2: Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

print(f"NaNs in target: {df['target'].isna().sum()}")
print(f"Target class distribution:\n{df['target'].value_counts()}")


# Step 3: Initialize and train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X)

# Step 4: Predict
y_pred = model.predict(X)

# Isolation Forest returns: 1 for normal, -1 for anomaly
# So map it to: 0 for normal, 1 for anomaly
y_pred = [0 if p == 1 else 1 for p in y_pred]

# Step 5: Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Step 6: Save the model
joblib.dump(model, "models/isolation_forest.pkl")
print("\n✅ Model saved to models/isolation_forest.pkl")
