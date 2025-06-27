# predict_autoencoder.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Step 1: Load saved model and scaler
model = load_model("models/autoencoder_model.h5")
scaler = joblib.load("models/scaler.pkl")

# Step 2: Load new input data
new_data = pd.read_csv("data/new_input.csv")

# Step 3: Preprocess new data
X_new_scaled = scaler.transform(new_data)

# Step 4: Predict using autoencoder
X_pred = model.predict(X_new_scaled)
mse = np.mean(np.power(X_new_scaled - X_pred, 2), axis=1)

# Step 5: Set threshold (same as in training)
threshold = np.percentile(mse, 95)

# Step 6: Classify: 0 = normal, 1 = anomaly
y_pred = (mse > threshold).astype(int)

# Step 7: Print results
print("\nAnomaly Prediction Results:")
for i, val in enumerate(y_pred):
    status = "Anomaly" if val == 1 else "Normal"
    print(f"Row {i+1}: {status} (Reconstruction error: {mse[i]:.6f})")
