import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import joblib
import os

# Step 1: Load data
df = pd.read_csv("data/cleaned_kdd.csv")
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)

# Step 2: Prepare features and target
X = df.drop("target", axis=1).values
y = df["target"].values

# Step 3: Normalize input
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# Step 4: Define Autoencoder model
input_dim = X_scaled.shape[1]
encoding_dim = 14  # compression layer
hidden_dim = int(encoding_dim / 2)

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train only on normal data (unsupervised)
print("✅ Normal samples (y==0):", np.sum(y == 0))
print("⚠️  Anomalies (y==1):", np.sum(y == 1))

if np.sum(y == 0) == 0:
    print("⚠️ No normal samples found. Training on full dataset (not ideal).")
    X_train = X_scaled
else:
    X_train = X_scaled[y == 0]

autoencoder.fit(X_train, X_train,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_split=0.1 if len(X_train) > 10 else 0,  # Avoid error if too small
                verbose=1)

# Step 6: Get reconstruction error for full dataset
X_pred = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

# Step 7: Define threshold and predict
threshold = np.percentile(mse[y == 0], 95) if np.sum(y == 0) > 0 else np.percentile(mse, 95)
y_pred = (mse > threshold).astype(int)

# Step 8: Evaluate
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\n📈 Classification Report:")
print(classification_report(y, y_pred))

# Step 9: Save the model
autoencoder.save("models/autoencoder_model.h5")
print("\n✅ Autoencoder model saved to models/autoencoder_model.h5")
