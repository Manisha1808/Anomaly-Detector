from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('models/autoencoder_model.h5')
scaler = joblib.load('models/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename.endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            X_scaled = scaler.transform(df)
            preds = model.predict(X_scaled)
            mse = np.mean(np.power(X_scaled - preds, 2), axis=1)
            threshold = np.percentile(mse, 95)
            errors = mse.tolist()
            results = [(i+1, "Anomaly" if e > threshold else "Normal", round(e, 6)) for i, e in enumerate(errors)]

            return render_template("index.html", predictions=results, 
                       errors=errors, 
                       threshold=threshold)


    return render_template("index.html", predictions=None)
if __name__ == '__main__':
    app.run(debug=True)
