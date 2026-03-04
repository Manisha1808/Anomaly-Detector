# 🚨 Anomaly Detector with Streamlit

An interactive anomaly detection web application built with **Isolation Forest and Streamlit** that allows users to detect anomalies in their datasets without writing code.

Users can upload a CSV file, run anomaly detection, visualize results, and download the processed dataset with anomaly labels.

---

## 🚀 Live Demo

🔗 https://anomaly-detector-app.streamlit.app/

---

## ✨ Features

- Upload CSV datasets for anomaly detection  
- Uses **Isolation Forest** for unsupervised anomaly detection  
- Visualizes **anomaly vs normal data distribution**  
- Highlights detected anomalies in the dataset table  
- Download the processed dataset with anomaly labels  
- Fully interactive **Streamlit interface**  
- Deployed on **Streamlit Cloud**

---

## 🖼️ Output Screenshots

### Dataset Upload Interface
*(Add screenshot here)*

![Upload Interface](screenshots/upload.png)

---

### Anomaly Detection Results
*(Add screenshot here)*

![Detection Results](screenshots/results.png)

---

### Visualization of Anomalies
*(Add screenshot here)*

![Visualization](screenshots/chart.png)

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Framework
- Streamlit

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Joblib

---

## 📂 How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Manisha1808/anomaly-detector-streamlit.git
cd anomaly-detector-streamlit
```

---

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## ⚙️ How It Works

### Model Training (`train.py`)

- Trains an **Isolation Forest model** on the dataset
- Scales features using a **StandardScaler**
- Saves the trained model and scaler using **Joblib**

### Inference (`app.py`)

- User uploads a CSV dataset
- Data is scaled using the saved scaler
- Isolation Forest predicts anomalies
- Anomalies are labeled as **-1**
- Results are displayed with visualizations
- Users can download the processed dataset

---

## 🔮 Future Improvements

- Add support for multiple anomaly detection algorithms
- Enable **hyperparameter tuning directly from the UI**
- Add authentication for secure dataset uploads
- Export visualizations for reporting

---

## 👩‍💻 Author

**Manisha Sen**  
Computer Science Engineering Student  
Interested in **Data Science, Machine Learning, and AI-based applications**

---

## 📄 License

This project is open-source and available under the **MIT License**.
