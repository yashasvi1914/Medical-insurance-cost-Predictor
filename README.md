# 🏥 Medical Insurance Cost Predictor

> **AI-powered insurance cost estimation using ML regression — FastAPI + HTML/CSS frontend**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---



> Built a full-stack Medical Insurance Cost Prediction system using Python and scikit-learn, training and comparing Linear Regression, Random Forest, and Gradient Boosting models (best R² = 0.94) on 1,338 real-world records. Developed a production-ready REST API with FastAPI exposing a `/predict` endpoint, and a modern HTML/CSS frontend for real-time cost estimation. Deployed end-to-end on Render with model serialization via pickle, complete EDA dashboard with Matplotlib/Seaborn visualizations, and feature importance analysis identifying smoking status as the dominant predictor (81.8% importance).

---

## 🎯 Project Overview

This project predicts annual medical insurance charges for an individual based on their health profile. It covers the full ML lifecycle:

- **Data Exploration** — EDA with distribution plots, correlation heatmaps, boxplots
- **Feature Engineering** — Label encoding, train/test split, standard scaling
- **Model Training** — 3 regression algorithms compared by R² and RMSE
- **API Deployment** — FastAPI serving predictions via POST `/predict`
- **Frontend UI** — Clean dark-themed interface with real-time predictions

---

## 🏗️ Project Structure

```
medical_insurance_project/
│
├── data/
│   └── insurance.csv              # Dataset (1,338 records)
│
├── model/
│   ├── model_bundle.pkl           # Serialized model + encoders
│   └── model_results.json         # Model comparison metrics
│
├── notebooks/
│   └── insurance_eda.ipynb        # Full EDA + training notebook
│
├── app/
│   └── main.py                    # FastAPI application
│
├── templates/
│   ├── index.html                 # Prediction UI
│   └── dashboard.html             # EDA dashboard
│
├── static/
│   ├── eda_charges.png            # Charge distribution chart
│   ├── correlation_heatmap.png    # Feature correlations
│   ├── feature_importance.png     # RF feature importance
│   └── model_comparison.png       # Model R²/RMSE comparison
│
└── requirements.txt
```

---

## 📊 Dataset

| Feature    | Type        | Description                           |
|------------|-------------|---------------------------------------|
| `age`      | Integer     | Age of the insured individual (18–64) |
| `sex`      | Categorical | `male` / `female`                     |
| `bmi`      | Float       | Body Mass Index (15.96 – 53.13)       |
| `children` | Integer     | Number of dependents (0–5)            |
| `smoker`   | Categorical | `yes` / `no`                          |
| `region`   | Categorical | US region: SW / SE / NW / NE          |
| `charges`  | Float       | **Target** — Annual insurance cost    |

---

## 🤖 Model Results

| Model               | R² Score | RMSE      |
|---------------------|----------|-----------|
| Linear Regression   | 0.9008   | $4,833    |
| **Random Forest** ✅| **0.9402** | **$3,753** |
| Gradient Boosting   | 0.9386   | $3,802    |

**Selected model: Random Forest Regressor** (100 estimators, random_state=42)

### Feature Importance
| Feature | Importance |
|---------|------------|
| Smoker  | **81.8%**  |
| Age     | 12.2%      |
| BMI     | 3.3%       |
| Region  | 1.0%       |
| Children| 0.9%       |
| Sex     | 0.9%       |

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/medical-insurance-predictor.git
cd medical-insurance-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (if needed)
```bash
cd notebooks
jupyter notebook insurance_eda.ipynb
```

### 5. Start the API server
```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open in browser
```
http://localhost:8000          # Prediction UI
http://localhost:8000/dashboard # EDA Dashboard
http://localhost:8000/docs      # Swagger API docs
```

---

## 🌐 API Reference

### `POST /predict`

**Request body:**
```json
{
  "age": 35,
  "sex": "male",
  "bmi": 28.5,
  "children": 2,
  "smoker": "no",
  "region": "southwest"
}
```

**Response:**
```json
{
  "predicted_charge": 12450.32,
  "model_used": "Random Forest",
  "r2_score": 0.9402,
  "rmse": 3752.95,
  "input_summary": { ... }
}
```

### `GET /health`
Returns API health status.

### `GET /model-info`
Returns all model comparison metrics.

---

## 🛠️ Tech Stack

| Layer        | Technology           |
|--------------|----------------------|
| ML/DS        | scikit-learn, pandas, numpy |
| Visualization| matplotlib, seaborn  |
| Backend API  | FastAPI, uvicorn     |
| Frontend     | HTML5, CSS3, Vanilla JS |
| Serialization| pickle               |
| Notebook     | Jupyter              |

---

## 📄 License

MIT License — free to use for educational and commercial projects.

---

*Built as a production-ready Data Science portfolio project.*
