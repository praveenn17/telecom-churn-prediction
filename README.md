# ChurnPredict — Telecom Customer Churn Prediction System

ML-powered churn prediction with per-customer explainability, batch evaluation, and live retention scenario simulation.

---

## 🌐 Live Demo

- **Web Application:** [https://frontend-eight-alpha-74.vercel.app](https://frontend-eight-alpha-74.vercel.app)
- **API Documentation (Swagger UI):** [https://telecom-churn-prediction-api-yb0f.onrender.com/docs](https://telecom-churn-prediction-api-yb0f.onrender.com/docs)

> **Note:** The backend is hosted on Render's free tier and spins down after ~15 minutes of inactivity. The first request after idle may take 30–60 seconds to wake up.

---

## 📖 Overview

ChurnPredict predicts whether a telecom customer will churn using supervised machine learning trained on the IBM/Kaggle Telco Customer Churn dataset (7,043 customers, 19 features). 

Going beyond a standard prediction demo, the system provides both global and per-prediction SHAP explainability, batch CSV evaluation for up to 500 customer records at once, and an interactive What-If Retention Simulator to test retention strategies in real time.

---

## ✨ Key Features

- **Real-Time Single Prediction:** Instant churn probability calculation with mapped risk levels (*Low*, *Medium*, *High*, *Very High*).
- **Per-Prediction SHAP Explanations:** Dynamic "Why this prediction" breakdown highlighting the top 5 contributing factors and their directional impact (*increases risk* vs. *decreases risk*).
- **Global Feature Importance Dashboard:** Ranks the most influential churn drivers across the entire dataset using mean absolute SHAP values.
- **Batch CSV Evaluation:** Upload and evaluate batches of up to 500 customer records at once, complete with a sortable/filterable results table, skip-and-report validation for invalid rows, and CSV export.
- **What-If Retention Simulator:** Interactive levers (*Tenure*, *Monthly Charges*, *Contract Type*) with debounced re-prediction to model counterfactual retention scenarios live.
- **RESTful API:** Full FastAPI backend featuring strict Pydantic schemas, validation error reporting, and interactive OpenAPI/Swagger documentation.

---

## 📊 Model Performance

Four classification models were evaluated using 5-fold stratified cross-validation and a held-out 80/20 test split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (selected)** | **73.81%** | **50.43%** | **78.34%** | **0.6136** | **0.8417** |
| Decision Tree | 72.68% | 49.00% | 72.19% | 0.5838 | 0.7605 |
| Random Forest | 78.85% | 62.75% | 50.00% | 0.5565 | 0.8238 |
| XGBoost | 76.65% | 55.53% | 60.43% | 0.5787 | 0.8154 |

> **Selection Rationale:** Logistic Regression was selected despite having lower raw accuracy than Random Forest because it achieved the highest **F1 score (0.6136)** and highest **Recall (78.34%)**. For an imbalanced dataset (~27% churn rate), F1 score and recall are the appropriate metrics since failing to detect a customer at risk of churn is significantly costlier than a false alarm. `class_weight='balanced'` was applied to handle class distribution skew.

---

## 🛠️ Tech Stack

- **Machine Learning:** Python 3.13, scikit-learn, XGBoost, SHAP, pandas, NumPy, joblib
- **Backend:** FastAPI, Pydantic v2, Uvicorn (Deployed on Render)
- **Frontend:** React 19, TypeScript, Vite (Deployed on Vercel)
- **Dataset:** IBM / Kaggle Telco Customer Churn (7,043 rows, 19 features)

---

## 🏗️ Architecture

```text
┌────────────────────────────────────────────────────────┐
│               Customer Data (CSV)                      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│              ML Training Pipeline                      │
│   (ColumnTransformer + Preprocessing + Estimator)      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                   model.pkl                            │
│        (Serialized End-to-End Pipeline)                │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               FastAPI Backend                          │
│   - POST /predict        (Single Inference + SHAP)     │
│   - POST /predict-batch  (Batch CSV Inference)         │
│   - GET  /feature-importance                           │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               React + Vite Frontend                    │
│   - Single Customer Evaluation Form                    │
│   - "Why this prediction" SHAP Waterfall               │
│   - What-If Retention Simulator                        │
│   - Batch CSV Processing & Export                      │
└────────────────────────────────────────────────────────┘
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check confirming model and feature schema status |
| `POST` | `/predict` | Single customer prediction returning probability, risk band, and top 5 SHAP factors |
| `POST` | `/predict-batch` | Batch CSV evaluation supporting both downloadable CSV and JSON formats |
| `GET` | `/feature-importance` | Global top-10 SHAP feature importance rankings from model training |

---

## 📁 Project Structure

```text
├── ml/
│   ├── train.py                  # End-to-end preprocessing, training, & SHAP pipeline
│   ├── requirements.txt          # ML training dependencies
│   ├── model.pkl                 # Fitted sklearn pipeline artifact
│   ├── feature_schema.json       # Feature metadata & expected categories
│   ├── feature_importance.json   # Pre-computed top-10 global SHAP drivers
│   └── shap_summary.png          # Generated SHAP summary plot
├── backend/
│   ├── main.py                   # FastAPI server, CORS, & prediction endpoints
│   ├── schemas.py                # Pydantic v2 request & response models
│   └── requirements.txt          # Backend deployment dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChurnForm.tsx              # Input form with dynamic add-on rules
│   │   │   ├── ResultPanel.tsx            # Probability gauge & SHAP waterfall
│   │   │   ├── WhatIfSimulator.tsx        # Interactive retention simulator
│   │   │   ├── BatchPrediction.tsx        # Drag-and-drop batch CSV evaluator
│   │   │   └── FeatureImportancePanel.tsx # Global driver rankings
│   │   ├── api.ts                # API client functions
│   │   ├── types.ts              # TypeScript interfaces mirroring backend schemas
│   │   ├── index.css             # Design tokens & styling
│   │   └── App.tsx               # Root view with tabbed navigation
│   └── package.json
└── render.yaml                   # Render Blueprint configuration
```

---

## 💻 Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/praveenn17/telecom-churn-prediction.git
cd telecom-churn-prediction
```

### 2. ML Pipeline Setup (Optional: to re-train model)
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and train
pip install -r ml/requirements.txt
python ml/train.py
```

### 3. Backend Setup
```bash
# Install backend dependencies
pip install -r backend/requirements.txt

# Start FastAPI dev server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
API docs available at `http://localhost:8000/docs`.

### 4. Frontend Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
Frontend available at `http://localhost:5173`.

---

## 👤 Author

**Praveen**
- GitHub: [github.com/praveenn17](https://github.com/praveenn17)
- Portfolio: [portfolio-vert-xi-79.vercel.app](https://portfolio-vert-xi-79.vercel.app)
