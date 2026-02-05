# Telco Customer Churn Prediction

## Overview

Customer churn is a critical business problem for subscription-based companies such as telecom, SaaS, and streaming platforms. Acquiring a new customer is significantly more expensive than retaining an existing one.

This project implements a **production-grade, end-to-end machine learning system** to:

* Predict customer churn probability
* Identify churn drivers
* Serve predictions via APIs and dashboards (future-ready)

The system is intentionally designed to scale from a **modeling project** to a **deployable ML product**.

---

## System Architecture

📐 **Complete system design and scalability roadmap** is documented in:

👉 **`ARCHITECTURE.md`**

It covers:

* End-to-end ML pipeline design
* Separation of training, evaluation, and inference
* FastAPI inference service integration
* Streamlit dashboard integration
* Uplift modeling roadmap

---

## Project Structure

---

customer-churn-prediction/
├── ARCHITECTURE.md        # System design & future roadmap
├── README.md
├── src/
│   ├── data_preprocessing.py   # Data cleaning & splitting
│   ├── feature_engineering.py  # Feature pipelines
│   ├── train.py                # Model training
│   ├── evaluate.py             # Metrics & evaluation
│   └── predict.py              # Centralized inference logic
├── api/                        # FastAPI layer
├── dashboard/                  # Streamlit UI
├── uplift/                     # Uplift modeling
├── models/                # Trained model artifacts
├── reports/               # Evaluation reports & analysis
├── requirements.txt
├── main.py                # Project entry point
└── artifacts

---

## Machine Learning Workflow

1. **Data Preprocessing**
   Raw data is cleaned, validated, and split into training and test sets.

2. **Feature Engineering**
   Numerical scaling and categorical encoding using reusable pipelines.

3. **Model Training**
   Supervised ML models trained with reproducibility and persistence.

4. **Evaluation**
   ROC-AUC, classification metrics, and business-aligned KPIs.

5. **Inference**
   Single-source prediction logic reusable across batch, API, and UI.

---

## Current Capabilities

* Binary churn prediction
* Feature pipeline with enforced feature order
* Reproducible training & evaluation
* Clean Git history & modular codebase

---

## Planned Enhancements

### 🔌 FastAPI Inference Service

* REST endpoints for real-time predictions
* Model versioning support

### 📊 Streamlit Dashboard

* Churn probability visualization
* Feature-level insights
* Business-friendly UI

### 📈 Uplift Modeling

* Treatment vs control modeling
* Qini / AUUC evaluation
* Targeted retention strategies

---

## How to Run

```bash
# Train model
python src/train.py

# Evaluate model
python src/evaluate.py
```

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Joblib
* (Planned) FastAPI, Streamlit, MLflow

---

## Author & Notes

**Priya Choure**
Data Science & Artificial Intelligence Practitioner

This project is structured using **industry best practices** for ML systems, focusing on maintainability, scalability, and real-world deployment readiness.
