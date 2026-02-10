# 📊 Telco Customer Churn Prediction

## End-to-End Machine Learning System for Retention Intelligence**

An end-to-end **customer churn prediction and analytics system** designed to identify **at-risk customers**, uncover **root churn drivers**, and enable **proactive retention strategies**.

Built on the **IBM Telco Customer Churn dataset**, this project reflects **real-world telecom churn workflows**, from data ingestion and modeling to explainability and deployment readiness.

---

## 🚀 Business Context

Customer churn has a **direct and measurable impact on revenue**.
In subscription businesses, **retaining an existing customer is far cheaper than acquiring a new one**.

### 🎯 Core Business Question

> **Which customers are most likely to churn — and what actions can prevent it?**

### 🎯 Business Objectives

* Predict churn with **high recall** to minimize missed at-risk customers
* Identify **behavioral, service, and contract-based churn drivers**
* Support **targeted, data-driven retention campaigns**

---

## 🧾 Dataset Overview

Each record represents one customer, with demographic, service usage, contract, and billing information.

### 👥 Demographics

* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`

### 🔧 Services

* `PhoneService`, `MultipleLines`
* `InternetService`
* `OnlineSecurity`, `OnlineBackup`
* `DeviceProtection`, `TechSupport`
* `StreamingTV`, `StreamingMovies`

### 💳 Account & Billing

* `tenure`
* `Contract`
* `PaymentMethod`
* `PaperlessBilling`
* `MonthlyCharges`
* `TotalCharges`

### 🎯 Target

* **`Churn`** (Yes / No)

---

## 🗂️ Project Structure (Production-Ready)

```text
customer-churn-prediction/
│
├── artifacts/                  # Model evaluation outputs
│   ├── evaluation_metrics.json
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── decile_lift.csv
│   └── roc_auc.txt
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned & engineered datasets
│
├── models/                     # Trained models & pipelines
│   ├── churn_pipeline.joblib
│   ├── churn_model.joblib
│   ├── feature_columns.joblib
│   └── training_metrics.json
│
├── notebooks/                  # Exploratory & modeling notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── src/                        # Production scripts
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── reports/figures/             # EDA & insights visuals
│
├── api/                        # FastAPI inference service
│
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Modeling:**

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * XGBoost
  * LightGBM
  * CatBoost
* **Imbalance Handling:** Class weighting / SMOTE
* **Explainability:** SHAP
* **Deployment:** FastAPI
* **Version Control:** Git, GitHub

---

## 🔬 Methodology

### 1️⃣ Data Cleaning

* Corrected data types (e.g., `TotalCharges`)
* Handled missing values
* Standardized categorical labels

### 2️⃣ Exploratory Data Analysis

* Overall churn distribution
* Churn vs tenure, contract type, charges
* Identification of **high-risk customer segments**

### 3️⃣ Feature Engineering

* Binary encoding (Yes/No)
* One-hot encoding for multi-class features
* Derived tenure & billing features

### 4️⃣ Modeling Strategy

* **Baseline:** Logistic Regression, Decision Tree
* **Advanced:** Random Forest, XGBoost, LightGBM, CatBoost
* Class imbalance handled via **weighted loss / SMOTE**

### 5️⃣ Evaluation Metrics

* ROC-AUC
* Precision, Recall, F1-Score
* Confusion Matrix

📌 **Primary business metric:** **Recall (Churn class)**

### 6️⃣ Explainability

* SHAP global feature importance
* Individual customer-level explanations

---

## 📈 Key Insights

### 🔑 Top Churn Drivers

* Month-to-month contracts
* High monthly charges
* Lack of TechSupport & OnlineSecurity
* Low customer tenure

### 💡 Insight

Customers with **short tenure**, **high bills**, and **no support services** exhibit the **highest churn probability**.

---

## 💼 Business Recommendations

* Incentivize **contract upgrades** for month-to-month users
* Bundle **TechSupport & Security services**
* Offer **loyalty discounts** for long-tenure customers
* Trigger **targeted retention campaigns** using churn scores

---

## ▶️ How to Run (Step-by-Step)

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd customer-churn-prediction
```

### 2️⃣ Create Environment & Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Analysis (Recommended Order)

```text
01_data_understanding.ipynb
02_eda.ipynb
03_feature_engineering.ipynb
04_model_training.ipynb
```

---

### 4️⃣ Train Model via Script (Production)

```bash
python src/train.py
```

Outputs:

* Trained model → `models/`
* Evaluation artifacts → `artifacts/`

---

### 5️⃣ Run Predictions

```bash
python src/predict.py
```

---

### 6️⃣ Run API (Optional – Deployment Ready)

```bash
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🔮 Future Enhancements

* Cost-sensitive churn optimization
* Customer Lifetime Value (CLV) modeling
* Uplift modeling for retention actions
* Streamlit executive dashboard
* Cloud deployment (Docker + AWS/GCP)

---

## 👩‍💻 Author

**Priya Choure**
Data Science & Artificial Intelligence Practitioner
