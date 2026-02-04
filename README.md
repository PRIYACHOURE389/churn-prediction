# 📊 Telco Customer Churn Prediction

An **end-to-end machine learning project** that predicts customer churn and uncovers **key business drivers** behind churn to support **data-driven retention strategies**.

Built using the **IBM Telco Customer Churn dataset**, this project mirrors **real-world churn analytics workflows** used in telecom and subscription-based businesses.

---

## 🚀 Business Problem

Customer churn directly impacts revenue. Retaining an existing customer is **significantly cheaper** than acquiring a new one.

### **Primary Question**

> Which customers are most likely to churn — and why?

### **Business Objectives**

* Predict churn with **high recall** to minimize missed at-risk customers
* Identify **behavioral, service, and contract-based churn drivers**
* Enable **proactive retention campaigns**

---

## 🧾 Dataset Overview

Each row represents a customer; columns represent demographics, services, and billing information.

### Feature Groups

## 🔧 Services**

* PhoneService, MultipleLines
* InternetService
* OnlineSecurity, OnlineBackup
* DeviceProtection, TechSupport
* StreamingTV, StreamingMovies

## 💳 Account Information**

* tenure
* Contract
* PaymentMethod
* PaperlessBilling
* MonthlyCharges
* TotalCharges

## 👥 Demographics**

* gender
* SeniorCitizen
* Partner
* Dependents

## 🎯 Target**

* `Churn` (Yes / No)

---

## 🗂️ Project Structure

```text
customer-churn-prediction/
│── artifacts/
│   ├──model_evaluation_results.json
│   ├──model_evaluation_results.csv
│   ├── evaluation_metrics.json
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── decile_lift.csv
│   └── roc_auc.txt
│   
├── data/
│   ├── raw/
│   │   └── churn_raw.csv
│   └── processed/
│       ├── churn_clean.csv
│       ├── featured_telco.csv
│       └──  retention_targets.csv
│
├── models/
│   ├── catboost.joblib
│   ├── decision_tree.joblib
│   ├── feature_columns.joblib
│   ├── lightgbm.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── training_metrics.json
│   ├── churn_model.joblib
│   └── churn_pipeline.joblib
│
├── notebooks/
|   └── catboost_info/
|       └──├── 01_data_understanding.ipynb
│          ├── 02_eda.ipynb
│          ├── 03_feature_engineering.ipynb
│          └── 04_model_training.ipynb
│   
├── src/
|   ├── _init_.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── reports/
│   └── figures/
│       ├── churn_distribution_pie.png
│       ├── contract_churn_barh.png
│       └── tenure_churn_violin.png
│
├── requirements.txt
├── main.py
├── README.md
└── ARCHITECTURE.md

```

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **ML Models:** Scikit-learn, XGBoost, LightGBM, CatBoost
* **Imbalance Handling:** Classimbalance weight
* **Explainability:** SHAP
* **Environment:** Jupyter Notebook
* **Version Control:** Git & GitHub

---

## 🔬 Methodology

### 1️⃣ Data Cleaning

* Data type correction (e.g., `TotalCharges`)
* Missing value handling
* Standardized categorical values

### 2️⃣ Exploratory Data Analysis

* Overall churn distribution
* Churn vs contract, tenure, monthly charges
* Identification of high-risk segments

### 3️⃣ Feature Engineering

* Binary encoding for Yes/No features
* One-hot encoding for categorical variables
* Derived tenure and billing features

### 4️⃣ Modeling

* Baselines: Logistic Regression, Decision Tree
* Advanced: Random Forest, XGBoost, LightGBM, CatBoost
* Class imbalance handled using **SMOTE**

### 5️⃣ Evaluation

* ROC-AUC
* Precision, Recall, F1-score
* Confusion Matrix

📌 **Business Priority:** Recall for churn class

### 6️⃣ Explainability

* SHAP global feature importance
* Individual prediction interpretation

---

## 📈 Key Results & Insights

## Top Churn Drivers**

* Month-to-month contracts
* High monthly charges
* Lack of TechSupport & OnlineSecurity
* Low tenure

**Insight:**
Customers with **short tenure**, **high bills**, and **no support services** show the highest churn probability.

---

## 💡 Business Recommendations

* Incentivize contract upgrades for high-risk users
* Bundle support services for churn-prone segments
* Offer loyalty discounts to long-tenure customers
* Trigger targeted retention campaigns using churn scores

---

## ▶️ How to Run

```bash
git clone <repo-url>
cd customer-churn-prediction
pip install -r requirements.txt
```

Run notebooks in sequence:

---
01 → 02 → 03 → 04 → 05

---

Or train via script:

```bash
python src/train.py
```

---

## 🔮 Future Enhancements

* Cost-sensitive churn modeling
* CLV-based retention optimization
* Uplift modeling
* FastAPI inference service
* Streamlit dashboard

---

## 👩‍💻 Author

**Priya Choure**
Data Science & Artificial Intelligence Practitioner
