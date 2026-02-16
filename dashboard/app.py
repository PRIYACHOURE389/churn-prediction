import streamlit as st
import pandas as pd
import joblib
import requests

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# API Configuration (NON-INTRUSIVE)
# ------------------------------
API_URL = "http://127.0.0.1:8000"

@st.cache_data(ttl=10)
def api_available():
    try:
        r = requests.get(
            "http://127.0.0.1:8000/health",
            timeout=3
        )
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False

# ------------------------------
# Dashboard Entrance / Landing Section
# ------------------------------
st.markdown(
    """
    <style>
    .dashboard-header {
        background: linear-gradient(90deg, #2A9D8F, #F4A261);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .dashboard-header:hover {
        transform: scale(1.02);
    }
    .dashboard-logo {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin-bottom: 15px;
    }
    </style>

    <div class="dashboard-header">
        <img src="https://img.icons8.com/ios-filled/100/ffffff/data-sheet.png" class="dashboard-logo"/>
        <h1>Telco Customer Churn Prediction Dashboard</h1>
        <p style="font-size:16px; max-width:720px; margin:auto;">
        An end-to-end <b>customer churn prediction and analytics system</b> designed to identify
        <b>at-risk customers</b>, uncover <b>root churn drivers</b>, and enable
        <b>proactive retention strategies</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------
# Sidebar (Filters + API Status)
# ------------------------------
st.sidebar.header("🎛 Controls")

st.sidebar.subheader("🧠 ML Service Status")
if api_available():
    st.sidebar.success("API Connected")
else:
    st.sidebar.warning("API Offline")

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "churn_clean.csv"

# ------------------------------
# Load Model & Data (LOCAL, FAST)
# ------------------------------
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# ------------------------------
# Helper Functions
# ------------------------------
def assign_risk(prob):
    if prob >= 0.75:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

RISK_COLORS = {
    "High Risk": "#E63946",
    "Medium Risk": "#F4A261",
    "Low Risk": "#2A9D8F"
}

# ------------------------------
# Local Predictions (Dashboard Analytics)
# ------------------------------
df["churn_probability"] = model.predict_proba(df)[:, 1]
df["risk_segment"] = df["churn_probability"].apply(assign_risk)

# ------------------------------
# Sidebar Filters
# ------------------------------
selected_risk = st.sidebar.multiselect(
    "Select Risk Segment",
    options=["High Risk", "Medium Risk", "Low Risk"],
    default=["High Risk", "Medium Risk", "Low Risk"]
)

filtered_df = df[df["risk_segment"].isin(selected_risk)]

# ------------------------------
# Risk Segment Metrics
# ------------------------------
st.subheader("📊 Risk Segments Overview")

risk_counts = filtered_df["risk_segment"].value_counts()

if len(risk_counts) > 0:
    cols = st.columns(len(risk_counts))
    for i, (risk, count) in enumerate(risk_counts.items()):
        cols[i].metric(
            label=risk,
            value=count,
            delta=f"{count / len(df) * 100:.1f}% of total"
        )
else:
    st.warning("No customers found for selected filters.")

st.markdown(
    """
    **Interpretation:**  
    - **High Risk** customers are most likely to churn imminently  
    - **Medium Risk** customers require proactive engagement  
    - **Low Risk** customers are stable  

    **Recommendation:** Prioritize High & Medium risk customers for retention offers.
    """
)

# ------------------------------
# Batch Churn Prediction (API)
# ------------------------------
st.divider()
st.subheader("📤 Batch Churn Prediction (via API)")

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file is not None:
    files = {"file": uploaded_file}
    response = requests.post(
        "http://127.0.0.1:8000/predict/batch",
        files=files,
        timeout=30
    )

    if response.status_code == 200:
        pred_df = pd.DataFrame(response.json()["predictions"])
        pred_df.to_csv("artifacts/predictions.csv", index=False)
        st.success("Batch prediction completed successfully.")
    else:
        st.error("Prediction API failed.")

# ------------------------------
# Churn Probability Distribution
# ------------------------------
st.subheader("📈 Churn Probability Distribution")

fig, ax = plt.subplots(figsize=(8, 3))
sns.histplot(
    filtered_df,
    x="churn_probability",
    hue="risk_segment",
    palette=RISK_COLORS,
    multiple="stack",
    bins=20,
    edgecolor="black"
)
ax.set_xlabel("Churn Probability")
ax.set_ylabel("Customers")
st.pyplot(fig)

if pred_df.empty:
    st.info("Run batch prediction to view churn distribution.")
else:
    fig, ax = plt.subplots()
    sns.histplot(pred_df["churn_probability"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

# ------------------------------
# Monthly Charges vs Tenure
# ------------------------------
st.subheader("💳 Monthly Charges vs Tenure")

fig, ax = plt.subplots(figsize=(8, 3))
sns.scatterplot(
    data=filtered_df,
    x="tenure",
    y="MonthlyCharges",
    hue="risk_segment",
    palette=RISK_COLORS,
    alpha=0.7
)
ax.set_xlabel("Tenure (Months)")
ax.set_ylabel("Monthly Charges")
st.pyplot(fig)

# ------------------------------
# Top 10 Churn Drivers
# ------------------------------
st.subheader("📌 Top 10 Churn Drivers")

model_lr = model.named_steps["model"]
preprocessor = model.named_steps["preprocessing"]

cat_features = preprocessor.named_transformers_["categorical"].get_feature_names_out()
num_features = preprocessor.named_transformers_["numerical"].get_feature_names_out()

features = list(cat_features) + list(num_features)

importance = pd.Series(
    model_lr.coef_[0],
    index=features
).sort_values(key=abs, ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 3))
importance.sort_values().plot(kind="barh", ax=ax, color="#F4A261")
ax.set_xlabel("Impact on Churn")
st.pyplot(fig)

# ------------------------------
# Risk Segmentation Heatmap
# ------------------------------
st.subheader("🔥 Risk Segmentation Heatmap")

risk_summary = df.groupby("risk_segment")[["tenure", "MonthlyCharges"]].mean()
risk_summary = risk_summary.loc[["High Risk", "Medium Risk", "Low Risk"]]

fig, ax = plt.subplots(figsize=(6, 2))
sns.heatmap(
    risk_summary,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    cbar=False,
    linewidths=0.5,
    ax=ax
)
st.pyplot(fig)

# ------------------------------
# Risk Distribution Pie
# ------------------------------
st.subheader("🎯 Customer Risk Distribution")

risk_counts_all = df["risk_segment"].value_counts()

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(
    risk_counts_all,
    labels=risk_counts_all.index,
    autopct="%1.1f%%",
    colors=[RISK_COLORS[r] for r in risk_counts_all.index]
)
st.pyplot(fig)
