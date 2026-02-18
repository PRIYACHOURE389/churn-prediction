import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

# ============================================================
# API HEALTH CHECK
# ============================================================

@st.cache_data(ttl=10)
def api_available():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

# ============================================================
# LOAD MODEL & DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "churn_clean.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# ============================================================
# RISK SEGMENT LOGIC
# ============================================================

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

df["churn_probability"] = model.predict_proba(df)[:, 1]
df["risk_segment"] = df["churn_probability"].apply(assign_risk)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="background: linear-gradient(90deg,#2A9D8F,#F4A261);
padding:30px;border-radius:15px;color:white;text-align:center">
<h1>Telco Customer Churn Intelligence Dashboard</h1>
<p>End-to-end churn intelligence system for proactive retention strategy.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛 Controls")
st.sidebar.subheader("🧠 ML Service Status")

if api_available():
    st.sidebar.success("API Connected")
else:
    st.sidebar.warning("API Offline")

selected_risk = st.sidebar.multiselect(
    "Select Risk Segment",
    ["High Risk", "Medium Risk", "Low Risk"],
    default=["High Risk", "Medium Risk", "Low Risk"]
)

filtered_df = df[df["risk_segment"].isin(selected_risk)]

# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

st.subheader("📊 Risk Segments Overview")

risk_counts = filtered_df["risk_segment"].value_counts()
cols = st.columns(len(risk_counts))

for i, (risk, count) in enumerate(risk_counts.items()):
    cols[i].metric(
        label=risk,
        value=count,
        delta=f"{count / len(df) * 100:.1f}% of total"
    )

# ---------------- Risk Pie ----------------

st.subheader("🎯 Customer Risk Distribution")

c1, c2, c3 = st.columns([1,2,1])
with c2:
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(
        df["risk_segment"].value_counts(),
        labels=df["risk_segment"].value_counts().index,
        autopct="%1.1f%%",
        colors=[RISK_COLORS[r] for r in df["risk_segment"].value_counts().index]
    )
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# PORTFOLIO ANALYTICS
# ============================================================

st.subheader("📈 Churn Probability Distribution")

c1, c2, c3 = st.columns([1,2,1])
with c2:
    fig, ax = plt.subplots(figsize=(6,3))
    sns.histplot(
        filtered_df,
        x="churn_probability",
        hue="risk_segment",
        palette=RISK_COLORS,
        multiple="stack",
        bins=20,
        edgecolor="black"
    )
    ax.set_title("Portfolio Risk Distribution")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Scatter ----------------

st.subheader("💳 Monthly Charges vs Tenure")

c1, c2, c3 = st.columns([1,2,1])
with c2:
    fig, ax = plt.subplots(figsize=(6,3.5))
    sns.scatterplot(
        data=filtered_df,
        x="tenure",
        y="MonthlyCharges",
        hue="risk_segment",
        palette=RISK_COLORS,
        alpha=0.6
    )
    ax.set_title("Behavioral Risk Clustering")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Heatmap ----------------

st.subheader("🔥 Risk Segmentation Heatmap")

risk_summary = df.groupby("risk_segment")[["tenure", "MonthlyCharges"]].mean()
risk_summary = risk_summary.loc[["High Risk", "Medium Risk", "Low Risk"]]

c1, c2, c3 = st.columns([1,2,1])
with c2:
    fig, ax = plt.subplots(figsize=(5,2.5))
    sns.heatmap(
        risk_summary,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar=False,
        linewidths=0.5
    )
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# GLOBAL MODEL DRIVERS
# ============================================================

st.subheader("📌 Global Model Drivers")

model_lr = model.named_steps["model"]
preprocessor = model.named_steps["preprocessing"]

cat_features = preprocessor.named_transformers_["categorical"].get_feature_names_out()
num_features = preprocessor.named_transformers_["numerical"].get_feature_names_out()
features = list(cat_features) + list(num_features)

importance = pd.Series(
    model_lr.coef_[0],
    index=features
).sort_values(key=abs, ascending=False).head(10)

c1, c2, c3 = st.columns([1,2,1])
with c2:
    fig, ax = plt.subplots(figsize=(6,3.5))
    importance.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Global Churn Drivers")
    ax.set_xlabel("Coefficient Impact")
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# BATCH PREDICTION
# ============================================================

st.divider()
st.subheader("📤 Batch Churn Prediction")

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file is not None:

    df_upload = pd.read_csv(uploaded_file)

    response = requests.post(
        f"{API_URL}/predict/batch",
        json=df_upload.to_dict(orient="records"),
        timeout=30
    )

    if response.status_code == 200:

        pred_df = pd.DataFrame(response.json()["predictions"])
        st.success("Batch prediction completed.")

        # KPI
        high = (pred_df["churn_probability"] >= 0.75).sum()
        medium = ((pred_df["churn_probability"] >= 0.4) &
                  (pred_df["churn_probability"] < 0.75)).sum()
        low = (pred_df["churn_probability"] < 0.4).sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 High Risk", high)
        col2.metric("🟠 Medium Risk", medium)
        col3.metric("🟢 Low Risk", low)

        # Distribution
        st.subheader("📈 Batch Probability Distribution")

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(pred_df["churn_probability"], bins=20, kde=True)
            plt.tight_layout()
            st.pyplot(fig)

        # Risk Bar
        st.subheader("📊 Batch Risk Segmentation")

        risk_labels = pd.cut(
            pred_df["churn_probability"],
            bins=[0, 0.4, 0.75, 1],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )

        risk_counts = risk_labels.value_counts().sort_index()

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            fig, ax = plt.subplots(figsize=(5,3))
            risk_counts.plot(
                kind="bar",
                color=["#2A9D8F", "#F4A261", "#E63946"],
                ax=ax
            )
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("📄 Prediction Output")
        st.dataframe(pred_df, use_container_width=True)