import streamlit as st
import pandas as pd
import joblib
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
# Dashboard Entrance / Landing Section
# ------------------------------
st.markdown(
    """
    <style>
    /* Gradient header */
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
    .get-started {
        background-color: white;
        color: #2A9D8F;
        font-weight: bold;
        padding: 10px 25px;
        border-radius: 25px;
        text-decoration: none;
        display: inline-block;
        margin-top: 15px;
        transition: transform 0.2s ease;
    }
    .get-started:hover {
        transform: scale(1.05);
        background-color: #F4A261;
        color: white;
    }
    </style>

    <div class="dashboard-header">
        <img src="https://img.icons8.com/ios-filled/100/ffffff/data-sheet.png" class="dashboard-logo"/>
        <h1>Telco Customer Churn Prediction Dashboard</h1>
        <p style="font-size:16px; max-width:700px; margin:auto;">
        An end-to-end <b>customer churn prediction and analytics system</b> designed to identify <b>at-risk customers</b>, 
        uncover <b>root churn drivers</b>, and enable <b>proactive retention strategies</b>.<br>
        Built on the <b>IBM Telco Customer Churn dataset</b>, reflecting real-world telecom churn workflows.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br><br>", unsafe_allow_html=True)

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "churn_clean.csv"

# ------------------------------
# Load Model & Data
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

RISK_COLORS = {"High Risk": "#E63946", "Medium Risk": "#F4A261", "Low Risk": "#2A9D8F"}

# ------------------------------
# Real-time Predictions & Risk
# ------------------------------
df["churn_probability"] = model.predict_proba(df)[:, 1]
df["risk_segment"] = df["churn_probability"].apply(assign_risk)

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("Filter Customers")
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
            delta=f"{count/len(df)*100:.1f}% of total",
        )
else:
    st.warning("No customers found in the selected risk segment(s).")

st.markdown("**Interpretation:** High Risk customers are at immediate risk of churn. Medium risk need proactive engagement, while Low risk are stable.")
st.markdown("**Advice:** Prioritize retention campaigns for High & Medium risk customers; offer loyalty discounts or service bundles.")

# ------------------------------
# Churn Probability Histogram
# ------------------------------
st.subheader("📈 Churn Probability Distribution")
if len(filtered_df) > 0:
    fig, ax = plt.subplots(figsize=(8,3))
    sns.histplot(filtered_df, x="churn_probability", hue="risk_segment",
                 palette=RISK_COLORS, multiple="stack", bins=20, edgecolor="black")
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.markdown("**Interpretation:** High risk customers mostly have probabilities > 0.75. Medium risk 0.4–0.75. Low risk < 0.4.")
    st.markdown("**Advice:** Allocate retention resources according to risk probabilities.")
else:
    st.info("No data to display in the histogram for selected risk segment(s).")

# ------------------------------
# Monthly Charges vs Tenure Scatter
# ------------------------------
st.subheader("💳 Monthly Charges vs Tenure")
if len(filtered_df) > 0:
    fig, ax = plt.subplots(figsize=(8,3))
    sns.scatterplot(data=filtered_df, x="tenure", y="MonthlyCharges",
                    hue="risk_segment", palette=RISK_COLORS, alpha=0.7)
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Monthly Charges")
    st.pyplot(fig)

    st.markdown("**Interpretation:** High risk customers often have short tenure and high monthly charges.")
    st.markdown("**Advice:** Use loyalty programs or bundled services to retain short-tenure, high-bill customers.")
else:
    st.info("No data to display in scatter plot for selected risk segment(s).")

# ------------------------------
# Top 10 Churn Drivers
# ------------------------------
st.subheader("📌 Top 10 Churn Drivers (Global)")
model_lr = model.named_steps['model']
preprocessor = model.named_steps['preprocessing']

cat_features = preprocessor.named_transformers_['categorical'].get_feature_names_out()
num_features = preprocessor.named_transformers_['numerical'].get_feature_names_out()
all_features = list(cat_features) + list(num_features)

feature_importance = pd.Series(model_lr.coef_[0], index=all_features).sort_values(key=abs, ascending=False)[:10]

fig, ax = plt.subplots(figsize=(8,3))
feature_importance.sort_values().plot(kind='barh', color="#F4A261", ax=ax)
ax.set_xlabel("Coefficient (Impact on Churn)")
ax.set_title("Top 10 Features Driving Churn")
st.pyplot(fig)

st.markdown("**Interpretation:** Features with large coefficients strongly influence churn probability.")
st.markdown("**Advice:** Focus on modifiable features such as Contract, TechSupport, MonthlyCharges, or OnlineSecurity.")

# ------------------------------
# Risk Segmentation Heatmap
# ------------------------------
st.subheader("🔥 Risk Segmentation Heatmap")
risk_summary = df.groupby('risk_segment')[['tenure', 'MonthlyCharges']].mean().loc[['High Risk','Medium Risk','Low Risk']]
fig, ax = plt.subplots(figsize=(6,2))
sns.heatmap(risk_summary, annot=True, fmt=".1f", cmap="YlOrRd", cbar=False, linewidths=0.5, ax=ax)
st.pyplot(fig)

st.markdown("**Interpretation:** High-risk customers usually have low tenure and high bills. Medium risk are moderate; Low risk have long tenure or low bills.")
st.markdown("**Advice:** Design retention campaigns by segment: loyalty offers for low-tenure/high-bill customers, upselling for medium risk.")

# ------------------------------
# Risk Distribution Pie Chart
# ------------------------------
st.subheader("🎯 Customer Risk Distribution")
risk_counts_all = df['risk_segment'].value_counts()
fig, ax = plt.subplots(figsize=(4,4))
ax.pie(risk_counts_all, labels=risk_counts_all.index, autopct="%1.1f%%",
       colors=[RISK_COLORS[i] for i in risk_counts_all.index])
st.pyplot(fig)

st.markdown("**Interpretation:** Quickly see the proportion of High, Medium, and Low risk customers.")
st.markdown("**Advice:** Use this to prioritize resources for retention campaigns.")
