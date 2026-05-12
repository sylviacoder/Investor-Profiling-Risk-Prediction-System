import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Investor Risk Profiler", layout="wide")

st.title("📊 Investor Risk Profiling System")
st.markdown("Explainable AI-based investor classification system")

# =========================
# LOAD MODEL + ENCODER
# =========================
@st.cache_resource
def load_models():
    model = joblib.load("models/investor_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return model, le

model, le = load_models()

# =========================
# FEATURE IMPORTANCE
# =========================
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
importances = np.abs(model.named_steps['model'].coef_[0])

feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# =========================
# SIDEBAR INPUTS (MATCH TRAINING EXACTLY)
# =========================
st.sidebar.header("Investor Inputs")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)

investment_avenues = st.sidebar.selectbox(
    "Investment Avenues",
    ["Mutual_Funds", "Equity_Market", "Fixed_Deposits", "Government_Bonds", "Gold"]
)

mutual_funds = st.sidebar.slider("Mutual Funds", 1, 5, 3)
debentures = st.sidebar.slider("Debentures", 1, 5, 3)
government_bonds = st.sidebar.slider("Government Bonds", 1, 5, 3)
ppf = st.sidebar.slider("PPF", 1, 5, 3)
gold = st.sidebar.slider("Gold", 1, 5, 3)

factor = st.sidebar.selectbox("Factor", ["Returns", "Risk", "Tax Benefits", "Lock-in Period"])
objective = st.sidebar.selectbox("Objective", ["Capital Appreciation", "Income", "Growth"])
purpose = st.sidebar.selectbox("Purpose", ["Wealth Creation", "Retirement", "Education", "Health Care"])

duration = st.sidebar.selectbox("Duration", ["Short Term", "Medium Term", "Long Term"])
invest_monitor = st.sidebar.selectbox("Invest Monitor", ["Daily", "Weekly", "Monthly", "Rarely"])

expect = st.sidebar.slider("Expected Return (%)", 1, 100, 10)

avenue = st.sidebar.selectbox(
    "Avenue",
    ["Mutual_Funds", "Equity_Market", "Fixed_Deposits", "PPF", "Gold"]
)

savings_obj = st.sidebar.selectbox(
    "What are your savings objectives?",
    ["Retirement", "Health Care", "Education", "Wealth Creation"]
)

source = st.sidebar.selectbox(
    "Source",
    ["Financial Consultants", "Newspapers", "Internet", "Television"]
)

# =========================
# INPUT DATAFRAME (MUST MATCH TRAINING EXACTLY)
# =========================
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'Investment_Avenues': [investment_avenues],
    'Mutual_Funds': [mutual_funds],
    'Debentures': [debentures],
    'Government_Bonds': [government_bonds],
    'PPF': [ppf],
    'Gold': [gold],
    'Factor': [factor],
    'Objective': [objective],
    'Purpose': [purpose],
    'Duration': [duration],
    'Invest_Monitor': [invest_monitor],
    'Expect': [expect],
    'Avenue': [avenue],
    'What are your savings objectives?': [savings_obj],
    'Source': [source]
})

# =========================
# FORCE NUMERIC TYPES (CRITICAL FIX)
# =========================
numeric_cols = ['age', 'Mutual_Funds', 'Debentures',
                'Government_Bonds', 'PPF', 'Gold', 'Expect']

input_data[numeric_cols] = input_data[numeric_cols].apply(pd.to_numeric)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Investor Type"):

    pred_encoded = model.predict(input_data)
    pred_label = le.inverse_transform(pred_encoded)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(input_data)) * 100

    st.markdown("---")

    # =========================
    # RESULT
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Prediction")

        if "risk" in str(pred_label).lower():
            st.error("Risk-Oriented Investor")
        else:
            st.success("Conservative Investor")

    with col2:
        st.subheader("📊 Confidence")

        if confidence:
            st.info(f"{confidence:.2f}%")
        else:
            st.warning("Not available")

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    st.markdown("---")
    st.subheader("📌 Key Behavioral Drivers")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=feat_imp_df.head(10),
        x="Importance",
        y="Feature",
        ax=ax
    )
    ax.set_title("Top Influencing Features")
    st.pyplot(fig)

    # =========================
    # INSIGHT
    # =========================
    st.markdown("---")
    st.subheader("🧠 Insight")

    st.write("""
    Your investor profile is mainly driven by behavioral financial choices such as 
    return expectations, investment purpose, and asset allocation preferences.
    """)

    # =========================
    # STRATEGY
    # =========================
    st.markdown("---")
    st.subheader("📌 Investment Strategy")

    if "risk" in str(pred_label).lower():
        st.write("""
        - 60% Equity / Mutual Funds  
        - 25% Growth assets  
        - 15% Fixed income  
        """)
    else:
        st.write("""
        - 50% Fixed Deposits / Bonds  
        - 30% PPF  
        - 20% Low-risk Mutual Funds  
        """)
