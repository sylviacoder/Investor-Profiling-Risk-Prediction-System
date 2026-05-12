import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Investor Risk Profiler", layout="wide")

st.title("📊 Investor Risk Profiling System")
st.markdown("Explainable AI for investor behavior classification")

@st.cache_resource
def load_models():
    model = joblib.load("models/investor_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return model, le

model, le = load_models()

feature_names = model.named_steps['preprocessor'].get_feature_names_out()
importances = np.abs(model.named_steps['model'].coef_[0])

feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.sidebar.header("Investor Inputs")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)

investment_avenues = st.sidebar.selectbox(
    "Investment Avenues",
    ["Mutual Funds", "Equity", "Fixed Deposits", "Bonds", "Gold"]
)

mutual_funds = st.sidebar.slider("Mutual Funds Preference", 1, 5, 3)
debentures = st.sidebar.slider("Debentures Preference", 1, 5, 3)
government_bonds = st.sidebar.slider("Government Bonds Preference", 1, 5, 3)
ppf = st.sidebar.slider("PPF Preference", 1, 5, 3)
gold = st.sidebar.slider("Gold Preference", 1, 5, 3)

factor = st.sidebar.selectbox("Factor", ["Returns", "Risk", "Tax Benefit", "Lock-in Period"])
objective = st.sidebar.selectbox("Objective", ["Capital Appreciation", "Income", "Growth"])
purpose = st.sidebar.selectbox("Purpose", ["Retirement", "Wealth Creation", "Education", "Health"])

duration = st.sidebar.selectbox("Duration", ["Short Term", "Medium Term", "Long Term"])
invest_monitor = st.sidebar.selectbox("Investment Monitoring", ["Daily", "Weekly", "Monthly", "Rarely"])

expect = st.sidebar.slider("Expected Return (%)", 1, 100, 10)

avenue = st.sidebar.selectbox("Preferred Avenue", ["Mutual Funds", "Equity", "Fixed Deposits", "PPF", "Gold"])

savings_obj = st.sidebar.selectbox(
    "Savings Objectives",
    ["Retirement", "Health Care", "Education", "Wealth Creation"]
)

source = st.sidebar.selectbox(
    "Information Source",
    ["Financial Consultants", "Newspapers", "Internet", "Television"]
)

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

if st.button("🔍 Predict Investor Type"):

    pred_encoded = model.predict(input_data)
    pred_label = le.inverse_transform(pred_encoded)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(input_data)) * 100

    st.markdown("---")
 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Prediction")

        if "risk" in pred_label.lower():
            st.error("Risk-Oriented Investor")
        else:
            st.success("Conservative Investor")

    with col2:
        st.subheader("📊 Confidence")

        if confidence:
            st.info(f"{confidence:.2f}%")
        else:
            st.warning("Not available")
         
    st.markdown("---")
    st.subheader("📌 Key Drivers of Your Profile")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=feat_imp_df.head(10),
        x="Importance",
        y="Feature",
        ax=ax
    )
    ax.set_title("Top Behavioral Drivers")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🧠 Behavioral Insight")

    st.write("""
    Your investor profile is primarily driven by **financial behavior and expectations**, 
    especially return expectations, investment purpose, and asset allocation choices.
    Demographics play a smaller role in classification.
    """)

    st.markdown("---")
    st.subheader("📊 Influence Breakdown")

    st.markdown("""
    - 🔴 **Behavioral Factors:** High impact (Expectations, Purpose)
    - 🟠 **Asset Allocation:** Medium impact (PPF, Mutual Funds, Debentures)
    - 🔵 **Demographics:** Low impact (Age, Gender)
    """)

    st.markdown("---")
    st.subheader("📌 Investment Strategy")

    if "risk" in pred_label.lower():
        st.write("""
        Suggested Portfolio:
        - 60% Equity / Mutual Funds  
        - 25% Growth Assets  
        - 15% Fixed Income  
        """)
    else:
        st.write("""
        Suggested Portfolio:
        - 50% Fixed Deposits / Bonds  
        - 30% PPF / Stable Instruments  
        - 20% Low-risk Mutual Funds  
        """)
