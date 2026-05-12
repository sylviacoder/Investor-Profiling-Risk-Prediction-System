import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Investor Risk Profiler", layout="wide")

st.title("📊 Investor Risk Profiling System")
st.markdown("### Explainable AI-based investor classification system")

@st.cache_resource
def load_models():
    model = joblib.load("models/investor_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return model, le

try:
    model, le = load_models()
except Exception as e:
    st.error(f"❌ Could not load models. Please ensure 'models/' folder contains your .pkl files. Error: {e}")
    st.stop()

st.sidebar.header("Step 1: Demographics & Behavior")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)
source = st.sidebar.selectbox("Information Source", ["Financial Consultants", "Newspapers", "Internet", "Television"])

st.sidebar.header("Step 2: Investment Preferences")
investment_avenues = st.sidebar.selectbox("Do you currently have Investment Avenues?", ["Yes", "No"])
avenue = st.sidebar.selectbox("Preferred Investment Avenue", ["Mutual Funds", "Equity Market", "Fixed Deposits", "Public Provident Fund", "Gold"])

st.sidebar.header("Step 3: Asset Ranking (1-5)")
mutual_funds = st.sidebar.slider("Mutual Funds Preference", 1, 5, 3)
debentures = st.sidebar.slider("Debentures Preference", 1, 5, 3)
government_bonds = st.sidebar.slider("Government Bonds Preference", 1, 5, 3)
ppf = st.sidebar.slider("PPF Preference", 1, 5, 3)
gold = st.sidebar.slider("Gold Preference", 1, 5, 3)

st.sidebar.header("Step 4: Objectives & Duration")
factor = st.sidebar.selectbox("Primary Factor", ["Returns", "Risk", "Tax Benefit", "Lock-in Period"])
objective = st.sidebar.selectbox("Investment Objective", ["Capital Appreciation", "Income", "Growth"])
purpose = st.sidebar.selectbox("Investment Purpose", ["Wealth Creation", "Retirement", "Education", "Healthcare"])
duration = st.sidebar.selectbox("Investment Duration", ["1-3 years", "3-5 years", "More than 5 years", "Less than 1 year"])
invest_monitor = st.sidebar.selectbox("Monitoring Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
expect = st.sidebar.selectbox("Expected Return", ["10%-20%", "20%-30%", "30%-40%"])
savings_obj = st.sidebar.selectbox("What are your savings objectives?", ["Retirement", "Health Care", "Education", "Wealth Creation"])

input_data = pd.DataFrame({
    'gender': [gender],
    'age': [float(age)], # Forced float
    'Investment_Avenues': [investment_avenues],
    'Mutual_Funds': [float(mutual_funds)],
    'Debentures': [float(debentures)],
    'Government_Bonds': [float(government_bonds)],
    'PPF': [float(ppf)],
    'Gold': [float(gold)],
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

if st.button("🔍 Predict Investor Risk Profile"):
    try:
        # pipeline to predict
        pred_encoded = model.predict(input_data)
        pred_label = le.inverse_transform(pred_encoded)[0]

        # confidence level
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(input_data)) * 100

        st.divider()

        # Results
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("🎯 Prediction")
            if "risk" in str(pred_label).lower():
                st.error(f"**{pred_label.upper()}**")
            else:
                st.success(f"**{pred_label.upper()}**")

        with res_col2:
            st.subheader("📈 Model Confidence")
            if confidence:
                st.info(f"**{confidence:.2f}%**")
            else:
                st.warning("N/A")


        st.divider()
        st.subheader("📌 Behavioral Drivers (Explainable AI)")
        
        step_name = 'model' if 'model' in model.named_steps else 'classifier'
        
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = np.abs(model.named_steps[step_name].coef_[0])

        feat_imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Plotly
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feat_imp_df.head(10), x="Importance", y="Feature", palette="coolwarm", ax=ax)
        ax.set_title("Top 10 Features Driving This Prediction")
        st.pyplot(fig)

        # Insight
        st.info(f"**Insight:** Your profile is most heavily influenced by features like **{feat_imp_df.iloc[0]['Feature']}**. Based on this, your recommended allocation leans toward assets that match your {purpose} goals.")

    except Exception as e:
        st.error("⚠️ Prediction Failed")
        st.write(f"**Technical Error:** {e}")
        st.info("Check if your dropdown values (e.g., '10%-20%') match your training dataset exactly.")
