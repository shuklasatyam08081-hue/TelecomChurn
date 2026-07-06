import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CUSTOM STYLING
# =====================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }
    .metric-card {
        background: #ffffff10;
        border: 1px solid #ffffff22;
        border-radius: 12px;
        padding: 1rem;
    }
    .result-churn {
        background: linear-gradient(135deg, #7f1d1d, #b91c1c);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .result-stay {
        background: linear-gradient(135deg, #14532d, #15803d);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
        border-left: 4px solid #2c5364;
        padding-left: 0.6rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# HEADER
# =====================================================================
st.markdown("""
<div class="main-header">
    <h1>📞 Customer Churn Predictor</h1>
    <p>Decision Tree Pipeline • Real-time inference • Powered by best_model.pkl</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# MODEL LOADING
# =====================================================================
MODEL_PATH = "best_model.pkl"

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None, f"Model file not found at '{path}'. Please place best_model.pkl in the app directory."
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

model, load_error = load_model(MODEL_PATH)

# Feature order the trained pipeline expects
EXPECTED_FEATURES = [
    "State", "Account length", "International plan", "Voice mail plan",
    "Number vmail messages", "Total day minutes", "Total day calls",
    "Total day charge", "Total eve minutes", "Total eve calls", "Total eve charge",
    "Total night minutes", "Total night calls", "Total night charge",
    "Total intl minutes", "Total intl calls", "Total intl charge",
    "Customer service calls", "Total charges", "Total_Usage", "Service_Stress",
]

US_STATES = sorted([
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
])

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app loads a pre-trained **Decision Tree** pipeline "
        "(`StandardScaler` + tuned `DecisionTreeClassifier`) and predicts "
        "whether a telecom customer is likely to churn."
    )

    st.divider()
    st.header("📦 Model Status")
    if model is not None:
        st.success("Model loaded successfully ✅")
        st.caption(f"Expected features: {len(EXPECTED_FEATURES)}")
    else:
        st.error("Model not loaded ❌")
        st.caption(load_error)

    st.divider()
    st.header("🕒 Session")
    st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    st.divider()
    st.caption(
        "⚠️ Categorical encodings (State, plan Yes/No) are approximated via "
        "alphabetical label-encoding to match typical `LabelEncoder` behavior "
        "used at training time. If your model was trained on different "
        "category orderings, retrain and re-export for exact parity."
    )

if model is None:
    st.error(f"🚫 {load_error}")
    st.stop()

# =====================================================================
# INPUT FORM
# =====================================================================
st.markdown('<div class="section-title">👤 Customer Details</div>', unsafe_allow_html=True)

with st.form("churn_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox("State", US_STATES, index=US_STATES.index("NY"))
        account_length = st.number_input("Account Length (days)", min_value=0, max_value=400, value=100)
        intl_plan = st.selectbox("International Plan", ["No", "Yes"])
        vmail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
        vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=60, value=0)
        service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=15, value=1)

    with col2:
        day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=400.0, value=180.0)
        day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, value=100)
        day_charge = st.number_input("Total Day Charge", min_value=0.0, max_value=70.0, value=30.0)
        eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
        eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=200, value=100)
        eve_charge = st.number_input("Total Evening Charge", min_value=0.0, max_value=40.0, value=17.0)

    with col3:
        night_minutes = st.number_input("Total Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
        night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, value=100)
        night_charge = st.number_input("Total Night Charge", min_value=0.0, max_value=20.0, value=9.0)
        intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=30.0, value=10.0)
        intl_calls = st.number_input("Total International Calls", min_value=0, max_value=20, value=4)
        intl_charge = st.number_input("Total International Charge", min_value=0.0, max_value=10.0, value=2.7)

    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

# =====================================================================
# PREDICTION LOGIC
# =====================================================================
if submitted:
    try:
        # --- Encode categoricals (alphabetical order, matching LabelEncoder convention) ---
        state_encoded = US_STATES.index(state)
        intl_plan_encoded = 0 if intl_plan == "No" else 1
        vmail_plan_encoded = 0 if vmail_plan == "No" else 1

        # --- Derived features (must match training feature engineering) ---
        total_charges = day_charge + eve_charge + night_charge + intl_charge
        total_usage = day_minutes + eve_minutes + night_minutes + intl_minutes
        service_stress = service_calls / (account_length + 1)

        input_dict = {
            "State": state_encoded,
            "Account length": account_length,
            "International plan": intl_plan_encoded,
            "Voice mail plan": vmail_plan_encoded,
            "Number vmail messages": vmail_messages,
            "Total day minutes": day_minutes,
            "Total day calls": day_calls,
            "Total day charge": day_charge,
            "Total eve minutes": eve_minutes,
            "Total eve calls": eve_calls,
            "Total eve charge": eve_charge,
            "Total night minutes": night_minutes,
            "Total night calls": night_calls,
            "Total night charge": night_charge,
            "Total intl minutes": intl_minutes,
            "Total intl calls": intl_calls,
            "Total intl charge": intl_charge,
            "Customer service calls": service_calls,
            "Total charges": total_charges,
            "Total_Usage": total_usage,
            "Service_Stress": service_stress,
        }

        input_df = pd.DataFrame([input_dict])[EXPECTED_FEATURES]  # enforce exact column order

        # --- Validate shape against model expectations ---
        if hasattr(model, "n_features_in_") and input_df.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Feature mismatch: model expects {model.n_features_in_} features, "
                f"got {input_df.shape[1]}."
            )

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

        res_col, prob_col, gauge_col = st.columns([1.3, 1, 1])

        with res_col:
            if prediction == 1:
                st.markdown(
                    f'<div class="result-churn">⚠️ Likely to CHURN</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-stay">✅ Likely to STAY</div>',
                    unsafe_allow_html=True
                )

        with prob_col:
            st.metric("Churn Probability", f"{probability:.1%}")

        with gauge_col:
            st.metric("Retention Probability", f"{1 - probability:.1%}")

        st.progress(float(probability))

        # --- Prepare downloadable result ---
        result_df = input_df.copy()
        result_df.insert(0, "State_Label", state)
        result_df["Predicted_Churn"] = int(prediction)
        result_df["Churn_Probability"] = round(float(probability), 4)
        result_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.download_button(
            label="⬇️ Download This Prediction (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name=f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("🔍 View raw feature vector sent to model"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"🚫 Prediction failed: {e}")
        st.info("Double-check your inputs and ensure best_model.pkl matches the expected feature schema.")

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.caption("Built with Streamlit • Decision Tree Pipeline • For demo / portfolio purposes")