import streamlit as st
import pandas as pd
import joblib

# Load model and encoder

model = joblib.load("fraud_detection_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Page Configuration

st.set_page_config(
page_title="Fraud Detection System",
page_icon="🛡️",
layout="wide"
)

st.title("🛡️ Fraud Detection System")
st.write("Predict whether a transaction is Fraudulent or Legitimate")

# User Inputs

transaction_time_step = st.number_input(
"Transaction Time Step",
min_value=0,
value=1
)

transaction_type = st.selectbox(
"Transaction Type",
encoder.classes_.tolist()
)

transaction_amount = st.number_input(
"Transaction Amount",
min_value=0.0,
value=0.0
)

sender_balance_before = st.number_input(
"Sender Balance Before",
min_value=0.0,
value=0.0
)

sender_balance_after = st.number_input(
"Sender Balance After",
min_value=0.0,
value=0.0
)

receiver_balance_before = st.number_input(
"Receiver Balance Before",
min_value=0.0,
value=0.0
)

receiver_balance_after = st.number_input(
"Receiver Balance After",
min_value=0.0,
value=0.0
)

# Prediction Button

if st.button("Predict Fraud"):

    encoded_transaction_type = encoder.transform(
        [transaction_type]
    )[0]

    balanceDiffOrg = sender_balance_before - sender_balance_after
    balanceDiffDest = receiver_balance_after - receiver_balance_before

    input_data = pd.DataFrame({
        'transaction_time_step': [transaction_time_step],
        'transaction_type': [encoded_transaction_type],
        'transaction_amount': [transaction_amount],
        'sender_balance_before': [sender_balance_before],
        'sender_balance_after': [sender_balance_after],
        'receiver_balance_before': [receiver_balance_before],
        'receiver_balance_after': [receiver_balance_after],
        'balanceDiffOrg': [balanceDiffOrg],
        'balanceDiffDest': [balanceDiffDest]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Legitimate Transaction\n\nProbability: {probability:.2%}")
