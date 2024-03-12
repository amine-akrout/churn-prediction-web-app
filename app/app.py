"""
Streamlit app for predicting customer churn using a trained model.
"""

from dataclasses import dataclass

import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model
from config import AREAS, STATES, YES_NO


# Load the model
@st.cache_data
def load_and_cache_model():
    """Load the trained model from the artifacts directory."""
    model = load_model("../artifacts/model")
    return model


@dataclass
class CustomerData:
    """Dataclass to hold customer data for prediction."""

    state: str
    account_length: int
    area_code: str
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_eve_minutes: float
    total_eve_calls: int
    total_night_minutes: float
    total_night_calls: int
    total_intl_minutes: float
    total_intl_calls: int
    number_customer_service_calls: int
    total_day_charge: float
    total_eve_charge: float
    total_night_charge: float
    total_intl_charge: float


def predict_customer_churn(model, customer_data: CustomerData):
    """Predict customer churn using the trained model and the customer data."""
    input_df = pd.DataFrame([customer_data.__dict__])
    predictions_df = predict_model(estimator=model, data=input_df)
    return predictions_df["prediction_label"].values[0].capitalize()


def online_prediction_form():
    """Render an online form for users to input customer data for prediction."""
    state_options = STATES
    area_code_options = AREAS
    yes_no_options = YES_NO

    customer_data = CustomerData(
        state=st.selectbox("State of customer residence:", state_options),
        account_length=st.number_input("Account length:", 0, 240, 0),
        area_code=st.selectbox("Area code:", area_code_options),
        international_plan=st.selectbox("International plan:", yes_no_options),
        voice_mail_plan=st.selectbox("Voice mail plan:", yes_no_options),
        number_vmail_messages=st.slider("Number of voice-mail messages:", 0, 60, 0),
        total_day_minutes=st.slider("Total day minutes:", 0.0, 360.0, 100.0),
        total_day_calls=st.slider("Total day calls:", 0, 200, 50),
        total_eve_minutes=st.slider("Total evening minutes:", 0.0, 400.0, 200.0),
        total_eve_calls=st.slider("Total evening calls:", 0, 200, 100),
        total_night_minutes=st.slider("Total night minutes:", 0.0, 400.0, 200.0),
        total_night_calls=st.slider("Total night calls:", 0, 200, 100),
        total_intl_minutes=st.slider("Total international minutes:", 0.0, 60.0, 10.0),
        total_intl_calls=st.slider("Total international calls:", 0, 20, 5),
        number_customer_service_calls=st.slider("Customer service calls:", 0, 10, 1),
        total_day_charge=st.slider("Total day charge:", 0.0, 60.0, 10.0),
        total_eve_charge=st.slider("Total evening charge:", 0.0, 60.0, 10.0),
        total_night_charge=st.slider("Total night charge:", 0.0, 60.0, 10.0),
        total_intl_charge=st.slider("Total international charge:", 0.0, 60.0, 10.0),
    )

    return customer_data


def main():
    """Main function for the Streamlit app."""
    st.title("Predicting Customer Churn")

    # Sidebar content
    st.sidebar.info("This app is created to predict Customer Churn")
    st.sidebar.image("../images/icone.png", use_column_width=True)
    st.sidebar.image("../images/image.png", use_column_width=True)

    prediction_method = st.sidebar.selectbox("Prediction method:", ["Online", "Batch"])
    model = load_and_cache_model()
    if prediction_method == "Online":
        customer_data = online_prediction_form()
        if st.button("Predict"):
            prediction = predict_customer_churn(model, customer_data)
            st.success(f"Churn Prediction: {prediction}")

    elif prediction_method == "Batch":
        file_upload = st.file_uploader("Upload CSV for batch prediction:", type=["csv"])
        if file_upload:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)


if __name__ == "__main__":
    main()
