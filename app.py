import streamlit as st
import pandas as pd
import joblib

# Load saved model and supporting files
model = joblib.load('models/xgb_clv_model.joblib')
feature_names = joblib.load('models/feature_names.joblib')
segment_bins = joblib.load('models/segment_bins.joblib')

st.title("Customer Lifetime Value Predictor")
st.write("Enter customer purchase behavior to predict their lifetime value.")

# Input fields
frequency = st.number_input("Frequency (number of repeat purchases)", min_value=0, value=1)
recency = st.number_input("Recency (days since last purchase)", min_value=0, value=100)
T = st.number_input("Customer Age (days since first purchase)", min_value=0, value=200)

# Validate inputs
if recency > T:
    st.error("⚠️ Recency cannot be greater than Customer Age (T). Please check your inputs.")
else:
    if st.button("Predict CLV"):
        input_data = pd.DataFrame([[frequency, recency, T]], columns=feature_names)

        # Predict
        prediction = model.predict(input_data)[0]

        # Determine segment
        if prediction <= segment_bins[1]:
            segment = "Low Value"
        elif prediction <= segment_bins[2]:
            segment = "Medium Value"
        else:
            segment = "High Value"

        st.success(f"Predicted CLV: ${prediction:,.2f}")
        st.info(f"Customer Segment: {segment}")
        st.markdown("---")
st.header("Bulk Prediction (CSV Upload)")
st.write("Upload a CSV with columns: frequency, recency, T")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    bulk_data = pd.read_csv(uploaded_file)

    # Check required columns exist
    required_cols = ['frequency', 'recency', 'T']
    if not all(col in bulk_data.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {required_cols}")
    else:
        # Validate recency <= T for all rows
        invalid_rows = bulk_data[bulk_data['recency'] > bulk_data['T']]
        if len(invalid_rows) > 0:
            st.warning(f"⚠️ {len(invalid_rows)} row(s) have recency > T and will be skipped.")
            bulk_data = bulk_data[bulk_data['recency'] <= bulk_data['T']]

        # Predict for all remaining rows
        X_bulk = bulk_data[required_cols]
        bulk_data['predicted_clv'] = model.predict(X_bulk)

        # Segment each customer
        def get_segment(value):
            if value <= segment_bins[1]:
                return "Low Value"
            elif value <= segment_bins[2]:
                return "Medium Value"
            else:
                return "High Value"

        bulk_data['segment'] = bulk_data['predicted_clv'].apply(get_segment)

        st.success(f"✓ Predictions complete for {len(bulk_data)} customers")
        st.dataframe(bulk_data)

        # Download button
        csv_output = bulk_data.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv_output,
            file_name="clv_predictions.csv",
            mime="text/csv"
        )