# Customer Lifetime Value (CLV) Prediction

An end-to-end data science project predicting customer lifetime value using purchase behavior data. Includes exploratory data analysis, RFM feature engineering, probabilistic modeling (BG/NBD), regression modeling (XGBoost), and a deployed Streamlit app for interactive and bulk predictions.

🔗 **Live App:** [https://customer-lifetime-value-prediction-mjjpvqedvmwwuldynhkwyr.streamlit.app](https://customer-lifetime-value-prediction-mjjpvqedvmwwuldynhkwyr.streamlit.app)
## Overview

This project predicts how much revenue a customer is likely to generate, using only their purchase history (frequency, recency, and customer age). It's designed to help a business identify high-value customers worth retaining versus low-value customers.

The project covers the full pipeline:
1. Data exploration and cleaning
2. Feature engineering (RFM: Recency, Frequency, Monetary)
3. Probabilistic modeling with BG/NBD (lifetimes library)
4. Model evaluation and calibration
5. Regression modeling (Linear Regression, Random Forest, XGBoost)
6. Model deployment prep (joblib serialization)
7. Interactive Streamlit web app
## Model Journey & Key Findings

**BG/NBD Model:** Initial predictions significantly overestimated customer value (98.5% of customers overpredicted). Applied a calibration factor (0.123) to correct this, reducing Mean Absolute Error from $1,533.79 to $163.30.

**Regression Models:** Tested Linear Regression, Random Forest, and XGBoost to directly predict monetary value from frequency, recency, and customer age.

| Model | MAE | R² |
|---|---|---|
| BG/NBD (calibrated) | $163.30 | — |
| Linear Regression | $156.69 | 0.202 |
| Random Forest | $164.41 | -0.156 |
| XGBoost | $150.76 | -0.010 |

**XGBoost was selected** for the final app, as it had the lowest MAE. Feature importance showed recency (57%) as the dominant predictor, followed by T (31%) and frequency (24%... wait — these should sum close to 1, double check your actual numbers before pasting).

### Limitations
- Low R² scores across all regression models indicate that frequency, recency, and customer age alone don't fully explain spending variation — additional features (product category, price, marketing exposure) would likely improve accuracy.
- The model is best suited for **ranking and segmenting customers** by relative value, not for precise dollar-amount forecasting.
- Feature importance shows recency (56.8%) dominates the model's decisions, followed by frequency (24.3%) and customer age (18.9%). This means predictions can appear counterintuitive for individual customers — two customers with different purchase frequency but similar recency may receive similar or even inverted scores, since recency outweighs frequency in the model's logic.y
## Project Structure

```
Customer-lifetime-value-prediction/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_clv_bgnbd_modeling.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_model_building_and_evaluation.ipynb
│   └── 07_model_deployment_prep.ipynb
├── data/
│   └── processed/
├── models/
│   ├── xgb_clv_model.joblib
│   ├── feature_names.joblib
│   └── segment_bins.joblib
├── app.py
├── requirements.txt
└── README.md
```
## Running Locally

1. Clone the repo:
```bash
   git clone https://github.com/marywarimi/Customer-lifetime-value-prediction.git
   cd Customer-lifetime-value-prediction
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Run the app:
```bash
   streamlit run app.py
```
## Tech Stack

- **Python** — pandas, numpy, scikit-learn, xgboost, lifetimes
- **Visualization** — matplotlib, seaborn
- **Modeling** — BG/NBD (lifetimes), Linear Regression, Random Forest, XGBoost
- **Deployment** — Streamlit, joblib
## Author

**Mary Warimi**
Data Science student at Meru University of Science and Technology (MUST)
