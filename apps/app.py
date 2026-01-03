import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Demand Prediction App", layout="centered")

st.title("📊 Demand Prediction App")
st.write("Enter input values to predict customer demand")

# -------------------------------
# SAMPLE TRAINING DATA (SIMULATED)
# -------------------------------
# This avoids external dataset/model issues
np.random.seed(42)

train_data = pd.DataFrame({
    "Price": np.random.uniform(50, 500, 200),
    "Discount": np.random.uniform(0, 0.5, 200),
    "Inventory_Level": np.random.randint(50, 1000, 200),
    "Promotion": np.random.randint(0, 2, 200),
    "Competitor_Pricing": np.random.uniform(50, 500, 200),
    "Seasonality": np.random.randint(0, 2, 200),
    "Epidemic": np.random.randint(0, 2, 200)
})

train_data["Demand"] = (
    0.4 * train_data["Inventory_Level"]
    - 0.3 * train_data["Price"]
    + 200 * train_data["Discount"]
    + 150 * train_data["Promotion"]
    - 0.2 * train_data["Competitor_Pricing"]
    + 100 * train_data["Seasonality"]
    - 120 * train_data["Epidemic"]
)

X = train_data.drop("Demand", axis=1)
y = train_data["Demand"]

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
model.fit(X, y)

# -------------------------------
# USER INPUTS
# -------------------------------
st.sidebar.header("Input Features")

price = st.sidebar.number_input("Price", min_value=1.0, value=100.0)
discount = st.sidebar.slider("Discount (0–0.5)", 0.0, 0.5, 0.1)
inventory = st.sidebar.number_input("Inventory Level", min_value=0, value=500)
promotion = st.sidebar.selectbox("Promotion (0 = No, 1 = Yes)", [0, 1])
competitor_price = st.sidebar.number_input("Competitor Pricing", min_value=1.0, value=95.0)
seasonality = st.sidebar.selectbox("Seasonality (0 = No, 1 = Yes)", [0, 1])
epidemic = st.sidebar.selectbox("Epidemic (0 = No, 1 = Yes)", [0, 1])

input_df = pd.DataFrame({
    "Price": [price],
    "Discount": [discount],
    "Inventory_Level": [inventory],
    "Promotion": [promotion],
    "Competitor_Pricing": [competitor_price],
    "Seasonality": [seasonality],
    "Epidemic": [epidemic]
})

st.subheader("User Input Data")
st.write(input_df)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Demand"):
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Demand: {int(prediction)}")
