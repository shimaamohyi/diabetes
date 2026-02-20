import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
data = pd.read_csv("diabetes.csv")

st.title("🩺 Diabetes Prediction App (Using Pre-Trained Models)")

# Show dataset
with st.expander("📊 Show Dataset"):
    st.write(data)

# Select model
st.subheader("⚙️ Select Pre-Trained Model")
model_option = st.selectbox("Choose Model", ["Decision Tree", "Random Forest", "XGBoost"])

# Load the corresponding saved model
if model_option == "Decision Tree":
    with open("dt_model.pkl", "rb") as f:
        model = pickle.load(f)
elif model_option == "Random Forest":
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
else:  # XGBoost
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

# Split data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Threshold selection
st.subheader("⚖️ Adjust Prediction Threshold")
threshold = st.slider("Select Threshold for Diabetic Prediction", 0.0, 1.0, 0.5, 0.01)

# Predict on the whole dataset with Threshold
y_proba = model.predict_proba(X)[:,1]
y_pred = np.array([1 if p >= threshold else 0 for p in y_proba])

# Metrics
accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_proba)
st.metric("Model Accuracy", f"{accuracy:.2%}")
st.metric("Model AUC", f"{auc:.2%}")

# Confusion Matrix
with st.expander("📌 Confusion Matrix"):
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Diabetic", "Diabetic"],
                yticklabels=["Non-Diabetic", "Diabetic"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)

# Classification Report
with st.expander("📄 Classification Report"):
    st.text(classification_report(y, y_pred))

# User input for prediction
st.subheader("📝 Enter Patient Data for Prediction")
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 120)
    bp = st.number_input("BloodPressure", 0, 122, 70)
    skin_thickness = st.number_input("SkinThickness", 0, 99, 20)
with col2:
    insulin = st.number_input("Insulin", 0, 846, 79)
    bmi = st.number_input("BMI", 0.0, 67.0, 25.0)
    dpf = st.number_input("DiabetesPedigreeFunction", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 0, 120, 33)

input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [bp],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Prediction with Threshold
if st.button("🔍 Predict"):
    proba = model.predict_proba(input_data)[0][1]
    prediction = 1 if proba >= threshold else 0
    st.subheader("Prediction Result")
    st.write("✅ **Diabetic**" if prediction == 1 else "🟢 **Non-Diabetic**")
    st.write(f"Prediction Probability: {proba:.2f}")
    st.write(f"Used Threshold: {threshold}")

# Compare with real data
st.subheader("📌 Check Prediction Against Real Data")
row_index = st.number_input("Choose Row Index from Dataset", 0, len(data)-1, 0)
selected_row = X.iloc[[row_index]]
real_outcome = y.iloc[row_index]
row_proba = model.predict_proba(selected_row)[:,1][0]
row_prediction = 1 if row_proba >= threshold else 0

st.write("Selected Data:", selected_row)
st.write(f"Real Outcome: {'Diabetic' if real_outcome == 1 else 'Non-Diabetic'}")
st.write(f"Model Prediction: {'Diabetic' if row_prediction == 1 else 'Non-Diabetic'}")
st.write("✅ Correct" if real_outcome == row_prediction else "❌ Incorrect")
