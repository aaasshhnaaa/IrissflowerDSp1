import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Classification")

st.write("Enter the flower measurements below:")

sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.success(f"Predicted Species: {prediction[0]}")
