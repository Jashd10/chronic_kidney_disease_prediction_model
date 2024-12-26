import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
with open("model.pkl", "wb") as file:
    pickle.dump(rf, file)

# Load the trained model
# Replace with the correct path where the trained model is saved
# Assuming the Random Forest model is saved as 'chronic_kidney_model.pkl'
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, min_samples_split=5, min_samples_leaf=1)

# Define the main function
def main():
    st.title("Chronic Kidney Disease Prediction")
    st.write("Provide the following details to check for CKD risk:")

    # Create input fields for all features
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    bp = st.number_input("Blood Pressure (bp)", min_value=0, max_value=200, step=1)
    sg = st.selectbox("Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.number_input("Albumin (al)", min_value=0, max_value=5, step=1)
    su = st.number_input("Sugar (su)", min_value=0, max_value=5, step=1)
    rbc = st.selectbox("Red Blood Cells (rbc)", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell (pc)", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps (pcc)", ["notpresent", "present"])
    ba = st.selectbox("Bacteria (ba)", ["notpresent", "present"])
    bgr = st.number_input("Blood Glucose Random (bgr)", min_value=0, step=1)
    bu = st.number_input("Blood Urea (bu)", min_value=0, step=1)
    sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, step=0.1)
    sod = st.number_input("Sodium (sod)", min_value=0.0, step=0.1)
    pot = st.number_input("Potassium (pot)", min_value=0.0, step=0.1)
    hemo = st.number_input("Hemoglobin (hemo)", min_value=0.0, step=0.1)
    pcv = st.number_input("Packed Cell Volume (pcv)", min_value=0, step=1)
    wc = st.number_input("White Blood Cell Count (wc)", min_value=0, step=1)
    rc = st.number_input("Red Blood Cell Count (rc)", min_value=0.0, step=0.1)
    htn = st.selectbox("Hypertension (htn)", ["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus (dm)", ["yes", "no"])
    cad = st.selectbox("Coronary Artery Disease (cad)", ["yes", "no"])
    appet = st.selectbox("Appetite (appet)", ["good", "poor"])
    pe = st.selectbox("Pedal Edema (pe)", ["yes", "no"])
    ane = st.selectbox("Anemia (ane)", ["yes", "no"])

    # Convert categorical inputs to numerical as done during model training
    categorical_map = {
        "rbc": {"normal": 1, "abnormal": 0},
        "pc": {"normal": 1, "abnormal": 0},
        "pcc": {"notpresent": 0, "present": 1},
        "ba": {"notpresent": 0, "present": 1},
        "htn": {"yes": 1, "no": 0},
        "dm": {"yes": 1, "no": 0},
        "cad": {"yes": 1, "no": 0},
        "appet": {"good": 0, "poor": 1},
        "pe": {"yes": 1, "no": 0},
        "ane": {"yes": 1, "no": 0},
    }

    inputs = [
        age,
        bp,
        sg,
        al,
        su,
        categorical_map["rbc"][rbc],
        categorical_map["pc"][pc],
        categorical_map["pcc"][pcc],
        categorical_map["ba"][ba],
        bgr,
        bu,
        sc,
        sod,
        pot,
        hemo,
        pcv,
        wc,
        rc,
        categorical_map["htn"][htn],
        categorical_map["dm"][dm],
        categorical_map["cad"][cad],
        categorical_map["appet"][appet],
        categorical_map["pe"][pe],
        categorical_map["ane"][ane],
    ]

    # Predict button
    if st.button("Predict"):
        prediction = model.predict([inputs])
        if prediction[0] == 1:
            st.error("Prediction: Chronic Kidney Disease Detected!")
        else:
            st.success("Prediction: No Chronic Kidney Disease Detected.")

# Run the app
if __name__ == "_main_":
    main()