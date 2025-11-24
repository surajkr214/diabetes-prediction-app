import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="Diabetes Predictor", layout="wide")
st.title("🩺 Diabetes Prediction System (KNN)")

# --- LOAD & CLEAN DATA ---
try:
    # Using the absolute path as requested
    dataset = pd.read_csv('diabetes.csv')
    
    # FIX: Using np.nan (lowercase) instead of np.NaN (uppercase/deprecated)
    list_no_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for column in list_no_zero:
        dataset[column] = dataset[column].replace(0, np.nan)
        mean = int(dataset[column].mean(skipna=True))
        dataset[column] = dataset[column].replace(np.nan, mean)

    X = dataset.iloc[:, 0:8]
    Y = dataset.iloc[:, 8]

    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("⚙️ Model Settings")
    # Using the optimized settings you found earlier
    best_state = st.sidebar.number_input("Random State (Seed)", 0, 1000, 42)
    best_k = st.sidebar.slider("Number of Neighbors (K)", 1, 21, 13, step=2)

    # --- TRAIN MODEL ---
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=best_state, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=best_k, p=2, metric='euclidean')
    classifier.fit(X_train, Y_train)
    
    # Show accuracy
    acc = accuracy_score(Y_test, classifier.predict(X_test))
    st.sidebar.success(f"Current Accuracy: {acc*100:.2f}%")

    # --- PATIENT INPUT UI ---
    st.subheader("Enter Patient Data")
    col1, col2 = st.columns(2)
    
    with col1:
        preg = st.number_input("Pregnancies", 0, 20, 1)
        gluc = st.slider("Glucose Level", 40, 200, 110)
        bp = st.slider("Blood Pressure", 40, 140, 70)
        skin = st.slider("Skin Thickness", 10, 100, 20)
        
    with col2:
        ins = st.slider("Insulin Level", 10, 900, 79)
        bmi = st.slider("BMI", 10.0, 70.0, 32.0)
        dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
        age = st.slider("Age", 15, 100, 33)

    # --- PREDICTION LOGIC ---
    if st.button("Predict Result"):
        # Scale the single input using the same scaler as training
        user_input = np.array([[preg, gluc, bp, skin, ins, bmi, dpf, age]])
        user_input_scaled = scaler.transform(user_input)
        
        prediction = classifier.predict(user_input_scaled)
        prob = classifier.predict_proba(user_input_scaled)
        
        st.write("---")
        if prediction[0] == 1:
            st.error(f"⚠️ **Result: DIABETIC**")
            st.write(f"Confidence: {prob[0][1] * 100:.2f}%")
        else:
            st.success(f"✅ **Result: HEALTHY (Non-Diabetic)**")
            st.write(f"Confidence: {prob[0][0] * 100:.2f}%")

except FileNotFoundError:
    st.error("CRITICAL ERROR: /content/diabetes.csv not found.")
    st.info("Please drag and drop 'diabetes.csv' into the Files folder on the left.")
except Exception as e:
    st.error(f"An error occurred: {e}")