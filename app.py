import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download nltk once
nltk.download('stopwords')
nltk.download('wordnet')

# load model + tfidf
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# SAME clean_text function (unchanged)
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# column definitions (same as your code)
categorical_cols = ["Group", "Sex", "Injury", "Pain", "Arrival mode"]
numeric_cols = ["Age", "Mental", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation", "KTAS_RN"]

st.title("🏥 Patient Triage Prediction App")

st.header("Enter Patient Details")

# ----------- UI INPUTS -----------

# categorical inputs
Group = st.selectbox("Group", ["Adult", "Pediatric"])
Sex = st.selectbox("Sex", ["Male", "Female"])
Injury = st.selectbox("Injury", ["Yes", "No"])
Pain = st.selectbox("Pain", ["Yes", "No"])
Arrival_mode = st.selectbox("Arrival mode", ["Walk-in", "Ambulance", "Other"])

# numeric inputs
Age = st.number_input("Age", 0, 120, 30)
Mental = st.number_input("Mental Score", 0, 10, 5)
NRS_pain = st.number_input("Pain Score", 0, 10, 5)
SBP = st.number_input("SBP", 50, 200, 120)
DBP = st.number_input("DBP", 30, 150, 80)
HR = st.number_input("Heart Rate", 30, 200, 80)
RR = st.number_input("Respiratory Rate", 5, 50, 18)
BT = st.number_input("Body Temp", 30.0, 45.0, 37.0)
Saturation = st.number_input("Oxygen Saturation", 50, 100, 98)
KTAS_RN = st.number_input("KTAS RN", 1, 5, 3)

# text input
Chief_complain = st.text_area("Chief Complaint")

# ----------- PREDICTION -----------

if st.button("Predict"):

    structured_data = {
        "Group": Group,
        "Sex": Sex,
        "Injury": Injury,
        "Pain": Pain,
        "Arrival mode": Arrival_mode,
        "Age": Age,
        "Mental": Mental,
        "NRS_pain": NRS_pain,
        "SBP": SBP,
        "DBP": DBP,
        "HR": HR,
        "RR": RR,
        "BT": BT,
        "Saturation": Saturation,
        "KTAS_RN": KTAS_RN
    }

    # convert to dataframe
    row = pd.DataFrame([structured_data])

    # clean text
    cleaned = clean_text(Chief_complain)
    text_vec = tfidf.transform([cleaned])

    # combine
    combined = hstack([csr_matrix(row[categorical_cols + numeric_cols].values), text_vec])

    prediction = model.predict(combined)[0]

    st.success(f"Predicted KTAS Level: {prediction}")
