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
encoder = joblib.load("encoder.pkl")
svd = joblib.load("svd.pkl")


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# clean_text function 
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# column definitions
categorical_cols = ["Group", "Sex", "Injury", "Pain", "Arrival mode"]
numeric_cols = ["Age", "Mental", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation", "KTAS_RN"]

st.title("The Queue Cure")

st.header("Enter Patient Details")

# ---------- ROW 1 ----------
col1, col2, col3 = st.columns(3)

with col1:
    Group = st.selectbox("Group", ["", "Local ER", "Regional ER"])

with col2:
    Arrival_mode = st.selectbox("Arrival", ["", "Walk-in", "Ambulance", "Other"])


with col3:
    Sex = st.selectbox("Sex", ["", "Male", "Female"])


# ---------- ROW 2 ----------
col4, col5, col6 = st.columns(3)

with col4:
    Age = st.number_input("Age", 0, value=None)


with col5:
    Injury = st.selectbox("Injury", ["", "Yes", "No"])


with col6:
    Pain = st.selectbox("Pain", ["", "Yes", "No"])


# ---------- ROW 3 ----------
col7, col8, col9 = st.columns(3)

with col7:
    pain_options = {
        "0: No pain": 0,
        "1: Very Mild Pain, barely noticeable. Most of the time you don't think about it.": 1,
        "2: Minor pain. It's annoying. You may have sharp pain now and then.": 2,
        "3: Noticeable pain. It may distract you, but you can get used to it.": 3,
        "4: Moderate pain. You can ignore the pain for a while, but it is still distracting.": 4,
        "5: Moderately strong pain. You can't ignore it for more than a few minutes. But with effort you can still do work.": 5,
        "6: Moderately stronger pain. You avoid some of your normal daily activities. You have trouble concentrating.": 6,
        "7: Strong pain. It keeps you from doing normal activities": 7,
        "8: Very strong pain. It's hard to do anything at all.": 8,
        "9: Pain that is very hard to bear. You can't carry on a conversation": 9,
        "10: Worst pain imaginable": 10,

    }
    pain_keys = [""] + list(pain_options.keys())
    pain_label = st.selectbox("Pain Level", pain_keys)
    NRS_pain = pain_options[pain_label] if pain_label != "" else np.nan
with col8:
    mental_options = {
        "Alert (Normal)": 0,
        "Verbal response": 1,
        "Pain response": 2,
        "Unresponsive": 3
    }
    mental_keys = [""] + list(mental_options.keys())
    mental_label = st.selectbox("Mental State", mental_keys)
    Mental = mental_options[mental_label] if mental_label != "" else np.nan


with col9:
    BT = st.number_input("Body temperature", 30.0, 45.0, value=None)


# ---------- ROW 4 ----------
col10, col11, col12 = st.columns(3)

with col10:
    SBP = st.number_input("Systolic Blood Pressure", 50, 200, value=None)

with col11:
    DBP = st.number_input("Diastolic Blood Pressure", 30, 150, value=None)

with col12:
    Saturation = st.number_input("O2 Saturation", 50, 100, value=None)


# ---------- ROW 5 ----------
col13, col14, col15 = st.columns(3)

with col13:
    HR = st.number_input("Heart rate", 30, 200, value=None)

with col14:
    RR = st.number_input("Respiration rate", 5, 50, value=None)
with col15:
    KTAS_RN = st.number_input("Current KTAS Level", 1, 5, value=None)

# ---------- TEXT ----------
Chief_complain = st.text_area("Chief Complaint", height=80, placeholder="Enter your chief complaint...")
# ----------- PREDICTION -----------

if st.button("Calculate"):

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

    # ✅ APPLY ENCODER (this fixes dtype issue)
    row[categorical_cols] = encoder.transform(row[categorical_cols])

    # clean text
    cleaned = clean_text(Chief_complain)

    # TF-IDF + SVD (same as training)
    text_vec = tfidf.transform([cleaned])
    text_vec = svd.transform(text_vec)

    # combine
    combined = hstack([
        csr_matrix(row[categorical_cols + numeric_cols].astype(float).values),
        text_vec
    ])

    prediction = model.predict(combined)[0]
    st.success(f"AI KTAS Level: {prediction}")
