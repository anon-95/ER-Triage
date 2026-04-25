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

# Page configuration
st.set_page_config(
    page_title="The Queue Cure",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    :root {
        --primary-color: #0066CC;
        --danger-color: #E84C3D;
        --success-color: #27AE60;
        --light-bg: #F5F7FA;
    }
    
    .main {
        background-color: #FFFFFF;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .section-header {
        color: #0066CC;
        font-size: 18px;
        font-weight: 600;
        border-bottom: 2px solid #0066CC;
        padding-bottom: 10px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    .info-box {
        background-color: #F5F7FA;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0066CC;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

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

# Header
col_logo, col_title = st.columns([1, 3])
with col_logo:
    st.markdown("🏥")
with col_title:
    st.markdown("<h1 style='color: #0066CC; margin: 0;'>The Queue Cure</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; margin: 0; font-size: 14px;'>Where AI Meets Emergency Care</p>", unsafe_allow_html=True)

st.divider()

st.markdown("<div class='section-header'>👤 Patient Information</div>", unsafe_allow_html=True)

# ---------- ROW 1 - Demographics ----------
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    Group = st.selectbox(
        "🏢 Facility Group",
        ["", "Local ER", "Regional ER"],
        help="Select the type of emergency facility"
    )

with col2:
    Arrival_mode = st.selectbox(
        "🚑 Arrival Mode",
        ["", "Walk-in", "Ambulance", "Other"],
        help="How the patient arrived at the facility"
    )

with col3:
    Sex = st.selectbox(
        "👥 Sex",
        ["", "Male", "Female"],
        help="Patient's biological sex"
    )

# ---------- ROW 2 - Injury Assessment ----------
col4, col5, col6 = st.columns(3, gap="medium")

with col4:
    Age = st.number_input(
        "👶 Age (years)",
        min_value=0,
        max_value=150,
        value=None,
        help="Patient's age in years"
    )

with col5:
    Injury = st.selectbox(
        "🤕 Injury Present",
        ["", "Yes", "No"],
        help="Does the patient have an injury?"
    )

with col6:
    Pain = st.selectbox(
        "⚡ Pain Reported",
        ["", "Yes", "No"],
        help="Does the patient report pain?"
    )

# ---------- ROW 3 - Pain & Mental Assessment ----------
col7, col8, col9 = st.columns(3, gap="medium")

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
    pain_label = st.selectbox(
        "📊 Pain Level (NRS)",
        pain_keys,
        help="Numeric Rating Scale for pain (0-10)"
    )
    NRS_pain = pain_options[pain_label] if pain_label != "" else np.nan

with col8:
    mental_options = {
        "Alert (Normal)": 0,
        "Verbal response": 1,
        "Pain response": 2,
        "Unresponsive": 3
    }
    mental_keys = [""] + list(mental_options.keys())
    mental_label = st.selectbox(
        "🧠 Mental State",
        mental_keys,
        help="Patient's level of consciousness"
    )
    Mental = mental_options[mental_label] if mental_label != "" else np.nan

with col9:
    BT = st.number_input(
        "🌡️ Body Temp (°C)",
        min_value=30.0,
        max_value=45.0,
        value=None,
        step=0.1,
        help="Patient's body temperature"
    )

# ---------- ROW 4 - Vital Signs ----------
st.markdown("<div class='section-header'>❤️ Vital Signs</div>", unsafe_allow_html=True)

col10, col11, col12 = st.columns(3, gap="medium")

with col10:
    SBP = st.number_input(
        "Systolic BP (mmHg)",
        min_value=50,
        max_value=200,
        value=None,
        help="Systolic blood pressure"
    )

with col11:
    DBP = st.number_input(
        "Diastolic BP (mmHg)",
        min_value=30,
        max_value=150,
        value=None,
        help="Diastolic blood pressure"
    )

with col12:
    Saturation = st.number_input(
        "O₂ Saturation (%)",
        min_value=50,
        max_value=100,
        value=None,
        help="Blood oxygen saturation"
    )

# ---------- ROW 5 - Additional Vitals ----------
col13, col14, col15 = st.columns(3, gap="medium")

with col13:
    HR = st.number_input(
        "Heart Rate (bpm)",
        min_value=30,
        max_value=200,
        value=None,
        help="Beats per minute"
    )

with col14:
    RR = st.number_input(
        "Respiration Rate (bpm)",
        min_value=5,
        max_value=50,
        value=None,
        help="Breaths per minute"
    )

with col15:
    KTAS_RN = st.number_input(
        "Current KTAS Level",
        min_value=1,
        max_value=5,
        value=None,
        help="Kanadian Triage and Acuity Scale (1-5)"
    )

# ---------- TEXT ----------
st.markdown("<div class='section-header'>📝 Chief Complaint</div>", unsafe_allow_html=True)
Chief_complain = st.text_area(
    "Describe the patient's chief complaint",
    height=100,
    placeholder="Enter patient's main complaint or symptoms...",
    help="Provide details about the patient's primary complaint"
)

# ----------- PREDICTION -----------
st.divider()
col_button, col_info = st.columns([1, 3])

with col_button:
    submit_button = st.button(
        "🔍 Calculate KTAS Level",
        use_container_width=True,
        type="primary"
    )

with col_info:
    st.markdown("<p style='font-size: 12px; color: #666;'>AI-powered triage recommendation based on patient assessment</p>", unsafe_allow_html=True)

if submit_button:

    # Check if all required fields are filled
    if not all([Group, Arrival_mode, Sex, Age, Injury, Pain, NRS_pain, Mental, BT, 
                SBP, DBP, Saturation, HR, RR, KTAS_RN, Chief_complain]):
        st.warning("⚠️ Please fill in all required fields before calculating.")
    else:
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

        fatal_levels = []

        if pd.notnull(Mental) and Mental == 3:
            fatal_levels.append(1)
        if pd.notnull(SBP) and SBP < 80:
            fatal_levels.append(1)
        if pd.notnull(Saturation) and Saturation < 85:
            fatal_levels.append(1)  
        if pd.notnull(HR) and (HR < 40 or HR > 150):
            fatal_levels.append(2)
        if pd.notnull(RR) and (RR < 8 or RR > 35):
            fatal_levels.append(2)
        if pd.notnull(BT) and (BT < 32 or BT > 40):
            fatal_levels.append(2)
        
        fatal_level = min(fatal_levels) if fatal_levels else None
        
        st.divider()
        
        if fatal_level is not None:
            st.error(f"🚨 CRITICAL CONDITION DETECTED\n\nRecommended KTAS Level: {fatal_level}\n\nImmediate physician assessment required!")
        else:
            prediction = model.predict(combined)[0]
            
            # Color coding for KTAS levels
            ktas_colors = {
                1: "🔴 Resuscitation",
                2: "🟠 Emergent",
                3: "🟡 Urgent",
                4: "🟢 Semi-urgent",
                5: "🔵 Non-urgent"
            }
            
            st.success(f"✅ Assessment Complete")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric(
                    label="AI-Recommended KTAS Level",
                    value=int(prediction),
                    delta="Triage algorithm result"
                )
            
            with col_result2:
                st.info(f"**Classification:** {ktas_colors.get(int(prediction), 'Unknown')}")
            
            st.markdown("""
            ---
            ### KTAS Scale Overview:
            - **Level 1 (🔴 Red):** Resuscitation - Immediate life-threat
            - **Level 2 (🟠 Orange):** Emergent - High risk of deterioration  
            - **Level 3 (🟡 Yellow):** Urgent - Significant complexity/risk
            - **Level 4 (🟢 Green):** Semi-urgent - Lower severity
            - **Level 5 (🔵 Blue):** Non-urgent - Minor injuries/illness
            """)
