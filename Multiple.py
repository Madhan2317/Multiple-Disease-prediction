import streamlit as st
import pickle
import numpy as np

# Load models and scalers
kidney_model = pickle.load(open('D:/MDTE21/Multi Diseses Prediction/D2_random_forest_model.pkl', 'rb'))
kidney_scaler = pickle.load(open('D:/MDTE21/Multi Diseses Prediction/D2_scaler.pkl', 'rb'))

liver_model = pickle.load(open('D:/MDTE21/Multi Diseses Prediction/D1_random_forest_model.pkl', 'rb'))
liver_scaler = pickle.load(open('D:/MDTE21/Multi Diseses Prediction/D1_scaler.pkl', 'rb'))

parkinson_model = pickle.load(open('D:/MDTE21/Multi Diseses Prediction/D3_random_forest_model.pkl', 'rb'))
parkinson_scaler = pickle.load(open('D:/MDTE21/Multi Diseses Prediction/D3_scaler.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

# Set background image using CSS
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQjWEuBZke7cq9CeebCAoc_hE4EM7jSlDxwtA&s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;    
    }
    .main {
        background-color: reduced transparency from 0.7 to 0 
        padding: 2rem;
        border-radius: 10px;
    }
     .block-container {
        background-color: rgba(30,25,40, 0.75);
        padding: 2rem;
        border-radius: 12px;
        layout: wide;
    }
    </style>
""", unsafe_allow_html=True)


# Disease selection UI (top buttons)
st.markdown("""
    <style>
        .disease-buttons {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .disease-buttons button {
            margin: 0 10px;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #FF4B4B;'>🧠 Multiple Disease Prediction</h1>
        <h4>Select a disease to begin</h4>
    </div>
""", unsafe_allow_html=True)

# Custom-styled selectbox label
st.markdown("<h3 style='text-align: center; font-size: 28px;'>🔍 Select Disease for Prediction:</h3>", unsafe_allow_html=True)
disease_option = st.selectbox("", ["Kidney Disease", "Liver Disease", "Parkinson's Disease"], index=0)

# Kidney Disease Prediction
if disease_option == "Kidney Disease":
    st.subheader("🩸 Kidney Disease Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1.0)
        bp = st.number_input("Blood Pressure")
        sg = st.number_input("Specific Gravity")
        al = st.number_input("Albumin")
        su = st.number_input("Sugar")
        pcv = st.number_input("Packed Cell Volume")
        htn = st.selectbox("Hypertension", ["0", "1"])
        dm = st.selectbox("Diabetes Mellitus", ["0", "1"])
    with col2:
        rbc = st.selectbox("Red Blood Cells", ["0", "1"])
        rc = st.number_input("Red Blood Cell Count")
        wc = st.number_input("White Blood Cell Count")
        pc = st.selectbox("Pus Cell", ["0", "1"])
        pcc = st.selectbox("Pus Cell Clumps", ["0", "1"])
        ba = st.selectbox("Bacteria", ["0", "1"])
        bgr = st.number_input("Blood Glucose Random")
        ane = st.selectbox("Anemia", ["0", "1"])       
    with col3:
        bu = st.number_input("Blood Urea")
        sc = st.number_input("Serum Creatinine")
        sod = st.number_input("Sodium")
        pot = st.number_input("Potassium")
        hemo = st.number_input("Hemoglobin")
        cad = st.selectbox("Coronary Artery Disease", ["0", "1"])
        appet = st.selectbox("Appetite", ["0", "1"])
        pe = st.selectbox("Pedal Edema", ["0", "1"])
        
    if st.button("Predict"):
        input_data = np.array([[age, bp, sg, al, su, int(rbc), int(pc), int(pcc), int(ba),
                                bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                                int(htn), int(dm), int(cad), int(appet), int(pe), int(ane)]])

        input_data_scaled = kidney_scaler.transform(input_data)
        pred = kidney_model.predict(input_data_scaled)[0]

        st.success("✅ CKD Detected" if pred == 1 else "❎ No CKD Detected")

# Liver Disease Prediction
elif disease_option == "Liver Disease":
    st.subheader("🧬 Liver Disease Prediction")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tb = st.number_input("Total Bilirubin (mg/dL)", format="%.2f")
        db = st.number_input("Direct Bilirubin (mg/dL)", format="%.2f")
        alkphos = st.number_input("Alkaline Phosphotase (IU/L)", format="%.2f")
    with col2:
        sgpt = st.number_input("Alanine Aminotransferase (IU/L)", format="%.2f")
        sgot = st.number_input("Aspartate Aminotransferase (IU/L)", format="%.2f")
        tp = st.number_input("Total Proteins (g/dL)", format="%.2f")
        alb = st.number_input("Albumin (g/dL)", format="%.2f")
        ag_ratio = st.number_input("Albumin and Globulin Ratio", format="%.2f")

    gender_encoded = 1 if gender == "Male" else 0

    if st.button("Predict"):
        input_data = np.array([[age, gender_encoded, tb, db, alkphos, sgot, sgpt, tp, alb, ag_ratio]])
        input_scaled = liver_scaler.transform(input_data)
        prediction = liver_model.predict(input_scaled)[0]
        st.success("✅ Liver Disease Detected" if prediction == 1 else "❎ No Liver Disease Detected")

# Parkinson's Disease Prediction
elif disease_option == "Parkinson's Disease":
    st.subheader("🗣️ Parkinson's Disease Prediction")

    cols = st.columns(6)
    fo = cols[0].number_input("MDVP:Fo(Hz)")
    fhi = cols[1].number_input("MDVP:Fhi(Hz)")
    flo = cols[2].number_input("MDVP:Flo(Hz)")
    jitter_percent = cols[3].number_input("MDVP:Jitter(%)")
    rpde = cols[4].number_input("RPDE")
    dfa = cols[5].number_input("DFA")

    cols2 = st.columns(4)
    jitter_abs = cols2[0].number_input("MDVP:Jitter(Abs)")
    rap = cols2[1].number_input("MDVP:RAP")
    ppq = cols2[2].number_input("MDVP:PPQ")
    ddp = cols2[3].number_input("Jitter:DDP")

    cols3 = st.columns(6)
    shimmer = cols3[0].number_input("MDVP:Shimmer")
    shimmer_db = cols3[1].number_input("MDVP:Shimmer(dB)")
    apq3 = cols3[2].number_input("Shimmer:APQ3")
    apq5 = cols3[3].number_input("Shimmer:APQ5")
    spread1 = cols3[4].number_input("spread1")
    spread2 = cols3[5].number_input("spread2")

    cols4 = st.columns(6)
    apq = cols4[0].number_input("MDVP:APQ")
    dda = cols4[1].number_input("Shimmer:DDA")
    nhr = cols4[2].number_input("NHR")
    hnr = cols4[3].number_input("HNR")
    d2 = cols4[4].number_input("D2")
    ppe = cols4[5].number_input("PPE")

    if st.button("Predict"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                                rpde, dfa, spread1, spread2, d2, ppe]])
        input_scaled = parkinson_scaler.transform(input_data)
        result = parkinson_model.predict(input_scaled)[0]
        st.success("✅ Parkinson's Disease Detected" if result == 1 else "❎ No Parkinson's Disease Detected")
