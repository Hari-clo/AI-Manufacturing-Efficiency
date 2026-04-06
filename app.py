# =========================================================
# 🏭 AI MANUFACTURING INTELLIGENCE DASHBOARD
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# LOAD MODEL
# -------------------------------
rf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Manufacturing AI System", layout="wide")

# -------------------------------
# TITLE
# -------------------------------
st.title("🏭 Smart Manufacturing Efficiency Dashboard")

st.markdown("""
### 🎯 What This System Solves
Traditional factories rely on manual monitoring.  
This AI system provides **instant efficiency classification** and highlights **root causes of inefficiency**.

👉 Helps reduce:
- Production loss  
- Machine instability  
- Network inefficiencies  
""")

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("⚙️ Machine Inputs")

temp = st.sidebar.slider("Temperature", 0.0, 200.0, 50.0)
vibration = st.sidebar.slider("Vibration", 0.0, 100.0, 10.0)
power = st.sidebar.slider("Power", 1.0, 500.0, 100.0)
latency = st.sidebar.slider("Latency", 0.0, 500.0, 50.0)
packet_loss = st.sidebar.slider("Packet Loss", 0.0, 100.0, 5.0)
production = st.sidebar.slider("Production Speed (units/hr)", 0.0, 1000.0, 200.0)


operation_mode = st.sidebar.selectbox("Operation Mode", ["Normal", "High Load"])
operation_mode_encoded = 0 if operation_mode == "Normal" else 1

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
energy_eff = production / power if power != 0 else 0
network_quality = 100 - (packet_loss + latency / 10)

temp_stability = 0
vibration_stability = 0

# -------------------------------
# INPUT PREPARATION
# -------------------------------
# -------------------------------
# INPUT FEATURES (FIXED)
# -------------------------------
feature_names = scaler.feature_names_in_

input_dict = {f: 0 for f in feature_names}

input_dict["Temperature_C"] = temp
input_dict["Vibration_Hz"] = vibration
input_dict["Power_Consumption_kW"] = power
input_dict["Network_Latency_ms"] = latency
input_dict["Packet_Loss_%"] = packet_loss
input_dict["Operation_Mode"] = operation_mode_encoded

input_dict["Production_Speed_units_per_hr"] = production

input_dict["Energy_Efficiency"] = production / power if power != 0 else 0
input_dict["Network_Quality"] = 100 - (packet_loss + latency / 10)

input_dict["Temp_Stability"] = 0
input_dict["Vibration_Stability"] = 0

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)
# -------------------------------
# PREDICTION
# -------------------------------
if st.sidebar.button("🚀 Analyze"):

    prediction = rf.predict(input_scaled)
    probs = rf.predict_proba(input_scaled)

    confidence = np.max(probs) * 100

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    result = label_map[prediction[0]]

    # -------------------------------
    # RESULT
    # -------------------------------
    st.subheader("🎯 Efficiency Result")

    if result == "High":
        st.success(f"✅ HIGH Efficiency")
    elif result == "Medium":
        st.warning(f"⚠️ MEDIUM Efficiency")
    else:
        st.error(f"❌ LOW Efficiency")

    st.metric("Confidence", f"{confidence:.2f}%")

    # -------------------------------
    # GRAPH 1: SENSOR ANALYSIS
    # -------------------------------
    st.subheader("📊 Sensor Analysis")

    sensor_df = pd.DataFrame({
        "Metric": ["Temperature", "Vibration", "Power"],
        "Value": [temp, vibration, power]
    })

    fig1, ax1 = plt.subplots()
    ax1.bar(sensor_df["Metric"], sensor_df["Value"])
    ax1.set_title("Sensor Conditions")
    st.pyplot(fig1)

    # -------------------------------
    # GRAPH 2: NETWORK ANALYSIS
    # -------------------------------
    st.subheader("🌐 Network Analysis")

    net_df = pd.DataFrame({
        "Metric": ["Latency", "Packet Loss", "Network Quality"],
        "Value": [latency, packet_loss, network_quality]
    })

    fig2, ax2 = plt.subplots()
    ax2.bar(net_df["Metric"], net_df["Value"])
    ax2.set_title("Network Performance")
    st.pyplot(fig2)

    # -------------------------------
    # GRAPH 3: PERFORMANCE SCORE
    # -------------------------------
    st.subheader("📈 Performance Overview")

    performance = pd.DataFrame({
        "Category": ["Energy Efficiency", "Network Quality"],
        "Score": [energy_eff, network_quality]
    })

    fig3, ax3 = plt.subplots()
    ax3.bar(performance["Category"], performance["Score"])
    ax3.set_title("System Efficiency Indicators")
    st.pyplot(fig3)

    # -------------------------------
    # INSIGHTS
    # -------------------------------
    st.subheader("🔍 Insights")

    if temp > 120:
        st.write("• Overheating detected → impacts efficiency")

    if vibration > 50:
        st.write("• High vibration → machine instability")

    if latency > 200:
        st.write("• Network delay → affects coordination")

    if packet_loss > 20:
        st.write("• Data loss → unreliable communication")

    # -------------------------------
    # BUSINESS IMPACT
    # -------------------------------
    st.subheader("📉 Impact on Manufacturing")

    if result == "Low":
        st.write("""
        ❌ Production loss likely  
        ❌ Increased maintenance cost  
        ❌ Reduced output quality  
        """)
    else:
        st.write("""
        ✅ Stable production  
        ✅ Optimized performance  
        """)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("AI-Powered Smart Factory System 🚀")
