# =========================================================
# 🏭 AI MANUFACTURING EFFICIENCY INTELLIGENCE SYSTEM
# =========================================================

# -------------------------------
# 📦 IMPORT LIBRARIES
# -------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# 📂 LOAD MODEL & SCALER
# -------------------------------
rf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# ⚙️ PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="AI Manufacturing System", layout="wide")

# -------------------------------
# 🏷️ TITLE & DESCRIPTION
# -------------------------------
st.title("🏭 AI-Based Manufacturing Efficiency Intelligence System")

st.markdown("""
This system provides **real-time efficiency classification** using sensor and network data.  
It also offers **insights and recommendations** for improving manufacturing performance.
""")

# -------------------------------
# 🎛️ SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("⚙️ Machine Inputs")

temp = st.sidebar.slider("Temperature (°C)", 0.0, 200.0, 50.0)
vibration = st.sidebar.slider("Vibration (Hz)", 0.0, 100.0, 10.0)
power = st.sidebar.slider("Power Consumption (kW)", 1.0, 500.0, 100.0)
latency = st.sidebar.slider("Network Latency (ms)", 0.0, 500.0, 50.0)
packet_loss = st.sidebar.slider("Packet Loss (%)", 0.0, 100.0, 5.0)

operation_mode = st.sidebar.selectbox("Operation Mode", ["Normal", "High Load"])
operation_mode_encoded = 0 if operation_mode == "Normal" else 1

# -------------------------------
# 🧠 FEATURE ENGINEERING
# -------------------------------
energy_eff = 100 / power if power != 0 else 0
network_quality = 100 - (packet_loss + latency / 10)

temp_stability = 0
vibration_stability = 0

# -------------------------------
# 📊 CREATE INPUT USING TRAINING FEATURES
# -------------------------------
feature_names = scaler.feature_names_in_

input_dict = {feature: 0 for feature in feature_names}

# Fill known features
if "Temperature_C" in input_dict:
    input_dict["Temperature_C"] = temp

if "Vibration_Hz" in input_dict:
    input_dict["Vibration_Hz"] = vibration

if "Power_Consumption_kW" in input_dict:
    input_dict["Power_Consumption_kW"] = power

if "Network_Latency_ms" in input_dict:
    input_dict["Network_Latency_ms"] = latency

if "Packet_Loss_%" in input_dict:
    input_dict["Packet_Loss_%"] = packet_loss

if "Operation_Mode" in input_dict:
    input_dict["Operation_Mode"] = operation_mode_encoded

if "Energy_Efficiency" in input_dict:
    input_dict["Energy_Efficiency"] = energy_eff

if "Network_Quality" in input_dict:
    input_dict["Network_Quality"] = network_quality

if "Temp_Stability" in input_dict:
    input_dict["Temp_Stability"] = temp_stability

if "Vibration_Stability" in input_dict:
    input_dict["Vibration_Stability"] = vibration_stability

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale input
input_scaled = scaler.transform(input_df)

# -------------------------------
# 🔮 PREDICTION
# -------------------------------
if st.sidebar.button("🚀 Analyze Efficiency"):

    prediction = rf.predict(input_scaled)
    probs = rf.predict_proba(input_scaled)

    confidence = np.max(probs) * 100

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    result = label_map[prediction[0]]

    st.markdown("---")

    # -------------------------------
    # 🎯 RESULT DISPLAY
    # -------------------------------
    st.subheader("🎯 Efficiency Status")

    if result == "High":
        st.success(f"✅ {result} Efficiency")
        st.balloons()
    elif result == "Medium":
        st.warning(f"⚠️ {result} Efficiency")
    else:
        st.error(f"❌ {result} Efficiency")

    st.metric("Confidence Score", f"{confidence:.2f}%")

    # -------------------------------
    # 📊 INPUT VISUALIZATION
    # -------------------------------
    st.subheader("📊 Machine Condition Overview")

    df_plot = pd.DataFrame({
        "Metric": ["Temperature", "Vibration", "Power", "Latency", "Packet Loss"],
        "Value": [temp, vibration, power, latency, packet_loss]
    })

    fig, ax = plt.subplots()
    ax.bar(df_plot["Metric"], df_plot["Value"])
    ax.set_ylabel("Values")
    ax.set_title("Current System State")
    st.pyplot(fig)

    # -------------------------------
    # 🔍 INSIGHTS
    # -------------------------------
    st.subheader("🔍 Key Insights")

    if temp > 120:
        st.write("• High temperature may indicate overheating risk")

    if vibration > 50:
        st.write("• Excessive vibration suggests machine instability")

    if latency > 200:
        st.write("• Network latency is high, may affect coordination")

    if packet_loss > 20:
        st.write("• Packet loss detected, communication reliability reduced")

    if power > 300:
        st.write("• High power usage may reduce efficiency")

    # -------------------------------
    # 💡 RECOMMENDATIONS
    # -------------------------------
    st.subheader("💡 Recommendations")

    st.write("""
    • Optimize machine calibration to reduce vibration  
    • Improve cooling mechanisms to control temperature  
    • Upgrade network infrastructure to reduce latency  
    • Monitor power consumption and optimize usage  
    """)

# -------------------------------
# 📌 FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Developed as an AI-driven smart manufacturing solution 🚀")
