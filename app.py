import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 📂 LOAD MODEL & SCALER
rf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AI Manufacturing System", layout="wide")

# 🏷️ TITLE & DESCRIPTION
st.title("🏭 AI-Based Manufacturing Efficiency Intelligence System")

st.markdown("""
This system provides **real-time efficiency classification** using sensor and network data.  
It also offers **insights and recommendations** for improving manufacturing performance.
""")

# 🎛️ SIDEBAR INPUTS
st.sidebar.header("⚙️ Machine Inputs")

temp = st.sidebar.slider("Temperature (°C)", 0.0, 200.0, 50.0)
vibration = st.sidebar.slider("Vibration (Hz)", 0.0, 100.0, 10.0)
power = st.sidebar.slider("Power Consumption (kW)", 1.0, 500.0, 100.0)
latency = st.sidebar.slider("Network Latency (ms)", 0.0, 500.0, 50.0)
packet_loss = st.sidebar.slider("Packet Loss (%)", 0.0, 100.0, 5.0)

operation_mode = st.sidebar.selectbox("Operation Mode", ["Normal", "High Load"])

# Encode operation mode
operation_mode_encoded = 0 if operation_mode == "Normal" else 1

# 🧠 FEATURE ENGINEERING
energy_eff = 100 / power if power != 0 else 0
network_quality = 100 - (packet_loss + latency / 10)

# Stability (no rolling in real-time)
temp_stability = 0
vibration_stability = 0


# 📊 INPUT DATA (MATCH TRAINING ORDER)

input_data = np.array([[
    temp,
    vibration,
    power,
    latency,
    packet_loss,
    operation_mode_encoded,
    energy_eff,
    network_quality,
    temp_stability,
    vibration_stability
]])

# Scale input
input_scaled = scaler.transform(input_data)

# 🔮 PREDICTION BUTTON
if st.sidebar.button("🚀 Analyze Efficiency"):

    prediction = rf.predict(input_scaled)
    probs = rf.predict_proba(input_scaled)

    confidence = np.max(probs) * 100

    # Map labels
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    result = label_map[prediction[0]]

    st.markdown("---")

   
    # 🎯 RESULT DISPLAY
    st.subheader("🎯 Efficiency Status")

    if result == "High":
        st.success(f"✅ {result} Efficiency")
        st.balloons()
    elif result == "Medium":
        st.warning(f"⚠️ {result} Efficiency")
    else:
        st.error(f"❌ {result} Efficiency")

    st.metric("Confidence Score", f"{confidence:.2f}%")

   
    # 📊 INPUT VISUALIZATION
    st.subheader("📊 Input Metrics Overview")

    df_plot = pd.DataFrame({
        "Metric": ["Temperature", "Vibration", "Power", "Latency", "Packet Loss"],
        "Value": [temp, vibration, power, latency, packet_loss]
    })

    fig, ax = plt.subplots()
    ax.bar(df_plot["Metric"], df_plot["Value"])
    ax.set_ylabel("Values")
    ax.set_title("Current Machine Condition")
    st.pyplot(fig)

   
    # 🔍 INSIGHTS
   
    st.subheader("🔍 Key Insights")

    if temp > 120:
        st.write("• High temperature detected → possible overheating risk")

    if vibration > 50:
        st.write("• High vibration → machine instability")

    if latency > 200:
        st.write("• High network latency → communication delay")

    if packet_loss > 20:
        st.write("• Packet loss detected → unreliable data transmission")

    if power > 300:
        st.write("• High power usage → possible inefficiency")

   
    # 💡 RECOMMENDATIONS
   
    st.subheader("💡 Recommendations")

    st.write("""
    • Optimize machine calibration to reduce vibration  
    • Improve cooling systems to manage temperature  
    • Upgrade network infrastructure to reduce latency  
    • Monitor power consumption vs output  
    """)


# 📌 FOOTER

st.markdown("---")
st.markdown("Developed as an AI-driven smart manufacturing solution 🚀")
