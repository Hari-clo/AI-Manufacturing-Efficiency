import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

rf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AI Manufacturing Intelligence System", layout="wide")

st.title("🏭 AI-Based Manufacturing Efficiency Intelligence System")

st.markdown("""
This system provides **real-time efficiency classification and insights** using sensor and network data.
""")

# SIDEBAR INPUT
st.sidebar.header("⚙️ Input Parameters")

temp = st.sidebar.slider("Temperature (°C)", 0.0, 200.0, 50.0)
vibration = st.sidebar.slider("Vibration (Hz)", 0.0, 100.0, 10.0)
power = st.sidebar.slider("Power Consumption (kW)", 1.0, 500.0, 100.0)
latency = st.sidebar.slider("Network Latency (ms)", 0.0, 500.0, 50.0)
packet_loss = st.sidebar.slider("Packet Loss (%)", 0.0, 100.0, 5.0)

# FEATURE ENGINEERING

energy_eff = 100 / power
network_quality = 100 - (packet_loss + latency / 10)

input_data = np.array([[temp, vibration, power, latency, packet_loss, energy_eff, network_quality]])
input_scaled = scaler.transform(input_data)


# PREDICTION
if st.sidebar.button("Analyze Efficiency"):

    prediction = rf.predict(input_scaled)
    probs = rf.predict_proba(input_scaled)

    confidence = np.max(probs) * 100

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    result = label_map[prediction[0]]

    st.subheader("🎯 Efficiency Status")

    if result == "High":
        st.success(f"✅ {result} Efficiency")
    elif result == "Medium":
        st.warning(f"⚠️ {result} Efficiency")
    else:
        st.error(f"❌ {result} Efficiency")

    st.metric("Confidence Score", f"{confidence:.2f}%")

# GRAPH SECTION
st.subheader("📊 Sensor & Network Impact Analysis")

data = {
    "Metric": ["Temperature", "Vibration", "Power", "Latency", "Packet Loss"],
    "Value": [temp, vibration, power, latency, packet_loss]
}

df_plot = pd.DataFrame(data)

fig, ax = plt.subplots()
ax.bar(df_plot["Metric"], df_plot["Value"])
st.pyplot(fig)

st.subheader("🔍 Key Insights")

st.write("""
• High vibration and temperature fluctuations reduce efficiency  
• Increased latency and packet loss impact coordination  
• Power usage without proportional output reduces efficiency  
""")

# RECOMMENDATIONS
st.subheader("💡 Recommendations")

st.write("""
• Optimize machine calibration to reduce vibration  
• Improve network reliability to reduce latency  
• Monitor power-to-output ratio for efficiency  
""")
