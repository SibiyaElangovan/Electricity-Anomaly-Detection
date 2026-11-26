
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Electricity Anomaly Detection",
    page_icon="⚡",
    layout="wide"
)

# Load saved files
@st.cache_data
def load_data():
    return pd.read_csv("electricity_anomaly_results.csv", parse_dates=["Date"])

df = load_data()
hmm_model = joblib.load("electricity_hmm.pkl")
scaler = joblib.load("electricity_scaler.pkl")

# Dashboard title
st.title("⚡ Electricity Consumption Anomaly Detection")
st.markdown(
    "### Hidden Markov Model (HMM) Based Analysis"
)
st.write(
    "This system identifies hidden electricity consumption "
    "states and detects potentially unusual consumption patterns."
)

st.divider()

# Summary metrics
total_readings = len(df)
total_anomalies = int(df["Anomaly"].sum())
normal_readings = total_readings - total_anomalies
anomaly_percentage = total_anomalies / total_readings * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Readings", f"{total_readings:,}")
col2.metric("Normal Readings", f"{normal_readings:,}")
col3.metric("Potential Anomalies", f"{total_anomalies:,}")
col4.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")

st.divider()

# Electricity consumption graph
st.subheader("📈 Electricity Consumption Pattern")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["Date"], df["Consumption"], linewidth=0.7)
ax.set_xlabel("Date")
ax.set_ylabel("Electricity Consumption")
ax.set_title("Electricity Consumption Over Time")
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.divider()

# Hidden states
st.subheader("🔍 Hidden Consumption States")

state_summary = (
    df.groupby("Hidden_State")["Consumption"]
    .agg(["mean", "count"])
    .reset_index()
)

state_summary.columns = [
    "Hidden State", "Average Consumption", "Number of Readings"
]

state_summary["Consumption Level"] = state_summary[
    "Hidden State"
].map({
    0: "Lower Consumption",
    1: "Higher Consumption"
})

st.dataframe(state_summary, use_container_width=True)

# State distribution chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    state_summary["Consumption Level"],
    state_summary["Number of Readings"]
)
ax.set_title("Distribution of Hidden Consumption States")
ax.set_ylabel("Number of Readings")
st.pyplot(fig)
plt.close(fig)

st.divider()

# Anomaly detection results
st.subheader("🚨 Potential Anomaly Detection")

anomalies = df[df["Anomaly"] == True].copy()

st.write(f"Total potential anomalies detected: **{len(anomalies):,}**")

if not anomalies.empty:
    st.dataframe(
        anomalies.sort_values("Probability"),
        use_container_width=True
    )

    csv = anomalies.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Anomaly Report",
        data=csv,
        file_name="potential_anomalies.csv",
        mime="text/csv"
    )
else:
    st.info("No potential anomalies detected.")

st.divider()

# Project information
st.subheader("📌 About This Project")

st.write("""
**Project:** Detection of Unusual Electricity Consumption Patterns Using HMM.

**Dataset:** UCI Electricity Load Diagrams 2011–2014.

**Algorithm:** Gaussian Hidden Markov Model (HMM).

**Objective:** Identify hidden electricity consumption states
and detect potentially unusual consumption readings.
""")

st.caption("Developed as an individual academic project.")
