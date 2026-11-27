
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Electricity Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# WEBSITE DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #fce7f3, #e0f2fe);
    color: #1e1b4b;
}

h1 { color: #4c1d95 !important; font-weight: 850 !important; }
h2 { color: #6d28d9 !important; font-weight: 750 !important; }
h3 { color: #7c3aed !important; font-weight: 700 !important; }

.stApp p, .stApp li, .stApp label {
    color: #312e81;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b, #4c1d95, #7e22ce);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* KPI CARDS */
[data-testid="stMetric"] {
    background: white;
    border: 2px solid #c4b5fd;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 5px 15px #4338ca20;
}

[data-testid="stMetricLabel"] {
    color: #4338ca !important;
}

[data-testid="stMetricValue"] {
    color: #312e81 !important;
    font-weight: 850 !important;
}

/* ALL ALERT BOXES: DARK BACKGROUND + WHITE TEXT */
[data-testid="stAlert"] {
    background: linear-gradient(135deg, #312e81, #581c87) !important;
    border: 1px solid #a78bfa !important;
    border-radius: 14px !important;
}

[data-testid="stAlert"] *,
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] strong,
[data-testid="stAlert"] div {
    color: #ffffff !important;
}

/* BUTTONS */
div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #db2777);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: bold;
}

div[data-testid="stDownloadButton"] button {
    background: linear-gradient(90deg, #0891b2, #2563eb);
    color: white !important;
    border-radius: 10px;
}

/* TABLES */
[data-testid="stDataFrame"] {
    border: 2px solid #c4b5fd;
    border-radius: 12px;
}

hr { border-color: #c4b5fd; }

</style>
""", unsafe_allow_html=True)


# =====================================================
# KPI CARD
# =====================================================

def kpi_card(title, value, icon, color):

    html = (
        f'<div style="background:linear-gradient(135deg,#ffffff,{color});'
        'padding:22px;border-radius:18px;'
        'border-left:7px solid #6d28d9;'
        'box-shadow:0 5px 18px #312e811c;'
        'min-height:125px;margin-bottom:12px;">'
        f'<div style="font-size:17px;color:#312e81;font-weight:700;">'
        f'{icon} {title}</div>'
        f'<div style="font-size:29px;color:#312e81;'
        f'font-weight:850;margin-top:12px;">{value}</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# =====================================================
# TECHNOLOGY CARD
# =====================================================

def technology_card(name, icon, background):

    html = (
        f'<div style="background:{background};'
        'padding:24px 18px;border-radius:14px;'
        'margin-bottom:12px;min-height:75px;'
        'box-shadow:0 4px 12px #00000018;">'
        f'<span style="font-size:20px;color:white;'
        f'font-weight:700;">{icon} &nbsp; {name}</span>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# =====================================================
# CHART FUNCTION
# =====================================================

COLORS = ["#7c3aed", "#ec4899", "#06b6d4", "#f59e0b", "#10b981"]


def make_donut(labels, values, title, colors):

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")

    if sum(values) > 0:
        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={
                "width": 0.42,
                "edgecolor": "white",
                "linewidth": 3
            },
            textprops={"fontsize": 10}
        )
    else:
        ax.text(0, 0, "No data available", ha="center", va="center")

    ax.set_title(title, fontsize=14, fontweight="bold", color="#312e81")
    plt.tight_layout()

    return fig


# =====================================================
# LOAD FILES
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "electricity_anomaly_results.csv",
        parse_dates=["Date"]
    )


@st.cache_data
def load_forecast_results():
    return pd.read_csv(
        "forecasting_results.csv",
        parse_dates=["Date"]
    )


try:
    df = load_data()
    rf_model = joblib.load("random_forest_model.pkl")
    forecast_features = joblib.load("forecasting_features.pkl")
    forecast_results = load_forecast_results()

except Exception as e:
    st.error("Unable to load project files. Please check that all required files are present.")
    st.exception(e)
    st.stop()


# =====================================================
# DATA PREPARATION
# =====================================================

required_columns = [
    "Date", "Consumption", "Hidden_State", "Anomaly", "Probability"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

if df["Anomaly"].dtype == object:
    df["Anomaly"] = df["Anomaly"].astype(str).str.lower().eq("true")
else:
    df["Anomaly"] = df["Anomaly"].astype(bool)

total_readings = len(df)
total_anomalies = int(df["Anomaly"].sum())
normal_readings = total_readings - total_anomalies

anomaly_percentage = (
    total_anomalies / total_readings * 100
    if total_readings > 0 else 0
)

start_date = df["Date"].min()
end_date = df["Date"].max()
number_of_days = (end_date - start_date).days + 1
total_consumption = df["Consumption"].sum()
number_of_states = df["Hidden_State"].nunique()


# =====================================================
# NAVIGATION
# =====================================================

st.sidebar.markdown(
    "<h1 style='color:white;'>⚡ POWER ANALYTICS</h1>",
    unsafe_allow_html=True
)

st.sidebar.markdown("### Electricity Intelligence System")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🔮 Future Prediction",
        "🚨 Anomaly Detection",
        "🧠 HMM Analysis",
        "📚 Project Details"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### 📌 Project Information")
st.sidebar.write("Dataset: UCI Electricity Load Diagrams")
st.sidebar.write("Client: MT_330")
st.sidebar.write("Interval: 15 Minutes")
st.sidebar.divider()
st.sidebar.caption("Individual Academic Project")


# =====================================================
# FUTURE PREDICTION FUNCTION
# =====================================================

def predict_future(model, history, feature_names, steps=96):

    history = history.copy().astype(float)
    predictions = []

    for _ in range(steps):

        next_time = history.index[-1] + pd.Timedelta(minutes=15)

        row = {
            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_4": history.iloc[-4],
            "lag_96": history.iloc[-96],
            "lag_672": history.iloc[-672],
            "rolling_mean_4": history.iloc[-4:].mean(),
            "rolling_mean_96": history.iloc[-96:].mean(),
            "hour": next_time.hour,
            "day_of_week": next_time.dayofweek
        }

        features = pd.DataFrame([row], index=[next_time])
        features = features[feature_names]

        predicted_value = float(model.predict(features)[0])

        predictions.append({
            "Date": next_time,
            "Predicted Consumption": predicted_value
        })

        history.loc[next_time] = predicted_value

    return pd.DataFrame(predictions)


# =====================================================
# PAGE 1: DASHBOARD
# =====================================================

if page == "🏠 Dashboard":

    st.title("⚡ Electricity Consumption Dashboard")
    st.markdown("### Intelligent Electricity Analysis Using HMM & Random Forest")

    st.write(
        "Explore electricity consumption patterns, hidden states, "
        "potential anomalies and future consumption estimates."
    )

    st.divider()
    st.subheader("📊 Key Performance Indicators")

    c1, c2, c3 = st.columns(3)

    with c1:
        kpi_card("Total Readings", f"{total_readings:,}", "📈", "#ddd6fe")
    with c2:
        kpi_card("Dataset Coverage", f"{number_of_days:,} Days", "📅", "#fbcfe8")
    with c3:
        kpi_card("Potential Anomalies", f"{total_anomalies:,}", "🚨", "#fed7aa")

    c4, c5, c6 = st.columns(3)

    with c4:
        kpi_card("Normal Readings", f"{normal_readings:,}", "✅", "#a7f3d0")
    with c5:
        kpi_card("Recorded Consumption Sum", f"{total_consumption:,.2f}", "⚡", "#bae6fd")
    with c6:
        kpi_card("Hidden States", str(number_of_states), "🧠", "#c4b5fd")

    st.divider()
    st.subheader("🍩 Consumption Analysis")

    chart1, chart2 = st.columns(2)

    with chart1:
        fig = make_donut(
            ["Normal Readings", "Potential Anomalies"],
            [normal_readings, total_anomalies],
            "Normal vs Potential Anomalies",
            ["#10b981", "#f43f5e"]
        )
        st.pyplot(fig)
        plt.close(fig)

    with chart2:
        state_counts = df["Hidden_State"].value_counts().sort_index()

        state_labels = [
            "Lower Consumption" if x == 0
            else "Higher Consumption" if x == 1
            else f"State {x}"
            for x in state_counts.index
        ]

        fig = make_donut(
            state_labels,
            state_counts.values,
            "Hidden State Distribution",
            COLORS[:len(state_counts)]
        )
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("📈 Electricity Consumption Trend")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["Date"], df["Consumption"], color="#7c3aed", linewidth=0.7)
    ax.set_title("Electricity Consumption Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption")
    ax.grid(alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("📊 Consumption by Hidden State")

    state_summary = (
        df.groupby("Hidden_State")["Consumption"]
        .agg(["mean", "count"])
        .reset_index()
    )

    state_summary["Consumption Level"] = (
        state_summary["Hidden_State"]
        .map({0: "Lower Consumption", 1: "Higher Consumption"})
        .fillna("Other State")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        state_summary["Consumption Level"],
        state_summary["count"],
        color=COLORS[:len(state_summary)]
    )
    ax.bar_label(bars)
    ax.set_title("Number of Readings in Each Hidden State")
    ax.set_ylabel("Number of Readings")
    ax.grid(axis="y", alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("📅 Dataset Information")

    d1, d2, d3 = st.columns(3)
    d1.info(f"Starting Date\n\n{start_date.date()}")
    d2.info(f"Ending Date\n\n{end_date.date()}")
    d3.info("Sampling Interval\n\n15 Minutes")


# =====================================================
# PAGE 2: FUTURE PREDICTION
# =====================================================

elif page == "🔮 Future Prediction":

    st.title("🔮 Future Electricity Prediction")

    st.write(
        "Random Forest Regression estimates future electricity "
        "consumption using historical readings and time-based features."
    )

    st.divider()

    history = (
        df.set_index("Date")["Consumption"]
        .dropna()
        .sort_index()
    )

    if len(history) < 672:
        st.error("Insufficient historical readings for prediction.")

    else:

        with st.spinner("Generating future predictions..."):
            future = predict_future(
                rf_model,
                history,
                forecast_features,
                steps=96
            )

        st.subheader("⏳ Short-Term Prediction")

        p1, p2, p3 = st.columns(3)

        p1.metric("Next 15 Minutes", f"{future.iloc[0]['Predicted Consumption']:.2f}")
        p2.metric("Next 30 Minutes", f"{future.iloc[1]['Predicted Consumption']:.2f}")
        p3.metric("Next 1 Hour", f"{future.iloc[3]['Predicted Consumption']:.2f}")

        st.divider()
        st.subheader("📈 Future Consumption Forecast")

        fig, ax = plt.subplots(figsize=(14, 5))

        recent = history.tail(96)

        ax.plot(
            recent.index, recent.values,
            color="#2563eb", label="Historical Consumption"
        )

        ax.plot(
            future["Date"], future["Predicted Consumption"],
            color="#ec4899", label="Future Prediction",
            linewidth=2, linestyle="--"
        )

        ax.axvline(
            history.index[-1], color="#f59e0b",
            linestyle=":", label="Forecast Starting Point"
        )

        ax.set_title("Historical and Predicted Electricity Consumption")
        ax.set_xlabel("Date")
        ax.set_ylabel("Consumption")
        ax.legend()
        ax.grid(alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("📋 Future Prediction Results")

        st.dataframe(future, use_container_width=True)

        st.download_button(
            "⬇️ Download Future Prediction Report",
            data=future.to_csv(index=False),
            file_name="future_predictions.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("🎯 Random Forest Model Evaluation")

        mae = np.mean(
            np.abs(
                forecast_results["Actual_Consumption"]
                - forecast_results["Predicted_Consumption"]
            )
        )

        rmse = np.sqrt(
            np.mean(
                (
                    forecast_results["Actual_Consumption"]
                    - forecast_results["Predicted_Consumption"]
                ) ** 2
            )
        )

        m1, m2 = st.columns(2)
        m1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
        m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")

        st.info(
            "These evaluation metrics are from the held-out test dataset "
            "using actual previous readings. They do not guarantee the "
            "accuracy of recursive future predictions."
        )


# =====================================================
# PAGE 3: ANOMALY DETECTION
# =====================================================

elif page == "🚨 Anomaly Detection":

    st.title("🚨 Potential Anomaly Detection")

    st.write(
        "The Gaussian Hidden Markov Model identifies unusual "
        "electricity consumption patterns using emission probabilities."
    )

    st.divider()

    a1, a2, a3 = st.columns(3)

    a1.metric("Total Readings", f"{total_readings:,}")
    a2.metric("Potential Anomalies", f"{total_anomalies:,}")
    a3.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")

    st.divider()
    st.subheader("🍩 Anomaly Distribution")

    fig = make_donut(
        ["Normal", "Potential Anomalies"],
        [normal_readings, total_anomalies],
        "Anomaly Distribution",
        ["#10b981", "#f43f5e"]
    )

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    anomalies = df[df["Anomaly"] == True].copy()

    st.subheader("📋 Potential Anomaly Records")
    st.write(f"Potential anomalies detected: {len(anomalies):,}")

    if not anomalies.empty:

        st.dataframe(
            anomalies.sort_values("Probability"),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Anomaly Report",
            data=anomalies.to_csv(index=False),
            file_name="potential_anomalies.csv",
            mime="text/csv"
        )

    else:
        st.success("No potential anomalies detected.")

    st.warning(
        "The dataset does not contain confirmed anomaly labels. "
        "These readings are potential anomalies, not confirmed faults."
    )


# =====================================================
# PAGE 4: HMM ANALYSIS
# =====================================================

elif page == "🧠 HMM Analysis":

    st.title("🧠 Hidden Markov Model Analysis")

    st.write(
        "The Gaussian Hidden Markov Model identifies hidden "
        "electricity consumption states."
    )

    st.divider()

    state_summary = (
        df.groupby("Hidden_State")["Consumption"]
        .agg(["mean", "count", "min", "max"])
        .reset_index()
    )

    state_summary["Consumption Level"] = (
        state_summary["Hidden_State"]
        .map({0: "Lower Consumption", 1: "Higher Consumption"})
        .fillna("Other State")
    )

    st.subheader("🔍 Identified Hidden States")

    for _, row in state_summary.iterrows():

        state = int(row["Hidden_State"])
        label = row["Consumption Level"]
        background = "#ddd6fe" if state == 0 else "#bae6fd"

        html = (
            f'<div style="background:{background};padding:20px;'
            'margin-bottom:15px;border-radius:15px;'
            'border-left:7px solid #6d28d9;">'
            f'<h3 style="color:#312e81;">State {state}: {label}</h3>'
            f'<p style="color:#312e81;"><b>Average Consumption:</b> {row["mean"]:.2f}</p>'
            f'<p style="color:#312e81;"><b>Number of Readings:</b> {int(row["count"]):,}</p>'
            f'<p style="color:#312e81;"><b>Minimum Consumption:</b> {row["min"]:.2f}</p>'
            f'<p style="color:#312e81;"><b>Maximum Consumption:</b> {row["max"]:.2f}</p>'
            '</div>'
        )

        st.markdown(html, unsafe_allow_html=True)

    st.divider()
    st.subheader("📊 Hidden State Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        state_summary["Consumption Level"],
        state_summary["count"],
        color=COLORS[:len(state_summary)]
    )

    ax.bar_label(bars)
    ax.set_ylabel("Number of Readings")
    ax.set_title("Distribution of Hidden Consumption States")
    ax.grid(axis="y", alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("📋 State Summary Table")

    st.dataframe(state_summary, use_container_width=True)


# =====================================================
# PAGE 5: PROJECT DETAILS
# =====================================================

elif page == "📚 Project Details":

    st.title("📚 About the Project")

    st.markdown(
        "### Detection of Unusual Electricity Consumption Patterns Using HMM"
    )

    st.divider()
    st.subheader("🎯 Project Objective")

    st.write("""
    The objective of this project is to analyze electricity
    consumption patterns using a Hidden Markov Model, identify
    hidden consumption states, detect potentially unusual
    consumption readings and estimate future electricity consumption.
    """)

    st.divider()
    st.subheader("📂 Dataset Information")

    st.write("""
    - **Dataset:** UCI Electricity Load Diagrams 2011–2014
    - **Selected Client:** MT_330
    - **Time Interval:** 15 minutes
    - **Number of Clients in Original Dataset:** 370
    - **Data Type:** Time-series electricity consumption data
    """)

    st.divider()
    st.subheader("🧠 Algorithms Used")

    st.markdown("""
    **1. Gaussian Hidden Markov Model (HMM)**

    - Identifies hidden electricity consumption states.
    - Models transitions between consumption states.
    - Helps identify potentially unusual readings.

    **2. Random Forest Regression**

    - Uses historical consumption values.
    - Uses lag-based and time-based features.
    - Estimates future electricity consumption.
    """)

    st.divider()
    st.subheader("⚙️ Technologies Used")

    t1, t2, t3 = st.columns(3)

    with t1:
        technology_card("Python", "🐍", "#1e293b")
    with t2:
        technology_card("Pandas", "📊", "#312e81")
    with t3:
        technology_card("NumPy", "🔢", "#164e63")

    t4, t5, t6 = st.columns(3)

    with t4:
        technology_card("Scikit-learn", "🤖", "#064e3b")
    with t5:
        technology_card("Hmmlearn", "🧠", "#831843")
    with t6:
        technology_card("Matplotlib", "📈", "#7c2d12")

    technology_card("Streamlit", "🌐", "#1e293b")

    st.divider()
    st.subheader("📌 Project Outcome")

    st.write("""
    The developed system provides an interactive electricity
    analytics dashboard that displays consumption patterns,
    hidden consumption states, potential anomalies and
    future electricity consumption estimates.
    """)

    st.warning(
        "The dataset does not contain ground-truth anomaly labels. "
        "Therefore, detected unusual readings are interpreted as "
        "potential anomalies."
    )

    st.divider()
    st.subheader("👩‍💻 Project Information")

    st.write("**Project Type:** Individual Academic Project")
    st.write("**Domain:** Data Science and Machine Learning")
    st.write("**Application:** Electricity Consumption Analytics")

    st.caption("Developed using Python and Streamlit.")
