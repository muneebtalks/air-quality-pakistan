import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import pickle

# ==================== CONFIG ====================
st.set_page_config(page_title="Pakistan AQI Forecaster", layout="wide", page_icon="Cloud")
st.title("Pakistan Real-Time Air Quality Forecaster")
st.markdown("**Lahore • July–Dec 2024 → Live 60-Day Forecast**")

MODEL_PATH = Path("models/lahore_prophet_model.pkl")
DATA_PATH = "cleaned_lahore_aqi.csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_historical():
    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    df = df[['datetime', 'main_aqi']].rename(columns={'datetime': 'ds', 'main_aqi': 'y'})
    return df

model = load_model()
historical = load_historical()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Forecast Settings")
    forecast_days = st.slider("Forecast Days Ahead", 1, 60, 30)
    show_components = st.checkbox("Show Seasonality & Holidays", True)
    st.markdown("---")
    st.success("Detects Eid, Independence Day & winter smog peaks!")

# ==================== FORECAST ====================
future = model.make_future_dataframe(periods=forecast_days * 24, freq='h')  # 'h' instead of 'H'
forecast = model.predict(future)

# Combine historical + forecast (Pandas 2.0+ way)
plot_df = pd.concat([
    historical[['ds', 'y']].assign(type="Historical"),
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].assign(type="Forecast")
], ignore_index=True)

last_actual = historical.iloc[-1]
last_aqi = int(last_actual['y'])

# ==================== DASHBOARD ====================
col1, col2 = st.columns([1, 3])

with col1:
    st.metric("Latest Recorded AQI", last_aqi)
    if last_aqi <= 50:
        st.success("Good")
    elif last_aqi <= 100:
        st.info("Moderate")
    elif last_aqi <= 150:
        st.warning("Unhealthy for Sensitive")
    elif last_aqi <= 200:
        st.error("Unhealthy")
    else:
        st.error("**Hazardous**")

with col2:
    fig = go.Figure()

    # Historical
    hist = plot_df[plot_df['type'] == "Historical"]
    fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'],
                             name="Historical AQI", line=dict(color="#1f77b4")))

    # Forecast line
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                             name="Forecast", line=dict(color="#ff7f0e")))

    # Confidence interval
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                             mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                             mode='lines', line_color='rgba(0,0,0,0)',
                             fill='tonexty', fillcolor='rgba(255,127,0,0.2)',
                             name="95% Confidence"))

    fig.update_layout(
        title=f"Lahore AQI – Historical + {forecast_days}-Day Forecast",
        xaxis_title="Date", yaxis_title="AQI",
        hovermode="x unified", height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== COMPONENTS ====================
if show_components:
    st.markdown("### Seasonality, Trends & Pakistan Holiday Effects")
    fig_comp = model.plot_components(forecast)
    st.pyplot(fig_comp)

# ==================== DOWNLOAD ====================
st.download_button(
    "Download Forecast CSV",
    data=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days*24).to_csv(index=False),
    file_name=f"lahore_aqi_forecast_{forecast_days}days.csv",
    mime="text/csv"
)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("**Muneeb • Portfolio Project** | Real 2024 Lahore Data")