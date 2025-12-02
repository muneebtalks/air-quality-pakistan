import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

DATA_FILE = 'cleaned_lahore_aqi.csv'
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / 'lahore_prophet_model.pkl'
FORECAST_IMG = "forecast_preview.png"
COMPONENTS_IMG = "forecast_components.png"


def load_and_preprocess_data(filepath):
    """Loads data, renames columns for Prophet, and removes extreme outliers."""
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    df = df[['datetime', 'main_aqi']].rename(columns={'datetime': 'ds', 'main_aqi': 'y'})

    # Remove extreme outliers (optional but makes Prophet happier)
    q99 = df['y'].quantile(0.99)
    df = df[df['y'] <= q99]
    return df


def train_model(df):
    """Initializes and trains the Prophet model with specific seasonality settings."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    model.add_country_holidays(country_name='Pakistan')  # Fixed for Prophet ≥ 1.1
    model.fit(df)
    return model


def save_model(model, path):
    """Saves the trained model to a pickle file."""
    path.parent.mkdir(exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model trained and saved to {path}!")


def generate_and_plot_forecast(model, periods=720, freq='H'):
    """Generates a forecast and saves visualization plots."""
    # Quick 30-day forecast to test
    future = model.make_future_dataframe(periods=periods, freq=freq)  # 30 days ahead (default)
    forecast = model.predict(future)

    # Plot Forecast
    fig = model.plot(forecast)
    plt.title("Lahore AQI Forecast (Next 30 Days)")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(FORECAST_IMG, dpi=300)
    plt.show()

    # Plot components (trend, weekly, daily)
    model.plot_components(forecast)
    plt.savefig(COMPONENTS_IMG, dpi=300)
    plt.show()


if __name__ == "__main__":
    df = load_and_preprocess_data(DATA_FILE)
    model = train_model(df)
    save_model(model, MODEL_PATH)
    generate_and_plot_forecast(model)
