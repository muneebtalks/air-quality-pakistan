from statsmodels.tsa.arima.model import ARIMA
import pickle

df = pd.read_csv('cleaned_lahore_aqi.csv', parse_dates=['datetime'], index_col='datetime')
series = df['main_aqi'].resample('D').mean().dropna()

model_arima = ARIMA(series, order=(5,1,0))
model_arima_fit = model_arima.fit()

with open('models/lahore_arima_model.pkl', 'wb') as f:
    pickle.dump(model_arima_fit, f)

print("ARIMA model also saved!")