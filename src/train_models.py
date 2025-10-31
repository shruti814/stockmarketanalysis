"""
Train baseline models and print metrics.
Usage:
  python train_models.py --ticker AAPL --model all
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import timedelta

from preprocess import load_data, basic_features
from evaluate import rmse, mae, plot_forecast

# --- ARIMA (statsmodels) ---
def train_arima(series, steps=30):
    import statsmodels.api as sm
    # Simple auto_arima suggestion could be used (pmdarima) but we keep this basic:
    model = sm.tsa.ARIMA(series, order=(5,1,0))
    res = model.fit()
    forecast = res.forecast(steps=steps)
    return forecast, res

# --- Prophet ---
def train_prophet(df, periods=30):
    from prophet import Prophet
    prophet_df = df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=periods)
    fcst = m.predict(future)
    pred = fcst.set_index('ds')['yhat'][-periods:]
    return pred, m

# --- LSTM sequence model (Keras) ---
def train_lstm(series, n_steps=60, epochs=10, batch_size=16):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler

    # scale
    scaler = MinMaxScaler()
    values = series.values.reshape(-1,1)
    scaled = scaler.fit_transform(values)

    # prepare sequences
    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i-n_steps:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # train/test split: use last 20% as test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, input_shape=(n_steps,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    # forecast next `n_forecast` points using the last window
    n_forecast = 30
    last_window = scaled[-n_steps:].reshape(1, n_steps, 1)
    preds = []
    cur_window = last_window.copy()
    for _ in range(n_forecast):
        p = model.predict(cur_window, verbose=0)[0,0]
        preds.append(p)
        # slide
        cur_window = np.roll(cur_window, -1)
        cur_window[0, -1, 0] = p
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    return preds, model

def main(ticker, model_choice):
    df = load_data(ticker)
    df = basic_features(df)
    close = df['Close']

    # train/test split for evaluation: hold-out last 30 days
    train = close[:-60]
    test = close[-60:]

    results = {}

    if model_choice in ('arima','all'):
        try:
            arima_forecast, arima_model = train_arima(train, steps=len(test))
            results['arima'] = np.array(arima_forecast)
            print("ARIMA trained.")
        except Exception as e:
            print("ARIMA failed:", e)

    if model_choice in ('prophet','all'):
        try:
            prop_pred, prop_model = train_prophet(df, periods=len(test))
            # prophet forecast contains also the train period; take last len(test)
            results['prophet'] = np.array(prop_pred[-len(test):])
            print("Prophet trained.")
        except Exception as e:
            print("Prophet failed:", e)

    if model_choice in ('lstm','all'):
        try:
            lstm_pred, lstm_model = train_lstm(close, n_steps=60, epochs=8)
            # lstm_pred is next 30 predictions; if test is longer adjust (here basic)
            results['lstm'] = np.array(lstm_pred[:len(test)])
            print("LSTM trained.")
        except Exception as e:
            print("LSTM failed:", e)

    # Evaluate
    for name, preds in results.items():
        # align lengths with test
        preds = preds[-len(test):]
        print(f"Model: {name} | RMSE: {rmse(test.values, preds):.4f} | MAE: {mae(test.values, preds):.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--model", default="all", choices=['all','arima','prophet','lstm'])
    args = p.parse_args()
    main(args.ticker, args.model)
