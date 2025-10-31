"""
Run:
streamlit run src/streamlit_app.py -- --ticker AAPL
"""
import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, basic_features
from train_models import train_arima, train_prophet, train_lstm
from evaluate import plot_forecast

st.title("Stock Forecasting Playground")

ticker = st.sidebar.text_input("Ticker", value="AAPL")
periods = st.sidebar.number_input("Forecast days", min_value=7, max_value=180, value=30)
if st.sidebar.button("Load & Forecast"):
    df = load_data(ticker)
    df = basic_features(df)
    st.subheader(f"{ticker} — Latest data")
    st.line_chart(df['Close'])

    with st.spinner("Training Prophet..."):
        try:
            p_pred, _ = train_prophet(df, periods=periods)
            p_series = p_pred[-periods:]
            st.write("Prophet forecast (last points):")
            st.line_chart(pd.Series(p_series, index=pd.date_range(df.index.max(), periods=periods+1)[1:]))
        except Exception as e:
            st.error(f"Prophet error: {e}")

    # (Optional) ARIMA and LSTM could be run similarly; left out to keep UI snappy.
