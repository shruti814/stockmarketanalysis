"""
Load CSV, basic cleaning, feature engineering (moving averages, returns).
Returns a cleaned DataFrame with Date index.
"""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    return df

def basic_features(df):
    df = df.sort_index()
    df['Close'] = df['Close'].astype(float)
    # returns
    df['ret_1d'] = df['Close'].pct_change()
    # moving averages
    df['ma_7'] = df['Close'].rolling(7).mean()
    df['ma_21'] = df['Close'].rolling(21).mean()
    df = df.dropna()
    return df

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv)>1 else 'AAPL'
    df = load_data(ticker)
    df2 = basic_features(df)
    print(df2.tail())
