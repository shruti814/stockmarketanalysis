"""
Download historical OHLCV data using yfinance and save to data/{TICKER}.csv
Usage:
  python data_fetch.py --ticker AAPL --start 2015-01-01 --end 2025-10-30
"""
import argparse
import os
import yfinance as yf
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def fetch_save(ticker, start, end, interval="1d"):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker} in given date range.")
    df.index.name = 'Date'
    out_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(out_path)
    print(f"Saved {ticker} to {out_path}")
    return out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="1d")
    args = p.parse_args()
    fetch_save(args.ticker, args.start, args.end, args.interval)