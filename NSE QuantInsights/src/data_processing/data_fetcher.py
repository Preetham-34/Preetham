import yfinance as yf
import pandas as pd

def fetch_nse_data(tickers=['RELIANCE.NS'], start='2020-01-01', end='2024-01-01'):
    """
    Fetches OHLCV data from Yahoo Finance
    Returns: DataFrame with DatetimeIndex
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False
    )
    return data['Close']

if __name__ == "__main__":
    df = fetch_nse_data()
    df.to_csv('../data/raw/nse_stocks.csv')
