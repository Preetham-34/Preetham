import pandas as pd
import numpy as np

def add_technical_features(df, target_col='RELIANCE.NS'):
    """Calculate 14+ technical indicators"""
    close = df[target_col]
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # ... [rest of your RSI code]...
    
    # Bollinger Bands
    df['SMA_20'] = close.rolling(20).mean()
    df['STD_20'] = close.rolling(20).std()
    df['Upper_Bollinger'] = df['SMA_20'] + 2*df['STD_20']
    
    return df
