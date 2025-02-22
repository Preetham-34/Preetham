import pytest
import pandas as pd
from src.features.technical_indicators import add_technical_features

def test_rsi_calculation():
    # Create test data
    prices = pd.Series([100, 101, 102, 101, 100])
    df = pd.DataFrame({'RELIANCE.NS': prices})
    
    # Calculate RSI
    df = add_technical_features(df)
    
    # Assert RSI values are within 0-100 range
    assert df['RSI'].between(0, 100).all()
