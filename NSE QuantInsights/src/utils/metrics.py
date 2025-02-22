import numpy as np

def calculate_sharpe(returns, risk_free_rate=0):
    """Annualized Sharpe ratio calculation"""
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def max_drawdown(cum_returns):
    """Calculate maximum drawdown from cumulative returns"""
    peak = np.maximum.accumulate(cum_returns)
    trough = np.minimum.accumulate(cum_returns)
    return (peak - trough).max()
