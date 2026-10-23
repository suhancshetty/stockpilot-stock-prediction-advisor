"""
Risk Analyzer Module for StockPilot.

Calculates key risk and return metrics from historical price series:
- Annualized Historical Volatility (assuming 252 trading sessions per year)
- Maximum Drawdown (percentage peak-to-trough decline)
- Average Daily Return (percentage daily arithmetic mean)
"""

from typing import Dict
import numpy as np
import pandas as pd


def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute annualized volatility, maximum drawdown, and average daily return.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Close' column.

    Returns:
        dict:
            - annualized_volatility (float): Annualized standard deviation of daily returns (%)
            - max_drawdown (float): Maximum peak-to-trough drop (%)
            - average_daily_return (float): Mean daily percentage return (%)

    Raises:
        ValueError: If 'Close' column is missing or there are fewer than 2 data points.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    close_series = pd.to_numeric(df["Close"], errors="coerce").dropna()

    if len(close_series) < 2:
        raise ValueError("At least 2 price observations are required to calculate risk metrics.")

    daily_returns = close_series.pct_change().dropna()

    if len(daily_returns) == 0:
        raise ValueError("No valid daily returns could be calculated.")

    # Annualized volatility: std(daily_returns) * sqrt(252) * 100
    std_dev = daily_returns.std()
    annualized_volatility = float(std_dev * np.sqrt(252) * 100.0) if not np.isnan(std_dev) else 0.0

    # Average daily return in percent
    average_daily_return = float(daily_returns.mean() * 100.0)

    # Maximum Drawdown: lowest percentage drop from the historical running peak
    running_peak = close_series.cummax()
    drawdown = (close_series / running_peak - 1.0) * 100.0
    max_drawdown = float(drawdown.min())

    return {
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "average_daily_return": average_daily_return,
    }