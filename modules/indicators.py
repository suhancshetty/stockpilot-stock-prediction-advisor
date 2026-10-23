"""
Technical Indicators Module for StockPilot.

Calculates Daily Returns, Moving Averages (20-day, 50-day),
Relative Strength Index (RSI 14-session), and MACD (12, 26, 9),
along with rule-based technical signals.
"""

from typing import Tuple, Dict, Any
import pandas as pd


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns, 20-day MA, 50-day MA, 14-day RSI, and MACD indicators.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least a numeric 'Close' column.

    Returns:
        pd.DataFrame: A new DataFrame copy containing the original data plus:
                      - Daily_Return: Percentage daily change
                      - MA_20: 20-session Simple Moving Average
                      - MA_50: 50-session Simple Moving Average
                      - RSI: 14-session Relative Strength Index (0 to 100)
                      - MACD: 12-session EMA minus 26-session EMA
                      - MACD_Signal: 9-session EMA of the MACD line
    """
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    df_out = df.copy()

    # Daily percentage returns
    df_out["Daily_Return"] = df_out["Close"].pct_change() * 100.0

    # Moving averages
    df_out["MA_20"] = df_out["Close"].rolling(window=20, min_periods=20).mean()
    df_out["MA_50"] = df_out["Close"].rolling(window=50, min_periods=50).mean()

    # Relative Strength Index (RSI, 14-period Wilder-style EWM)
    delta = df_out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

    # Handle division by zero when loss is zero
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df_out["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # In flat periods where gain and loss are both zero, RSI is 50.0
    df_out["RSI"] = df_out["RSI"].fillna(50.0)

    # MACD: 12-day EMA minus 26-day EMA
    ema12 = df_out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_out["Close"].ewm(span=26, adjust=False).mean()
    df_out["MACD"] = ema12 - ema26

    # MACD Signal Line: 9-day EMA of the MACD line
    df_out["MACD_Signal"] = df_out["MACD"].ewm(span=9, adjust=False).mean()

    return df_out


def generate_technical_signal(latest_row: pd.Series) -> Dict[str, str]:
    """
    Evaluate rule-based technical indicator signal for the latest trading session.

    Rules:
    - BUY: Close > MA_20 AND MA_20 > MA_50 AND RSI > 50 AND MACD > MACD_Signal
    - SELL: Close < MA_20 AND MA_20 < MA_50 AND RSI < 50 AND MACD < MACD_Signal
    - HOLD: Indicators are mixed or inconclusive

    Parameters:
        latest_row (pd.Series): A row containing Close, MA_20, MA_50, RSI, MACD, MACD_Signal.

    Returns:
        dict: {"signal": "BUY"|"SELL"|"HOLD", "explanation": str}
    """
    required_cols = ["Close", "MA_20", "MA_50", "RSI", "MACD", "MACD_Signal"]
    for col in required_cols:
        if col not in latest_row or pd.isna(latest_row[col]):
            return {
                "signal": "HOLD",
                "explanation": "Insufficient indicator history to compute directional market signal."
            }

    close = float(latest_row["Close"])
    ma20 = float(latest_row["MA_20"])
    ma50 = float(latest_row["MA_50"])
    rsi = float(latest_row["RSI"])
    macd = float(latest_row["MACD"])
    macd_signal = float(latest_row["MACD_Signal"])

    if close > ma20 and ma20 > ma50 and rsi > 50.0 and macd > macd_signal:
        return {
            "signal": "BUY",
            "explanation": "Bullish alignment: Price is above 20-day MA, 20-day MA is above 50-day MA, RSI > 50, and MACD is above its signal line."
        }
    elif close < ma20 and ma20 < ma50 and rsi < 50.0 and macd < macd_signal:
        return {
            "signal": "SELL",
            "explanation": "Bearish alignment: Price is below 20-day MA, 20-day MA is below 50-day MA, RSI < 50, and MACD is below its signal line."
        }
    else:
        return {
            "signal": "HOLD",
            "explanation": "Neutral / Mixed: The technical indicators do not agree on a clear directional trend."
        }