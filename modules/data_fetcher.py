"""
Data Fetcher Module for StockPilot.

Downloads historical price data using yfinance, handles ticker symbols,
MultiIndex columns, data cleaning, and validation.
"""

import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Download historical stock data and return a clean DataFrame with a single 'Close' column.

    Parameters:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS').
        period (str): Historical period to fetch (e.g., '6mo', '1y', '2y', '5y').

    Returns:
        pd.DataFrame: Clean DataFrame indexed by date containing a numeric 'Close' column.

    Raises:
        ValueError: If ticker is invalid, data is empty, or insufficient records (<60 rows) are found.
    """
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("A valid stock ticker symbol must be provided.")

    clean_ticker = ticker.strip().upper()

    try:
        data = yf.download(
            clean_ticker,
            period=period,
            auto_adjust=True,
            progress=False
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch data for ticker '{clean_ticker}': {exc}"
        ) from exc

    if data is None or data.empty:
        raise ValueError(
            f"No historical data found for ticker '{clean_ticker}'. "
            "Please check the symbol or choose a different time period."
        )

    # Safely extract Close prices (handling MultiIndex or single column)
    if "Close" not in data.columns:
        raise ValueError(f"Historical data for '{clean_ticker}' does not contain 'Close' prices.")

    close = data["Close"]

    # Handle yfinance MultiIndex columns when downloading a single or multi ticker
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close_series = pd.to_numeric(close, errors="coerce").dropna()

    if close_series.empty:
        raise ValueError(f"No valid numeric closing prices found for '{clean_ticker}'.")

    df = pd.DataFrame({"Close": close_series})

    # Ensure index is datetime and sorted chronologically
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if len(df) < 60:
        raise ValueError(
            f"Not enough historical data ({len(df)} rows). "
            "At least 60 trading sessions are required. Select a longer period."
        )

    return df