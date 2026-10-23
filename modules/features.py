"""
Feature Engineering Module for StockPilot.

Constructs strictly lagged features to prevent lookahead bias / data leakage:
- Previous_Close (Close at t-1)
- MA_5 (Average of Close from t-5 to t-1)
Target variable is the session closing price (Close at t).
"""

from typing import List
import pandas as pd

FEATURE_COLUMNS: List[str] = ["Previous_Close", "MA_5"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct machine learning features using lagged historical data.

    To ensure zero data leakage:
    - 'Previous_Close' is Close shifted by 1 session (price at t-1).
    - 'MA_5' is the 5-session rolling mean of the shifted Close (prices t-5 to t-1).
    - The target is 'Close' at day t.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Close' column.

    Returns:
        pd.DataFrame: Clean DataFrame with 'Previous_Close' and 'MA_5' features and no NaNs.

    Raises:
        ValueError: If 'Close' column is missing or insufficient rows exist.
    """
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    df_features = df.copy()

    # Lagged previous session closing price (t-1)
    df_features["Previous_Close"] = df_features["Close"].shift(1)

    # 5-session rolling moving average of past closing prices (t-5 to t-1)
    df_features["MA_5"] = df_features["Close"].shift(1).rolling(window=5).mean()

    # Drop initial rows that contain NaN due to shifting and rolling window
    clean_df = df_features.dropna(subset=FEATURE_COLUMNS + ["Close"])

    if len(clean_df) < 20:
        raise ValueError(
            f"Insufficient data ({len(clean_df)} usable rows) after feature preparation. "
            "A minimum of 20 usable sessions is required."
        )

    return clean_df