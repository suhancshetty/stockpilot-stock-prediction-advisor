"""
Predictor Module for StockPilot.

Constructs next-session feature vectors matching training alignment and generates
next-session closing price estimates using the trained machine learning model.
"""

from typing import Dict, Any
import pandas as pd

from .features import FEATURE_COLUMNS


def predict_next_session(model: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Estimate the next session's closing price using the trained regression model.

    Features are constructed using the same methodology as training:
    - Previous_Close: The most recent session's closing price
    - MA_5: The mean of the most recent 5 sessions' closing prices

    Parameters:
        model: Trained scikit-learn regressor (e.g. LinearRegression or RandomForestRegressor).
        df (pd.DataFrame): DataFrame containing at least 5 historical 'Close' prices.

    Returns:
        dict:
            - latest_close (float): Most recent closing price
            - next_price (float): Estimated next session closing price
            - change (float): Projected price difference
            - change_pct (float): Projected percentage change
            - direction (str): Directional label ("UP 📈", "DOWN 📉", "UNCHANGED")
            - direction_message (str): Explanatory summary

    Raises:
        ValueError: If input DataFrame has fewer than 5 rows or invalid/zero price.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    if len(df) < 5:
        raise ValueError("At least 5 historical closing prices are required to predict next session.")

    latest_close = float(df["Close"].iloc[-1])
    if latest_close <= 0:
        raise ValueError("Latest closing price must be positive.")

    # 5-session average of the most recent closing prices
    latest_ma5 = float(df["Close"].tail(5).mean())

    # Build DataFrame matching the exact training feature columns and order
    next_features = pd.DataFrame(
        [[latest_close, latest_ma5]],
        columns=FEATURE_COLUMNS
    )

    next_price = float(model.predict(next_features)[0])
    change = next_price - latest_close
    change_pct = (change / latest_close) * 100.0

    if change_pct > 0.05:
        direction = "UP 📈"
        direction_message = f"The model estimates an increase of {change_pct:+.2f}%."
    elif change_pct < -0.05:
        direction = "DOWN 📉"
        direction_message = f"The model estimates a decrease of {abs(change_pct):.2f}%."
    else:
        direction = "UNCHANGED"
        direction_message = "The model estimates little to no price change (within ±0.05%)."

    return {
        "latest_close": latest_close,
        "next_price": next_price,
        "change": change,
        "change_pct": change_pct,
        "direction": direction,
        "direction_message": direction_message,
        "features_used": next_features.to_dict(orient="records")[0],
    }