"""
Unit tests for modules.indicators.
"""

import pandas as pd
import pytest

from modules.indicators import calculate_indicators, generate_technical_signal


def test_calculate_indicators_missing_close():
    """Verify that a DataFrame without 'Close' raises ValueError."""
    bad_df = pd.DataFrame({"Open": [10.0, 11.0, 12.0]})
    with pytest.raises(ValueError, match="must contain a 'Close' column"):
        calculate_indicators(bad_df)


def test_calculate_indicators_non_mutating(sample_stock_df):
    """Verify that the original DataFrame is not modified in-place."""
    cols_before = list(sample_stock_df.columns)
    result = calculate_indicators(sample_stock_df)

    assert list(sample_stock_df.columns) == cols_before
    assert "Daily_Return" not in sample_stock_df.columns
    assert "Daily_Return" in result.columns


def test_calculate_indicators_column_presence(sample_indicators_df):
    """Verify all expected indicator columns are created."""
    expected_cols = [
        "Close", "Daily_Return", "MA_20", "MA_50", "RSI", "MACD", "MACD_Signal"
    ]
    for col in expected_cols:
        assert col in sample_indicators_df.columns


def test_rsi_bounded_values(sample_indicators_df):
    """Verify RSI remains bounded between 0 and 100."""
    valid_rsi = sample_indicators_df["RSI"].dropna()
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_moving_averages_calculation(sample_stock_df):
    """Verify moving averages calculate properly over expected windows."""
    df_ind = calculate_indicators(sample_stock_df)

    # First 19 rows of MA_20 should be NaN
    assert df_ind["MA_20"].iloc[:19].isna().all()
    # 20th row (index 19) should match rolling mean of first 20 close prices
    expected_ma20_first = sample_stock_df["Close"].iloc[:20].mean()
    assert pytest.approx(df_ind["MA_20"].iloc[19], rel=1e-5) == expected_ma20_first


def test_generate_technical_signal_rules():
    """Verify BUY, SELL, and HOLD technical signals."""
    # Bullish scenario
    bull_row = pd.Series({
        "Close": 120.0,
        "MA_20": 115.0,
        "MA_50": 110.0,
        "RSI": 62.0,
        "MACD": 1.5,
        "MACD_Signal": 0.8
    })
    bull_result = generate_technical_signal(bull_row)
    assert bull_result["signal"] == "BUY"

    # Bearish scenario
    bear_row = pd.Series({
        "Close": 90.0,
        "MA_20": 95.0,
        "MA_50": 105.0,
        "RSI": 38.0,
        "MACD": -2.0,
        "MACD_Signal": -1.0
    })
    bear_result = generate_technical_signal(bear_row)
    assert bear_result["signal"] == "SELL"

    # Mixed / Neutral scenario
    mixed_row = pd.Series({
        "Close": 100.0,
        "MA_20": 105.0,
        "MA_50": 95.0,
        "RSI": 52.0,
        "MACD": 0.5,
        "MACD_Signal": 0.5
    })
    mixed_result = generate_technical_signal(mixed_row)
    assert mixed_result["signal"] == "HOLD"

    # Missing column scenario
    incomplete_row = pd.Series({"Close": 100.0})
    missing_result = generate_technical_signal(incomplete_row)
    assert missing_result["signal"] == "HOLD"
