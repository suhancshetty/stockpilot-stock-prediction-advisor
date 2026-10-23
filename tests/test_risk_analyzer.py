"""
Unit tests for modules.risk_analyzer.
"""

import numpy as np
import pandas as pd
import pytest

from modules.risk_analyzer import calculate_risk_metrics


def test_calculate_risk_metrics_missing_close():
    """Verify ValueError when 'Close' column is missing."""
    bad_df = pd.DataFrame({"Return": [0.01, 0.02]})
    with pytest.raises(ValueError, match="must contain a 'Close' column"):
        calculate_risk_metrics(bad_df)


def test_calculate_risk_metrics_insufficient_data():
    """Verify ValueError when fewer than 2 price observations are passed."""
    single_row = pd.DataFrame({"Close": [150.0]})
    with pytest.raises(ValueError, match="At least 2 price observations"):
        calculate_risk_metrics(single_row)


def test_calculate_risk_metrics_keys(sample_stock_df):
    """Verify expected keys in the output dictionary."""
    metrics = calculate_risk_metrics(sample_stock_df)

    assert "annualized_volatility" in metrics
    assert "max_drawdown" in metrics
    assert "average_daily_return" in metrics

    assert metrics["annualized_volatility"] >= 0.0
    assert metrics["max_drawdown"] <= 0.0  # Drawdown is non-positive by definition


def test_zero_volatility_for_constant_prices():
    """Verify that a series of flat prices yields 0 volatility and 0 drawdown."""
    flat_df = pd.DataFrame(
        {"Close": [100.0] * 50},
        index=pd.date_range("2025-01-01", periods=50)
    )
    metrics = calculate_risk_metrics(flat_df)

    assert metrics["annualized_volatility"] == pytest.approx(0.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
    assert metrics["average_daily_return"] == pytest.approx(0.0)


def test_exact_max_drawdown_calculation():
    """Verify drawdown calculation on a known peak-to-trough sequence."""
    # Prices: 100 -> 200 (peak) -> 100 (50% drop) -> 150
    prices = [100.0, 200.0, 100.0, 150.0]
    df = pd.DataFrame({"Close": prices}, index=pd.date_range("2025-01-01", periods=4))

    metrics = calculate_risk_metrics(df)

    # From peak of 200 down to 100 is a -50.0% drawdown
    assert metrics["max_drawdown"] == pytest.approx(-50.0)
