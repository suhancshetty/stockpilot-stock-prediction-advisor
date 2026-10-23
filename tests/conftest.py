"""
Pytest configuration and deterministic fixtures for offline testing.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from modules.indicators import calculate_indicators
from modules.features import prepare_features


@pytest.fixture
def sample_stock_df() -> pd.DataFrame:
    """
    Generate 100 sessions of deterministic synthetic stock prices.
    """
    np.random.seed(42)
    n = 100
    daily_returns = np.random.normal(loc=0.0008, scale=0.015, size=n)
    prices = 100.0 * np.cumprod(1.0 + daily_returns)

    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n)]
    df = pd.DataFrame({"Close": prices}, index=pd.DatetimeIndex(dates))
    return df


@pytest.fixture
def insufficient_stock_df() -> pd.DataFrame:
    """
    Generate a small DataFrame with only 10 rows to test error boundaries.
    """
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({"Close": [100.0 + i for i in range(10)]}, index=pd.DatetimeIndex(dates))
    return df


@pytest.fixture
def sample_indicators_df(sample_stock_df) -> pd.DataFrame:
    """
    Generate DataFrame with technical indicators computed.
    """
    return calculate_indicators(sample_stock_df)


@pytest.fixture
def sample_features_df(sample_stock_df) -> pd.DataFrame:
    """
    Generate DataFrame with lagged features prepared.
    """
    return prepare_features(sample_stock_df)
