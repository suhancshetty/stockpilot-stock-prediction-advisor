"""
Unit tests for modules.data_fetcher.
"""

from unittest.mock import patch
import pandas as pd
import pytest

from modules.data_fetcher import fetch_stock_data


def test_fetch_stock_data_invalid_ticker():
    """Verify that empty or invalid ticker symbols raise ValueError."""
    with pytest.raises(ValueError, match="valid stock ticker"):
        fetch_stock_data("")

    with pytest.raises(ValueError, match="valid stock ticker"):
        fetch_stock_data("   ")

    with pytest.raises(ValueError, match="valid stock ticker"):
        fetch_stock_data(None)  # type: ignore


@patch("yfinance.download")
def test_fetch_stock_data_empty_response(mock_download):
    """Verify that an empty response raises ValueError."""
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="No historical data found"):
        fetch_stock_data("NONEXISTENT_TICKER")


@patch("yfinance.download")
def test_fetch_stock_data_insufficient_data(mock_download):
    """Verify that fewer than 60 records raises ValueError."""
    mock_df = pd.DataFrame(
        {"Close": [150.0 + i for i in range(25)]},
        index=pd.date_range("2025-01-01", periods=25)
    )
    mock_download.return_value = mock_df

    with pytest.raises(ValueError, match="Not enough historical data"):
        fetch_stock_data("TEST", period="1y")


@patch("yfinance.download")
def test_fetch_stock_data_multiindex_handling(mock_download):
    """Verify proper handling of yfinance MultiIndex columns."""
    dates = pd.date_range("2024-01-01", periods=70)
    # Construct a MultiIndex columns DataFrame mimicking yfinance
    cols = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Volume", "AAPL")])
    data = [[150.0 + i, 1000 + i] for i in range(70)]
    mock_df = pd.DataFrame(data, index=dates, columns=cols)
    mock_download.return_value = mock_df

    result = fetch_stock_data("AAPL", period="1y")

    assert isinstance(result, pd.DataFrame)
    assert "Close" in result.columns
    assert len(result) == 70
    assert result["Close"].iloc[0] == 150.0


@patch("yfinance.download")
def test_fetch_stock_data_valid(mock_download):
    """Verify clean DataFrame output with standard columns and datetime index."""
    dates = pd.date_range("2024-01-01", periods=80)
    mock_df = pd.DataFrame({"Close": [200.0 + i for i in range(80)]}, index=dates)
    mock_download.return_value = mock_df

    result = fetch_stock_data("MSFT", period="1y")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Close"]
    assert len(result) == 80
    assert isinstance(result.index, pd.DatetimeIndex)
