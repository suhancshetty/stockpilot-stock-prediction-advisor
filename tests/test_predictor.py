"""
Unit tests for modules.predictor.
"""

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from modules.predictor import predict_next_session
from modules.features import FEATURE_COLUMNS


def test_predict_next_session_missing_close():
    """Verify ValueError when 'Close' is missing."""
    mock_model = MagicMock()
    bad_df = pd.DataFrame({"Price": [100.0, 101.0, 102.0, 103.0, 104.0]})
    with pytest.raises(ValueError, match="must contain a 'Close' column"):
        predict_next_session(mock_model, bad_df)


def test_predict_next_session_fewer_than_five_rows():
    """Verify ValueError when fewer than 5 rows are provided."""
    mock_model = MagicMock()
    small_df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0]})
    with pytest.raises(ValueError, match="At least 5 historical closing prices"):
        predict_next_session(mock_model, small_df)


def test_predict_next_session_zero_or_negative_price():
    """Verify ValueError when latest price is zero or negative."""
    mock_model = MagicMock()
    zero_df = pd.DataFrame({"Close": [10.0, 11.0, 12.0, 13.0, 0.0]})
    with pytest.raises(ValueError, match="must be positive"):
        predict_next_session(mock_model, zero_df)


def test_predict_next_session_feature_columns_and_calculations():
    """Verify feature alignment and return dictionary contents."""
    mock_model = MagicMock()
    # Mock model prediction output
    mock_model.predict.return_value = np.array([110.0])

    prices = [100.0, 102.0, 104.0, 106.0, 108.0]
    df = pd.DataFrame({"Close": prices}, index=pd.date_range("2025-01-01", periods=5))

    result = predict_next_session(mock_model, df)

    # Latest close was 108.0; predicted is 110.0 -> change = +2.0
    assert result["latest_close"] == 108.0
    assert result["next_price"] == 110.0
    assert result["change"] == pytest.approx(2.0)
    assert result["change_pct"] == pytest.approx((2.0 / 108.0) * 100.0)
    assert "UP" in result["direction"]

    # Verify features passed to mock_model.predict
    assert mock_model.predict.call_count == 1
    features_called = mock_model.predict.call_args[0][0]
    assert isinstance(features_called, pd.DataFrame)
    assert list(features_called.columns) == FEATURE_COLUMNS
    assert features_called["Previous_Close"].iloc[0] == 108.0
    assert features_called["MA_5"].iloc[0] == pytest.approx(np.mean(prices))


def test_predict_next_session_direction_labels():
    """Verify UP, DOWN, and UNCHANGED directional classification."""
    df = pd.DataFrame(
        {"Close": [100.0, 100.0, 100.0, 100.0, 100.0]},
        index=pd.date_range("2025-01-01", periods=5)
    )

    # Up scenario
    mock_up = MagicMock()
    mock_up.predict.return_value = np.array([105.0])
    res_up = predict_next_session(mock_up, df)
    assert "UP" in res_up["direction"]

    # Down scenario
    mock_down = MagicMock()
    mock_down.predict.return_value = np.array([95.0])
    res_down = predict_next_session(mock_down, df)
    assert "DOWN" in res_down["direction"]

    # Unchanged scenario (within 0.05%)
    mock_flat = MagicMock()
    mock_flat.predict.return_value = np.array([100.01])
    res_flat = predict_next_session(mock_flat, df)
    assert res_flat["direction"] == "UNCHANGED"
