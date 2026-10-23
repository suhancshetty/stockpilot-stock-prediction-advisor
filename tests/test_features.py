"""
Unit tests for modules.features.
"""

import pandas as pd
import pytest

from modules.features import prepare_features, FEATURE_COLUMNS


def test_prepare_features_missing_close():
    """Verify that a DataFrame without 'Close' raises ValueError."""
    bad_df = pd.DataFrame({"Volume": [100, 200, 300]})
    with pytest.raises(ValueError, match="must contain a 'Close' column"):
        prepare_features(bad_df)


def test_prepare_features_insufficient_rows():
    """Verify that fewer than 20 rows raises ValueError."""
    small_df = pd.DataFrame(
        {"Close": [100.0 + i for i in range(12)]},
        index=pd.date_range("2025-01-01", periods=12)
    )
    with pytest.raises(ValueError, match="Insufficient data"):
        prepare_features(small_df)


def test_prepare_features_columns_and_no_nans(sample_stock_df):
    """Verify that FEATURE_COLUMNS exist and no NaNs remain."""
    df_feat = prepare_features(sample_stock_df)

    for col in FEATURE_COLUMNS:
        assert col in df_feat.columns
    assert "Close" in df_feat.columns
    assert not df_feat[FEATURE_COLUMNS + ["Close"]].isna().any().any()


def test_zero_data_leakage_alignment(sample_stock_df):
    """
    Critically verify that row t's features use ONLY data prior to session t:
    - Previous_Close at t must equal Close at t-1
    - MA_5 at t must equal mean(Close at t-5 through t-1)
    """
    df_feat = prepare_features(sample_stock_df)

    # Check multiple arbitrary rows in the prepared DataFrame
    for target_date in df_feat.index[5:15]:
        target_row = df_feat.loc[target_date]

        # Locate historical positions in original unshifted sample_stock_df
        target_loc = sample_stock_df.index.get_loc(target_date)

        expected_prev_close = sample_stock_df["Close"].iloc[target_loc - 1]
        expected_ma5 = sample_stock_df["Close"].iloc[target_loc - 5 : target_loc].mean()

        assert target_row["Previous_Close"] == pytest.approx(expected_prev_close, rel=1e-5)
        assert target_row["MA_5"] == pytest.approx(expected_ma5, rel=1e-5)

        # The current close must NOT equal Previous_Close (unless by pure coincidence)
        # and features must be computed strictly from past indices.
