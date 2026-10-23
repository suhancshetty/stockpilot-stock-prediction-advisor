"""
Unit tests for modules.model_trainer.
"""

import numpy as np
import pandas as pd
import pytest

from modules.model_trainer import train_and_evaluate_models


def test_train_and_evaluate_missing_columns():
    """Verify ValueError when required feature columns are missing."""
    df_missing = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    with pytest.raises(ValueError, match="Required column"):
        train_and_evaluate_models(df_missing)


def test_train_and_evaluate_insufficient_samples(insufficient_stock_df):
    """Verify ValueError when fewer than 20 rows are passed."""
    # Add dummy feature columns to short df
    df = insufficient_stock_df.copy()
    df["Previous_Close"] = df["Close"]
    df["MA_5"] = df["Close"]

    with pytest.raises(ValueError, match="At least 20 samples"):
        train_and_evaluate_models(df)


def test_chronological_split_no_shuffling(sample_features_df):
    """
    Verify chronological train/test partitioning:
    All training dates must precede all testing dates.
    """
    results = train_and_evaluate_models(sample_features_df, split_ratio=0.8)

    X_train = results["X_train"]
    X_test = results["X_test"]

    assert len(X_train) + len(X_test) == len(sample_features_df)
    assert X_train.index.max() < X_test.index.min(), "Data was shuffled! Split must be chronological."


def test_model_training_metrics_validity(sample_features_df):
    """Verify that both models train and produce sensible non-negative error metrics."""
    results = train_and_evaluate_models(sample_features_df)

    assert results["best_model_name"] in ["Linear Regression", "Random Forest"]
    assert results["mae"] > 0
    assert results["rmse"] > 0
    assert results["mape"] > 0

    assert results["linear_mae"] > 0
    assert results["linear_rmse"] > 0
    assert results["rf_mae"] > 0
    assert results["rf_rmse"] > 0

    # Ensure best_model selected corresponds to lower MAE
    if results["rf_mae"] < results["linear_mae"]:
        assert results["best_model_name"] == "Random Forest"
        assert results["mae"] == results["rf_mae"]
    else:
        assert results["best_model_name"] == "Linear Regression"
        assert results["mae"] == results["linear_mae"]

    assert isinstance(results["comparison_df"], pd.DataFrame)
    assert list(results["comparison_df"].columns) == ["MAE", "RMSE", "MAPE (%)"]


def test_reproducibility_with_seed(sample_features_df):
    """Verify that Random Forest results are deterministic with fixed random_state."""
    res1 = train_and_evaluate_models(sample_features_df)
    res2 = train_and_evaluate_models(sample_features_df)

    assert res1["rf_mae"] == pytest.approx(res2["rf_mae"])
    assert np.allclose(res1["rf_predictions"], res2["rf_predictions"])
