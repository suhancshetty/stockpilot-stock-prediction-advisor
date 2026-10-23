"""
Model Trainer Module for StockPilot.

Trains and evaluates Linear Regression and Random Forest Regressor models
using strict chronological time-series splitting (no shuffling).
Calculates MAE, RMSE, and MAPE, and selects the best-performing model.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import FEATURE_COLUMNS


def train_and_evaluate_models(
    df: pd.DataFrame,
    split_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    Train Linear Regression and Random Forest models using chronological splitting.

    Parameters:
        df (pd.DataFrame): DataFrame containing feature columns and target 'Close'.
        split_ratio (float): Fraction of data to use for training (default 0.8 for 80/20 split).

    Returns:
        dict: Comprehensive results dictionary containing:
            - model: Trained best model object
            - best_model_name: "Linear Regression" or "Random Forest"
            - predictions: Array of predictions from the best model on X_test
            - mae: MAE of the best model
            - rmse: RMSE of the best model
            - mape: MAPE of the best model (%)
            - linear_model: Trained LinearRegression instance
            - linear_mae: MAE of Linear Regression
            - linear_rmse: RMSE of Linear Regression
            - linear_mape: MAPE of Linear Regression (%)
            - linear_predictions: Array of Linear Regression predictions
            - rf_model: Trained RandomForestRegressor instance
            - rf_mae: MAE of Random Forest
            - rf_rmse: RMSE of Random Forest
            - rf_mape: MAPE of Random Forest (%)
            - rf_predictions: Array of Random Forest predictions
            - X_train, X_test, y_train, y_test: Time-series train/test splits
            - comparison_df: Summary DataFrame comparing both models

    Raises:
        ValueError: If features or target are missing, or there are insufficient samples.
    """
    for col in FEATURE_COLUMNS + ["Close"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from DataFrame.")

    clean_df = df.dropna(subset=FEATURE_COLUMNS + ["Close"])

    if len(clean_df) < 20:
        raise ValueError("At least 20 samples are required to train and evaluate models.")

    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["Close"]

    # Strict chronological split: historical past trains future predictions
    split_idx = int(len(clean_df) * split_ratio)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    if len(X_train) < 10 or len(X_test) < 5:
        raise ValueError(
            f"Split resulted in too few samples: {len(X_train)} train, {len(X_test)} test. "
            "Select a longer historical period."
        )

    # 1. Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)

    linear_mae = float(mean_absolute_error(y_test, linear_predictions))
    linear_rmse = float(mean_squared_error(y_test, linear_predictions) ** 0.5)
    linear_mape = float(
        (np.abs(y_test.values - linear_predictions) / np.maximum(np.abs(y_test.values), 1e-8)).mean() * 100.0
    )

    # 2. Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    rf_mae = float(mean_absolute_error(y_test, rf_predictions))
    rf_rmse = float(mean_squared_error(y_test, rf_predictions) ** 0.5)
    rf_mape = float(
        (np.abs(y_test.values - rf_predictions) / np.maximum(np.abs(y_test.values), 1e-8)).mean() * 100.0
    )

    # Model Selection Rule: select the model with the lowest test Mean Absolute Error (MAE)
    if rf_mae < linear_mae:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_predictions = rf_predictions
        best_mae = rf_mae
        best_rmse = rf_rmse
        best_mape = rf_mape
    else:
        best_model = linear_model
        best_model_name = "Linear Regression"
        best_predictions = linear_predictions
        best_mae = linear_mae
        best_rmse = linear_rmse
        best_mape = linear_mape

    comparison_df = pd.DataFrame(
        {
            "MAE": [linear_mae, rf_mae],
            "RMSE": [linear_rmse, rf_rmse],
            "MAPE (%)": [linear_mape, rf_mape],
        },
        index=["Linear Regression", "Random Forest"]
    )

    return {
        "model": best_model,
        "best_model_name": best_model_name,
        "predictions": best_predictions,
        "mae": best_mae,
        "rmse": best_rmse,
        "mape": best_mape,
        "linear_model": linear_model,
        "linear_mae": linear_mae,
        "linear_rmse": linear_rmse,
        "linear_mape": linear_mape,
        "linear_predictions": linear_predictions,
        "rf_model": rf_model,
        "rf_mae": rf_mae,
        "rf_rmse": rf_rmse,
        "rf_mape": rf_mape,
        "rf_predictions": rf_predictions,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "comparison_df": comparison_df,
    }