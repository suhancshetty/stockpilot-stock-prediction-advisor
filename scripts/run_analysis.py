"""
Command-Line Analysis and Report Generation Script for StockPilot.

Runs the complete machine learning and technical analysis workflow, generates
evaluation metrics, and saves static charts and reports to the outputs/ directory.

Usage:
    python scripts/run_analysis.py --ticker AAPL --period 1y
    python scripts/run_analysis.py --synthetic
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# Handle Windows cp1252 console encoding issues with emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.data_fetcher import fetch_stock_data
from modules.indicators import calculate_indicators, generate_technical_signal
from modules.features import prepare_features
from modules.model_trainer import train_and_evaluate_models
from modules.predictor import predict_next_session
from modules.risk_analyzer import calculate_risk_metrics


def generate_synthetic_data(n_sessions: int = 150, initial_price: float = 100.0) -> pd.DataFrame:
    """
    Generate deterministic synthetic stock data for offline testing and evaluation.
    """
    np.random.seed(42)
    # Geometric Brownian motion-like daily returns
    daily_returns = np.random.normal(loc=0.0005, scale=0.018, size=n_sessions)
    price_path = initial_price * np.cumprod(1 + daily_returns)

    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_sessions)]

    df = pd.DataFrame({"Close": price_path}, index=pd.DatetimeIndex(dates))
    return df


def run_pipeline(ticker: str = "AAPL", period: str = "1y", use_synthetic: bool = False):
    """
    Execute full StockPilot pipeline and export reports & charts.
    """
    print(f"\n==================================================")
    print(f" StockPilot — Automated Pipeline Execution")
    print(f"==================================================")

    reports_dir = os.path.join(PROJECT_ROOT, "outputs", "reports")
    graphs_dir = os.path.join(PROJECT_ROOT, "outputs", "graphs")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    data_source_label = f"Yahoo Finance ({ticker}, period: {period})"

    if use_synthetic:
        print("[INFO] Generating deterministic synthetic stock series for offline run...")
        df = generate_synthetic_data(n_sessions=150)
        data_source_label = f"Synthetic Deterministic Series ({ticker} simulation)"
    else:
        print(f"[INFO] Fetching historical data for '{ticker}' (period: {period})...")
        try:
            df = fetch_stock_data(ticker, period=period)
        except Exception as exc:
            print(f"[WARNING] Live fetch failed ({exc}). Falling back to synthetic offline data.")
            df = generate_synthetic_data(n_sessions=150)
            data_source_label = f"Synthetic Series (Fallback for {ticker})"

    print(f"[INFO] Retrieved {len(df)} price observations.")

    # 1. Technical Indicators
    df_indicators = calculate_indicators(df)

    # 2. Features Preparation
    df_features = prepare_features(df_indicators)

    # 3. Model Training & Evaluation
    results = train_and_evaluate_models(df_features)

    # 4. Next-Session Prediction
    pred = predict_next_session(results["model"], df_features)

    # 5. Risk Analytics
    risk = calculate_risk_metrics(df_features)

    # 6. Technical Signal
    latest_row = df_indicators.iloc[-1]
    signal_info = generate_technical_signal(latest_row)

    print("\n--------------------------------------------------")
    print(" Pipeline Results Summary")
    print("--------------------------------------------------")
    print(f"Data Source       : {data_source_label}")
    print(f"Best Model        : {results['best_model_name']}")
    print(f"Linear Reg MAE    : {results['linear_mae']:.4f} | RMSE: {results['linear_rmse']:.4f} | MAPE: {results['linear_mape']:.2f}%")
    print(f"Random Forest MAE : {results['rf_mae']:.4f} | RMSE: {results['rf_rmse']:.4f} | MAPE: {results['rf_mape']:.2f}%")
    print(f"Latest Close      : ${pred['latest_close']:.2f}")
    print(f"Next Est. Close   : ${pred['next_price']:.2f} ({pred['change_pct']:+.2f}%) -> {pred['direction']}")
    print(f"Annual Volatility : {risk['annualized_volatility']:.2f}%")
    print(f"Max Drawdown      : {risk['max_drawdown']:.2f}%")
    print(f"Technical Signal  : {signal_info['signal']} — {signal_info['explanation']}")

    # =========================================================
    # EXPORT REPORTS
    # =========================================================
    results_csv_path = os.path.join(reports_dir, "model_results.csv")
    results["comparison_df"].to_csv(results_csv_path, index=True)
    print(f"\n[SAVED] Model results table -> {results_csv_path}")

    evaluation_txt_path = os.path.join(reports_dir, "model_evaluation.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(evaluation_txt_path, "w", encoding="utf-8") as f:
        f.write("===================================================================\n")
        f.write(" STOCKPILOT MACHINE LEARNING & RISK EVALUATION REPORT\n")
        f.write("===================================================================\n")
        f.write(f"Generated At      : {timestamp}\n")
        f.write(f"Target Symbol     : {ticker}\n")
        f.write(f"Data Source       : {data_source_label}\n")
        f.write(f"Usable Sessions   : {len(df_features)}\n")
        f.write(f"Train/Test Split  : 80% Train ({len(results['X_train'])} obs) / 20% Test ({len(results['X_test'])} obs)\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(" MODEL PERFORMANCE METRICS (CHRONOLOGICAL TEST SET)\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(f"Selected Best Model: {results['best_model_name']}\n\n")
        f.write(f"Linear Regression:\n")
        f.write(f"  - Mean Absolute Error (MAE)       : {results['linear_mae']:.4f}\n")
        f.write(f"  - Root Mean Squared Error (RMSE)  : {results['linear_rmse']:.4f}\n")
        f.write(f"  - Mean Absolute Pct Error (MAPE)  : {results['linear_mape']:.2f}%\n\n")
        f.write(f"Random Forest Regressor:\n")
        f.write(f"  - Mean Absolute Error (MAE)       : {results['rf_mae']:.4f}\n")
        f.write(f"  - Root Mean Squared Error (RMSE)  : {results['rf_rmse']:.4f}\n")
        f.write(f"  - Mean Absolute Pct Error (MAPE)  : {results['rf_mape']:.2f}%\n\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(" NEXT SESSION PRICE ESTIMATE\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(f"Latest Recorded Close  : {pred['latest_close']:.2f}\n")
        f.write(f"Projected Next Close   : {pred['next_price']:.2f}\n")
        f.write(f"Estimated Change       : {pred['change']:+.2f} ({pred['change_pct']:+.2f}%)\n")
        f.write(f"Directional Sentiment  : {pred['direction']}\n")
        f.write(f"Model Summary Note     : {pred['direction_message']}\n\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(" RISK & TECHNICAL MOMENTUM METRICS\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(f"Annualized Volatility (252) : {risk['annualized_volatility']:.2f}%\n")
        f.write(f"Maximum Drawdown            : {risk['max_drawdown']:.2f}%\n")
        f.write(f"Average Daily Return        : {risk['average_daily_return']:.3f}%\n")
        f.write(f"Technical Indicator Signal  : {signal_info['signal']}\n")
        f.write(f"Signal Rationale            : {signal_info['explanation']}\n\n")
        f.write("DISCLAIMER: For educational and benchmarking purposes only. Not investment advice.\n")
    print(f"[SAVED] Evaluation text report -> {evaluation_txt_path}")

    # =========================================================
    # EXPORT MATPLOTLIB CHARTS
    # =========================================================
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # 1. Actual vs Predicted Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    y_test_series = results["y_test"]
    ax.plot(y_test_series.index, y_test_series.values, label="Actual Price", color="#1685d1", linewidth=2.0)
    ax.plot(y_test_series.index, results["predictions"], label=f"Predicted ({results['best_model_name']})", color="#e65100", linestyle="--", linewidth=2.0)
    ax.set_title(f"Actual vs Predicted Stock Price — {ticker} ({results['best_model_name']})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Price ($)", fontsize=10)
    ax.legend(loc="upper left")
    plt.tight_layout()
    actual_vs_pred_path = os.path.join(graphs_dir, "actual_vs_predicted.png")
    fig.savefig(actual_vs_pred_path)
    plt.close(fig)
    print(f"[SAVED] Actual vs Predicted chart -> {actual_vs_pred_path}")

    # 2. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    models = ["Linear Regression", "Random Forest"]
    x = np.arange(len(models))
    width = 0.35
    maes = [results["linear_mae"], results["rf_mae"]]
    rmses = [results["linear_rmse"], results["rf_rmse"]]

    rects1 = ax.bar(x - width/2, maes, width, label="MAE", color="#1976d2")
    rects2 = ax.bar(x + width/2, rmses, width, label="RMSE", color="#ff7043")

    ax.set_title(f"Model Error Comparison: MAE vs RMSE — {ticker}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Error Metric Value", fontsize=10)
    ax.legend()
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    model_comp_path = os.path.join(graphs_dir, "model_comparison.png")
    fig.savefig(model_comp_path)
    plt.close(fig)
    print(f"[SAVED] Model comparison chart -> {model_comp_path}")

    # 3. Stock Price & Moving Averages Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(df_indicators.index, df_indicators["Close"], label="Stock Price (Close)", color="#0d47a1", linewidth=1.8)
    if "MA_20" in df_indicators.columns and df_indicators["MA_20"].notna().any():
        ax.plot(df_indicators.index, df_indicators["MA_20"], label="20-Day MA", color="#2e7d32", linestyle="-.", linewidth=1.5)
    if "MA_50" in df_indicators.columns and df_indicators["MA_50"].notna().any():
        ax.plot(df_indicators.index, df_indicators["MA_50"], label="50-Day MA", color="#c62828", linestyle=":", linewidth=1.5)
    ax.set_title(f"Historical Price & Moving Average Trends — {ticker}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Price ($)", fontsize=10)
    ax.legend(loc="upper left")
    plt.tight_layout()
    trends_path = os.path.join(graphs_dir, "stock_price_trends.png")
    fig.savefig(trends_path)
    plt.close(fig)
    print(f"[SAVED] Stock price trends chart -> {trends_path}")

    print("\n[SUCCESS] Pipeline execution finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockPilot Analysis and Reporting Script")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--period", type=str, default="1y", help="Historical period (default: 1y)")
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic synthetic data for offline execution")
    args = parser.parse_args()

    run_pipeline(ticker=args.ticker, period=args.period, use_synthetic=args.synthetic)
