"""
StockPilot — AI-Powered Stock Prediction Advisor
Streamlit Web Application & Interactive Trading Dashboard
"""

import streamlit as st
import pandas as pd
import requests

from modules.data_fetcher import fetch_stock_data
from modules.indicators import calculate_indicators, generate_technical_signal
from modules.features import prepare_features
from modules.model_trainer import train_and_evaluate_models
from modules.predictor import predict_next_session
from modules.risk_analyzer import calculate_risk_metrics


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StockPilot | Stock Prediction Advisor",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# DARK TRADING TERMINAL STYLE
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1220;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111c2d;
    }

    [data-testid="stMetric"] {
        background-color: #142136;
        border: 1px solid #263449;
        padding: 16px;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #eaf2ff;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #1685d1;
        color: white;
        border-radius: 8px;
        border: none;
        min-height: 42px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #0868ad;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# STOCK UNIVERSE
# =========================================================

INDIAN_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tata Motors": "TMPV.NS",
    "Larsen & Toubro": "LT.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
}

US_STOCKS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Alphabet (Google)": "GOOGL",
    "Meta Platforms": "META",
    "Tesla": "TSLA",
    "Broadcom": "AVGO",
    "JPMorgan Chase": "JPM",
    "Walmart": "WMT",
}


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.title("StockPilot")
st.sidebar.caption("STOCK PREDICTION ADVISOR")
st.sidebar.divider()

st.sidebar.subheader("Analysis Settings")

market = st.sidebar.selectbox(
    "Select Market",
    ["Indian Market", "US Market"]
)

stocks = INDIAN_STOCKS if market == "Indian Market" else US_STOCKS

selected_name = st.sidebar.selectbox(
    "Select Company",
    list(stocks.keys())
)

ticker = stocks[selected_name]

period = st.sidebar.selectbox(
    "Historical Period",
    ["3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    index=2
)


# =========================================================
# MAIN DASHBOARD HEADER
# =========================================================

st.title("Stock Prediction Advisor")
st.caption("MARKET INTELLIGENCE  /  MACHINE LEARNING  /  RISK ANALYTICS")
st.markdown(f"### {selected_name} · `{ticker}`")

st.info(
    "Choose a market, company, and historical period in the sidebar. "
    "Click Fetch Data and Analyze to generate the machine-learning forecast and risk metrics."
)

if st.button("↻ Fetch Data and Analyze", use_container_width=True):
    st.session_state.analysis_done = True


# =========================================================
# ANALYSIS EXECUTION & PRESENTATION
# =========================================================

if st.session_state.analysis_done:
    with st.spinner(f"Downloading historical data and training models for {ticker}..."):
        try:
            # 1. Fetch historical data
            df = fetch_stock_data(ticker, period=period)

            # 2. Compute technical indicators
            df = calculate_indicators(df)

            # 3. Prepare lagged machine learning features (zero lookahead bias)
            df = prepare_features(df)

            # 4. Train & evaluate regression models (chronological split)
            trainer_results = train_and_evaluate_models(df)

            # 5. Predict next-session price estimate
            pred_results = predict_next_session(trainer_results["model"], df)

            # 6. Compute portfolio risk analytics
            risk_metrics = calculate_risk_metrics(df)

            # 7. Evaluate latest technical signal
            latest = df.iloc[-1]
            signal_data = generate_technical_signal(latest)

            st.success("Historical data retrieved and machine-learning models evaluated.")

            # -------------------------------------------------
            # SECTION 1: STOCK OVERVIEW
            # -------------------------------------------------
            st.subheader("Stock Overview")

            latest_close = pred_results["latest_close"]
            period_start_price = float(df["Close"].iloc[0])
            period_high = float(df["Close"].max())
            period_low = float(df["Close"].min())
            period_change_pct = ((latest_close - period_start_price) / period_start_price) * 100.0

            info1, info2, info3, info4 = st.columns(4)
            info1.metric("First Price in Period", f"{period_start_price:,.2f}")
            info2.metric("Highest Price", f"{period_high:,.2f}")
            info3.metric("Lowest Price", f"{period_low:,.2f}")
            info4.metric("Overall Price Change", f"{period_change_pct:+.2f}%")

            # -------------------------------------------------
            # SECTION 2: PREDICTION SUMMARY
            # -------------------------------------------------
            st.subheader("Prediction Summary")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latest Close", f"{latest_close:,.2f}")
            col2.metric(
                "Model Estimate",
                f"{pred_results['next_price']:,.2f}",
                f"{pred_results['change_pct']:+.2f}%"
            )
            col3.metric("Best Model", trainer_results["best_model_name"])
            col4.metric("Test MAE", f"{trainer_results['mae']:,.2f}")

            st.markdown("### Model's Price Direction")
            st.write(f"**{pred_results['direction']}**")
            st.write(pred_results["direction_message"])
            st.caption(
                "Note: This is an experimental statistical estimate based on past prices. "
                "It is NOT a guaranteed future price or personalized financial advice."
            )

            # -------------------------------------------------
            # SECTION 3: MODEL COMPARISON
            # -------------------------------------------------
            st.subheader("Machine Learning Model Comparison")

            comparison_display = trainer_results["comparison_df"][["MAE", "RMSE"]]
            st.dataframe(
                comparison_display.style.format({
                    "MAE": "{:,.3f}",
                    "RMSE": "{:,.3f}"
                }),
                use_container_width=True
            )
            st.caption(
                "Chronological 80/20 train/test evaluation. Lower MAE and RMSE indicate smaller "
                "historical test errors. Model selection is based on the lowest test MAE."
            )

            # -------------------------------------------------
            # SECTION 4: ERROR EVALUATION (MAPE)
            # -------------------------------------------------
            st.subheader("How Close Were the Predictions?")

            accuracy_col1, accuracy_col2 = st.columns(2)
            accuracy_col1.metric(
                "Linear Regression Average Error",
                f"{trainer_results['linear_mape']:.2f}%"
            )
            accuracy_col2.metric(
                "Random Forest Average Error",
                f"{trainer_results['rf_mape']:.2f}%"
            )
            st.caption(
                "These percentages are Mean Absolute Percentage Errors (MAPE), "
                "not classification accuracy scores. Lower values indicate closer fit."
            )

            # -------------------------------------------------
            # SECTION 5: TECHNICAL INDICATORS
            # -------------------------------------------------
            st.subheader("Technical Indicators")

            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.metric("20-Day Average", f"{latest['MA_20']:,.2f}")
            t_col2.metric("50-Day Average", f"{latest['MA_50']:,.2f}")
            t_col3.metric("RSI (14)", f"{latest['RSI']:.2f}")
            t_col4.metric("MACD", f"{latest['MACD']:.2f}")

            # -------------------------------------------------
            # SECTION 6: RULE-BASED MARKET SIGNAL
            # -------------------------------------------------
            st.subheader("Rule-Based Market Signal")

            sig = signal_data["signal"]
            if sig == "BUY":
                st.success(f"Signal: {sig}")
            elif sig == "SELL":
                st.error(f"Signal: {sig}")
            else:
                st.warning(f"Signal: {sig}")

            st.write(signal_data["explanation"])
            st.caption(
                "The signal is generated strictly by technical momentum indicators. "
                "It is not personalized investment advice."
            )

            # -------------------------------------------------
            # SECTION 7: RISK ANALYTICS
            # -------------------------------------------------
            st.subheader("Risk Analytics")

            risk_col1, risk_col2, risk_col3 = st.columns(3)
            risk_col1.metric(
                "Annualized Volatility",
                f"{risk_metrics['annualized_volatility']:.2f}%"
            )
            risk_col2.metric(
                "Maximum Drawdown",
                f"{risk_metrics['max_drawdown']:.2f}%"
            )
            risk_col3.metric(
                "Average Daily Return",
                f"{risk_metrics['average_daily_return']:.3f}%"
            )
            st.caption(
                "Annualized volatility assumes 252 trading sessions. "
                "Maximum drawdown represents the greatest drop from a historical peak."
            )

            # -------------------------------------------------
            # SECTION 8: DOWNLOAD STOCK DATA
            # -------------------------------------------------
            st.subheader("Download Stock Data")

            csv_data = df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="Download Analysis as CSV",
                data=csv_data,
                file_name=f"{ticker}_stock_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

            # -------------------------------------------------
            # SECTION 9: VIRTUAL INVESTMENT SIMULATOR
            # -------------------------------------------------
            st.subheader("Virtual Investment Simulator")
            st.write("Explore a hypothetical investment using historical prices.")

            native_currency = "INR" if ticker.endswith(".NS") else "USD"

            currency_options = {
                "Indian Rupee (₹)": ("INR", "₹"),
                "US Dollar ($)": ("USD", "$"),
                "Euro (€)": ("EUR", "€"),
                "British Pound (£)": ("GBP", "£"),
                "Japanese Yen (¥)": ("JPY", "¥"),
                "Canadian Dollar (C$)": ("CAD", "C$"),
                "Australian Dollar (A$)": ("AUD", "A$"),
                "Singapore Dollar (S$)": ("SGD", "S$")
            }

            currency_label = st.selectbox(
                "Display results in",
                list(currency_options.keys()),
                index=(0 if native_currency == "INR" else 1),
                key="sim_currency"
            )

            display_currency, display_symbol = currency_options[currency_label]

            investment_amount = st.number_input(
                f"Investment amount ({native_currency})",
                min_value=1000.0,
                max_value=10000000.0,
                value=10000.0,
                step=1000.0,
                key="sim_investment_amount"
            )

            available_dates = df.index.strftime("%Y-%m-%d").tolist()

            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                start_date = st.selectbox(
                    "Investment start date",
                    available_dates,
                    index=0,
                    key="sim_start_date"
                )
            with sim_col2:
                end_date = st.selectbox(
                    "Investment end date",
                    available_dates,
                    index=len(available_dates) - 1,
                    key="sim_end_date"
                )

            date_prices = df.copy()
            date_prices.index = date_prices.index.strftime("%Y-%m-%d")

            start_price = float(date_prices.loc[start_date, "Close"])
            end_price = float(date_prices.loc[end_date, "Close"])

            if end_date <= start_date:
                st.warning("Please select an end date later than the start date.")
            else:
                try:
                    if display_currency == native_currency:
                        exchange_rate = 1.0
                    else:
                        response = requests.get(
                            f"https://open.er-api.com/v6/latest/{native_currency}",
                            timeout=10
                        )
                        response.raise_for_status()
                        fx_data = response.json()

                        if fx_data.get("result") != "success":
                            raise ValueError("Exchange-rate service responded with an error.")

                        exchange_rate = float(fx_data["rates"][display_currency])

                    shares = investment_amount / start_price
                    ending_value_native = shares * end_price

                    starting_display = investment_amount * exchange_rate
                    ending_display = ending_value_native * exchange_rate
                    profit_loss = ending_display - starting_display
                    return_pct = (profit_loss / starting_display) * 100.0

                    st.markdown("#### Simulation Results")
                    result1, result2, result3 = st.columns(3)
                    result1.metric("Starting Investment", f"{display_symbol}{starting_display:,.2f}")
                    result2.metric("Ending Value", f"{display_symbol}{ending_display:,.2f}")
                    result3.metric(
                        "Profit / Loss",
                        f"{display_symbol}{profit_loss:+,.2f}",
                        f"{return_pct:+.2f}%"
                    )

                    st.caption(
                        f"Exchange rate: 1 {native_currency} = {exchange_rate:.6f} {display_currency}. "
                        "Excludes brokerage fees, taxes, and dividends. "
                        "This is a hypothetical simulation, not financial advice."
                    )
                except Exception as error:
                    st.error(f"Could not retrieve exchange rate: {error}")

            # -------------------------------------------------
            # SECTION 10: HISTORICAL PRICE CHART
            # -------------------------------------------------
            st.subheader("Historical Price and Trends")

            price_chart = df[["Close", "MA_20", "MA_50"]].rename(columns={
                "Close": "Stock Price",
                "MA_20": "20-Day Average",
                "MA_50": "50-Day Average"
            })
            st.line_chart(price_chart, use_container_width=True)
            st.caption(
                "Closing price shown alongside 20-day (short-term) and 50-day (intermediate) moving averages."
            )

            # -------------------------------------------------
            # SECTION 11: ACTUAL VS PREDICTED CHART
            # -------------------------------------------------
            st.subheader("Actual vs Predicted")

            comparison = pd.DataFrame({
                "Actual": trainer_results["y_test"],
                "Predicted": trainer_results["predictions"]
            }, index=trainer_results["y_test"].index)

            st.line_chart(comparison, use_container_width=True)
            st.caption(
                f"Comparison on chronological test partition using best model "
                f"({trainer_results['best_model_name']}). Past fit does not guarantee future results."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

else:
    st.markdown("### Explore the dashboard")
    st.write(
        f"Selected stock: **{selected_name}** (`{ticker}`). "
        "Click **Fetch Data and Analyze** above to generate the full analysis."
    )