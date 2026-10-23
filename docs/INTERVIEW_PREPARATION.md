# StockPilot — Placement & Technical Interview Preparation Guide

This guide contains beginner-friendly, technically accurate interview questions and structured answers designed for B.Tech Information Science / Computer Science students presenting StockPilot in campus placements, technical rounds, and ML engineer internships.

---

### Q1: What problem does StockPilot solve?
**Answer:**
StockPilot is an educational market analysis platform that bridges the gap between raw historical stock market data, quantitative technical analysis, and machine learning regression modeling. It enables users to study stock price trends, compare two classic regression models (Linear Regression vs. Random Forest) on an unbiased chronological timeline, evaluate momentum indicators (RSI, MACD, moving averages), analyze historical risk (volatility and maximum drawdown), and simulate hypothetical multi-currency investments.

---

### Q2: Why did you choose Python for this project?
**Answer:**
Python is the standard language for quantitative finance and machine learning. Its rich ecosystem allowed me to build an end-to-end pipeline efficiently:
- `yfinance` for automated market data ingestion.
- `pandas` and `numpy` for vector-based time-series transformations and moving window computations.
- `scikit-learn` for regression training, hyperparameter configuration, and mathematical evaluation metrics.
- `streamlit` for a rapid, reactive web dashboard without needing complex JavaScript frontend code.

---

### Q3: Why did you select Streamlit over traditional frameworks like Django or Flask?
**Answer:**
Streamlit is built specifically for data science and machine learning applications. It provides direct, reactive two-way binding with Python data structures (like DataFrames and charts). Building a full frontend in Django or React would require writing separate REST APIs, serializing data, and managing web state. Streamlit allowed me to concentrate on core analytical logic, feature engineering, and model validation while still delivering a professional dark trading-terminal user interface.

---

### Q4: How does `yfinance` work, and how did you handle edge cases?
**Answer:**
`yfinance` queries Yahoo Finance public API endpoints to fetch adjusted historical price series. During development, I implemented defensive handling for several edge cases:
1. **MultiIndex Columns:** In newer pandas/yfinance versions, columns are returned as MultiIndex tuples (e.g., `('Close', 'AAPL')`). I extracted the relevant series defensively using `iloc[:, 0]`.
2. **Missing Tickers & Empty Responses:** If a user selects an invalid or delisted symbol, the module catches empty returns and raises clear `ValueError` exceptions with guidance.
3. **Data Length Check:** The pipeline verifies a minimum threshold of 60 trading days so moving averages (like 50-day SMA) have sufficient data to initialize properly.

---

### Q5: What is feature engineering in this project, and how did you prevent data leakage?
**Answer:**
Feature engineering is the process of transforming raw historical prices into structured numerical inputs for machine learning models.
In StockPilot, we predict day $t$'s closing price. To prevent **data leakage (lookahead bias)**, features must only contain information known prior to session $t$:
- `Previous_Close`: The closing price at day $t-1$ (`Close.shift(1)`).
- `MA_5`: The rolling average of the 5 preceding sessions from day $t-5$ to $t-1$ (`Close.shift(1).rolling(5).mean()`).
Because both features are explicitly shifted by 1 session, row $t$'s features never access day $t$'s closing price during training.

---

### Q6: How does Linear Regression work in StockPilot?
**Answer:**
Linear Regression models the relationship between the target price ($y_t$) and the lagged predictors ($X_1, X_2$) as a linear equation:
$$\hat{y}_t = \beta_0 + \beta_1 \cdot \text{Previous\_Close} + \beta_2 \cdot \text{MA\_5}$$
Using Ordinary Least Squares (OLS), the model calculates the coefficients $\beta$ that minimize the sum of squared residuals on the chronological training split. It serves as our primary transparent baseline.

---

### Q7: How does the Random Forest Regressor work here?
**Answer:**
Random Forest is an ensemble learning method that constructs a collection of 100 de-correlated decision trees (`n_estimators=100`). Each tree is trained on a bootstrap sample of the training data and splits nodes using random feature subsets. The final price prediction is the average across all 100 trees:
$$\hat{y} = \frac{1}{100} \sum_{i=1}^{100} T_i(X)$$
To prevent the model from memorizing noise or overfitting on training data, I set `max_depth=8` and `min_samples_leaf=3`.

---

### Q8: Why did you compare two different models instead of just using Random Forest?
**Answer:**
In scientific machine learning, complex models are not guaranteed to outperform simpler ones, especially in financial time series. By comparing Linear Regression and Random Forest side-by-side on identical test sets, we test whether non-linear tree splits add genuine predictive value over a linear baseline. In fact, on trending price series, Linear Regression often achieves a lower MAE because tree models cannot extrapolate values beyond their training range.

---

### Q9: Why did you choose MAE and RMSE instead of classification accuracy?
**Answer:**
Stock price prediction is a **continuous regression task**, not a discrete classification task. Metrics like accuracy, precision, and recall are mathematically undefined for continuous prices.
- **MAE (Mean Absolute Error):** Represents the average dollar miss in intuitive currency units.
- **RMSE (Root Mean Squared Error):** Squaring errors before averaging penalizes larger forecast deviations more heavily, which is critical for financial risk assessment.
- We also display **MAPE (Mean Absolute Percentage Error)**, but label it strictly as a percentage error measure, not accuracy.

---

### Q10: Why did you avoid random train/test splitting (like `train_test_split(shuffle=True)`)?
**Answer:**
Random shuffling violates the fundamental assumption of time-series data. In finance, tomorrow's price is autocorrelated with today's price. If we shuffle data randomly, the model trains on future dates (e.g. day 50) to predict past dates (e.g. day 49). This leaks future regime information, giving artificially low test errors that fail completely in reality. StockPilot strictly uses **chronological splitting** (first 80% train, subsequent 20% test).

---

### Q11: What is the Relative Strength Index (RSI), and how is it used?
**Answer:**
RSI is a momentum oscillator with values bounded between 0 and 100, calculated over a 14-day rolling window using Wilder's exponential smoothing. It measures the ratio of average gains to average losses.
- RSI above 70 indicates potentially overbought conditions.
- RSI below 30 indicates potentially oversold conditions.
- In StockPilot, an RSI above 50 confirms positive short-term bullish momentum in our rule-based signal.

---

### Q12: What is MACD, and what does it indicate?
**Answer:**
MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator.
1. **MACD Line:** The difference between a short-term 12-day EMA and a longer-term 26-day EMA of closing prices.
2. **Signal Line:** A 9-day EMA of the MACD line itself.
When the MACD line crosses above the Signal line, it signals accelerating upward momentum.

---

### Q13: What is Maximum Drawdown, and how is it calculated?
**Answer:**
Maximum Drawdown (MDD) is a measure of downside risk. It quantifies the largest percentage drop from a historical peak to a subsequent trough before a new peak is reached:
$$\text{Drawdown} = \frac{\text{Price} - \text{Cumulative Peak}}{\text{Cumulative Peak}} \times 100\%$$
MDD answers the question: *"What is the worst loss an investor would have suffered if they bought at the absolute peak and sold at the lowest bottom?"*

---

### Q14: How does the Virtual Investment Simulator handle multi-currency conversions?
**Answer:**
The simulator takes a hypothetical starting capital, checks the stock's native currency (INR for NSE `.NS` stocks, USD for US stocks), and computes the share purchase quantity at the selected start date. At the end date, it values the holding and converts the returns to the user's chosen display currency (USD, INR, EUR, GBP, JPY, CAD, AUD, SGD) by querying a real-time exchange rates endpoint (`open.er-api.com`). If the foreign exchange API is unreachable, it catches the exception and informs the user rather than breaking the application.

---

### Q15: What are the fundamental limitations of this model?
**Answer:**
1. **Market Non-Stationarity & Exogenous Shocks:** Asset prices are influenced by interest rates, quarterly earnings, regulatory policies, and macroeconomic news—none of which are captured by price-lag features alone.
2. **Tree Extrapolation:** Random Forest decision trees predict values as leaf node averages, meaning they cannot forecast prices higher or lower than the historical training range.
3. **No Direct Trading Advice:** The project demonstrates regression modeling and indicator mathematics for academic benchmarking, not financial advisory.

---

### Q16: What challenges did you encounter and resolve?
**Answer:**
1. **Time-Series Lookahead Bias:** Ensuring rolling indicators and features did not include session $t$'s price when building input matrices.
2. **Windows Console Character Encoding:** When exporting command-line reports containing directional emojis (`📈`), PowerShell on Windows cp1252 threw `UnicodeEncodeError`. I resolved this by reconfiguring standard streams to UTF-8.
3. **MultiIndex DataFrame Structures:** Handling yfinance updates that returned MultiIndex columns by flattening them into standard single-index DataFrames.
4. **Deterministic Testing:** Building a 100% offline pytest suite with 30 tests that generates reproducible synthetic price paths without depending on external web APIs.

---

### Q17: What improvements would you implement next?
**Answer:**
1. **Additional Exogenous Features:** Incorporate trading volume, volatility indices (VIX), and sector ETF momentum.
2. **Walk-Forward Validation:** Implement expanding-window (rolling) backtesting instead of a single static 80/20 train/test split.
3. **Sequence Models:** Benchmark recurrent architectures such as LSTM or GRU networks against the linear and tree baselines.
