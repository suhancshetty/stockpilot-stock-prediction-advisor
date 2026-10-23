# StockPilot — Machine Learning & Financial Methodology

This document outlines the theoretical foundations, mathematical formulas, feature engineering principles, evaluation metrics, and intrinsic limitations of the machine learning and technical analysis models implemented in **StockPilot**.

---

## 1. Feature Engineering & Preventing Data Leakage

In financial time-series forecasting, **data leakage (lookahead bias)** occurs when information from the target timestep $t$ or future timesteps $t+k$ inadvertently leaks into the training feature matrix.

### Formulation
Let $P_t$ represent the closing price of a stock at trading day $t$.
- We want to predict target $y_t = P_t$.
- Features must be strictly derived from the information set $\mathcal{F}_{t-1} = \{P_{t-1}, P_{t-2}, \dots, P_0\}$.

StockPilot implements two lagged predictors:
1. **`Previous_Close`**:
   $$X_{1, t} = P_{t-1}$$
   The immediate prior session closing price, capturing short-term price memory and base level.
2. **`MA_5` (5-Session Moving Average)**:
   $$X_{2, t} = \frac{1}{5} \sum_{i=1}^{5} P_{t-i}$$
   The simple moving average of the preceding five sessions ($t-5$ to $t-1$), capturing weekly short-term trend and smoothing out daily noise.

By computing features using `.shift(1)` before applying rolling windows, StockPilot mathematically guarantees zero target leakage for each historical observation $t$.

---

## 2. Chronological Train/Test Partitioning

### Why Random Splitting is Invalid for Financial Time Series
Standard machine learning cross-validation techniques (such as random `train_test_split` or standard k-fold cross-validation) assume samples are Independent and Identically Distributed (i.i.d.). 

Financial prices are **autocorrelated and non-stationary**. If we randomly shuffle data:
- A model might train on day $t+1$ and test on day $t$.
- The model memorizes future regime states and interpolated boundaries, producing deceptively low error metrics that completely collapse in live forward deployment.

### StockPilot's Approach
StockPilot enforces a **chronological split** (80% train / 20% test):
- **Training Set**: Sessions $0$ to $K = \lfloor 0.8 \times N \rfloor$
- **Testing Set**: Sessions $K+1$ to $N$
- **Temporal Constraint**: $\max(\text{Date}_{\text{train}}) < \min(\text{Date}_{\text{test}})$

---

## 3. Regression Algorithms

StockPilot trains and compares two complementary algorithms:

### 1. Ordinary Least Squares Linear Regression
- **Mechanism**: Models the target price as a linear combination of input features:
  $$\hat{y}_t = \beta_0 + \beta_1 X_{1, t} + \beta_2 X_{2, t}$$
- **Strengths**: Highly interpretable, fast to fit, resistant to complex overfitting, serves as an essential statistical baseline.
- **Weakness**: Cannot capture non-linear market interactions or sudden volatility regime shifts.

### 2. Random Forest Regressor
- **Mechanism**: An ensemble of $B = 100$ decision trees trained via bootstrap aggregation (bagging) and random feature subspace selection:
  $$\hat{y}_t = \frac{1}{B} \sum_{b=1}^{B} T_b(X_t)$$
- **Hyperparameter Constraints**:
  - `max_depth = 8`: Constrains individual tree depth to prevent memorizing historical noise.
  - `min_samples_leaf = 3`: Requires minimum support in terminal leaves.
  - `random_state = 42`: Ensures deterministic, reproducible forest generation.
- **Strengths**: Robust to outliers, captures non-linear relationships.
- **Limitation**: Tree-based models cannot extrapolate beyond the minimum and maximum price levels observed during training.

---

## 4. Evaluation Metrics

Because stock price forecasting is a continuous regression task, classification metrics (like Precision, Recall, or F1-Score) are not applicable.

### 1. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- Measures the average dollar error in original currency units (e.g. \$, ₹).
- Scale-dependent and resilient to isolated extreme outliers.

### 2. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- Penalizes large forecast errors disproportionately due to squaring.
- Essential for risk management where rare large forecast misses carry high financial risk.

### 3. Mean Absolute Percentage Error (MAPE)
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
- Normalized percentage error metric allowing scale-free comparison across stocks of differing share prices.
- **Important**: Labeled strictly as a percentage error metric, never as a classification accuracy score.

---

## 5. Technical Indicators & Market Signals

### Relative Strength Index (RSI, 14-period)
Measures the velocity and magnitude of recent price changes to evaluate overbought or oversold conditions:
$$\text{RS} = \frac{\text{Average Gain over 14 sessions}}{\text{Average Loss over 14 sessions}}$$
$$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$
- $\text{RSI} > 70$: Overbought condition.
- $\text{RSI} < 30$: Oversold condition.
- $\text{RSI} > 50$: Positive price momentum.

### Moving Average Convergence Divergence (MACD)
Trend-following momentum indicator capturing the relationship between two exponential moving averages:
- $\text{MACD Line} = \text{EMA}_{12}(Close) - \text{EMA}_{26}(Close)$
- $\text{Signal Line} = \text{EMA}_{9}(\text{MACD Line})$
- A bullish crossover occurs when the MACD line rises above the Signal line.

### Rule-Based Signal Formulation
- **BUY**: Price $> MA_{20} \land MA_{20} > MA_{50} \land \text{RSI} > 50 \land \text{MACD} > \text{MACD\_Signal}$
- **SELL**: Price $< MA_{20} \land MA_{20} < MA_{50} \land \text{RSI} < 50 \land \text{MACD} < \text{MACD\_Signal}$
- **HOLD**: When indicator conditions are mixed or conflicting.

---

## 6. Risk Analytics

### Annualized Volatility
Measures the dispersion of daily returns annualized over a standard trading year:
$$\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252} \times 100\%$$
Higher annualized volatility indicates higher uncertainty and potential fluctuation.

### Maximum Drawdown (MDD)
The peak-to-trough drop observed over the selected historical period:
$$\text{Drawdown}_t = \frac{P_t - \max_{0 \le s \le t} P_s}{\max_{0 \le s \le t} P_s} \times 100\%$$
$$\text{MDD} = \min_{t} \text{Drawdown}_t$$
Quantifies capital preservation risk under worst-case historical market timing.

---

## 7. Theoretical Limitations & Ethical Considerations

1. **Efficient Market Hypothesis (EMH)**: Asset prices largely reflect all available public information. Technical price history alone cannot predict corporate earnings shocks, monetary policy shifts, or geopolitical disruptions.
2. **Non-Stationarity**: Market distribution parameters (mean return, variance, correlation) shift continuously across economic regimes.
3. **No Financial Advice Disclaimer**: StockPilot is an educational benchmarking and analytical software project. It is strictly not intended for real-money trade execution or investment advisement.
