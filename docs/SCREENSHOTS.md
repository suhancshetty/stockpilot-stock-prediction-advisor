# Visual Capture Guide — Capturing Genuine Project Screenshots

This guide walks you through capturing real, high-resolution screenshots from your live **StockPilot** dashboard to showcase on GitHub and in project presentations.

---

## Prerequisites
1. Start the Streamlit application in Windows PowerShell:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
2. Your browser will automatically open at `http://localhost:8501`.
3. In the sidebar:
   - Select **US Market** -> **Apple** (`AAPL`) or **Indian Market** -> **Reliance Industries** (`RELIANCE.NS`).
   - Select Historical Period: **1y**.
   - Click the blue **↻ Fetch Data and Analyze** button.
4. Wait for the green success message: *"Historical data retrieved and machine-learning models evaluated."*

---

## Recommended Screenshots to Capture

Use the Windows Snipping Tool shortcut: **`Windows Key + Shift + S`** to crop and save the following views:

### 1. Main Dashboard Header & Sidebar
- **Region to Crop:** Top area showing the sidebar controls (Market selector, company dropdown, historical period) and the title banner (*"Stock Prediction Advisor — Apple · AAPL"*).
- **Target Filename:** `assets/dashboard_overview.png`

### 2. Prediction Summary & Direction Card
- **Region to Crop:** The four top metric cards (*Latest Close, Model Estimate, Best Model, Test MAE*) and the **Model's Price Direction** box with the direction badge (*UP 📈* or *DOWN 📉*).
- **Target Filename:** `assets/prediction_summary.png`

### 3. Machine Learning Model Comparison
- **Region to Crop:** The **Machine Learning Model Comparison** table showing MAE and RMSE side-by-side for Linear Regression and Random Forest, and the **How Close Were the Predictions?** MAPE cards.
- **Target Filename:** `assets/model_comparison.png`

### 4. Technical Indicators & Rule-Based Signal
- **Region to Crop:** The 4 indicator cards (*20-Day Average, 50-Day Average, RSI (14), MACD*) and the highlighted **Rule-Based Market Signal** banner (*BUY*, *SELL*, or *HOLD*).
- **Target Filename:** `assets/technical_indicators.png`

### 5. Risk Analytics & Investment Simulator
- **Region to Crop:** The **Risk Analytics** metrics (*Annualized Volatility, Maximum Drawdown, Average Daily Return*) and the **Virtual Investment Simulator** cards with currency conversion.
- **Target Filename:** `assets/investment_simulator.png`

### 6. Historical & Actual-vs-Predicted Charts
- **Region to Crop:** The two interactive line charts:
  - Historical Price and Moving Average Trends (Close, MA 20, MA 50).
  - Actual vs Predicted test period comparison.
- **Target Filename:** `assets/actual_vs_predicted_chart.png`

---

## Adding Images to `README.md`
Once captured, save your images into the `assets/` folder. They will immediately display in the `README.md` screenshots section:
```markdown
![StockPilot Dashboard Overview](assets/dashboard_overview.png)
```
