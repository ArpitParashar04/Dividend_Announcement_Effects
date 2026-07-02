# Gold & Silver Price Prediction

### Time Series — ARIMA vs. Regularised Lag-Feature Regression

---

## Overview

This project forecasts daily closing prices for **Gold (GC=F)** and **Silver (SI=F)** futures
using two complementary modeling paradigms: a classical **univariate ARIMA** pipeline and a
**lag-feature regression** approach (Linear, Ridge, Lasso) that reframes forecasting as a
supervised learning problem. It began as a basic ARIMA script (`gold_forecasting.py`) and was
substantially extended into a full academic-grade notebook
(`Gold_Silver_Prediction_Enhanced.ipynb`) with rigorous EDA, dual stationarity testing, residual
diagnostics, and side-by-side multi-model evaluation.

Data is pulled live from Yahoo Finance via `yfinance`, with a reproducible synthetic fallback
generator if the API/network is unavailable.

---

## Key Findings

| Model                     | MAPE (Gold)  | MAPE (Silver) | Verdict                                             |
| ---------------------------- | -------------- | ---------------- | ------------------------------------------------------ |
| ARIMA                        | ~34% (high)    | ~12% (moderate)   | Structural divergence in multi-step forecasts            |
| Linear Regression             | ~0.6%          | ~1.2%             | Strong short-term accuracy                                |
| Ridge Regression               | ~0.7%          | ~1.2%             | Slight improvement over OLS via regularisation             |
| Lasso Regression               | ~0.6%          | ~1.2%             | Near-identical to Linear for this feature set               |

**Best model: lag-feature regression (Ridge/Lasso)** for short-horizon forecasting on this
dataset — ARIMA's pure autoregressive structure struggles over longer multi-step horizons,
especially for Gold.

- Both Gold and Silver prices are **I(1)** processes — non-stationary at levels, stationary after
  first-order differencing (confirmed by both ADF and KPSS).
- Daily log-returns exhibit **excess kurtosis** (fat tails) and are not normally distributed —
  typical of financial asset returns.
- Gold and Silver returns are **positively correlated**, consistent with their shared role as
  safe-haven assets.
- Silver's ARIMA residuals pass the Ljung-Box white-noise test; Gold's residuals retain some
  autocorrelation, suggesting a more complex specification (e.g., ARMA-GARCH) may be warranted.

---

## Methodology

The notebook follows a 10-stage pipeline:

| Step | Stage                                   | Purpose                                                          |
| ---- | ------------------------------------------ | --------------------------------------------------------------------- |
| 1    | Setup & Data Acquisition                  | Pull Gold/Silver daily closes via `yfinance` (synthetic fallback)        |
| 2    | Exploratory Data Analysis                 | Descriptive stats, price/return plots, rolling mean & std, seasonal decomposition |
| 3    | Stationarity & Statistical Testing        | ADF + KPSS (opposite null hypotheses, used jointly for stronger evidence) |
| 4    | ACF / PACF Analysis                       | Identify candidate AR(p)/MA(q) orders on the differenced series           |
| 5    | Train-Test Split                          | Temporal 85/15 split (~4.25 years train / ~0.75 year test)                |
| 6    | Model 1 — ARIMA                           | `auto_arima` order search (d=1 fixed) + `statsmodels` ARIMA fit             |
| 7    | Model 2 — Lag-Feature Regression           | Linear/Ridge/Lasso on engineered lag features + time index                 |
| 8    | Model Evaluation & Comparison             | RMSE, MAE, MAPE, R² across all models for both assets                      |
| 9    | Residual Diagnostics                      | Time plot, ACF of residuals, Q-Q plot, distribution check                  |
| 10   | Conclusions                               | Consolidated findings, limitations, future work                            |

### Data Acquisition & Preprocessing (`gold_forecasting.py`)

- Ticker: `GC=F` (Gold futures), business-day frequency (`asfreq('B')`), forward/backward-filled
  for missing sessions.
- Feature engineering: daily return (%), intraday volatility (`High − Low`), 50-day and 200-day
  simple moving averages.

### Stationarity Testing

Two tests with **opposite null hypotheses** are used jointly so that agreement between them gives
stronger evidence of the series' true order of integration:

| Test | H₀                          |
| ---- | ------------------------------ |
| ADF  | Series has a unit root (non-stationary) |
| KPSS | Series is stationary                     |

Both confirm **d = 1** is required for both Gold and Silver — consistent with the random-walk
hypothesis for commodity prices.

### Model 1 — ARIMA

Order selection via `pmdarima.auto_arima` (forcing `d=1`, non-seasonal, AIC-optimized, stepwise
search) on the training set only, then refit with `statsmodels.tsa.arima.model.ARIMA` for full
summary statistics and forecasting.

The standalone script (`gold_forecasting.py`) additionally benchmarks **ARMA**, **ARIMA**, and
**ARIMAX** (with `Open`, `High`, `Low`, `Volume` as exogenous regressors) side by side, using a
chronological train/test split (train `< 2025-01-01`, test `≥ 2025-01-01`) and generating a
30-day out-of-sample forecast with 95% confidence bands.

### Model 2 — Lag-Feature Regression

Converts the forecasting problem into a supervised-learning problem:

```python
def make_lag_features(df, lags=5):
    X = pd.DataFrame(index=df.index)
    for lag in range(1, lags + 1):
        X[f"lag_{lag}"] = df["close"].shift(lag)
    X["t"] = np.arange(len(df))   # trend index
    ...
```

Linear, Ridge, and Lasso regressors are trained on these lag + trend features for both assets.

### Evaluation Metrics

| Metric | Formula                    | Interpretation                          |
| -------- | ----------------------------- | -------------------------------------------- |
| RMSE     | √(mean((ŷ−y)²))                 | Penalizes large errors; same unit as price      |
| MAE      | mean(\|ŷ−y\|)                    | Average absolute error; robust to outliers      |
| MAPE     | mean(\|ŷ−y\|/y) × 100             | Scale-free percentage error                     |
| R²       | 1 − SS_res/SS_tot                | Proportion of variance explained                 |

### Residual Diagnostics

Each model's residuals are checked for white-noise behavior via a four-panel diagnostic: time
plot (heteroscedasticity check), ACF of residuals (unexplained autocorrelation), Q-Q plot
(normality), and distribution histogram.

---

## Tech Stack

| Category            | Tools / Libraries                                                       |
| ---------------------- | ------------------------------------------------------------------------- |
| Language                | Python 3                                                                  |
| Data Acquisition        | `yfinance`                                                                |
| Data Handling           | `pandas`, `numpy`                                                        |
| Time Series / Stats     | `statsmodels` (ARIMA, `seasonal_decompose`, ADF, KPSS, Ljung-Box), `pmdarima` (`auto_arima`) |
| Regression Models       | `scikit-learn` (`LinearRegression`, `Ridge`, `Lasso`, metrics)              |
| Visualization           | `matplotlib`, `seaborn`, `plotly` (interactive candlestick charts)           |
| Environment             | Jupyter Notebook                                                          |

---

## Installation

```bash
conda create -n gold-silver-forecast python=3.10
conda activate gold-silver-forecast
pip install yfinance pandas numpy matplotlib seaborn plotly statsmodels pmdarima scikit-learn scipy
```

---

## Usage

### Notebook (full academic pipeline)

Run `Gold_Silver_Prediction_Enhanced.ipynb` top to bottom to reproduce EDA plots, stationarity
tests, ARIMA and regression model fits, evaluation tables, and residual diagnostics for both
Gold and Silver.

---

## Project Structure

```
.
├── dataset 
├── Gold_Silver_Prediction_Enhanced.ipynb   # Full EDA + ARIMA + regression pipeline (Gold & Silver)
└── README.md
```

---

## Limitations

- Evaluation uses a single static train-test split rather than walk-forward/rolling-origin
  validation, which would give a more realistic estimate of out-of-sample MAPE.
- No volatility modeling (GARCH) despite clear evidence of heteroscedasticity in returns,
  particularly during 2020.
- The lag-feature regression models exploit only autoregressive structure and a linear trend —
  no exogenous macro features (USD index, VIX, oil) are included in the notebook (though the
  standalone script's ARIMAX variant does use OHLCV exogenous regressors).
- Gold's ARIMA residuals retain autocorrelation, indicating the linear ARIMA structure is
  insufficiently rich for this series.

---

## Recommended Next Steps

- **Walk-forward validation** — replace the static split with rolling-origin evaluation for a
  more robust accuracy estimate.
- **GARCH extensions** — model time-varying volatility explicitly given confirmed
  heteroscedasticity.
- **Multivariate models** — VAR/VECM to exploit Gold–Silver cointegration.
- **Exogenous features** — incorporate USD index, VIX, oil prices, and other macro indicators.
- **Deep learning** — LSTM/Transformer architectures for longer-horizon forecasting.

---

## License

Price data sourced from Yahoo Finance via the `yfinance` library; all issues identified in the original baseline ARIMA script have been
documented and corrected. Provided for educational/portfolio purposes.
