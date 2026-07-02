import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(ticker="GC=F", start_date="2010-01-01", end_date="2026-01-01"):
    print("Phase 1: Data Engineering & Preprocessing...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Ensure business days timestamp index (remove weekends/holidays)
    df = df.asfreq('B')
    df.fillna(method='ffill', inplace=True)
    df.index.name = 'Date'
    
    # Feature Engineering
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Intraday_Volatility'] = df['High'] - df['Low']
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df.dropna(inplace=True)
    return df

def perform_eda(df):
    print("Phase 2: Exploratory Data Analysis (EDA) & Visualizations...")
    
    # Candlestick Charts
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    fig.update_layout(title="Gold (GC=F) Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
    fig.show()
    
    # Trend & Seasonality Decompose
    decomposition = seasonal_decompose(df['Close'], model='additive', period=252) # 252 business days a year
    decomposition.plot()
    plt.suptitle("Decomposition of Closing Price (Trend, Seasonality, Residuals)", y=1.02)
    plt.show()
    
    # Volume Analysis
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df['Volume'], color='gray', alpha=0.7)
    plt.title("Trading Volume over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.show()

def perform_statistical_tests(df):
    print("Phase 3: Rigorous Statistical Testing...")
    
    # Augmented Dickey-Fuller (ADF) Test
    result = adfuller(df['Close'].dropna())
    print("\n[ ADF Test ]")
    print(f"ADF Statistic (Raw Close): {result[0]:.4f}, p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print(" -> Data is stationary")
    else:
        print(" -> Data is non-stationary. Differencing required.")
        df['Close_diff'] = df['Close'].diff().dropna()
        result_diff = adfuller(df['Close_diff'].dropna())
        print(f"ADF Statistic (Differenced Close): {result_diff[0]:.4f}, p-value: {result_diff[1]:.4f}")
        
    # ACF & PACF Plots
    fig, axes = plt.subplots(1, 2, figsize=(16,4))
    plot_acf(df['Close'].diff().dropna(), ax=axes[0], title='ACF of Differenced Close')
    plot_pacf(df['Close'].diff().dropna(), ax=axes[1], title='PACF of Differenced Close')
    plt.show()
    
    # Variance Comparison (F-Test proxy using Levene's test for robustness)
    pre_2020 = df[df.index < '2020-01-01']['Daily_Return'].dropna()
    post_2020 = df[df.index >= '2020-01-01']['Daily_Return'].dropna()
    stat, p_levene = stats.levene(pre_2020, post_2020)
    print("\n[ Variance Comparison (Levene Test) ]")
    print(f"Pre-2020 vs Post-2020 Variance -> p-value: {p_levene:.4f}")
    
    # Mean Reversion Check (T-Test)
    t_stat, p_val = stats.ttest_1samp(df['Daily_Return'].dropna(), popmean=0)
    print("\n[ Mean Reversion Check (T-Test) ]")
    print(f"T-Test for Daily Returns Mean=0 -> p-value: {p_val:.4f}")

def build_and_evaluate_models(df):
    print("\nPhase 4: Model Development & Training...")
    # Train / Test Split chronologically
    train = df[df.index < '2025-01-01']
    test = df[df.index >= '2025-01-01']
    
    print(f"Train sample size: {len(train)}, Test sample size: {len(test)}")
    
    # Model 1: ARMA (Using ARIMA(p,0,q))
    print("Training Model 1: ARMA(1, 0, 1) on differenced data...")
    try:
        arma_model = ARIMA(train['Close'].diff().dropna(), order=(1, 0, 1)).fit()
    except Exception as e:
        print("ARMA Error:", e)
    
    # Model 2: ARIMA
    print("Training Model 2: ARIMA(1, 1, 1) on raw closing data...")
    arima_model = ARIMA(train['Close'], order=(1, 1, 1)).fit()
    
    # Model 3: ARIMAX (Core Model)
    print("Training Model 3: ARIMAX(1, 1, 1) with exogenous regressors (Open, High, Low, Volume)...")
    exog_train = train[['Open', 'High', 'Low', 'Volume']]
    exog_test = test[['Open', 'High', 'Low', 'Volume']]
    arimax_model = ARIMA(train['Close'], order=(1, 1, 1), exog=exog_train).fit()
    
    print("\n--- ARIMAX SUMMARY ---\n")
    print(arimax_model.summary())
    
    # Phase 5: Model Diagnostics & Validation
    print("\nPhase 5: Model Diagnostics & Validation...")
    
    # Residual Analysis & QQ Plot
    arimax_model.plot_diagnostics(figsize=(12, 8))
    plt.suptitle("ARIMAX Residual Diagnostics", y=1.02)
    plt.show()
    
    # Ljung-Box Test
    lb_test = acorr_ljungbox(arimax_model.resid, lags=[10], return_df=True)
    print("\n[ Ljung-Box Test ]")
    print(lb_test)
    
    # Generate Predictions for Test Set comparison
    print("\nGenerating out-of-sample predictions for test set comparison...")
    arma_preds_diff = arma_model.forecast(steps=len(test))
    arma_preds = train['Close'].iloc[-1] + arma_preds_diff.cumsum()
    
    arima_preds = arima_model.forecast(steps=len(test))
    arimax_preds = arimax_model.forecast(steps=len(test), exog=exog_test)
    
    # Performance Metrics Comparison
    print("\n[ Performance Metrics Comparison (Test Data) ]")
    print(f"{'Model':<10} | {'RMSE':<10} | {'MAE':<10} | {'MAPE':<10}")
    print("-" * 47)
    
    for name, m_preds in zip(["ARMA", "ARIMA", "ARIMAX"], [arma_preds, arima_preds, arimax_preds]):
        rmse = np.sqrt(mean_squared_error(test['Close'], m_preds))
        mae = mean_absolute_error(test['Close'], m_preds)
        mape = mean_absolute_percentage_error(test['Close'], m_preds)
        print(f"{name:<10} | {rmse:<10.4f} | {mae:<10.4f} | {mape:<10.4f}")
        
    # Plot Comparison
    plt.figure(figsize=(14, 7))
    plt.plot(train.index[-100:], train['Close'].iloc[-100:], label='Train (Last 100 days)', color='black')
    plt.plot(test.index, test['Close'], label='Actual (Test)', color='blue', linewidth=2)
    plt.plot(test.index, arma_preds, label='ARMA Forecast', color='orange', linestyle='--')
    plt.plot(test.index, arima_preds, label='ARIMA Forecast', color='green', linestyle='--')
    plt.plot(test.index, arimax_preds, label='ARIMAX Forecast', color='red', linewidth=2)
    plt.title("Model Comparison: ARMA vs ARIMA vs ARIMAX on Test Set")
    plt.xlabel("Date")
    plt.ylabel("Gold Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return arimax_model, test, arimax_preds

def forecast_future(arimax_model, last_exog, steps=30):
    print(f"\nPhase 6: Forecasting & Final Output (Next {steps} periods)...")
    
    # Create simple extended exogenous variables (naive assumption: repeats last known row)
    future_exog = pd.DataFrame([last_exog]*steps, columns=['Open', 'High', 'Low', 'Volume'])
    
    forecast_res = arimax_model.get_forecast(steps=steps, exog=future_exog)
    pred_mean = forecast_res.predicted_mean
    pred_ci = forecast_res.conf_int(alpha=0.05)
    
    plt.figure(figsize=(12, 6))
    plt.plot(pred_mean.index, pred_mean, color='blue', label='Forecast')
    plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='red', alpha=0.15, label='95% Confidence Interval')
    plt.title("Out-of-Sample ARIMAX Forecast with 95% Confidence Bands")
    plt.xlabel("Periods Ahead")
    plt.ylabel("Gold Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Ensure data starts well before train/test split requirement and features computation
    df = load_and_preprocess_data(start_date="2010-01-01", end_date="2026-04-10")
    
    perform_eda(df)
    perform_statistical_tests(df)
    
    arimax_model, test_data, preds = build_and_evaluate_models(df)
    
    # Forecast out-of-sample (e.g. next 30 days)
    last_known_exog = test_data[['Open', 'High', 'Low', 'Volume']].iloc[-1].values
    forecast_future(arimax_model, last_known_exog, steps=30)
