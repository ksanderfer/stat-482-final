import os
import torch
import pickle
import numpy as np
import yfinance as yf
from typing import Callable
from scipy.optimize import minimize
from datetime import datetime, timedelta
from neural_network import predict_covariance, CovarianceNet
from util import sample_covariance, ledoit_wolf_shrinkage
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from regression import cov_estimator
import matplotlib.pyplot as plt

# Minimum variance portfolio weights
def minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    
    n = len(cov_matrix)
    
    # Objective: portfolio variance
    def objective(w):
        return w @ cov_matrix @ w
    
    # Solve optimization
    result = minimize(
        objective,
        x0=np.ones(n) / n,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, 
        bounds=tuple((0, 1) for _ in range(n)) # long only (can't short)
    )

    # Return weights
    return result.x

# Backtest portfolio strategy using covariance estimator
def backtest_portfolio (
        returns: np.ndarray, 
        estimator: Callable[[np.ndarray], np.ndarray],
        window_size: int = 252, # lookback window of 1 year
        rebalance_freq: int = 21 # rebalance monthly
        ) -> np.ndarray:

    days, _ = returns.shape
    portfolio_returns = []
    
    # Rolling window backtest
    for t in range(window_size, days, rebalance_freq):
        
        # Get estimation window
        train_returns = returns[t - window_size:t]
        
        # Estimate covariance matrix
        cov_matrix = estimator(train_returns)
        
        # Optimize portfolio weights
        weights = minimum_variance_portfolio(cov_matrix)
        
        # Compute returns until next rebalancing
        end_period = min(t + rebalance_freq, days)
        period_returns = returns[t:end_period]
        
        # Compute portfolio returns for holding period
        port_rets = period_returns @ weights
        portfolio_returns.extend(port_rets)
    
    return np.array(portfolio_returns)

# Calculate portfolio performance from returns
def calculate_portfolio_performance(portfolio_returns: np.ndarray) -> dict:
    
    annualization_factor = 252  # Trading days in a year

    total_log_return = np.sum(portfolio_returns)
    annualized_log_return = total_log_return * (annualization_factor / len(portfolio_returns))
    volatility = np.std(portfolio_returns) * np.sqrt(annualization_factor)
    sharpe_ratio = annualized_log_return / volatility if volatility > 0 else 0
    
    return {
        'Annualized Return': np.exp(annualized_log_return) - 1,
        'Total Return': np.exp(total_log_return) - 1,
        'Annualized Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
    }

# Evaluate an estimator on return data
def evaluate(returns: np.ndarray, estimator: Callable[[np.ndarray], np.ndarray]) -> dict:
    backtest_results = backtest_portfolio(returns, estimator)
    performance_metrics = calculate_portfolio_performance(backtest_results)
    return performance_metrics

if __name__ == "__main__":
   
    # Some tickers
    tickers = ['JNJ', 'PG', 'KO', 'MSFT', 'AAPL', 'ADBE', 'NVDA', 'TSLA', 'XLE', 'GLD']
    
    # Fetch historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if data is None:
        raise Exception("Failed to fetch data from yfinance.")
    
    # Prepare returns data
    returns = []
    for ticker in tickers:
        prices = data['Close', ticker]
        rets = np.diff(np.log(np.array(prices.values)))
        returns.append(rets)
    returns = np.column_stack(returns)

    # Define covariance estimators
    model = CovarianceNet(n_assets=10)
    model.load_state_dict(torch.load('models/neural_network.pt'))
    def neural_network (returns):
        return predict_covariance(model, returns.T)

    with open('models/random_forest.pkl', 'rb') as f:
        rf = pickle.load(f)
    if not isinstance(rf, RandomForestRegressor):
            raise TypeError("Loaded object is not a RandomForestRegressor.")
    def rf_model (returns):
        return cov_estimator(rf, returns)
    
    with open('models/linear_regression.pkl', 'rb') as f:
        lr = pickle.load(f)
    if not isinstance(lr, LinearRegression):
            raise TypeError("Loaded object is not a LinearRegression.")
    def lr_model (returns):
        return cov_estimator(lr, returns)
    
    # Compare strategies
    estimators = {
        'Sample Covariance': sample_covariance,
        'Ledoit-Wolf Shrinkage': ledoit_wolf_shrinkage,
        'Random Forest': rf_model,
        'Neural Network': neural_network,
        'Linear Regression': lr_model
    }
    
    performances = []
    for (name, estimator) in estimators.items():
        print(f"Evaluating estimator: {name}")
        performance = evaluate(returns, estimator)
        for metric , value in performance.items():
            print(f"  {metric}: {value:.4f}")
        print()
        performances.append((name, performance))

    # Plot results
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.style.use('seaborn-v0_8')

    labels = [name for name, _ in performances]
    annualized_returns = [perf['Annualized Return'] for _, perf in performances]
    plt.bar(labels, annualized_returns)
    plt.ylabel('Annualized Return')
    plt.title('Portfolio Returns by Covariance Estimator')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/portfolio_returns.png')

    annualized_volatility = [perf['Annualized Volatility'] for _, perf in performances]
    plt.clf()
    plt.bar(labels, annualized_volatility)
    plt.ylabel('Variance')
    plt.title('Portfolio Volatility by Covariance Estimator')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/portfolio_volatility.png')

    sharpe_ratio = [perf['Sharpe Ratio'] for _, perf in performances]
    plt.clf()
    plt.bar(labels, sharpe_ratio)
    plt.ylabel('Sharpe Ratio')
    plt.title('Portfolio Sharpe Ratio by Covariance Estimator')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/portfolio_sharpe_ratio.png')
        
        