import numpy as np
from scipy.optimize import minimize
from typing import Callable
import yfinance as yf
from datetime import datetime, timedelta

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
    mean_return = np.mean(portfolio_returns) * annualization_factor
    volatility = np.std(portfolio_returns) * np.sqrt(annualization_factor)
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    total_return = cumulative_returns[-1]
    
    return {
        'Annualized Return': mean_return,
        'Annualized Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Total Return': total_return,
    }

# Evaluate an estimator on return data
def evaluate(returns: np.ndarray, estimator: Callable[[np.ndarray], np.ndarray]) -> dict:
    backtest_results = backtest_portfolio(returns, estimator)
    performance_metrics = calculate_portfolio_performance(backtest_results)
    return performance_metrics

if __name__ == "__main__":
   
    # Some tickers
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'V']
    
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
        rets = prices.pct_change().dropna().values
        returns.append(rets)
    returns = np.column_stack(returns)

    # Define covariance estimators
    def sample_covariance (returns):
        return np.cov(returns.T)
    
    def identity_shrinkage(returns):
        sample_cov = sample_covariance(returns)
        target = np.eye(len(sample_cov)) * np.mean(np.diag(sample_cov))
        alpha = 0.25
        return alpha * sample_cov + (1 - alpha) * target
    
    # Compare strategies
    estimators = {
        'Sample Covariance': sample_covariance,
        'Identity Shrinkage': identity_shrinkage
    }
    
    for (name, estimator) in estimators.items():
        print(f"Evaluating estimator: {name}")
        performance = evaluate(returns, estimator)
        for metric , value in performance.items():
            print(f"  {metric}: {value:.4f}")
        print()
        