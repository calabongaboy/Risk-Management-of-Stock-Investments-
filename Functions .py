import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.optimize import minimize
import datetime

from dateutil import parser
from dateutil.tz import tzutc


# For conveniece, we write a separate function that calculates the annualized volatility of log return of stocks.
def volatility(path):
    path = pd.Series(path)

    log_returns = np.log(path / path.shift(1)).dropna()

    # Annualized volatility
    #As we assume the logarithm of the stock price satisfy a geometric Brownian motion, $\sqrt{252}$ is mutiplied.
    path_volatility = log_returns.std() * np.sqrt(252) 
    
    return path_volatility

# This block provides with an optimization function, which identifies the optimal weights needed to be put on each asset to guarantee a minimal 
# volatility of the overall portfolio.
def weight_optimization_1(ticker, stock: pd.DataFrame, n_days, min_weight):

    # Collect log returns for all tickers
    log_returns = pd.DataFrame()
    
    for t in ticker:
        close_prices = stock[t][:-n_days]
        log_ret = np.log(close_prices / close_prices.shift(1)).dropna()
        log_returns[t] = log_ret

    # Ensure all columns align in time
    log_returns = log_returns.dropna()

    n_assets = len(ticker)
    initial_weights = np.array([1/n_assets] * n_assets)  #Setting up initial weights for later optimization process.

    covariance_matrix = 252*(log_returns.cov())  #Computing the (annualized) covariance matrix of the underlying random variables behind the columns.
    
    min_constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights)-1},
               {'type': 'ineq','fun': lambda weights: np.min(weights) - min_weight}] 

    # Define the objective function to minimize portfolio variance
    def portfolio_volatility(weights):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        return portfolio_vol

    result = minimize(portfolio_volatility, initial_weights, constraints=min_constraints)

    return result.x if result.success else None


# This calculates the annualized Shapre Ratio of a portfolio.
def Ann_Sharpe_Ratio(stock_history: pd.DataFrame, weights, risk_free_rate: float):
    """
    Computes the annualized Sharpe Ratio for a portfolio of assets.

    Parameters:
    - stock_history: DataFrame of daily prices (rows = dates, columns = assets)
    - weights: list or array of portfolio weights
    - risk_free_rate: annual risk-free rate (e.g., 0.02 for 2%)

    Returns:
    - Annualized Sharpe Ratio (float)
    """
    weights = np.array(weights)

    # Compute daily log returns for all assets (vectorized)
    log_returns = np.log(stock_history / stock_history.shift(1)).dropna()

    # Compute portfolio daily log returns
    portfolio_log_return = log_returns.dot(weights)

    # Annualized expected return
    ann_return = 252 * portfolio_log_return.mean()

    # Annualized volatility using covariance matrix
    cov_matrix = log_returns.cov()
    ann_cov_matrix = 252 * cov_matrix
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(ann_cov_matrix, weights)))

    # Compute annualized Sharpe Ratio
    sharpe_ratio = (ann_return - risk_free_rate) / portfolio_vol

    return sharpe_ratio




















