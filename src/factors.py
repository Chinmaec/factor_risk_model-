import pandas as pd 
import numpy as np 
import os 
from pathlib import Path

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "sample_data" / "sample_data.csv"
file_path = Path(os.getenv("PRICES_CSV_PATH", str(DEFAULT_CSV)))
prices = pd.read_csv(file_path)

prices['Date'] = pd.to_datetime(prices['Date'])
prices.set_index('Date', inplace=True)

returns = np.log(prices / prices.shift(1)).dropna()

def construct_momentum(prices,long_window=252,short_window=21):
    if len(prices) < long_window:
        raise ValueError(f"Need at least {long_window} rows for momentum.")
    past_price = prices.iloc[-long_window]
    recent_price = prices.iloc[-short_window]
    momentum = recent_price / past_price - 1
    return momentum

# Market Cap (Barra-style, when shares outstanding is available)
def construct_size_with_shares(prices, shares_outstanding):
    market_cap = prices * shares_outstanding
    market_cap = market_cap.replace(0, np.nan)
    return np.log(market_cap)

# Market Cap proxy (fallback when shares outstanding is unavailable/unreliable)
def construct_size_proxy(prices: pd.DataFrame | pd.Series) -> pd.Series | float:
    size_proxy = prices.mean()
    size_proxy = size_proxy.replace(0, np.nan) if hasattr(size_proxy, "replace") else size_proxy
    return np.log(size_proxy)


def construct_size(prices, shares_outstanding=None, use_barra=False):
    if use_barra and shares_outstanding is not None:
        return construct_size_with_shares(prices, shares_outstanding)
    return construct_size_proxy(prices)

def construct_volatility(returns, window=60):
    volatility = returns.tail(window).std() * np.sqrt(252)
    return volatility

def construct_value(prices, lookback=252):
    if len(prices) < lookback:
        raise ValueError(f"Need at least {lookback} rows for value factor.")
    past_price = prices.iloc[-lookback]
    current_price = prices.iloc[-1]
    value = -(current_price / past_price - 1)
    return value


def construct_beta(returns, window=252):

    trailing_returns = returns.tail(window)

    market_returns = trailing_returns.mean(axis=1)

    betas = {}

    for stock in trailing_returns.columns:

        stock_returns = trailing_returns[stock]

        beta = (
            np.cov(stock_returns, market_returns)[0, 1]
            /
            np.var(market_returns)
        )

        betas[stock] = beta

    return pd.Series(betas)

def construct_factors(returns, prices):

    momentum = construct_momentum(prices)

    size = construct_size(prices)

    volatility = construct_volatility(returns)

    value = construct_value(prices)

    beta = construct_beta(returns)

    factor_df = pd.DataFrame({
        "MOMENTUM": momentum,
        "SIZE": size,
        "VOLATILITY": volatility,
        "VALUE": value,
        "BETA": beta
    })

    return factor_df

def standardize_factors(factor_df):

    standardized = (
        factor_df - factor_df.mean()
    ) / factor_df.std()

    standardized = standardized.clip(-3, 3)

    return standardized


factor_df = construct_factors(returns, prices)

X = standardize_factors(factor_df)
print(X.shape)