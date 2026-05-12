import pandas as pd 
import numpy as np 
import os 

file_path = os.getenv("PRICES_CSV_PATH", "factor_risk_model/data/prices.csv")
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

# Market Cap(proxy)

# real barra uses Market cap = price * shares outstanding
# but since we are using free data here it may not reliable give shares outstanding 
def construct_size(prices):
    size_proxy = prices.mean()
    size = np.log(size_proxy)
    return size 

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