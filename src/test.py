from factors import (
    construct_factors,
    standardize_factors
)


from risk import (
    estimate_factor_covariance,
    compute_idiosyncratic_variance,
    compute_portfolio_exposures,
    decompose_portfolio_risk,
    equal_weight_portfolio
)


import pandas as pd
import numpy as np




file_path = r"C:\users\chinmae\projects_latest\CSV_files\all_time_niftys.csv"
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['Date'])
# only take last 2 years data using iloc
data = data.iloc[-1500:]
data.set_index('Date', inplace=True)
prices = data
# price_df = price_df.fillna(0)
# returns_df = data.pct_change()

prices = prices.replace([np.inf, -np.inf], np.nan)

prices = prices.dropna(axis=1)

returns = np.log(
    prices / prices.shift(1)
).dropna()

returns = np.log(prices / prices.shift(1)).dropna()


factor_df = construct_factors(
    returns=returns,
    prices=prices
)

factor_exposures = standardize_factors(factor_df)


factor_returns, factor_cov, residuals = (
    estimate_factor_covariance(
        factor_exposures=factor_exposures,
        returns=returns
    )
)


idio_var = compute_idiosyncratic_variance(
    residuals
)


weights = equal_weight_portfolio(
    tickers=returns.columns.tolist()
)


portfolio_exposures = compute_portfolio_exposures(
    weights=weights,
    factor_exposures=factor_exposures
)


risk_report = decompose_portfolio_risk(
    weights=weights,
    factor_exposures=factor_exposures,
    factor_cov=factor_cov,
    idio_var=idio_var
)


print("\nFACTOR EXPOSURES")
print(portfolio_exposures)

print("\nFACTOR COVARIANCE MATRIX")
print(factor_cov)

print("\nIDIOSYNCRATIC VARIANCE")
print(idio_var.head())

print("\nPORTFOLIO RISK REPORT")
print(risk_report)