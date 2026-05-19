import pandas as pd
import numpy as np


def estimate_factor_covariance(factor_exposures,returns):

    factor_returns_list = []
    residuals_df = pd.DataFrame(
        index=returns.index,
        columns=returns.columns
    )
    B = factor_exposures.values

    for date in returns.index:
        r_t = returns.loc[date].values
        factor_return_t, _, _, _ = np.linalg.lstsq(B,r_t,rcond=None)
        predicted_returns = B @ factor_return_t
        residuals = r_t - predicted_returns
        factor_returns_list.append(factor_return_t)
        residuals_df.loc[date] = residuals

    factor_returns = pd.DataFrame(
        factor_returns_list,
        index=returns.index,
        columns=factor_exposures.columns
    )

    factor_cov = factor_returns.cov() * 252
    return factor_returns, factor_cov, residuals_df


def compute_idiosyncratic_variance(residuals):
    idio_var = residuals.var() * 252
    return idio_var


def compute_portfolio_exposures(weights,factor_exposures):
    portfolio_exposures = (factor_exposures.T @ weights)
    return portfolio_exposures


def decompose_portfolio_risk(
    weights,
    factor_exposures,
    factor_cov,
    idio_var):

    w = weights.values
    B = factor_exposures.values
    F = factor_cov.values
    D = np.diag(idio_var.values)

    systematic_cov = B @ F @ B.T
    factor_var = w.T @ systematic_cov @ w

    # x = B.T @ w
    # factor_var = float(x.T @ F @ x)
    # if factor_var < 0 and abs(factor_var) < 1e-12:
    #     factor_var = 0.0
    
    idio_var_port = w.T @ D @ w
    total_var = factor_var + idio_var_port
    total_vol = np.sqrt(total_var)
    factor_vol = np.sqrt(factor_var)
    idio_vol = np.sqrt(idio_var_port)
    factor_pct = factor_var / total_var
    idio_pct = idio_var_port / total_var
    factor_contributions = {}

    for i, factor in enumerate(factor_cov.columns):

        B_k = B[:, i].reshape(-1, 1)
        F_kk = F[i, i]
        contribution = (w.T @(B_k * F_kk @ B_k.T) @ w)
        factor_contributions[factor] = contribution

    return {
        "total_vol": total_vol,
        "factor_vol": factor_vol,
        "idio_vol": idio_vol,
        "factor_pct": factor_pct,
        "idio_pct": idio_pct,
        "factor_contributions": factor_contributions
    }


def equal_weight_portfolio(tickers):
    n = len(tickers)
    weights = np.repeat(1 / n, n)
    return pd.Series(weights,index=tickers)



