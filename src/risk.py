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


# def decompose_portfolio_risk(
#     weights,
#     factor_exposures,
#     factor_cov,
#     idio_var):

#     w = weights.values
#     B = factor_exposures.values
#     F = factor_cov.values
#     D = np.diag(idio_var.values)

#     systematic_cov = B @ F @ B.T
#     factor_var = w.T @ systematic_cov @ w

#     # x = B.T @ w
#     # factor_var = float(x.T @ F @ x)
#     # if factor_var < 0 and abs(factor_var) < 1e-12:
#     #     factor_var = 0.0
    
#     idio_var_port = w.T @ D @ w
#     total_var = factor_var + idio_var_port
#     total_vol = np.sqrt(total_var)
#     factor_vol = np.sqrt(factor_var)
#     idio_vol = np.sqrt(idio_var_port)
#     factor_pct = factor_var / total_var
#     idio_pct = idio_var_port / total_var
#     factor_contributions = {}

#     for i, factor in enumerate(factor_cov.columns):

#         B_k = B[:, i].reshape(-1, 1)
#         F_kk = F[i, i]
#         contribution = (w.T @(B_k * F_kk @ B_k.T) @ w)
#         factor_contributions[factor] = contribution

#     return {
#         "total_vol": total_vol,
#         "factor_vol": factor_vol,
#         "idio_vol": idio_vol,
#         "factor_pct": factor_pct,
#         "idio_pct": idio_pct,
#         "factor_contributions": factor_contributions
#     }

def decompose_portfolio_risk(
    weights,
    factor_exposures,
    factor_cov,
    idio_var,
):
    w = weights.values.astype(float)
    B = factor_exposures.values.astype(float)
    F = factor_cov.values.astype(float)

    # Align idio var to exposures index if needed
    idio = idio_var.reindex(factor_exposures.index).astype(float).fillna(0.0)
    D = np.diag(idio.values)

    # Stabilize factor covariance numerically
    F = 0.5 * (F + F.T)

    # Portfolio-level pieces
    x = B.T @ w                      # portfolio factor exposures (K,)
    factor_var = float(x.T @ F @ x)  # numerically stabler than w' B F B' w
    if factor_var < 0 and abs(factor_var) < 1e-12:
        factor_var = 0.0

    idio_var_port = float(w.T @ D @ w)
    total_var = factor_var + idio_var_port
    if total_var < 0 and abs(total_var) < 1e-12:
        total_var = 0.0

    total_vol = float(np.sqrt(total_var))
    factor_vol = float(np.sqrt(factor_var))
    idio_vol = float(np.sqrt(idio_var_port))

    factor_pct = (factor_var / total_var) if total_var > 0 else 0.0
    idio_pct = (idio_var_port / total_var) if total_var > 0 else 0.0

    # Euler factor variance contributions (add up to factor_var)
    marginal = F @ x
    contrib = x * marginal
    factor_contributions = pd.Series(contrib, index=factor_cov.columns).to_dict()

    return {
        "total_vol": total_vol,
        "factor_vol": factor_vol,
        "idio_vol": idio_vol,
        "factor_var": factor_var,
        "idio_var": idio_var_port,
        "total_var": total_var,
        "factor_pct": factor_pct,
        "idio_pct": idio_pct,
        "factor_contributions": factor_contributions,
    }

# def equal_weight_portfolio(tickers):
#     n = len(tickers)
#     weights = np.repeat(1 / n, n)
#     return pd.Series(weights,index=tickers)

def equal_weight_portfolio(tickers):
    n = len(tickers)
    weights = np.repeat(1 / n, n)
    return pd.Series(weights, index=tickers)


def compute_stock_mrc(
    weights: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    idio_var: pd.Series,
) -> pd.Series:
    """
    Euler decomposition of portfolio volatility by stock.

    MRC_i  =  w_i * (Σ @ w)_i / σ_p
    where Σ = B F B' + D  (full N×N covariance matrix).

    By Euler's theorem the MRC values sum exactly to σ_p (total portfolio vol).
    Dividing by σ_p and multiplying by 100 gives each stock's % contribution
    to total portfolio volatility.
    """
    # Cast and align
    w = (
        weights
        .reindex(factor_exposures.index)
        .astype(float)
        .fillna(0.0)
        .values
    )                                                       # (N,)
    B = factor_exposures.values.astype(float)              # (N, K)
    F = factor_cov.values.astype(float)                    # (K, K)
    F = 0.5 * (F + F.T)                                    # symmetrise numerically

    idio = (
        idio_var
        .reindex(factor_exposures.index)
        .astype(float)
        .fillna(0.0)
    )
    D = np.diag(idio.values)                               # (N, N) diagonal

    # Full covariance matrix (systematic + idiosyncratic)
    sigma_matrix = B @ F @ B.T + D                         # (N, N)

    # Marginal contribution vector
    sigma_w = sigma_matrix @ w                             # (N,) = Σ @ w
    total_var = float(w.T @ sigma_w)                       # scalar σ²_p
    total_vol = float(np.sqrt(max(total_var, 0.0)))        # scalar σ_p

    if total_vol < 1e-16:
        return pd.Series(0.0, index=factor_exposures.index)

    # MRC in vol units: w_i * (Σw)_i / σ_p  -> sum = σ_p
    mrc = w * sigma_w / total_vol
    return pd.Series(mrc, index=factor_exposures.index)

