# Factor Risk Model — Barra/Axioma Style

A from-scratch implementation of a multi-factor equity risk model following the
methodology used by commercial platforms like Barra (MSCI) and Axioma.

Most public “factor model” repos stop at simple return sorting or backtests.  
This project goes further by implementing the **risk-engine internals** that practitioners actually use:

- systematic vs idiosyncratic risk decomposition  
- exposure matrix engineering under noisy/free-data assumptions  
- factor risk contribution analysis  
- fallback logic for missing data (e.g., market-cap proxying)

It is intentionally designed as a **mini institutional-grade framework**: compact enough to understand end-to-end, yet structured enough to extend toward real production workflows.


## Project Structure
```text
factor_risk_model/
├─ app/
│  └─ app.py                # Interactive dashboard / visualization layer
├─ src/
│  ├─ factors.py            # Factor definitions and exposure construction
│  └─ risk.py               # Covariance, specific risk, and risk decomposition logic
├─ test.py                  # Lightweight validation / sanity checks
├─ requirements.txt         # Dependencies
└─ readme.md

## Factor Definitions
| Factor | Definition | Barra Equivalent |
|--------|-----------|-----------------|
| Momentum | 12-1 month return | Barra MOM factor |
| Size | Log(avg price) | Barra SIZE |
| Volatility | 60-day realized vol | Barra RESVOL |
| Value | Inverse trailing return | Barra BTOP proxy |
| Beta | OLS market beta | Barra BETA |

## Methodology
At each rebalancing date \(t\), the model estimates factor premia using a cross-sectional regression of asset returns on contemporaneous factor exposures. Let \(r_t \in \mathbb{R}^N\) be the vector of asset returns for \(N\) securities, and let \(X_t \in \mathbb{R}^{N \times K}\) be the exposure matrix for \(K\) factors (e.g., size and other style factors). The regression is:
\[
r_t = X_t f_t + \epsilon_t
\]
where \(f_t \in \mathbb{R}^K\) are factor returns and \(\epsilon_t\) are specific (idiosyncratic) returns. Repeating this through time gives a factor return history \(\{f_t\}\) and residual history \(\{\epsilon_t\}\), which are then used to estimate systematic and specific risk components.

Risk decomposition follows the standard linear factor model:
\[
\Sigma_t = X_t \Sigma_f X_t^\top + D_t
\]
where \(\Sigma_f = \mathrm{Cov}(f_t)\) is the factor covariance matrix and \(D_t\) is a diagonal matrix of specific variances estimated from residuals. For a portfolio with weights \(w\), total variance is \(w^\top \Sigma_t w\), which can be split into factor-driven variance \(w^\top X_t \Sigma_f X_t^\top w\) and specific variance \(w^\top D_t w\). This decomposition enables both top-level risk forecasting and granular attribution by factor.

## How to Run
### Notebook
pip install -r requirements.txt
jupyter notebook notebook/factor_model.ipynb

### Dashboard
streamlit run app/app.py