# Factor Risk Model — Barra/Axioma Style

A from-scratch implementation of a multi-factor equity risk model following the
methodology used by commercial platforms like Barra (MSCI) and Axioma.

## Project Structure
```text
barra_risk_model/
├── data/                  # auto-created; stores cached CSVs
├── fetch_data.py          # Step 1: download & cache price data
├── requirements.txt       # Python dependencies
├── README.md              # this file
├── <notebook>.ipynb       # TODO: add later
└── <dashboard>.py         # TODO: add later



## Factor Definitions
| Factor | Definition | Barra Equivalent |
|--------|-----------|-----------------|
| Momentum | 12-1 month return | Barra MOM factor |
| Size | Log(avg price) | Barra SIZE |
| Volatility | 60-day realized vol | Barra RESVOL |
| Value | Inverse trailing return | Barra BTOP proxy |
| Beta | OLS market beta | Barra BETA |

## How to Run
### Notebook
pip install -r requirements.txt
jupyter notebook notebook/factor_model.ipynb

### Dashboard
streamlit run app/app.py

## Methodology
[2 paragraphs on the math — the cross-sectional regression and risk decomposition]

## Limitations vs Commercial Platforms
[Honest list: no fundamental data, smaller universe, no shrinkage estimation, etc.]