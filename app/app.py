import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import factors as factor_engine
from src import risk as risk_engine


st.set_page_config(page_title="Factor Risk Model", layout="wide")
st.title("Factor Risk Model")
st.caption("Barra/Axioma-Style Risk Attribution")


@st.cache_data(show_spinner=False)
def load_model_inputs():
    prices = factor_engine.prices.copy()
    returns = factor_engine.returns.copy()

    factor_raw = factor_engine.construct_factors(returns=returns, prices=prices)
    factor_exposures = factor_engine.standardize_factors(factor_raw)

    factor_exposures = factor_exposures.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    common_tickers = [c for c in returns.columns if c in factor_exposures.index]
    if not common_tickers:
        raise ValueError("No overlapping tickers between returns and factor exposures.")

    prices = prices[common_tickers]
    returns = returns[common_tickers]
    factor_raw = factor_raw.loc[common_tickers]
    factor_exposures = factor_exposures.loc[common_tickers]

    return prices, returns, factor_raw, factor_exposures


@st.cache_data(show_spinner=False)
def load_risk_objects(returns: pd.DataFrame, factor_exposures: pd.DataFrame):
    _, factor_cov, residuals = risk_engine.estimate_factor_covariance(
        factor_exposures=factor_exposures,
        returns=returns,
    )
    residuals = residuals.apply(pd.to_numeric, errors="coerce")
    idio_var = risk_engine.compute_idiosyncratic_variance(residuals)
    idio_var = idio_var.reindex(factor_exposures.index).fillna(idio_var.median())
    return factor_cov, idio_var


def build_weights(
    method: str,
    tickers: list[str],
    factor_raw: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    custom_tickers: list[str],
) -> pd.Series:
    if method == "Equal Weight":
        return risk_engine.equal_weight_portfolio(tickers).reindex(tickers).fillna(0.0)

    if method == "Momentum Tilt":
        scores = factor_exposures.loc[tickers, "MOMENTUM"].copy()
        scores = scores - scores.min() + 1e-8
        scores = scores.fillna(0.0)
        if scores.sum() <= 0:
            return risk_engine.equal_weight_portfolio(tickers).reindex(tickers).fillna(0.0)
        return scores / scores.sum()

    if method == "Low Volatility Tilt":
        inv_vol = 1 / factor_raw.loc[tickers, "VOLATILITY"].replace(0, np.nan)
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if inv_vol.sum() <= 0:
            return risk_engine.equal_weight_portfolio(tickers).reindex(tickers).fillna(0.0)
        return inv_vol / inv_vol.sum()

    selected = [t for t in custom_tickers if t in tickers]
    if not selected:
        raise ValueError("Custom portfolio requires at least one valid stock.")
    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[selected] = 1.0 / len(selected)
    return weights


def make_download_payload(
    weights: pd.Series,
    stock_exposures: pd.DataFrame,
    portfolio_exposures: pd.Series,
    factor_contrib_pct: pd.Series,
    risk_report: dict,
) -> bytes:
    factor_share = max(float(risk_report["factor_pct"]), 0.0)
    idio_share = max(float(risk_report["idio_pct"]), 0.0)
    total_share = factor_share + idio_share
    if total_share > 0:
        factor_share = factor_share / total_share
        idio_share = idio_share / total_share

    summary = pd.DataFrame(
        {
            "Metric": [
                "Total Vol (%)",
                "Factor Risk (%)",
                "Idiosyncratic Risk (%)",
                "Factor Share of Variance (%)",
                "Idio Share of Variance (%)",
            ],
            "Value": [
                100 * float(risk_report["total_vol"]),
                100 * float(risk_report["factor_vol"]),
                100 * float(risk_report["idio_vol"]),
                100 * factor_share,
                100 * idio_share,
            ],
        }
    )

    exposure_table = portfolio_exposures.rename("Exposure").reset_index()
    exposure_table.columns = ["Factor", "Exposure"]

    contrib_table = factor_contrib_pct.rename("Contribution (% Total Variance)").reset_index()
    contrib_table.columns = ["Factor", "Contribution (% Total Variance)"]

    holdings_table = stock_exposures.copy()
    holdings_table.insert(0, "Weight", weights.loc[stock_exposures.index].values)
    holdings_table = holdings_table.reset_index().rename(columns={"index": "Ticker"})

    output = io.StringIO()
    output.write("Summary\n")
    summary.to_csv(output, index=False)

    output.write("\nPortfolio Factor Exposures\n")
    exposure_table.to_csv(output, index=False)

    output.write("\nFactor Contributions\n")
    contrib_table.to_csv(output, index=False)

    output.write("\nHoldings Factor Exposures\n")
    holdings_table.to_csv(output, index=False)

    return output.getvalue().encode("utf-8")


try:
    prices, returns, factor_raw, factor_exposures = load_model_inputs()
    factor_cov, idio_var = load_risk_objects(returns, factor_exposures)
except Exception as exc:
    st.error(f"Model load failed: {exc}")
    st.stop()

tickers = factor_exposures.index.tolist()

with st.sidebar:
    st.title("Factor Risk Model")
    st.caption("Barra/Axioma-Style Risk Attribution")

    construction = st.selectbox(
        "Select Portfolio Construction",
        options=["Equal Weight", "Momentum Tilt", "Low Volatility Tilt", "Custom"],
        index=0,
    )

    custom_tickers = []
    if construction == "Custom":
        custom_tickers = st.multiselect(
            "Select stocks",
            options=tickers,
            default=tickers[: min(10, len(tickers))],
        )

    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if run_clicked:
    try:
        weights = build_weights(
            method=construction,
            tickers=tickers,
            factor_raw=factor_raw,
            factor_exposures=factor_exposures,
            custom_tickers=custom_tickers,
        )

        portfolio_exposures = risk_engine.compute_portfolio_exposures(
            weights=weights,
            factor_exposures=factor_exposures,
        )

        risk_report = risk_engine.decompose_portfolio_risk(
            weights=weights,
            factor_exposures=factor_exposures,
            factor_cov=factor_cov,
            idio_var=idio_var,
        )

        total_var = float(risk_report["total_vol"]) ** 2
        factor_contrib = pd.Series(risk_report["factor_contributions"], dtype=float).sort_values(
            ascending=False
        )

        if total_var > 0:
            factor_contrib_pct = 100.0 * factor_contrib / total_var
        else:
            factor_contrib_pct = factor_contrib * 0.0

        selected_tickers = weights[weights > 0].index.tolist()
        stock_exposures = factor_exposures.loc[selected_tickers].copy()

        st.session_state.analysis = {
            "weights": weights,
            "portfolio_exposures": portfolio_exposures,
            "risk_report": risk_report,
            "factor_contrib_pct": factor_contrib_pct,
            "stock_exposures": stock_exposures,
        }

    except Exception as exc:
        st.session_state.analysis = None
        st.error(f"Analysis failed: {exc}")

if st.session_state.analysis is None:
    st.info("Choose portfolio construction in the sidebar and click Run Analysis.")
    st.stop()

a = st.session_state.analysis
risk_report = a["risk_report"]

metric_cols = st.columns(3)
metric_cols[0].metric("Total Vol", f"{100 * float(risk_report['total_vol']):.2f}%")
metric_cols[1].metric("Factor Risk", f"{100 * float(risk_report['factor_vol']):.2f}%")
metric_cols[2].metric("Idio Risk", f"{100 * float(risk_report['idio_vol']):.2f}%")

st.info(
    "Factor Risk is the portion of portfolio variance explained by common systematic factors, "
    "equivalent to what Barra reports as systematic risk in portfolio analytics."
)

left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.subheader("Factor Contributions")
    st.info("Contribution of each factor to total portfolio variance.")

    factor_contrib_df = a["factor_contrib_pct"].reset_index()
    factor_contrib_df.columns = ["Factor", "ContributionPct"]

    fig_contrib = px.bar(
        factor_contrib_df,
        x="Factor",
        y="ContributionPct",
        color="ContributionPct",
        color_continuous_scale="Blues",
        text=factor_contrib_df["ContributionPct"].map(lambda v: f"{v:.1f}%"),
    )
    fig_contrib.update_layout(
        xaxis_title="",
        yaxis_title="% of total variance",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_contrib.update_traces(textposition="outside")
    st.plotly_chart(fig_contrib, use_container_width=True)

    st.subheader("Portfolio Factor Exposures")
    st.info("Net standardized exposure of the portfolio to each factor.")

    pexp_df = a["portfolio_exposures"].reset_index()
    pexp_df.columns = ["Factor", "Exposure"]
    max_abs = max(0.01, float(np.abs(pexp_df["Exposure"]).max()))

    fig_pexp = px.bar(
        pexp_df,
        x="Factor",
        y="Exposure",
        color="Exposure",
        color_continuous_scale="RdBu",
        range_color=[-max_abs, max_abs],
        text=pexp_df["Exposure"].map(lambda v: f"{v:.2f}"),
    )
    fig_pexp.update_layout(
        xaxis_title="",
        yaxis_title="Standardized Exposure",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_pexp.update_traces(textposition="outside")
    st.plotly_chart(fig_pexp, use_container_width=True)

with right_col:
    st.subheader("Factor Exposure Heatmap")
    st.info("Stock-level standardized exposures for holdings in the selected portfolio.")

    heatmap_data = a["stock_exposures"].copy()
    heatmap_data.index.name = "Ticker"

    # fig_heatmap = px.imshow(
    #     heatmap_data,
    #     aspect="auto",
    #     color_continuous_scale="RdBu",
    #     zmid=0,
    #     labels={"x": "Factor", "y": "Stock", "color": "Exposure"},
    #     text_auto=".2f",
    # )
    # fig_heatmap.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    # st.plotly_chart(fig_heatmap, use_container_width=True)

    max_abs = float(np.nanmax(np.abs(heatmap_data.values)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    fig_heatmap = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        range_color=[-max_abs, max_abs],
        labels={"x": "Factor", "y": "Stock", "color": "Exposure"},
        text_auto=".2f",
    )
    fig_heatmap.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Risk Split")
    st.info(
        "Factor risk is systematic common-factor risk; idiosyncratic risk is stock-specific residual risk."
    )

    factor_share = max(float(risk_report["factor_pct"]), 0.0)
    idio_share = max(float(risk_report["idio_pct"]), 0.0)
    share_total = factor_share + idio_share
    if share_total > 0:
        factor_share = factor_share / share_total
        idio_share = idio_share / share_total

    split_df = pd.DataFrame(
        {
            "Risk Bucket": ["Factor", "Idiosyncratic"],
            "Share": [100 * factor_share, 100 * idio_share],
        }
    )

    fig_split = px.pie(
        split_df,
        names="Risk Bucket",
        values="Share",
        hole=0.60,
        color="Risk Bucket",
        color_discrete_map={"Factor": "#1f77b4", "Idiosyncratic": "#ff7f0e"},
    )
    fig_split.update_traces(textinfo="label+percent")
    fig_split.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_split, use_container_width=True)

st.subheader("Full Factor Exposure Table")
st.info("Holdings-level factor exposure matrix used in this run.")

table = a["stock_exposures"].copy()
table.insert(0, "Weight", a["weights"].loc[table.index].values)

factor_cols = list(a["stock_exposures"].columns)
fmt = {"Weight": "{:.2%}"}
fmt.update({col: "{:.2f}" for col in factor_cols})

styled_table = (
    table.style.format(fmt)
    .background_gradient(cmap="RdYlGn", subset=factor_cols, axis=0)
    .bar(subset=["Weight"], color="#BBD7F0")
)

st.dataframe(styled_table, use_container_width=True, height=420)

download_payload = make_download_payload(
    weights=a["weights"],
    stock_exposures=a["stock_exposures"],
    portfolio_exposures=a["portfolio_exposures"],
    factor_contrib_pct=a["factor_contrib_pct"],
    risk_report=risk_report,
)

st.download_button(
    "Download Risk Report (CSV)",
    data=download_payload,
    file_name="factor_risk_report.csv",
    mime="text/csv",
)
