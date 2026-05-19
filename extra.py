# factor_risk_model/streamlit_app.py
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import factors as factor_engine
from src import risk as risk_engine


st.set_page_config(page_title="Factor Risk Model", layout="wide")
st.title("Factor Risk Model")
st.caption("This app implements a Barra/Axioma-style factor risk model.")
st.warning("This tool is not intended as investment advice, so we do not provide real-time data.")


def build_price_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"],
            "STOCK_A": [100.0, 101.2, 100.5, 102.0],
            "STOCK_B": [50.0, 49.8, 50.6, 51.1],
            "STOCK_C": [200.0, 202.5, 201.0, 203.2],
        }
    )


def build_shares_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"],
            "STOCK_A": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
            "STOCK_B": [2_000_000, 2_000_000, 2_000_000, 2_000_000],
            "STOCK_C": [500_000, 500_000, 500_000, 500_000],
        }
    )


def read_uploaded_csv(uploaded_file, label: str) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError(f"{label} file was not uploaded.")
    raw = pd.read_csv(uploaded_file)
    return factor_engine.prepare_wide_timeseries(raw, label=label)


def build_model_inputs(
    prices: pd.DataFrame,
    shares_outstanding: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = factor_engine.compute_returns(prices)
    if len(returns) < 252:
        raise ValueError(f"Need at least 252 return rows; got {len(returns)}.")

    factor_raw = factor_engine.construct_factors(
        returns=returns,
        prices=prices,
        shares_outstanding=shares_outstanding,
        use_barra_size=shares_outstanding is not None,
    )
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


def filter_inputs_by_date(
    prices: pd.DataFrame,
    shares: pd.DataFrame | None,
    start_date,
    end_date,
):
    if end_date < start_date:
        raise ValueError("End date must be on or after start date.")

    mask = (prices.index.date >= start_date) & (prices.index.date <= end_date)
    prices_f = prices.loc[mask].copy()

    if len(prices_f) < 252:
        raise ValueError(
            f"Selected range has {len(prices_f)} price rows. Need at least 252 for factor construction."
        )

    shares_f = None
    if shares is not None:
        shares_mask = (shares.index.date >= start_date) & (shares.index.date <= end_date)
        shares_f = shares.loc[shares_mask].copy()

    return build_model_inputs(prices_f, shares_outstanding=shares_f)


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

with st.sidebar:
    st.title("Factor Risk Model")

    st.subheader("Portfolio Construction")
    construction = st.selectbox(
        "Select Portfolio Construction",
        options=["Equal Weight", "Momentum Tilt", "Low Volatility Tilt", "Custom Tickers"],
        index=0,
    )

    st.divider()

    data_mode = st.radio(
        "Price data source",
        options=["Built-in sample CSV", "Upload prices CSV"],
        index=0,
    )

    uploaded_prices = None
    uploaded_shares = None
    if data_mode == "Upload prices CSV":

        with st.expander("Input format (read this before upload)", expanded=False):
            st.write("Required: one date column + one column per ticker, wide format.")
            st.write("Date can be named `Date`, `date`, or any column containing `date`.")
            st.write("Optional shares file must use the same wide format and ticker names.")
            st.write("If shares data exists, SIZE uses log(price * shares). Missing shares fall back to proxy.")
            st.dataframe(build_price_template(), use_container_width=True)
            st.download_button(
                "Download Price Template CSV",
                build_price_template().to_csv(index=False).encode("utf-8"),
                file_name="price_template.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download Shares Template CSV (Optional)",
                build_shares_template().to_csv(index=False).encode("utf-8"),
                file_name="shares_template.csv",
                mime="text/csv",
            )

        uploaded_prices = st.file_uploader("Upload prices CSV", type=["csv"])
        uploaded_shares = st.file_uploader("Upload shares outstanding CSV (optional)", type=["csv"])

try:
    if data_mode == "Built-in sample CSV":
        full_prices = factor_engine.prices.copy()
        if full_prices.empty:
            raise ValueError(
                "Built-in sample data was not found. Upload a prices CSV or set PRICES_CSV_PATH."
            )
        full_shares = None
    else:
        if uploaded_prices is None:
            st.info("Upload a prices CSV in the sidebar to continue.")
            st.stop()
        full_prices = read_uploaded_csv(uploaded_prices, label="prices")

        full_shares = None
        if uploaded_shares is not None:
            parsed_shares = read_uploaded_csv(uploaded_shares, label="shares outstanding")
            overlap = [c for c in full_prices.columns if c in parsed_shares.columns]
            if overlap:
                full_shares = parsed_shares[overlap]
                if len(overlap) < len(full_prices.columns):
                    st.warning(
                        "Shares data does not cover all tickers. SIZE falls back to proxy where shares are missing."
                    )
            else:
                st.warning("No ticker overlap between prices and shares file. Ignoring shares file.")
except Exception as exc:
    st.error(f"Data load failed: {exc}")
    st.stop()

with st.sidebar:

    st.divider()
    st.subheader("Customizeable Features")

    use_full_history = st.checkbox("Use full date range", value=True)

    min_date = full_prices.index.min().date()
    max_date = full_prices.index.max().date()

    start_date = min_date
    end_date = max_date

    if not use_full_history:
        st.markdown("**Date Range**")
        # date_selection = st.date_input(
        #     "Select start and end date",
        #     value=(min_date, max_date),
        #     min_value=min_date,
        #     max_value=max_date,
        # )
        # if not isinstance(date_selection, (list, tuple)) or len(date_selection) != 2:
        #     st.error("Select both a start and end date.")
        #     st.stop()
        # start_date, end_date = date_selection
    start_date, end_date = st.slider(
        "Drag to select start and end date",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        key="date_range_slider",
    )

    if end_date < start_date:
        st.error("End date must be on or after start date.")
        st.stop()

    st.caption(f"Selected: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")

try:
    if use_full_history:
        prices, returns, factor_raw, factor_exposures = build_model_inputs(
            prices=full_prices,
            shares_outstanding=full_shares,
        )
        current_range = ("FULL",)
    else:
        prices, returns, factor_raw, factor_exposures = filter_inputs_by_date(
            prices=full_prices,
            shares=full_shares,
            start_date=start_date,
            end_date=end_date,
        )
        current_range = (start_date.isoformat(), end_date.isoformat())

    factor_cov, idio_var = load_risk_objects(returns, factor_exposures)
except Exception as exc:
    st.error(f"Model build failed: {exc}")
    st.stop()

tickers = factor_exposures.index.tolist()

data_signature = (
    data_mode,
    full_prices.index.min().isoformat(),
    full_prices.index.max().isoformat(),
    tuple(full_prices.columns.tolist()),
    full_shares is not None,
    current_range,
)
if st.session_state.get("data_signature") != data_signature:
    st.session_state.analysis = None
    st.session_state.data_signature = data_signature

with st.sidebar:
    custom_tickers = []
    if construction == "Custom Tickers":
        custom_tickers = st.multiselect(
            "Select stocks",
            options=tickers,
            default=tickers[: min(10, len(tickers))],
        )
    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.caption("Created with love, logic and a questionable amount of caffeine")
    st.caption("Chinmae Chittybabu")
    st.markdown(
        "LinkedIn: [www.linkedin.com/in/chinmae-c-bba900274](https://www.linkedin.com/in/chinmae-c-bba900274)"
    )
    st.markdown("GitHub: [github.com/Chinmaec](https://github.com/Chinmaec)")


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
        factor_contrib_pct = 100.0 * factor_contrib / total_var if total_var > 0 else factor_contrib * 0.0

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
    st.info("Choose settings in the sidebar and click Run Analysis.")
    st.stop()

a = st.session_state.analysis
risk_report = a["risk_report"]

metric_cols = st.columns(3)
metric_cols[0].metric("Total Volatility", f"{100 * float(risk_report['total_vol']):.2f}%")
metric_cols[1].metric("Factor Risk", f"{100 * float(risk_report['factor_vol']):.2f}%")
metric_cols[2].metric("Idiosyncratic Risk", f"{100 * float(risk_report['idio_vol']):.2f}%")

left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.subheader("Factor Contributions")
    factor_contrib_df = a["factor_contrib_pct"].reset_index()
    factor_contrib_df.columns = ["Factor", "ContributionPct"]

    fig_contrib = px.bar(
        factor_contrib_df,
        x="Factor",
        y="ContributionPct",
        color="ContributionPct",
        color_continuous_scale="Blues",
        text=factor_contrib_df["ContributionPct"].map(lambda v: f"{v:.4f}%"),
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
        text=pexp_df["Exposure"].map(lambda v: f"{v:.4f}"),
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
    heatmap_data = a["stock_exposures"].copy()
    heatmap_data.index.name = "Ticker"

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
table = a["stock_exposures"].copy()
table.insert(0, "Weight", a["weights"].loc[table.index].values)

factor_cols = list(a["stock_exposures"].columns)
fmt = {"Weight": "{:.2%}"}
fmt.update({col: "{:.4f}" for col in factor_cols})

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