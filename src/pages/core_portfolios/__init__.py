import numpy as np
import pandas as pd
import streamlit as st

from src.services.data import get_data
from src.services.monte_carlo import get_simulated_prices, get_stats
from src.services.scraper import scrape_core_portfolio
from src.services.strategy import calculate_performance_metrics

proper_tickers = {
    "XDEW": "XDEW.L",
    "CSPX": "CSPX.L",
    "EIMI": "EIMI.L",
    "IUAA": "IUAA.L",
    "AGGU": "AGGU.L",
}


def examine_core_portfolios_page():
    """
    This function is a placeholder for the core portfolios page.
    It currently does not contain any functionality or content.
    """
    end_date = (pd.Timestamp.now()).strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.offsets.DateOffset(years=10)).strftime(
        "%Y-%m-%d"
    )
    current_portfolios = [
        ("equity100", "15"),
        ("core-growth", "5"),
        ("core-balanced", "15"),
        ("core-defensive", "25"),
    ]
    curr_metrics = {}
    combined_weights = {}
    combined_price_df = pd.DataFrame()
    for portfolio, second_key in current_portfolios:
        portfolio_info = scrape_core_portfolio(portfolio)
        etf_allocation = portfolio_info["etfAllocation"][second_key]
        weights = pd.DataFrame([etf_allocation])
        tickers = [proper_tickers.get(x, x) for x in weights.columns.tolist()]
        weights = weights.rename(columns=proper_tickers)
        combined_weights[portfolio] = weights.iloc[0]
        price_df, risk_free_rate = get_data(
            tickers=tuple(tickers),
            start_date=start_date,
            end_date=end_date,
        )
        combined_price_df = pd.concat([combined_price_df, price_df], axis=1)
        perf_metrics = calculate_performance_metrics(price_df, weights, risk_free_rate)
        curr_metrics[portfolio] = perf_metrics

    combined_price_df = combined_price_df.loc[
        :, ~combined_price_df.columns.duplicated()
    ]
    combined_price_df = combined_price_df[combined_price_df > 0].dropna()
    combined_weights = (
        pd.DataFrame(combined_weights).reindex(combined_price_df.columns).fillna(0)
    )
    cube = np.array(
        [get_simulated_prices(combined_price_df[ticker]) for ticker in tickers]
    )
    stats = get_stats(cube, combined_weights)
    st.markdown(
        """
        # Core Portfolios metrics
        This section shows the performance metrics of the core portfolios.
        """
    )
    st.write(pd.DataFrame(curr_metrics))
    st.markdown(
        """
        # Core Portfolios Long-Term Simulation
        This section simulates the long-term performance of the core portfolios using Monte Carlo methods.  
        Each ticker's historical price data is used to train a GARCH model, which is then used to generate simulated prices for the next 10 years.
        """
    )
    st.write(stats)

    st.session_state["core_portfolios"] = {
        "metrics": curr_metrics,
        "long_term_simulation_metrics": stats.to_dict(),
    }
