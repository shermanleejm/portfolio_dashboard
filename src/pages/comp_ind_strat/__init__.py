import pandas as pd
import streamlit as st

from src.pages.comp_ind_strat.cumulative_returns import cumulative_returns
from src.pages.comp_ind_strat.long_term import long_term
from src.pages.comp_ind_strat.market import market_page
from src.services.data import get_data
from src.services.strategy import strategy


def comparing_investment_strategies_page():
    end_date = (pd.Timestamp.now()).strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.offsets.DateOffset(years=20)).strftime(
        "%Y-%m-%d"
    )

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(label="Start date", value=start_date)

    with col2:
        end_date = st.date_input(label="End date", value=end_date)

    tickers_input = st.text_input(
        value="SPY,TLT,GLD",
        label="Additional tickers",
        help="Enter additional tickers separated by commas",
    )

    tickers = []
    if tickers_input:
        tickers = [ticker.strip() for ticker in tickers_input.split(",")]

    st.write("Selected tickers:", tickers)
    strategy_tab, market_conditions_tab, long_term_tab = st.tabs(
        [
            "Cumulative Returns",
            "Historical Crisis Events",
            "Long-Term Portfolio",
        ]
    )

    price_df, risk_free_rate = get_data(
        tickers=tuple(tickers),
        start_date=start_date,
        end_date=end_date,
    )

    combined_weights = pd.concat(
        [
            # price_df.pct_change().dropna(),
            strategy(price_df, "max_sharpe", risk_free_rate=risk_free_rate),
            strategy(price_df, "min_volatility"),
            strategy(price_df, "risk_parity"),
            strategy(price_df, "hierarchical_risk_parity"),
        ],
        axis=1,
    ).dropna()

    st.session_state["current_investment_strategies"]["weights"] = (
        combined_weights.to_dict()
    )

    with strategy_tab:
        st.write("Investment strategies weights")
        st.dataframe(combined_weights)
        cumulative_returns(price_df.pct_change() @ combined_weights)

    with market_conditions_tab:
        market_page(price_df @ combined_weights)

    with long_term_tab:
        long_term(price_df, combined_weights)
