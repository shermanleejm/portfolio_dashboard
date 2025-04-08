import numpy as np
import pandas as pd
import streamlit as st

from src.services.monte_carlo import get_simulated_prices, get_stats


def long_term(price_df: pd.DataFrame, combined_weights: pd.DataFrame) -> None:
    tickers = price_df.columns.tolist()
    cube = np.array([get_simulated_prices(price_df[[ticker]]) for ticker in tickers])
    stats = get_stats(cube, combined_weights)
    st.markdown(""" 
    # Long-Term Portfolio Simulation
    This section simulates the long-term performance of the portfolio using Monte Carlo methods.  
    Each ticker's hitorical price data is use to train a GARCH model, which is then used to generate simulated prices for the next 10 years.
    """)
    st.write(stats)
    st.session_state["current_investment_strategies"][
        "investment_strategies_long_term_simulation"
    ] = stats.to_dict()
