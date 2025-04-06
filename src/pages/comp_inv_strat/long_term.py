import pandas as pd
import streamlit as st

from src.services.monte_carlo import get_simulated_returns


def long_term(price_df: pd.DataFrame, weights: pd.DataFrame) -> None:
    simulated_returns = get_simulated_returns(price_df, 1000)
    simulated_portfolio_values = simulated_returns @ weights
    st.write(simulated_portfolio_values)
