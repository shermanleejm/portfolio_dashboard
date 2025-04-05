import streamlit as st

from src.pages.comparing_investment_strategies import (
    comparing_investment_strategies_page,
)

st.set_page_config(layout="wide")

strategy_tab, syfe_tab, ai_tab = st.tabs(
    [
        "Comparing Investment Strategies",
        "Syfe Core portfolios",
        "AI",
    ]
)

with strategy_tab:
    comparing_investment_strategies_page()
