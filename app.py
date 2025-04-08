import streamlit as st

from src.pages.ai import ai_page
from src.pages.comp_ind_strat import (
    comparing_investment_strategies_page,
)
from src.pages.core_portfolios import examine_core_portfolios_page

st.set_page_config(layout="wide")

strategy_tab, core_portf_tab, ai_tab = st.tabs(
    [
        "Comparing Investment Strategies",
        "Core portfolios",
        "AI",
    ]
)


st.session_state["current_investment_strategies"] = {}
st.session_state["core_portfolios"] = {}


with strategy_tab:
    comparing_investment_strategies_page()
with core_portf_tab:
    examine_core_portfolios_page()
with ai_tab:
    ai_page()
