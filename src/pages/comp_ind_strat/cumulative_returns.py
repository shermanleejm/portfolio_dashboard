import pandas as pd
import plotly.express as px
import streamlit as st


def cumulative_returns(portf_df: pd.DataFrame):
    cumulative_returns = (1 + portf_df).cumprod()

    st.plotly_chart(px.line(cumulative_returns))

    with st.expander("Show data"):
        cumulative_returns[::-1]
