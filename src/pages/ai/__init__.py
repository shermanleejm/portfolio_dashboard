import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def ai_page():
    openai_ai_key_input = st.text_input(type="password", label="OpenAI API Key")
    openai_ai_key = os.getenv("OPENAI_API_KEY") or openai_ai_key_input
    if not openai_ai_key:
        st.warning("Please enter your OpenAI API key.")
        return

    if len(st.session_state["core_portfolios"]) == 0:
        return

    openai_client = OpenAI(api_key=openai_ai_key)

    st.button(
        "Clear chat history", on_click=lambda: st.session_state.pop("messages", None)
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if len(st.session_state["messages"]) == 0:
        st.session_state.messages = [
            {
                "role": "system",
                "content": f"""
                Here are some statistics of our core portfolios in json format. The names of the portfolios are equity100, core-growth, core-balanced, core-defensive.
                The statistics are the following:
                - Annualized Return
                - Annualized Volatility
                - Sharpe ratio
                - Sortino ratio
                - Calmar ratio
                - Maximum Drawdown

                ```json
                {json.dumps(st.session_state["core_portfolios"])}
                ```
                """,
            },
            {
                "role": "system",
                "content": f"""
                Here are some statistics of some new strategeis that we are experimenting with.  
                These strategies are based on the efficient frontier and risk parity.
                The names of the strategies are max_sharpe, min_volatility, risk_parity, hierarchical_risk_parity.
                The statistics are the following:
                - Annualized Return
                - Annualized Volatility
                - Sharpe ratio
                - Conditional Value at Risk (CVaR)
                - 10 year shortfall risk (SR10Y)
                - Probability of loss (PL)
                
                ```json
                {json.dumps(st.session_state["current_investment_strategies"])}
                ```
                """,
            },
            {
                "role": "system",
                "content": "You are a financial advisor with an expertise in portfolio management and asset allocation. You are very good at answering questions about finance and investment strategies.",
            },
        ]

    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = (
            openai_client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state["messages"],
            )
            .choices[0]
            .message.content
        )

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
