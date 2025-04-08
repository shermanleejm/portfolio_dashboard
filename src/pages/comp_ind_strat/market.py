import pandas as pd
import plotly.express as px
import streamlit as st


def market_page(price_df: pd.DataFrame):
    crisis_events = load_crisis_events()

    crisis_start = (
        crisis_events[["stress_event", "start_date"]]
        .merge(price_df, left_on="start_date", right_index=True)
        .set_index("stress_event")
        .drop(columns=["start_date"])
    )
    crisis_end = (
        crisis_events[["stress_event", "end_date"]]
        .merge(price_df, left_on="end_date", right_index=True)
        .set_index("stress_event")
        .drop(columns=["end_date"])
    )

    crisis_duration = (
        crisis_events["start_date"] - crisis_events["end_date"]
    ).dt.days / 252
    crisis_duration.index = crisis_start.index

    cagr = crisis_end / crisis_start
    for col in cagr.columns:
        cagr[col] = -(cagr[col] ** (1 / crisis_duration) - 1)

    st.write((cagr * 100).round(2).astype(str) + "%")
    st.session_state["current_investment_strategies"]["cagr_during_crisis"] = (
        cagr.to_dict()
    )

    fig = px.line((1 + price_df.pct_change()).cumprod())

    shapes = [
        dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=row["start_date"],
            x1=row["end_date"],
            y0=0,
            y1=1,
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
        for _, row in crisis_events.iterrows()
    ]
    annotations = [
        dict(
            x=row["start_date"],
            y=1.02,  # Slightly above the plot
            xref="x",
            yref="paper",
            text=row["stress_event"],
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=2,
        )
        for _, row in crisis_events.iterrows()
    ]

    fig.update_layout(shapes=shapes, annotations=annotations)
    st.plotly_chart(fig)


def load_crisis_events():
    crisis_events = pd.DataFrame(
        [
            {
                "stress_event": "2022 Inflation",
                "start_date": "2021-12-31",
                "end_date": "2022-09-30",
            },
            {
                "stress_event": "COVID-19 Pandemic",
                "start_date": "2019-12-31",
                "end_date": "2020-03-31",
            },
            {
                "stress_event": "Chinese Stock Market Crash",
                "start_date": "2015-06-01",
                "end_date": "2015-09-30",
            },
            {
                "stress_event": "Flash Crash 2010",
                "start_date": "2010-03-31",
                "end_date": "2010-06-30",
            },
            {
                "stress_event": "2008 Financial Crisis",
                "start_date": "2007-10-01",
                "end_date": "2009-03-31",
            },
            # {
            #     "stress_event": "Dot-com Bubble Burst",
            #     "start_date": "1999-12-31",
            #     "end_date": "2003-03-31",
            # },
        ]
    )
    crisis_events["start_date"] = pd.to_datetime(crisis_events["start_date"])
    crisis_events["end_date"] = pd.to_datetime(crisis_events["end_date"])

    return crisis_events
