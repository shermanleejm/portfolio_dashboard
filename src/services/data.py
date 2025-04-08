import pandas as pd
import yfinance as yf
from functools import lru_cache


@lru_cache
def get_data(
    tickers: tuple[str], start_date: str, end_date: str
) -> tuple[pd.DataFrame, float]:
    """
    Parameters
    ----------
    tickers : list[str]
        List of tickers to download data for.
    start_date : str
        Start date for the data download. ISO format (YYYY-MM-DD).
    end_date : str
        End date for the data download. ISO format (YYYY-MM-DD).
    """
    risk_free_rate = (
        yf.download("^TNX", start=start_date, end=end_date, auto_adjust=True)[
            "Close"
        ].mean()
        / 100
    ).iloc[0]
    price_df = (
        yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
        .reindex(tickers, axis=1)
        .fillna(0)
    )

    return price_df, risk_free_rate
