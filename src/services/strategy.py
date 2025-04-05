from typing import Literal

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, HRPOpt
from scipy.optimize import minimize


def get_cum_ret(df: pd.DataFrame) -> pd.DataFrame:
    return (1 + df).cumprod()


def risk_parity_strategy(price_df: pd.DataFrame):
    def risk_parity_objective(weights: pd.DataFrame, cov_matrix: pd.DataFrame):
        variance = weights @ cov_matrix @ weights.T
        marginal_contribution = cov_matrix @ weights
        risk_contribution = np.multiply(weights, marginal_contribution) / variance
        target_risk = np.mean(risk_contribution)
        return np.sum((risk_contribution - target_risk) ** 2)

    cov_matrix = price_df.pct_change().dropna().cov() * 252
    weights = np.ones(len(cov_matrix)) / len(cov_matrix)

    return minimize(
        risk_parity_objective,
        weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=[(0, 1)] * len(cov_matrix),
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1},
    ).x


def hierarchical_risk_parity_strategy(price_df: pd.DataFrame):
    daily_returns = price_df.pct_change().dropna()
    cov_matrix = price_df.pct_change().dropna().cov() * 252
    opt = HRPOpt(daily_returns, cov_matrix)
    opt.optimize()
    optimised_weights = opt.clean_weights()
    optimised_weights_df = pd.DataFrame.from_dict(
        optimised_weights, orient="index", columns=["weight"]
    )
    return optimised_weights_df


def efficient_frontier_strategy(
    price_df: pd.DataFrame,
    type_: Literal["max_sharpe", "min_volatility"],
    risk_free_rate: float = 0.02,
):
    mean_hist_ret = price_df.pct_change().dropna().mean() * 252
    cov_matrix = price_df.pct_change().dropna().cov() * 252
    opt = EfficientFrontier(mean_hist_ret, cov_matrix)
    match type_:
        case "max_sharpe":
            opt.max_sharpe(risk_free_rate=risk_free_rate)
        case "min_volatility":
            opt.min_volatility()
        case _:
            raise ValueError(f"Unknown strategy: {type_}")
    optimised_weights = opt.clean_weights()
    optimised_weights_df = pd.DataFrame.from_dict(
        optimised_weights, orient="index", columns=["weight"]
    )
    return optimised_weights_df


def strategy(
    price_df: pd.DataFrame,
    type_: Literal[
        "max_sharpe",
        "min_volatility",
        "risk_parity",
        "hierarchical_risk_parity",
    ],
    risk_free_rate: float = 0.02,
):
    weights: pd.DataFrame
    match type_:
        case "max_sharpe":
            weights = efficient_frontier_strategy(
                price_df, "max_sharpe", risk_free_rate
            )
        case "min_volatility":
            weights = efficient_frontier_strategy(price_df, "min_volatility")
        case "hierarchical_risk_parity":
            weights = hierarchical_risk_parity_strategy(price_df)
        case "risk_parity":
            weights = pd.DataFrame(
                risk_parity_strategy(price_df),
                index=price_df.columns,
                columns=["weight"],
            )
        case _:
            raise ValueError(f"Unknown strategy: {type_}")

    weights.rename(columns={"weight": type_}, inplace=True)

    return weights
