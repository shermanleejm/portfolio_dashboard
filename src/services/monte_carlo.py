import numpy as np
import pandas as pd
from arch import arch_model
from copulae import GaussianCopula

np.random.seed(1234)  # For reproducibility


def get_simulated_returns(
    price_df: pd.DataFrame, num_simulations: int = 1000
) -> pd.DataFrame:
    tickers = price_df.columns
    returns = price_df.pct_change().dropna() * 100

    standardized_residuals = pd.DataFrame(index=returns.index)

    for ticker in tickers:
        model = arch_model(returns[ticker], vol="ARCH", p=1)
        res = model.fit(disp="off")
        standardized_residuals[ticker] = res.std_resid.dropna()

    standardized_residuals.dropna(inplace=True)

    def ecdf_uniform(x):
        ranks = np.argsort(np.argsort(x))
        return ranks / (len(x) - 1)

    u_data = np.column_stack(
        [ecdf_uniform(standardized_residuals[col].values) for col in tickers]
    )

    copula = GaussianCopula(dim=3)
    copula.fit(u_data)

    u_sim = copula.random(num_simulations)

    def invert_ecdf(sample, uniform_data):
        sorted_sample = np.sort(sample)
        n = len(sorted_sample)
        return sorted_sample[(uniform_data * n).astype(int).clip(0, n - 1)]

    sim_residuals = {}
    for i, ticker in enumerate(tickers):
        sim_residuals[ticker] = invert_ecdf(
            standardized_residuals[ticker].values, u_sim[:, i]
        )

    future_returns = {}
    for ticker in tickers:
        model = arch_model(returns[ticker], vol="ARCH", p=1)
        res = model.fit(disp="off")
        latest_vol = np.sqrt(res.conditional_volatility.iloc[-1])
        future_returns[ticker] = sim_residuals[ticker] * latest_vol

    return pd.DataFrame(future_returns)
