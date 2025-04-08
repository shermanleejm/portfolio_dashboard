import numpy as np
import pandas as pd
from arch import arch_model
from copulas.univariate import StudentTUnivariate

np.random.seed(1234)


def get_simulated_prices(
    price_df: pd.DataFrame,
    n_simulations: int = 1000,
    n_years: int = 10,
    trading_days_in_year: int = 252,
) -> np.ndarray:
    returns = 100 * price_df.pct_change().dropna()
    model = arch_model(returns, vol="Garch", p=1, q=1, dist="t")
    res = model.fit(disp="off")

    standardized_resid = res.resid / res.conditional_volatility
    standardized_resid = standardized_resid.dropna()
    student_t = StudentTUnivariate()
    student_t.fit(standardized_resid.values)

    mu = res.params["mu"]
    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]

    trading_days = trading_days_in_year * n_years

    last_price = price_df.iloc[-1]
    last_vol = res.conditional_volatility.iloc[-1]
    last_resid = res.resid.iloc[-1]

    simulated_prices = np.zeros((trading_days + 1, n_simulations))
    simulated_prices[0, :] = last_price

    eps = np.full(n_simulations, last_resid)
    sigma = np.full(n_simulations, last_vol)

    for t in range(1, trading_days + 1):
        z = student_t.sample(n_simulations)
        sigma2 = omega + alpha * eps**2 + beta * sigma**2
        sigma = np.sqrt(sigma2)
        eps = sigma * z
        r = mu + eps
        simulated_prices[t] = simulated_prices[t - 1] * (1 + r / 100)

    return simulated_prices


def get_stats(cube: np.ndarray, combined_weights: pd.DataFrame) -> pd.DataFrame:
    sim_prices = np.tensordot(combined_weights.T.to_numpy(), cube, axes=([1], [0]))

    returns = sim_prices[:, 1:, :] / sim_prices[:, :-1, :] - 1
    returns_3y = returns[:, : 252 * 3, :]
    cum_returns = (1 + returns).prod(axis=1) - 1
    cum_returns_3y = (1 + returns_3y).prod(axis=1) - 1
    annualized_returns = (1 + cum_returns) ** (252 / returns.shape[1]) - 1
    cutoffs = np.percentile(cum_returns, 5, axis=1)

    stats = []
    for i in range(4):
        ar = annualized_returns[i]
        cr = cum_returns[i]
        cr3y = cum_returns_3y[i]
        cutoff = cutoffs[i]

        stats.append(
            {
                "cvar": float(ar[ar <= cutoff].mean()),
                "10y_shortfall": np.mean(cr < 0),
                "prob_3y_loss": np.mean(cr3y < -0.3),
                "mean": ar.mean(),
                "volatility": ar.std() * np.sqrt(returns.shape[1] / 252),
                "min": ar.min(),
                "p5": np.percentile(ar, 5.0),
                "p25": np.percentile(ar, 25.0),
                "median": np.median(ar),
                "p75": np.percentile(ar, 75.0),
                "p95": np.percentile(ar, 95.0),
                "max": ar.max(),
            }
        )

    stats_df = pd.DataFrame(stats).T
    stats_df.columns = combined_weights.columns

    return stats_df
