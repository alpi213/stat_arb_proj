"""Shared fixtures: synthetic price series with KNOWN properties.

Testing on synthetic data with known ground truth (true beta, true half-life)
is the only way to verify statistical code — real data can't tell you if
your estimator is biased.
"""

import numpy as np
import pandas as pd
import pytest

RNG_SEED = 12345


def make_cointegrated_pair(
    n: int = 1500,
    beta: float = 1.5,
    alpha: float = 0.5,
    spread_theta: float = 0.05,   # ~13-day half-life
    spread_sigma: float = 0.01,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """log_y = alpha + beta*log_x + OU(theta, sigma); log_x = random walk."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    log_x = np.cumsum(rng.normal(0, 0.012, n)) + np.log(50)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - spread_theta) + rng.normal(0, spread_sigma)
    log_y = alpha + beta * log_x + spread
    return pd.DataFrame(
        {"x": np.exp(log_x), "y": np.exp(log_y)}, index=dates
    )


def make_independent_walks(n: int = 1500, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    a = np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)) + np.log(40))
    b = np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)) + np.log(60))
    return pd.DataFrame({"x": a, "y": b}, index=dates)


@pytest.fixture
def coint_pair() -> pd.DataFrame:
    return make_cointegrated_pair()


@pytest.fixture
def indep_pair() -> pd.DataFrame:
    return make_independent_walks()
