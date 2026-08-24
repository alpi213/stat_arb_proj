import numpy as np
import pandas as pd

from statarb.pairs import engle_granger, half_life
from tests.conftest import make_cointegrated_pair


def test_detects_true_cointegration(coint_pair):
    log_y = np.log(coint_pair["y"]).rename("y")
    log_x = np.log(coint_pair["x"]).rename("x")
    res = engle_granger(log_y, log_x)
    assert res.pvalue < 0.05
    # Direction may flip (EG is run both ways); beta should match either
    # the true 1.5 or its inverse.
    assert np.isclose(res.beta, 1.5, atol=0.1) or np.isclose(res.beta, 1 / 1.5, atol=0.05)


def test_rejects_independent_walks(indep_pair):
    log_y = np.log(indep_pair["y"]).rename("y")
    log_x = np.log(indep_pair["x"]).rename("x")
    res = engle_granger(log_y, log_x)
    assert res.pvalue > 0.05


def test_half_life_recovers_true_value():
    # theta=0.05 per bar => HL = ln2/(-ln(0.95)) ~ 13.5 bars
    df = make_cointegrated_pair(n=4000, spread_theta=0.05)
    log_y, log_x = np.log(df["y"]), np.log(df["x"])
    resid = log_y - 1.5 * log_x - 0.5
    hl = half_life(resid)
    assert 9 < hl < 20


def test_half_life_untradeably_slow_for_random_walk():
    # On a finite sample the AR(1) coefficient of a pure random walk is
    # biased slightly negative (Dickey-Fuller bias), so the estimate comes
    # out large-but-finite rather than infinite. What matters for the
    # pipeline is that it lands far above max_half_life_days (60), so the
    # selection filter rejects it.
    rng = np.random.default_rng(7)
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 3000)))
    assert half_life(rw) > 100
