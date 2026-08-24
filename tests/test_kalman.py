import numpy as np
import pandas as pd

from statarb.hedge import KalmanHedge, StaticHedge


def test_kalman_converges_to_true_beta(coint_pair):
    log_y = np.log(coint_pair["y"])
    log_x = np.log(coint_pair["x"])
    kf = KalmanHedge(delta=1e-5, obs_var=1e-3)
    out = kf.estimate(log_y, log_x)
    tail_beta = out["beta"].iloc[-250:].mean()
    assert np.isclose(tail_beta, 1.5, atol=0.15)


def test_kalman_tracks_drifting_beta_better_than_static():
    """Regime where the true hedge ratio drifts from 1.0 to 2.0."""
    rng = np.random.default_rng(99)
    n = 2000
    dates = pd.bdate_range("2015-01-01", periods=n)
    log_x = np.cumsum(rng.normal(0, 0.012, n)) + np.log(50)
    true_beta = np.linspace(1.0, 2.0, n)
    log_y = 0.3 + true_beta * log_x + rng.normal(0, 0.01, n)
    ly = pd.Series(log_y, index=dates)
    lx = pd.Series(log_x, index=dates)

    kf_beta = KalmanHedge(delta=1e-4).estimate(ly, lx)["beta"]
    st_beta = StaticHedge(refit_every_days=63).estimate(ly, lx)["beta"]

    idx = kf_beta.dropna().index.intersection(st_beta.dropna().index)
    kf_err = np.abs(kf_beta.loc[idx] - pd.Series(true_beta, index=dates).loc[idx]).mean()
    st_err = np.abs(st_beta.loc[idx] - pd.Series(true_beta, index=dates).loc[idx]).mean()
    assert kf_err < st_err


def test_static_hedge_no_same_bar_fit(coint_pair):
    """The refit at bar i must not use bar i itself."""
    log_y = np.log(coint_pair["y"])
    log_x = np.log(coint_pair["x"])
    sh = StaticHedge(refit_every_days=63, min_obs=126)
    base = sh.estimate(log_y, log_x)

    # Perturb ONLY the last observation; every earlier alpha/beta must be
    # unchanged (the last bar can never be in its own fit window).
    log_y2 = log_y.copy()
    log_y2.iloc[-1] += 0.5
    pert = sh.estimate(log_y2, log_x)
    pd.testing.assert_frame_equal(
        base[["alpha", "beta"]].iloc[:-1], pert[["alpha", "beta"]].iloc[:-1]
    )
