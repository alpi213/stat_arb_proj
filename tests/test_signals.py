import numpy as np
import pandas as pd

from statarb.signals import generate_positions


def z(vals):
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)), dtype=float)


def test_entry_and_revert_exit():
    sig = generate_positions(z([0, 1, 2.5, 1.5, 0.4, 0]), entry_z=2, exit_z=0.5)
    assert list(sig["position"]) == [0, 0, -1, -1, 0, 0]
    assert sig["event"].iloc[2] == "entry_short"
    assert sig["event"].iloc[4] == "exit_revert"


def test_long_entry_on_negative_z():
    sig = generate_positions(z([0, -2.2, -1.0, -0.3]), entry_z=2, exit_z=0.5)
    assert list(sig["position"]) == [0, 1, 1, 0]


def test_stop_and_lockout():
    # Blows through stop; must NOT re-enter while |z| stays beyond entry.
    sig = generate_positions(
        z([0, 2.5, 3.6, 3.0, 2.5, 0.4, 2.5]), entry_z=2, exit_z=0.5, stop_z=3.5
    )
    assert sig["event"].iloc[2] == "exit_stop"
    assert list(sig["position"].iloc[2:5]) == [0, 0, 0]  # locked out
    assert sig["position"].iloc[5] == 0                   # z back inside, unlocked
    assert sig["position"].iloc[6] == -1                  # fresh entry allowed


def test_time_stop():
    vals = [0, 2.5] + [1.5] * 10
    sig = generate_positions(z(vals), entry_z=2, exit_z=0.5, max_holding_days=5)
    assert sig["event"].iloc[6] == "exit_time"
    assert all(sig["position"].iloc[7:] == 0) or sig["position"].iloc[7:].eq(0).all()


def test_nan_z_holds_state():
    sig = generate_positions(z([0, 2.5, np.nan, 1.5, 0.2]), entry_z=2, exit_z=0.5)
    assert sig["position"].iloc[2] == -1  # NaN bar: hold, don't flatten
