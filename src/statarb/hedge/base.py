"""Common interface for hedge-ratio estimators.

The contract that prevents look-ahead: `estimate(log_y, log_x)` must return,
for each date t, a (alpha_t, beta_t) computed from information available at
the END of day t-? — concretely, every implementation must guarantee that
row t of the output uses only observations up to and including t. The
backtester then applies its own execution lag on top.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class HedgeEstimator(ABC):
    @abstractmethod
    def estimate(self, log_y: pd.Series, log_x: pd.Series) -> pd.DataFrame:
        """Return DataFrame(index=dates, columns=["alpha", "beta", "spread"]).

        spread_t = log_y_t - beta_t * log_x_t - alpha_t
        """
        ...
