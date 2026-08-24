from statarb.hedge.base import HedgeEstimator
from statarb.hedge.kalman import KalmanHedge
from statarb.hedge.static import StaticHedge

__all__ = ["HedgeEstimator", "StaticHedge", "KalmanHedge"]


def make_hedge(method: str, **kwargs) -> HedgeEstimator:
    if method == "static":
        return StaticHedge(**kwargs)
    if method == "kalman":
        return KalmanHedge(**kwargs)
    raise ValueError(f"Unknown hedge method: {method!r}")
