"""Typed configuration loaded from YAML.

Every experiment is fully described by (config file, git commit hash).
Pydantic validates types and ranges at load time so a typo in YAML fails
loudly at startup instead of silently producing a wrong backtest.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    provider: str = "yfinance"
    cache_dir: Path = Path("data/processed")
    raw_dir: Path = Path("data/raw")
    start: str
    end: str
    frequency: str = "1d"
    universe_file: Path = Path("configs/universe.csv")
    min_history_days: int = 750
    max_missing_pct: float = 0.02


class KalmanConfig(BaseModel):
    delta: float = 1e-5
    obs_var: float = 1e-3
    spread_mode: str = "innovation"  # "innovation" | "residual"

    @field_validator("spread_mode")
    @classmethod
    def _check_spread_mode(cls, v: str) -> str:
        if v not in {"innovation", "residual"}:
            raise ValueError("kalman.spread_mode must be 'innovation' or 'residual'")
        return v


class StaticHedgeConfig(BaseModel):
    refit_every_days: int = 63


class HedgeConfig(BaseModel):
    method: str = "kalman"
    kalman: KalmanConfig = KalmanConfig()
    static: StaticHedgeConfig = StaticHedgeConfig()

    @field_validator("method")
    @classmethod
    def _check_method(cls, v: str) -> str:
        if v not in {"static", "kalman"}:
            raise ValueError(f"hedge.method must be 'static' or 'kalman', got {v!r}")
        return v


class PairsConfig(BaseModel):
    grouping: str = "sector"
    formation_window_days: int = 504
    adf_pvalue_max: float = 0.05
    min_half_life_days: float = 2.0
    max_half_life_days: float = 60.0
    top_n_pairs: int = 20
    use_johansen: bool = False


class SignalConfig(BaseModel):
    zscore_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_holding_days: int = 40

    @field_validator("stop_z")
    @classmethod
    def _stop_beyond_entry(cls, v: float, info) -> float:
        entry = info.data.get("entry_z")
        if entry is not None and v <= entry:
            raise ValueError("stop_z must be greater than entry_z")
        return v


class CostConfig(BaseModel):
    commission_bps: float = 1.0
    half_spread_bps: float = 2.5
    borrow_fee_annual_bps: float = 50.0


class RiskConfig(BaseModel):
    max_gross_leverage: float = 2.0
    max_pairs_concurrent: int = 20


class BacktestConfig(BaseModel):
    initial_capital: float = 1_000_000
    gross_per_pair: float = 0.10
    execution_lag_bars: int = Field(1, ge=1)  # ge=1 enforces no same-bar execution
    costs: CostConfig = CostConfig()
    risk: RiskConfig = RiskConfig()


class CapacityConfig(BaseModel):
    impact_model: str = "sqrt_law"
    impact_coef: float = 0.1
    adv_window_days: int = 21
    aum_grid: list[float] = Field(default_factory=lambda: [1e6, 1e7, 1e8, 1e9])


class WalkForwardConfig(BaseModel):
    formation_days: int = 504
    trading_days: int = 126
    step_days: int = 126


class ValidationConfig(BaseModel):
    walkforward: WalkForwardConfig = WalkForwardConfig()
    n_trials: int = 1
    regimes: dict[str, tuple[str, str]] = Field(default_factory=dict)


class Config(BaseModel):
    data: DataConfig
    pairs: PairsConfig = PairsConfig()
    hedge: HedgeConfig = HedgeConfig()
    signals: SignalConfig = SignalConfig()
    backtest: BacktestConfig = BacktestConfig()
    capacity: CapacityConfig = CapacityConfig()
    validation: ValidationConfig = ValidationConfig()
    output_dir: Path = Path("results")
    seed: int = 42


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
