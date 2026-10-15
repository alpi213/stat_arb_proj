"""Typed config, loaded from configs/*.yaml.

Pydantic checks types and ranges when the file is read, so a typo fails at
startup instead of producing a quietly wrong backtest. Unknown keys are
rejected too, so a stale or misspelled setting can't be silently ignored.
An experiment is fully described by (config file, git commit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(Strict):
    cache_dir: Path = Path("data/processed")
    start: str
    end: str
    universe_file: Path = Path("configs/universe.csv")
    min_history_days: int = 750
    max_missing_pct: float = 0.02


class KalmanConfig(Strict):
    delta: float = 1e-5
    obs_var: float = 1e-3
    spread_mode: Literal["innovation", "residual"] = "innovation"


class StaticHedgeConfig(Strict):
    refit_every_days: int = 63


class HedgeConfig(Strict):
    method: Literal["static", "kalman"] = "kalman"
    kalman: KalmanConfig = KalmanConfig()
    static: StaticHedgeConfig = StaticHedgeConfig()


class PairsConfig(Strict):
    formation_window_days: int = 504
    adf_pvalue_max: float = 0.05
    min_half_life_days: float = 2.0
    max_half_life_days: float = 60.0
    top_n_pairs: int = 20


class SignalConfig(Strict):
    zscore_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_holding_days: int = 40

    @model_validator(mode="after")
    def _stop_beyond_entry(self):
        if self.stop_z <= self.entry_z:
            raise ValueError("signals.stop_z must be greater than entry_z")
        return self


class CostConfig(Strict):
    commission_bps: float = 1.0
    half_spread_bps: float = 2.5
    borrow_fee_annual_bps: float = 50.0


class BacktestConfig(Strict):
    initial_capital: float = 1_000_000
    gross_per_pair: float = 0.10
    # ge=1 means a config typo can never produce same-bar execution.
    execution_lag_bars: int = Field(1, ge=1)
    max_gross_leverage: float = 2.0
    costs: CostConfig = CostConfig()


class CapacityConfig(Strict):
    impact_coef: float = 0.1
    adv_window_days: int = 21
    aum_grid: list[float] = Field(default_factory=lambda: [1e6, 1e7, 1e8, 1e9])


class WalkForwardConfig(Strict):
    formation_days: int = 504
    trading_days: int = 126
    step_days: int = 126


class ValidationConfig(Strict):
    walkforward: WalkForwardConfig = WalkForwardConfig()
    n_trials: int = 1
    regimes: dict[str, tuple[str, str]] = Field(default_factory=dict)


class Config(Strict):
    data: DataConfig
    pairs: PairsConfig = PairsConfig()
    hedge: HedgeConfig = HedgeConfig()
    signals: SignalConfig = SignalConfig()
    backtest: BacktestConfig = BacktestConfig()
    capacity: CapacityConfig = CapacityConfig()
    validation: ValidationConfig = ValidationConfig()
    output_dir: Path = Path("results")


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
