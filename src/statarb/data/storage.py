"""Local parquet price store.

Deliberately simple: one parquet file per field, wide format
(index=date, columns=tickers). At 100 names x 10y daily this is a few MB —
no database needed. The interface is narrow so it can be swapped for
ClickHouse/TimescaleDB later without touching research code.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PriceStore:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, field: str) -> Path:
        return self.cache_dir / f"{field}.parquet"

    def save(self, data: dict[str, pd.DataFrame], meta: dict | None = None) -> None:
        for field, df in data.items():
            df.to_parquet(self._path(field))
        manifest = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "fields": sorted(data.keys()),
            "tickers": sorted(next(iter(data.values())).columns.tolist()),
            "start": str(next(iter(data.values())).index.min().date()),
            "end": str(next(iter(data.values())).index.max().date()),
            **(meta or {}),
        }
        (self.cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Saved %s fields to %s", sorted(data.keys()), self.cache_dir)

    def load(self, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
        fields = fields or [p.stem for p in self.cache_dir.glob("*.parquet")]
        out = {}
        for field in fields:
            path = self._path(field)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run `python scripts/01_download_data.py` first"
                )
            out[field] = pd.read_parquet(path)
        return out

    def manifest(self) -> dict:
        p = self.cache_dir / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {}
