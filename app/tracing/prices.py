"""Loads the price table from `config/prices.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.tracing.metrics import Price


@dataclass(frozen=True)
class PriceTable:
    prices: dict[str, Price]
    unverified: tuple[str, ...]  # model ids whose prices are not confirmed against the official page

    def get(self, model_id: str | None) -> Price | None:
        if model_id is None:
            return None
        return self.prices.get(model_id)


def load_prices(path: str | Path) -> PriceTable:
    path = Path(path)
    if not path.is_file():
        return PriceTable(prices={}, unverified=())
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("unit", "usd_per_1m_tokens") != "usd_per_1m_tokens":
        raise ValueError(f"{path}: unit must be usd_per_1m_tokens")
    prices: dict[str, Price] = {}
    unverified: list[str] = []
    for model_id, entry in (data.get("models") or {}).items():
        if entry.get("input") is None:
            continue
        prices[model_id] = Price(
            input_per_m=float(entry["input"]),
            output_per_m=float(entry["output"]) if entry.get("output") is not None else None,
        )
        if not entry.get("verified", False):
            unverified.append(model_id)
    return PriceTable(prices=prices, unverified=tuple(unverified))
