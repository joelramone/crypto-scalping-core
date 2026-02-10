from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import Settings
from app.data.features import FeatureSnapshot

Signal = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class StrategyDecision:
    symbol: str
    signal: Signal
    confidence: float


class StrategyAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def decide(self, features: FeatureSnapshot) -> StrategyDecision:
        if not self.settings.strategy_enabled:
            return StrategyDecision(symbol=features.symbol, signal="hold", confidence=1.0)
        return StrategyDecision(symbol=features.symbol, signal="hold", confidence=0.0)
