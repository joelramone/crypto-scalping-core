from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.data.features import FeatureSnapshot

Signal = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class StrategyDecision:
    symbol: str
    signal: Signal
    confidence: float


class StrategyAgent:
    def decide(self, features: FeatureSnapshot) -> StrategyDecision:
        if features.momentum > 0.001:
            return StrategyDecision(symbol=features.symbol, signal="buy", confidence=0.6)
        if features.momentum < -0.001:
            return StrategyDecision(symbol=features.symbol, signal="sell", confidence=0.6)
        return StrategyDecision(symbol=features.symbol, signal="hold", confidence=0.2)
