from dataclasses import dataclass


@dataclass
class BreakoutStrategyConfig:
    lookback_period: int = 20
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 3.0
