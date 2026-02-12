from dataclasses import dataclass


@dataclass
class RSIStrategyConfig:
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
