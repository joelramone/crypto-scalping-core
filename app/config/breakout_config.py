from dataclasses import dataclass


@dataclass
class BreakoutStrategyConfig:
    lookback_period: int = 20
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    breakout_buffer_atr: float = 0.05
    min_take_profit_r: float = 1.5
    trailing_stop_enabled: bool = False
    trailing_stop_atr_multiplier: float = 1.0
