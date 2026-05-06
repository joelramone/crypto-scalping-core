from pydantic import BaseModel, Field


class BreakoutStrategyConfig(BaseModel):
    lookback_period: int = Field(default=20, ge=2)
    atr_sl_multiplier: float = Field(default=1.5, gt=0)
    atr_tp_multiplier: float = Field(default=2.0, gt=0)
    breakout_buffer_atr: float = Field(default=0.05, ge=0)
    min_take_profit_r: float = Field(default=1.5, gt=0)
    trailing_stop_enabled: bool = False
    trailing_stop_atr_multiplier: float = Field(default=1.0, gt=0)
    trend_ema_period: int = Field(default=200, ge=2)
    breakout_body_ratio_min: float = Field(default=0.70, gt=0, le=1)
    atr_avg_period: int = Field(default=20, ge=2)
