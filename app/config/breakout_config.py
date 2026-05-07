from pydantic import BaseModel, Field


class BreakoutStrategyConfig(BaseModel):
    rsi_period: int = Field(default=7, ge=2)
    rsi_oversold: float = Field(default=20.0, ge=0, le=100)
    rsi_overbought: float = Field(default=80.0, ge=0, le=100)
    ema_period: int = Field(default=20, ge=2)
    extension_atr_multiplier: float = Field(default=0.20, ge=0)
    volume_avg_period: int = Field(default=20, ge=2)
    volume_multiplier: float = Field(default=1.2, gt=0)
    exhaustion_wick_ratio_min: float = Field(default=0.45, gt=0, lt=1)
    min_body_ratio: float = Field(default=0.20, gt=0, lt=1)
    atr_sl_multiplier: float = Field(default=1.5, gt=0)
    atr_tp_multiplier: float = Field(default=2.0, gt=0)
    min_take_profit_r: float = Field(default=1.5, gt=0)
    trailing_stop_enabled: bool = False
    trailing_stop_atr_multiplier: float = Field(default=1.0, gt=0)
    atr_avg_period: int = Field(default=20, ge=2)
    min_atr_ratio: float = Field(default=1.0, ge=0)
