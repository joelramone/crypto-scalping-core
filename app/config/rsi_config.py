from pydantic import BaseModel, Field


class RSIStrategyConfig(BaseModel):
    enable_shorts: bool = False
    signal_timeframe: str = "5m"
    fee_rate: float = Field(default=0.0004, ge=0)
    fee_multiple_threshold: float = Field(default=3.0, ge=0)
    min_expected_move_pct: float = Field(default=0.006, ge=0)
    ema_period: int = Field(default=34, ge=2)
    extension_atr_multiplier: float = Field(default=0.35, ge=0)
    volume_multiplier: float = Field(default=1.3, gt=0)
    min_atr_ratio: float = Field(default=1.10, ge=0)
    rsi_oversold: float = Field(default=28.0, ge=0, le=100)
    rsi_overbought: float = Field(default=72.0, ge=0, le=100)
    atr_sl_multiplier: float = Field(default=1.4, gt=0)
    atr_tp_multiplier: float = Field(default=2.8, gt=0)
