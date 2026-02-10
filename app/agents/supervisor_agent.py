from __future__ import annotations

from app.config import Settings


class SupervisorAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    def should_stop(self, pnl_today: float, trades_today: int) -> bool:
        if pnl_today <= -self.settings.max_daily_loss:
            self._stopped = True
        if pnl_today >= self.settings.daily_profit_target:
            self._stopped = True
        if trades_today >= self.settings.max_trades_per_day:
            self._stopped = True
        return self._stopped

    def stop(self) -> None:
        self._stopped = True
