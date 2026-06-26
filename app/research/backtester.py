from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from types import SimpleNamespace
from app.research.config import ResearchBacktestConfig
from app.research.features import add_basic_features
from app.research.results import BacktestSummary, TradeResult, summarize_trades
from app.strategies.breakout_trend import BreakoutTrendStrategy
from app.strategies.rsi_mean_reversion import RSIMeanReversionStrategy


class HistoricalBacktester:
    """Small V2 research backtester for fast strategy iteration on CSV candles."""

    def __init__(self, config: ResearchBacktestConfig) -> None:
        self.config = config
        self.strategy = self._build_strategy(config.strategy)

    def run(self) -> BacktestSummary:
        candles = add_basic_features(load_ohlcv_csv(self.config.data_path, self.config.timestamp_column))
        trades: list[TradeResult] = []
        index = max(self.config.warmup_bars, 1)
        while index < len(candles) - 1:
            signal = self._generate_signal(candles.iloc[: index + 1])
            if signal is None:
                index += 1
                continue
            trade = self._simulate_trade(candles, index, signal)
            if trade is None:
                index += 1
                continue
            trades.append(trade)
            index = max(trade.exit_index + 1, index + 1)
        return summarize_trades(trades)

    def _generate_signal(self, history: pd.DataFrame) -> dict[str, Any] | None:
        data = {column: history[column].dropna().tolist() for column in history.columns}
        return self.strategy.generate_signal(data, regime=None)

    def _simulate_trade(self, candles: pd.DataFrame, entry_index: int, signal: dict[str, Any]) -> TradeResult | None:
        entry_price = float(signal.get("entry", candles.iloc[entry_index]["close"]))
        stop_loss = float(signal["sl"])
        take_profit = float(signal["tp"])
        side = str(signal["side"])
        quantity = self.config.trade_size_usdt / entry_price
        last_index = min(entry_index + self.config.max_holding_bars, len(candles) - 1)

        for exit_index in range(entry_index + 1, last_index + 1):
            candle = candles.iloc[exit_index]
            low = float(candle["low"])
            high = float(candle["high"])
            if side == "LONG":
                if low <= stop_loss:
                    return self._build_trade(entry_index, exit_index, side, entry_price, stop_loss, quantity, "stop_loss", signal)
                if high >= take_profit:
                    return self._build_trade(entry_index, exit_index, side, entry_price, take_profit, quantity, "take_profit", signal)
            else:
                if high >= stop_loss:
                    return self._build_trade(entry_index, exit_index, side, entry_price, stop_loss, quantity, "stop_loss", signal)
                if low <= take_profit:
                    return self._build_trade(entry_index, exit_index, side, entry_price, take_profit, quantity, "take_profit", signal)

        exit_price = float(candles.iloc[last_index]["close"])
        return self._build_trade(entry_index, last_index, side, entry_price, exit_price, quantity, "time_exit", signal)

    def _build_trade(
        self,
        entry_index: int,
        exit_index: int,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        exit_reason: str,
        signal: dict[str, Any],
    ) -> TradeResult:
        gross_pnl = (exit_price - entry_price) * quantity if side == "LONG" else (entry_price - exit_price) * quantity
        fees = (entry_price * quantity * self.config.fee_rate) + (exit_price * quantity * self.config.fee_rate)
        return TradeResult(
            entry_index=entry_index,
            exit_index=exit_index,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=gross_pnl,
            fees=fees,
            net_pnl=gross_pnl - fees,
            exit_reason=exit_reason,
            strategy_name=str(signal.get("strategy_name", "unknown")),
        )

    def _build_strategy(self, name: str) -> Any:
        common_config = {
            "enable_shorts": False,
            "signal_timeframe": "research",
            "fee_rate": self.config.fee_rate,
            "fee_multiple_threshold": 3.0,
            "min_expected_move_pct": 0.006,
            "ema_period": 34,
            "extension_atr_multiplier": 0.35,
            "volume_multiplier": 1.3,
            "min_atr_ratio": 1.10,
            "rsi_oversold": 28.0,
            "rsi_overbought": 72.0,
            "atr_sl_multiplier": 1.4,
            "atr_tp_multiplier": 2.8,
        }
        if name == "rsi":
            return RSIMeanReversionStrategy(SimpleNamespace(**common_config))

        breakout_config = {
            **common_config,
            "rsi_period": 10,
            "volume_avg_period": 20,
            "exhaustion_wick_ratio_min": 0.45,
            "min_body_ratio": 0.20,
            "min_take_profit_r": 2.0,
            "trailing_stop_enabled": False,
            "trailing_stop_atr_multiplier": 1.0,
            "atr_avg_period": 20,
        }
        return BreakoutTrendStrategy(SimpleNamespace(**breakout_config))


def load_ohlcv_csv(path: Path, timestamp_column: str = "timestamp") -> pd.DataFrame:
    candles = pd.read_csv(path)
    if timestamp_column in candles.columns:
        candles[timestamp_column] = pd.to_datetime(candles[timestamp_column], errors="coerce")
        candles = candles.sort_values(timestamp_column).reset_index(drop=True)
    return candles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V2 research backtester on OHLCV CSV data.")
    parser.add_argument("--data", required=True, type=Path, help="Path to CSV with open/high/low/close/volume columns.")
    parser.add_argument("--strategy", choices=["breakout", "rsi"], default="breakout")
    parser.add_argument("--trade-size-usdt", type=float, default=100.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--max-holding-bars", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ResearchBacktestConfig(
        data_path=args.data,
        strategy=args.strategy,
        trade_size_usdt=args.trade_size_usdt,
        fee_rate=args.fee_rate,
        max_holding_bars=args.max_holding_bars,
    )
    summary = HistoricalBacktester(config).run()
    print(summary.model_dump_json(indent=2, exclude={"trades"}))


if __name__ == "__main__":
    main()
