import random
from typing import Dict, List

from app.agents.strategy_agent import StrategyAgent
from app.trading.paper_wallet import PaperWallet, Trade

TRADE_COMMISSION_RATE = 0.001


def _closed_trade_pnls(trades: List[Trade]) -> List[float]:
    pnls: List[float] = []
    pending_buy: Trade | None = None

    for trade in trades:
        side = str(getattr(trade, "side", "")).lower()

        if side == "buy":
            pending_buy = trade
            continue

        if side == "sell" and pending_buy is not None:
            if getattr(pending_buy, "symbol", None) != getattr(trade, "symbol", None):
                continue

            buy_price = float(getattr(pending_buy, "price", 0.0) or 0.0)
            sell_price = float(getattr(trade, "price", 0.0) or 0.0)
            quantity = float(getattr(pending_buy, "quantity", 0.0) or 0.0)
            gross_profit = (sell_price - buy_price) * quantity
            pnl = gross_profit - (buy_price * TRADE_COMMISSION_RATE)
            pnls.append(pnl)
            pending_buy = None

    return pnls


def run_single_backtest(ticks: int = 1000, initial_price: float = 50000.0) -> Dict[str, float]:
    wallet = PaperWallet()
    strategy = StrategyAgent(wallet=wallet)

    bullish_ticks = 300
    sideways_ticks = 300

    price = initial_price
    for tick in range(ticks):
        if tick < bullish_ticks:
            drift = 0.0005
        elif tick < bullish_ticks + sideways_ticks:
            drift = 0.0
        else:
            drift = -0.0005

        price *= 1 + random.uniform(-0.005, 0.005) + drift
        strategy.on_price(price)

    btc_balance = wallet.get_balance("BTC")
    if btc_balance > 0:
        wallet.sell("BTC/USDT", price, btc_balance)

    trade_pnls = _closed_trade_pnls(wallet.trades)
    total_trades = len(trade_pnls)
    total_wins = sum(1 for pnl in trade_pnls if pnl > 0)
    total_losses = sum(1 for pnl in trade_pnls if pnl < 0)
    gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
    gross_loss = sum(abs(pnl) for pnl in trade_pnls if pnl < 0)
    net_profit = sum(trade_pnls)

    return {
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
    }


def run(simulations: int = 100):
    results = [run_single_backtest() for _ in range(simulations)]

    total_runs = len(results)
    total_net_profit = sum(result["net_profit"] for result in results)
    total_gross_profit = sum(result["gross_profit"] for result in results)
    total_gross_loss = sum(result["gross_loss"] for result in results)
    total_trades_all_runs = sum(result["total_trades"] for result in results)

    average_net_profit = total_net_profit / total_runs if total_runs else 0.0
    average_win_rate = (
        sum(
            (result["total_wins"] / result["total_trades"]) * 100
            for result in results
            if result["total_trades"] > 0
        )
        / total_runs
        if total_runs
        else 0.0
    )
    average_trades_per_run = total_trades_all_runs / total_runs if total_runs else 0.0
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0.0
    expectancy_per_trade = (
        total_net_profit / total_trades_all_runs if total_trades_all_runs > 0 else 0.0
    )

    print("--- MONTE CARLO ANALYSIS ---")
    print(f"Simulations: {simulations}")
    print(f"Avg net profit: {average_net_profit:.6f}")
    print(f"Avg win rate: {average_win_rate:.2f}%")
    print(f"Avg trades per run: {average_trades_per_run:.2f}")
    print(f"Profit factor: {profit_factor:.6f}")
    print(f"Expectancy per trade: {expectancy_per_trade:.6f}")


if __name__ == "__main__":
    run()
