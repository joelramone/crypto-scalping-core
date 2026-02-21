import os
import sys
import time
from collections import deque
from datetime import datetime

from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL


SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
CANDLE_LIMIT = 100
BREAKOUT_LOOKBACK = 20
CAPITAL_USDT = 100.0
RISK_PER_TRADE_PCT = 0.01
STOP_PCT = 0.003
TAKE_PROFIT_PCT = 0.006
MAX_TRADES_PER_HOUR = 3
MAX_CONSECUTIVE_LOSSES = 3
POLL_SECONDS = 10


api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")

if not api_key or not api_secret:
    print("Missing API keys. Set BINANCE_API_KEY and BINANCE_SECRET_KEY.")
    sys.exit(1)

client = Client(api_key, api_secret)


symbol_info = client.futures_exchange_info()
price_tick = 0.0
qty_step = 0.0
min_qty = 0.0

for s in symbol_info["symbols"]:
    if s["symbol"] == SYMBOL:
        for f in s["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                price_tick = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                qty_step = float(f["stepSize"])
                min_qty = float(f["minQty"])
        break

if qty_step == 0.0 or price_tick == 0.0:
    print(f"Could not load symbol precision for {SYMBOL}")
    sys.exit(1)


def floor_to_step(value: float, step: float) -> float:
    return (value // step) * step


def round_to_tick(value: float, tick: float) -> float:
    return round(round(value / tick) * tick, 8)


def get_closed_candle_data():
    klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=CANDLE_LIMIT)
    if len(klines) < BREAKOUT_LOOKBACK + 2:
        return None

    closed = klines[-2]
    history = klines[-(BREAKOUT_LOOKBACK + 2):-2]

    close_price = float(closed[4])
    close_volume = float(closed[5])
    close_time = int(closed[6])

    highest_high = max(float(c[2]) for c in history)
    avg_volume = sum(float(c[5]) for c in history) / len(history)

    return {
        "close_time": close_time,
        "close_price": close_price,
        "close_volume": close_volume,
        "highest_high": highest_high,
        "avg_volume": avg_volume,
    }


def has_open_position() -> bool:
    positions = client.futures_position_information(symbol=SYMBOL)
    if not positions:
        return False
    amt = float(positions[0]["positionAmt"])
    return abs(amt) > 0


def get_position_amt() -> float:
    positions = client.futures_position_information(symbol=SYMBOL)
    if not positions:
        return 0.0
    return float(positions[0]["positionAmt"])


def get_last_price() -> float:
    ticker = client.futures_symbol_ticker(symbol=SYMBOL)
    return float(ticker["price"])


def place_trade():
    entry_price = get_last_price()
    stop_price = entry_price * (1 - STOP_PCT)
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)

    risk_usdt = CAPITAL_USDT * RISK_PER_TRADE_PCT
    stop_distance = entry_price - stop_price

    if stop_distance <= 0:
        print("Invalid stop distance, skip trade.")
        return None

    raw_qty = risk_usdt / stop_distance
    qty = floor_to_step(raw_qty, qty_step)

    if qty < min_qty:
        print(f"Calculated qty {qty} is below minQty {min_qty}, skip trade.")
        return None

    stop_price = round_to_tick(stop_price, price_tick)
    take_profit_price = round_to_tick(take_profit_price, price_tick)

    print(
        f"ENTRY SIGNAL -> entry={entry_price:.2f} qty={qty} stop={stop_price:.2f} tp={take_profit_price:.2f}"
    )

    entry_order = client.futures_create_order(
        symbol=SYMBOL,
        side=SIDE_BUY,
        type="MARKET",
        quantity=qty,
    )

    client.futures_create_order(
        symbol=SYMBOL,
        side=SIDE_SELL,
        type="STOP_MARKET",
        stopPrice=stop_price,
        closePosition=True,
        workingType="MARK_PRICE",
        timeInForce="GTC",
    )

    client.futures_create_order(
        symbol=SYMBOL,
        side=SIDE_SELL,
        type="TAKE_PROFIT_MARKET",
        stopPrice=take_profit_price,
        closePosition=True,
        workingType="MARK_PRICE",
        timeInForce="GTC",
    )

    print("Orders placed: MARKET + STOP_MARKET + TAKE_PROFIT_MARKET")

    return {
        "entry_order": entry_order,
        "entry_time_ms": int(time.time() * 1000),
        "qty": qty,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "tp_price": take_profit_price,
    }


def get_realized_pnl_since(start_ms: int) -> float:
    incomes = client.futures_income_history(
        symbol=SYMBOL,
        incomeType="REALIZED_PNL",
        startTime=start_ms,
        limit=50,
    )
    pnl = sum(float(i["income"]) for i in incomes)
    return pnl


print("Starting BTCUSDT 1m breakout scalping bot...")
print(
    f"Capital={CAPITAL_USDT} USDT | Risk/trade={CAPITAL_USDT * RISK_PER_TRADE_PCT:.2f} USDT | "
    f"SL={STOP_PCT * 100:.2f}% | TP={TAKE_PROFIT_PCT * 100:.2f}%"
)

last_processed_close_time = None
trade_times = deque()
consecutive_losses = 0
active_trade = None

while True:
    try:
        now_ms = int(time.time() * 1000)

        while trade_times and now_ms - trade_times[0] > 3600 * 1000:
            trade_times.popleft()

        if active_trade:
            if not has_open_position():
                pnl = get_realized_pnl_since(active_trade["entry_time_ms"])
                print(f"TRADE CLOSED -> PnL: {pnl:.4f} USDT")

                if pnl < 0:
                    consecutive_losses += 1
                    print(f"Consecutive losses: {consecutive_losses}")
                else:
                    consecutive_losses = 0

                active_trade = None

                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    print("3 consecutive losses reached. Shutting down.")
                    break
            else:
                print("Position still open. Waiting...")
                time.sleep(POLL_SECONDS)
                continue

        if len(trade_times) >= MAX_TRADES_PER_HOUR:
            print("Trade limit reached (3 per hour). Waiting...")
            time.sleep(POLL_SECONDS)
            continue

        candle = get_closed_candle_data()
        if candle is None:
            print("Not enough candle data yet.")
            time.sleep(POLL_SECONDS)
            continue

        if candle["close_time"] == last_processed_close_time:
            print("No new closed candle.")
            time.sleep(POLL_SECONDS)
            continue

        last_processed_close_time = candle["close_time"]

        breakout = candle["close_price"] > candle["highest_high"]
        vol_ok = candle["close_volume"] > candle["avg_volume"]

        candle_ts = datetime.utcfromtimestamp(candle["close_time"] / 1000).isoformat()
        print(
            f"{candle_ts} | close={candle['close_price']:.2f} high20={candle['highest_high']:.2f} "
            f"vol={candle['close_volume']:.2f} avg20vol={candle['avg_volume']:.2f}"
        )

        if breakout and vol_ok and not has_open_position():
            print("Signal confirmed: breakout + volume.")
            trade = place_trade()
            if trade:
                trade_times.append(int(time.time() * 1000))
                active_trade = trade
                print(f"Trades this hour: {len(trade_times)}/{MAX_TRADES_PER_HOUR}")
        else:
            print("No trade: conditions not met.")

        time.sleep(POLL_SECONDS)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(POLL_SECONDS)
