import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

from binance.client import Client
from binance.exceptions import BinanceAPIException


SYMBOL = "BTCUSDT"
BASE_ASSET = "BTC"
QUOTE_ASSET = "USDT"
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
MIN_BTC_POSITION = 0.0001


api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")

if not api_key or not api_secret:
    print("Missing API keys. Set BINANCE_API_KEY and BINANCE_SECRET_KEY.")
    sys.exit(1)

client = Client(api_key, api_secret)


def sync_time_offset():
    server_ms = client.get_server_time()["serverTime"]
    local_ms = int(time.time() * 1000)
    offset = server_ms - local_ms
    client.timestamp_offset = offset
    print(f"Binance time offset synced: {offset} ms")


def call_with_time_sync(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except BinanceAPIException as e:
        if e.code == -1021:
            print("Received -1021 timestamp error. Resyncing time and retrying once...")
            sync_time_offset()
            return fn(*args, **kwargs)
        raise


def d(v) -> Decimal:
    return Decimal(str(v))


def round_down_to_step(value: float, step: float) -> float:
    value_d = d(value)
    step_d = d(step)
    rounded = (value_d / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
    return float(rounded)


symbol_info = client.get_symbol_info(SYMBOL)
if not symbol_info:
    print(f"Could not load symbol info for {SYMBOL}")
    sys.exit(1)

price_tick = 0.0
qty_step = 0.0
min_qty = 0.0

for flt in symbol_info["filters"]:
    if flt["filterType"] == "PRICE_FILTER":
        price_tick = float(flt["tickSize"])
    elif flt["filterType"] == "LOT_SIZE":
        qty_step = float(flt["stepSize"])
        min_qty = float(flt["minQty"])

if qty_step == 0.0 or price_tick == 0.0:
    print(f"Could not load symbol precision for {SYMBOL}")
    sys.exit(1)


def get_balances() -> tuple[float, float]:
    usdt_balance = float(
        call_with_time_sync(client.get_asset_balance, asset=QUOTE_ASSET)["free"]
    )
    btc_balance = float(
        call_with_time_sync(client.get_asset_balance, asset=BASE_ASSET)["free"]
    )
    return usdt_balance, btc_balance


def get_open_orders_count() -> int:
    return len(call_with_time_sync(client.get_open_orders, symbol=SYMBOL))


def is_in_position() -> bool:
    _, btc_free = get_balances()
    open_orders_count = get_open_orders_count()
    return btc_free > MIN_BTC_POSITION or open_orders_count > 0


def get_closed_candle_data():
    klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=CANDLE_LIMIT)
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


def get_last_price() -> float:
    ticker = client.get_symbol_ticker(symbol=SYMBOL)
    return float(ticker["price"])


def print_heartbeat(last_closed_candle_ms, trades_this_hour: int, losses_in_row: int):
    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    if last_closed_candle_ms is None:
        last_closed = "none"
    else:
        last_closed = datetime.fromtimestamp(
            last_closed_candle_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    usdt_free, btc_free = get_balances()
    open_orders_count = get_open_orders_count()
    in_position = btc_free > MIN_BTC_POSITION or open_orders_count > 0

    print(
        f"[{now_utc}] heartbeat: last_closed_candle={last_closed} "
        f"usdt_free={usdt_free:.4f} btc_free={btc_free:.6f} "
        f"in_position={in_position} open_orders={open_orders_count} "
        f"trades_this_hour={trades_this_hour} consecutive_losses={losses_in_row}"
    )


def place_trade():
    usdt_before, _ = get_balances()
    entry_price = get_last_price()
    stop_price = entry_price * (1 - STOP_PCT)
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)

    risk_usdt = CAPITAL_USDT * RISK_PER_TRADE_PCT
    stop_distance = entry_price - stop_price

    if stop_distance <= 0:
        print("Invalid stop distance, skip trade.")
        return None

    raw_qty = risk_usdt / stop_distance
    qty = round_down_to_step(raw_qty, qty_step)

    if qty < min_qty:
        print(f"Calculated qty {qty} is below minQty {min_qty}, skip trade.")
        return None

    if qty * entry_price > usdt_before:
        max_affordable_qty = round_down_to_step(usdt_before / entry_price, qty_step)
        if max_affordable_qty < min_qty:
            print("Insufficient USDT for minimum quantity, skip trade.")
            return None
        qty = max_affordable_qty

    stop_price = round_down_to_step(stop_price, price_tick)
    take_profit_price = round_down_to_step(take_profit_price, price_tick)
    stop_limit_price = round_down_to_step(stop_price * 0.999, price_tick)

    print(
        f"ENTRY SIGNAL -> entry={entry_price:.2f} qty={qty} stop={stop_price:.2f} "
        f"stop_limit={stop_limit_price:.2f} tp={take_profit_price:.2f}"
    )

    entry_order = call_with_time_sync(client.order_market_buy, symbol=SYMBOL, quantity=qty)

    call_with_time_sync(
        client.create_oco_order,
        symbol=SYMBOL,
        side="SELL",
        quantity=qty,
        price=f"{take_profit_price:.8f}",
        stopPrice=f"{stop_price:.8f}",
        stopLimitPrice=f"{stop_limit_price:.8f}",
        stopLimitTimeInForce="GTC",
    )

    print("Orders placed: MARKET BUY + OCO SELL")

    return {
        "entry_order": entry_order,
        "entry_time_ms": int(time.time() * 1000),
        "qty": qty,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "tp_price": take_profit_price,
        "usdt_before": usdt_before,
    }


print("Starting BTCUSDT 1m breakout scalping bot (SPOT)...")
sync_time_offset()
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
            usdt_now, btc_now = get_balances()
            open_orders_count = get_open_orders_count()
            trade_closed = btc_now <= MIN_BTC_POSITION and open_orders_count == 0

            if trade_closed:
                pnl = usdt_now - active_trade["usdt_before"]
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
                print("Trade still active (balance/orders). Waiting...")
                print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses)
                time.sleep(POLL_SECONDS)
                continue

        if len(trade_times) >= MAX_TRADES_PER_HOUR:
            print("Trade limit reached (3 per hour). Waiting...")
            print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses)
            time.sleep(POLL_SECONDS)
            continue

        candle = get_closed_candle_data()
        if candle is None:
            print("Not enough candle data yet.")
            print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses)
            time.sleep(POLL_SECONDS)
            continue

        if candle["close_time"] == last_processed_close_time:
            print("No new closed candle.")
            print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses)
            time.sleep(POLL_SECONDS)
            continue

        last_processed_close_time = candle["close_time"]

        breakout = candle["close_price"] > candle["highest_high"]
        vol_ok = candle["close_volume"] > candle["avg_volume"]

        candle_ts = datetime.fromtimestamp(
            candle["close_time"] / 1000, tz=timezone.utc
        ).isoformat()
        print(
            f"{candle_ts} | close={candle['close_price']:.2f} high20={candle['highest_high']:.2f} "
            f"vol={candle['close_volume']:.2f} avg20vol={candle['avg_volume']:.2f}"
        )

        if breakout and vol_ok and not is_in_position():
            print("Signal confirmed: breakout + volume.")
            trade = place_trade()
            if trade:
                trade_times.append(int(time.time() * 1000))
                active_trade = trade
                print(f"Trades this hour: {len(trade_times)}/{MAX_TRADES_PER_HOUR}")
        else:
            print("No trade: conditions not met or already in position.")

        print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses)
        time.sleep(POLL_SECONDS)

    except Exception as e:
        print(f"Error: {e}")
        print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses)
        time.sleep(POLL_SECONDS)
