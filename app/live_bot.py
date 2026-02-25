import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

from binance.client import Client
from binance.exceptions import BinanceAPIException


PAPER_MODE = True
USE_TESTNET = True
VALIDATION_MODE = True

SYMBOL = "BTCUSDT"
BASE_ASSET = "BTC"
QUOTE_ASSET = "USDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
CANDLE_LIMIT = 100
BREAKOUT_LOOKBACK = 10 if VALIDATION_MODE else 20
VOLUME_LOOKBACK = 10 if VALIDATION_MODE else 20
CAPITAL_USDT = 100.0
RISK_PER_TRADE_PCT = 0.01
STOP_PCT = 0.003
TAKE_PROFIT_PCT = 0.006
MAX_TRADES_PER_HOUR = 3
MAX_CONSECUTIVE_LOSSES = 3
POLL_SECONDS = 10
MIN_BTC_POSITION = 0.0001
TAKER_FEE_RATE = 0.001
MAKER_FEE_RATE = 0.001
SIMULATED_SLIPPAGE = 0.0002


api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")

if not api_key or not api_secret:
    print("Missing API keys. Set BINANCE_API_KEY and BINANCE_SECRET_KEY.")
    sys.exit(1)

if USE_TESTNET:
    client = Client(api_key, api_secret)
    client.API_URL = "https://testnet.binance.vision/api"
else:
    client = Client(api_key, api_secret)


def get_mode_label() -> str:
    if PAPER_MODE:
        return "paper"
    if USE_TESTNET:
        return "testnet"
    return "real"


def call_api(label, fn, *args, **kwargs):
    start = time.time()
    try:
        res = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[api-ok] {label} {elapsed:.2f}s")
        return res
    except Exception as e:
        elapsed = time.time() - start
        print(f"[api-err] {label} {elapsed:.2f}s {type(e).__name__}: {e}")
        raise


def sync_time_offset():
    server_ms = call_api("get_server_time", client.get_server_time)["serverTime"]
    local_ms = int(time.time() * 1000)
    offset = server_ms - local_ms
    client.timestamp_offset = offset
    print(f"Binance time offset synced: {offset} ms")


def call_with_time_sync(label, fn, *args, **kwargs):
    try:
        return call_api(label, fn, *args, **kwargs)
    except BinanceAPIException as e:
        if e.code == -1021:
            print("Received -1021 timestamp error. Resyncing time and retrying once...")
            sync_time_offset()
            return call_api(f"{label} (retry)", fn, *args, **kwargs)
        raise


def d(v) -> Decimal:
    return Decimal(str(v))


def round_down_to_step(value: float, step: float) -> float:
    value_d = d(value)
    step_d = d(step)
    rounded = (value_d / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
    return float(rounded)


symbol_info = call_api("get_symbol_info", client.get_symbol_info, SYMBOL)
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


paper_usdt_balance = CAPITAL_USDT


def get_balances(active_trade=None) -> tuple[float, float]:
    if PAPER_MODE:
        btc_balance = active_trade["position_size"] if active_trade else 0.0
        return paper_usdt_balance, btc_balance

    usdt_balance = float(
        call_with_time_sync("get_asset_balance(USDT)", client.get_asset_balance, asset=QUOTE_ASSET)["free"]
    )
    btc_balance = float(
        call_with_time_sync("get_asset_balance(BTC)", client.get_asset_balance, asset=BASE_ASSET)["free"]
    )
    return usdt_balance, btc_balance


def get_open_orders_count() -> int:
    if PAPER_MODE:
        return 0
    return len(call_with_time_sync("get_open_orders", client.get_open_orders, symbol=SYMBOL))


def is_in_position(active_trade=None) -> bool:
    if PAPER_MODE:
        return active_trade is not None

    _, btc_free = get_balances()
    open_orders_count = get_open_orders_count()
    return btc_free > MIN_BTC_POSITION or open_orders_count > 0


def get_closed_candle_data():
    max_lookback = max(BREAKOUT_LOOKBACK, VOLUME_LOOKBACK)
    klines = call_with_time_sync(
        "get_klines", client.get_klines, symbol=SYMBOL, interval=INTERVAL, limit=CANDLE_LIMIT
    )
    if len(klines) < max_lookback + 2:
        return None

    closed = klines[-2]
    breakout_history = klines[-(BREAKOUT_LOOKBACK + 2):-2]
    volume_history = klines[-(VOLUME_LOOKBACK + 2):-2]

    close_price = float(closed[4])
    close_volume = float(closed[5])
    close_time = int(closed[6])

    highest_high = max(float(c[2]) for c in breakout_history)
    avg_volume = sum(float(c[5]) for c in volume_history) / len(volume_history)

    return {
        "close_time": close_time,
        "close_price": close_price,
        "close_volume": close_volume,
        "highest_high": highest_high,
        "avg_volume": avg_volume,
    }


def get_last_price() -> float:
    ticker = call_with_time_sync("get_symbol_ticker", client.get_symbol_ticker, symbol=SYMBOL)
    return float(ticker["price"])


def print_heartbeat(last_closed_candle_ms, trades_this_hour: int, losses_in_row: int, active_trade=None):
    utc_now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if last_closed_candle_ms is None:
        last_closed = "none"
    else:
        last_closed = datetime.fromtimestamp(
            last_closed_candle_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    usdt_free, btc_free = get_balances(active_trade)
    in_position = is_in_position(active_trade)

    print(
        f"[heartbeat] {utc_now_iso} last_closed={last_closed} mode={get_mode_label()} "
        f"usdt_balance={usdt_free:.4f} btc_balance={btc_free:.6f} in_position={in_position} "
        f"trades_this_hour={trades_this_hour} consecutive_losses={losses_in_row}"
    )


def place_trade():
    global paper_usdt_balance

    usdt_before, _ = get_balances()
    close_price = get_last_price()
    entry_price = close_price * (1 + SIMULATED_SLIPPAGE)
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

    if PAPER_MODE:
        position_size_usdt = qty * entry_price
        entry_fee = position_size_usdt * TAKER_FEE_RATE

        if position_size_usdt + entry_fee > paper_usdt_balance:
            print("Insufficient paper USDT balance, skip trade.")
            return None

        paper_usdt_balance -= entry_fee
        print(
            f"[PAPER ENTRY] close={close_price:.2f} slipped_entry={entry_price:.2f} qty={qty:.6f} "
            f"position_size_usdt={position_size_usdt:.4f} entry_fee={entry_fee:.4f} "
            f"stop={stop_price:.2f} tp={take_profit_price:.2f} paper_usdt_balance={paper_usdt_balance:.4f}"
        )

        return {
            "entry_time_ms": int(time.time() * 1000),
            "position_size": qty,
            "position_size_usdt": position_size_usdt,
            "entry_price": entry_price,
            "entry_fee": entry_fee,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "usdt_before": usdt_before,
        }

    entry_order = call_with_time_sync(
        "order_market_buy", client.order_market_buy, symbol=SYMBOL, quantity=qty
    )

    call_with_time_sync(
        "create_oco_order",
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
if VALIDATION_MODE:
    print("VALIDATION_MODE enabled: using shorter lookbacks and relaxed volume threshold.")
sync_time_offset()
print(
    f"Mode={get_mode_label()} | Capital={CAPITAL_USDT} USDT | "
    f"Risk/trade={CAPITAL_USDT * RISK_PER_TRADE_PCT:.2f} USDT | "
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
            if PAPER_MODE:
                last_price = get_last_price()
                exit_reason = None
                exit_price = None

                if last_price <= active_trade["stop_price"]:
                    exit_reason = "LOSS"
                    exit_price = active_trade["stop_price"] * (1 - SIMULATED_SLIPPAGE)
                elif last_price >= active_trade["take_profit_price"]:
                    exit_reason = "WIN"
                    exit_price = active_trade["take_profit_price"] * (1 - SIMULATED_SLIPPAGE)

                if exit_reason:
                    gross_pnl = active_trade["position_size"] * (exit_price - active_trade["entry_price"])
                    exit_fee = active_trade["position_size_usdt"] * TAKER_FEE_RATE
                    net_pnl = gross_pnl - active_trade["entry_fee"] - exit_fee
                    paper_usdt_balance += net_pnl
                    total_fees = active_trade["entry_fee"] + exit_fee

                    if exit_reason == "LOSS":
                        consecutive_losses += 1
                        print(
                            f"[PAPER EXIT LOSS] exit={exit_price:.2f} qty={active_trade['position_size']:.6f} "
                            f"gross_pnl={gross_pnl:.4f} fees={total_fees:.4f} "
                            f"net_pnl={net_pnl:.4f} new_balance={paper_usdt_balance:.4f}"
                        )
                    else:
                        consecutive_losses = 0
                        print(
                            f"[PAPER EXIT WIN] exit={exit_price:.2f} qty={active_trade['position_size']:.6f} "
                            f"gross_pnl={gross_pnl:.4f} fees={total_fees:.4f} "
                            f"net_pnl={net_pnl:.4f} new_balance={paper_usdt_balance:.4f}"
                        )

                    active_trade = None

                    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                        print("3 consecutive losses reached. Shutting down.")
                        break
                else:
                    print(
                        f"Paper trade still active. price={last_price:.2f} "
                        f"stop={active_trade['stop_price']:.2f} tp={active_trade['take_profit_price']:.2f}"
                    )
                    print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses, active_trade)
                    time.sleep(POLL_SECONDS)
                    continue
            else:
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
                    print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses, active_trade)
                    time.sleep(POLL_SECONDS)
                    continue

        if len(trade_times) >= MAX_TRADES_PER_HOUR:
            print("Trade limit reached (3 per hour). Waiting...")
            print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses, active_trade)
            time.sleep(POLL_SECONDS)
            continue

        candle = get_closed_candle_data()
        if candle is None:
            print("Not enough candle data yet.")
            print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses, active_trade)
            time.sleep(POLL_SECONDS)
            continue

        if candle["close_time"] == last_processed_close_time:
            print("No new closed candle.")
            print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses, active_trade)
            time.sleep(POLL_SECONDS)
            continue

        last_processed_close_time = candle["close_time"]

        breakout = candle["close_price"] > candle["highest_high"]
        vol_ok = candle["close_volume"] >= candle["avg_volume"] * 1.0

        candle_ts = datetime.fromtimestamp(
            candle["close_time"] / 1000, tz=timezone.utc
        ).isoformat()
        print(
            f"{candle_ts} | close={candle['close_price']:.2f} high{BREAKOUT_LOOKBACK}={candle['highest_high']:.2f} "
            f"vol={candle['close_volume']:.2f} avgVol{VOLUME_LOOKBACK}={candle['avg_volume']:.2f}"
        )

        if breakout and vol_ok and not is_in_position(active_trade):
            print("Signal confirmed: breakout + volume.")
            trade = place_trade()
            if trade:
                trade_times.append(int(time.time() * 1000))
                active_trade = trade
                print(f"Trades this hour: {len(trade_times)}/{MAX_TRADES_PER_HOUR}")
        else:
            print("No trade: conditions not met or already in position.")

        print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses, active_trade)
        time.sleep(POLL_SECONDS)

    except Exception as e:
        print(f"Error: {e}")
        print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses, active_trade)
        time.sleep(POLL_SECONDS)
