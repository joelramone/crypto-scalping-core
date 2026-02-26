import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

from binance.client import Client
from binance.exceptions import BinanceAPIException


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


PAPER_MODE = env_bool("PAPER_MODE", True)
USE_TESTNET = env_bool("USE_TESTNET", True)
VALIDATION_MODE = env_bool("VALIDATION_MODE", True)

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
CANDLE_LIMIT = 100
BREAKOUT_LOOKBACK = 10 if VALIDATION_MODE else 20
VOLUME_LOOKBACK = 10 if VALIDATION_MODE else 20
CAPITAL_USDT = float(os.getenv("CAPITAL_USDT", "100"))
risk_per_trade_usdt = float(os.getenv("RISK_PER_TRADE_USDT", "1.0"))
STOP_PCT = float(os.getenv("STOP_PCT", "0.003"))
TP_PCT = float(os.getenv("TP_PCT", "0.006"))
MAX_TRADES_PER_HOUR = 3
MAX_CONSECUTIVE_LOSSES = 3
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
SIMULATED_SLIPPAGE = float(os.getenv("SIMULATED_SLIPPAGE", "0.0002"))
FUTURES_LEVERAGE = int(os.getenv("FUTURES_LEVERAGE", "3"))
FUTURES_TAKER_FEE_RATE = float(os.getenv("FUTURES_TAKER_FEE_RATE", "0.0004"))
MARGIN_TYPE = "ISOLATED"
WORKING_TYPE = os.getenv("FUTURES_WORKING_TYPE", "MARK_PRICE")

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")

if not api_key or not api_secret:
    print("Missing API keys. Set BINANCE_API_KEY and BINANCE_SECRET_KEY.")
    sys.exit(1)

client = Client(api_key, api_secret)
if USE_TESTNET:
    client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"


def get_mode_label() -> str:
    if PAPER_MODE:
        return "paper"
    if USE_TESTNET:
        return "futures_testnet"
    return "futures_real"


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
    if step <= 0:
        return value
    value_d = d(value)
    step_d = d(step)
    rounded = (value_d / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
    return float(rounded)


exchange_info = call_with_time_sync("futures_exchange_info", client.futures_exchange_info)
symbol_info = next((s for s in exchange_info["symbols"] if s["symbol"] == SYMBOL), None)
if not symbol_info:
    print(f"Could not load futures symbol info for {SYMBOL}")
    sys.exit(1)

price_tick = 0.0
qty_step = 0.0
min_qty = 0.0
min_notional = 0.0

for flt in symbol_info["filters"]:
    if flt["filterType"] == "PRICE_FILTER":
        price_tick = float(flt["tickSize"])
    elif flt["filterType"] == "LOT_SIZE":
        qty_step = float(flt["stepSize"])
        min_qty = float(flt["minQty"])
    elif flt["filterType"] == "MIN_NOTIONAL":
        min_notional = float(flt.get("notional", 0.0))

if qty_step == 0.0 or price_tick == 0.0:
    print(f"Could not load futures precision for {SYMBOL}")
    sys.exit(1)

paper_usdt_balance = CAPITAL_USDT


def ensure_futures_settings():
    if PAPER_MODE:
        return

    try:
        call_with_time_sync(
            "futures_change_margin_type",
            client.futures_change_margin_type,
            symbol=SYMBOL,
            marginType=MARGIN_TYPE,
        )
        print(f"Margin type set to {MARGIN_TYPE} for {SYMBOL}")
    except BinanceAPIException as e:
        if e.code == -4046:
            print(f"Margin type already {MARGIN_TYPE} for {SYMBOL}")
        else:
            raise

    call_with_time_sync(
        "futures_change_leverage",
        client.futures_change_leverage,
        symbol=SYMBOL,
        leverage=FUTURES_LEVERAGE,
    )
    print(f"Leverage set to {FUTURES_LEVERAGE}x for {SYMBOL}")


def get_usdt_balance() -> float:
    if PAPER_MODE:
        return paper_usdt_balance

    balances = call_with_time_sync("futures_account_balance", client.futures_account_balance)
    usdt = next((b for b in balances if b.get("asset") == "USDT"), None)
    if not usdt:
        return 0.0
    return float(usdt["balance"])


def get_position_amt() -> float:
    if PAPER_MODE:
        return 0.0
    positions = call_with_time_sync(
        "futures_position_information", client.futures_position_information, symbol=SYMBOL
    )
    if not positions:
        return 0.0
    return float(positions[0]["positionAmt"])


def get_open_orders_count() -> int:
    if PAPER_MODE:
        return 0
    return len(call_with_time_sync("futures_get_open_orders", client.futures_get_open_orders, symbol=SYMBOL))


def is_in_position(active_trade=None) -> bool:
    if PAPER_MODE:
        return active_trade is not None
    return get_position_amt() > 0


def get_closed_candle_data():
    max_lookback = max(BREAKOUT_LOOKBACK, VOLUME_LOOKBACK)
    klines = call_with_time_sync(
        "futures_klines", client.futures_klines, symbol=SYMBOL, interval=INTERVAL, limit=CANDLE_LIMIT
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
    ticker = call_with_time_sync("futures_symbol_ticker", client.futures_symbol_ticker, symbol=SYMBOL)
    return float(ticker["price"])


def print_heartbeat(last_closed_candle_ms, trades_this_hour: int, losses_in_row: int, active_trade=None):
    utc_now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if last_closed_candle_ms is None:
        last_closed = "none"
    else:
        last_closed = datetime.fromtimestamp(
            last_closed_candle_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    usdt_balance = get_usdt_balance()
    pos_amt = active_trade["position_size"] if PAPER_MODE and active_trade else get_position_amt()
    in_position = is_in_position(active_trade)

    print(
        f"[heartbeat] {utc_now_iso} last_closed={last_closed} mode={get_mode_label()} "
        f"margin={MARGIN_TYPE} leverage={FUTURES_LEVERAGE}x usdt_balance={usdt_balance:.4f} "
        f"position_amt={pos_amt:.6f} in_position={in_position} "
        f"trades_this_hour={trades_this_hour} consecutive_losses={losses_in_row}"
    )


def place_trade():
    global paper_usdt_balance

    usdt_before = get_usdt_balance()
    close_price = get_last_price()
    entry_price = close_price * (1 + SIMULATED_SLIPPAGE)
    stop_price = entry_price * (1 - STOP_PCT)
    take_profit_price = entry_price * (1 + TP_PCT)

    if STOP_PCT <= 0:
        print("STOP_PCT must be > 0.")
        return None

    position_notional_usdt = risk_per_trade_usdt / STOP_PCT
    raw_qty = position_notional_usdt / entry_price
    qty = round_down_to_step(raw_qty, qty_step)

    stop_price = round_down_to_step(stop_price, price_tick)
    take_profit_price = round_down_to_step(take_profit_price, price_tick)

    if qty < min_qty:
        print(f"Calculated qty {qty} is below minQty {min_qty}, skip trade.")
        return None

    if min_notional and qty * entry_price < min_notional:
        print(f"Calculated notional {qty * entry_price:.4f} below minNotional {min_notional:.4f}, skip trade.")
        return None

    margin_required = (qty * entry_price) / max(FUTURES_LEVERAGE, 1)
    if margin_required > usdt_before:
        max_qty = round_down_to_step((usdt_before * FUTURES_LEVERAGE) / entry_price, qty_step)
        if max_qty < min_qty:
            print("Insufficient USDT margin for minimum quantity, skip trade.")
            return None
        qty = max_qty

    notional = qty * entry_price
    print(
        f"ENTRY SIGNAL -> entry={entry_price:.2f} stop={stop_price:.2f} tp={take_profit_price:.2f} "
        f"qty={qty:.6f} notional={notional:.2f}"
    )

    if PAPER_MODE:
        entry_fee = notional * FUTURES_TAKER_FEE_RATE
        if entry_fee > paper_usdt_balance:
            print("Insufficient paper USDT balance for entry fee, skip trade.")
            return None

        paper_usdt_balance -= entry_fee
        print(
            f"[PAPER ENTRY] close={close_price:.2f} slipped_entry={entry_price:.2f} qty={qty:.6f} "
            f"notional={notional:.4f} entry_fee={entry_fee:.4f} stop={stop_price:.2f} tp={take_profit_price:.2f} "
            f"paper_usdt_balance={paper_usdt_balance:.4f}"
        )

        return {
            "entry_time_ms": int(time.time() * 1000),
            "position_size": qty,
            "position_notional": notional,
            "entry_price": entry_price,
            "entry_fee": entry_fee,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "usdt_before": usdt_before,
        }

    entry_order = call_with_time_sync(
        "futures_create_order(MARKET)",
        client.futures_create_order,
        symbol=SYMBOL,
        side="BUY",
        type="MARKET",
        quantity=qty,
    )

    call_with_time_sync(
        "futures_create_order(STOP_MARKET)",
        client.futures_create_order,
        symbol=SYMBOL,
        side="SELL",
        type="STOP_MARKET",
        stopPrice=f"{stop_price:.8f}",
        quantity=qty,
        reduceOnly=True,
        workingType=WORKING_TYPE,
    )

    call_with_time_sync(
        "futures_create_order(TAKE_PROFIT_MARKET)",
        client.futures_create_order,
        symbol=SYMBOL,
        side="SELL",
        type="TAKE_PROFIT_MARKET",
        stopPrice=f"{take_profit_price:.8f}",
        quantity=qty,
        reduceOnly=True,
        workingType=WORKING_TYPE,
    )

    print("Orders placed: MARKET BUY + STOP_MARKET + TAKE_PROFIT_MARKET")

    return {
        "entry_order": entry_order,
        "entry_time_ms": int(time.time() * 1000),
        "qty": qty,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "tp_price": take_profit_price,
        "usdt_before": usdt_before,
    }


print("Starting BTCUSDT 1m breakout scalping bot (BINANCE FUTURES USDT-M)...")
if VALIDATION_MODE:
    print("VALIDATION_MODE enabled: using shorter lookbacks and relaxed volume threshold.")
sync_time_offset()
ensure_futures_settings()
print(
    f"Mode={get_mode_label()} | Capital={CAPITAL_USDT} USDT | Risk/trade={risk_per_trade_usdt:.2f} USDT | "
    f"SL={STOP_PCT * 100:.2f}% | TP={TP_PCT * 100:.2f}% | leverage={FUTURES_LEVERAGE}x | margin={MARGIN_TYPE}"
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
                    exit_fee = active_trade["position_notional"] * FUTURES_TAKER_FEE_RATE
                    total_fees = active_trade["entry_fee"] + exit_fee
                    net_pnl = gross_pnl - total_fees
                    paper_usdt_balance += net_pnl

                    if exit_reason == "LOSS":
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                    print(
                        f"[PAPER EXIT {exit_reason}] exit={exit_price:.2f} qty={active_trade['position_size']:.6f} "
                        f"gross_pnl={gross_pnl:.4f} fees={total_fees:.4f} net_pnl={net_pnl:.4f} "
                        f"balance={paper_usdt_balance:.4f}"
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
                pos_amt = get_position_amt()
                if pos_amt <= 0:
                    usdt_now = get_usdt_balance()
                    gross_pnl = usdt_now - active_trade["usdt_before"]
                    fees = 0.0
                    net_pnl = gross_pnl - fees
                    print(
                        f"[FUTURES EXIT] gross_pnl={gross_pnl:.4f} fees={fees:.4f} "
                        f"net_pnl={net_pnl:.4f} balance={usdt_now:.4f}"
                    )

                    if net_pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                    active_trade = None

                    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                        print("3 consecutive losses reached. Shutting down.")
                        break
                else:
                    print("Futures trade still active. Waiting for stop/tp trigger...")
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

        candle_ts = datetime.fromtimestamp(candle["close_time"] / 1000, tz=timezone.utc).isoformat()
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
