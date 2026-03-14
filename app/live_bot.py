import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

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
LOOKBACK = BREAKOUT_LOOKBACK
TREND_TIMEFRAME = "15m"
EMA_FAST = 50
EMA_SLOW = 200
VOLUME_MULTIPLIER = 1.8
CAPITAL_USDT = float(os.getenv("CAPITAL_USDT", "100"))
RISK_PER_TRADE_USDT = float(os.getenv("RISK_PER_TRADE_USDT", "2.0"))
STOP_PCT = float(os.getenv("STOP_PCT", "0.005"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", os.getenv("TP_PCT", "0.012")))
MAX_TRADES_PER_HOUR = 3
MAX_CONSECUTIVE_LOSSES = 3
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
SIMULATED_SLIPPAGE = float(os.getenv("SIMULATED_SLIPPAGE", "0.0002"))
FUTURES_LEVERAGE = int(os.getenv("FUTURES_LEVERAGE", "5"))
FUTURES_TAKER_FEE_RATE = float(os.getenv("FUTURES_TAKER_FEE_RATE", "0.0004"))
MARGIN_TYPE = "ISOLATED"
WORKING_TYPE = os.getenv("FUTURES_WORKING_TYPE", "MARK_PRICE")

RUNTIME_DIR = Path("runtime")
DASHBOARD_STATE_FILE = RUNTIME_DIR / "dashboard_state.json"
TRADES_LOG_FILE = RUNTIME_DIR / "trades_log.json"


def ensure_runtime_files():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not TRADES_LOG_FILE.exists():
        TRADES_LOG_FILE.write_text("[]", encoding="utf-8")


recent_events = deque(maxlen=50)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(message: str):
    line = f"[{utc_now_iso()}] {message}"
    print(message)
    recent_events.append(line)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Metrics
total_trades = 0
wins = 0
losses = 0
gross_profit_sum = 0.0
gross_loss_sum = 0.0
fees_paid = 0.0
net_pnl_total = 0.0
equity_peak = CAPITAL_USDT
max_drawdown_pct = 0.0

last_signal = "NONE"
last_reason = "startup"
last_market_state = "TRADEABLE"
last_trade_snapshot = None

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
        print(f"[api-ok] {label} {time.time() - start:.2f}s")
        return res
    except Exception as e:
        print(f"[api-err] {label} {time.time() - start:.2f}s {type(e).__name__}: {e}")
        raise


def sync_time_offset():
    server_ms = call_api("get_server_time", client.get_server_time)["serverTime"]
    local_ms = int(time.time() * 1000)
    client.timestamp_offset = server_ms - local_ms
    log_event(f"Binance time offset synced: {client.timestamp_offset} ms")


def call_with_time_sync(label, fn, *args, **kwargs):
    try:
        return call_api(label, fn, *args, **kwargs)
    except BinanceAPIException as e:
        if e.code == -1021:
            log_event("Received -1021 timestamp error. Resyncing time and retrying once...")
            sync_time_offset()
            return call_api(f"{label} (retry)", fn, *args, **kwargs)
        raise


def d(v) -> Decimal:
    return Decimal(str(v))


def round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return float((d(value) / d(step)).to_integral_value(rounding=ROUND_DOWN) * d(step))


def load_trade_log():
    if not TRADES_LOG_FILE.exists():
        return []
    try:
        data = json.loads(TRADES_LOG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def append_trade_log(trade):
    rows = load_trade_log()
    rows.append(trade)
    TRADES_LOG_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def update_metrics(net_pnl_value: float, gross_pnl: float, fees: float, current_balance: float):
    global total_trades, wins, losses, gross_profit_sum, gross_loss_sum, fees_paid, net_pnl_total, equity_peak, max_drawdown_pct
    total_trades += 1
    if net_pnl_value >= 0:
        wins += 1
    else:
        losses += 1

    if gross_pnl >= 0:
        gross_profit_sum += gross_pnl
    else:
        gross_loss_sum += gross_pnl

    fees_paid += fees
    net_pnl_total += net_pnl_value
    equity_peak = max(equity_peak, current_balance)
    dd = (equity_peak - current_balance) / equity_peak if equity_peak > 0 else 0.0
    max_drawdown_pct = max(max_drawdown_pct, dd)


def metrics_snapshot():
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else None
    profit_factor = (gross_profit_sum / abs(gross_loss_sum)) if losses > 0 and gross_loss_sum != 0 else None
    expectancy = (net_pnl_total / total_trades) if total_trades > 0 else None
    return win_rate, profit_factor, expectancy


def export_dashboard_state(last_closed_candle_ms, trades_this_hour, losses_in_row, active_trade, market_state):
    win_rate, profit_factor, expectancy = metrics_snapshot()
    balance = get_usdt_balance()
    state = {
        "timestamp_utc": utc_now_iso(),
        "mode": "paper" if PAPER_MODE else "real",
        "symbol": SYMBOL,
        "timeframe": "1m",
        "trend_timeframe": TREND_TIMEFRAME,
        "margin_type": MARGIN_TYPE,
        "leverage": FUTURES_LEVERAGE,
        "paper_balance": balance,
        "in_position": is_in_position(active_trade),
        "position_side": active_trade["side"] if active_trade else None,
        "position_qty": float(active_trade.get("position_size", active_trade.get("qty", 0.0))) if active_trade else 0.0,
        "entry_price": safe_float(active_trade.get("entry_price")) if active_trade else None,
        "stop_price": safe_float(active_trade.get("stop_price")) if active_trade else None,
        "take_profit_price": safe_float(active_trade.get("take_profit_price", active_trade.get("tp_price"))) if active_trade else None,
        "trades_this_hour": trades_this_hour,
        "consecutive_losses": losses_in_row,
        "market_state": market_state,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "net_pnl": net_pnl_total,
        "fees_paid": fees_paid,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown_pct": max_drawdown_pct * 100,
        "last_signal": last_signal,
        "last_reason": last_reason,
        "recent_events": list(recent_events),
        "last_trade": last_trade_snapshot,
        "last_closed_candle": datetime.fromtimestamp(last_closed_candle_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if last_closed_candle_ms
        else None,
    }
    DASHBOARD_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def log_trade_exit(side, entry_price, exit_price, qty, gross_pnl, fees, net_pnl_value, balance_after):
    global last_trade_snapshot
    result = "WIN" if net_pnl_value >= 0 else "LOSS"
    trade = {
        "timestamp": utc_now_iso(),
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "qty": qty,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": net_pnl_value,
        "result": result,
        "balance_after": balance_after,
    }
    append_trade_log(trade)
    last_trade_snapshot = trade


exchange_info = call_with_time_sync("futures_exchange_info", client.futures_exchange_info)
symbol_info = next((s for s in exchange_info["symbols"] if s["symbol"] == SYMBOL), None)
if not symbol_info:
    print(f"Could not load futures symbol info for {SYMBOL}")
    sys.exit(1)

price_tick = qty_step = min_qty = min_notional = 0.0
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
        call_with_time_sync("futures_change_margin_type", client.futures_change_margin_type, symbol=SYMBOL, marginType=MARGIN_TYPE)
    except BinanceAPIException as e:
        if e.code != -4046:
            raise
    call_with_time_sync("futures_change_leverage", client.futures_change_leverage, symbol=SYMBOL, leverage=FUTURES_LEVERAGE)


def get_usdt_balance() -> float:
    if PAPER_MODE:
        return paper_usdt_balance
    balances = call_with_time_sync("futures_account_balance", client.futures_account_balance)
    usdt = next((b for b in balances if b.get("asset") == "USDT"), None)
    return float(usdt["balance"]) if usdt else 0.0


def get_position_amt() -> float:
    if PAPER_MODE:
        return 0.0
    positions = call_with_time_sync("futures_position_information", client.futures_position_information, symbol=SYMBOL)
    return float(positions[0]["positionAmt"]) if positions else 0.0


def is_in_position(active_trade=None) -> bool:
    if PAPER_MODE:
        return active_trade is not None
    return abs(get_position_amt()) > 0


def calculate_ema(prices, period: int):
    if len(prices) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_atr(klines, period=14):
    if len(klines) < period + 1:
        return None
    trs = []
    for i in range(1, len(klines)):
        high = float(klines[i][2])
        low = float(klines[i][3])
        prev_close = float(klines[i - 1][4])
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return sum(trs[-period:]) / period


last_trend_close_time = None
last_trend_value = "FLAT"
last_trend_ema_fast = None
last_trend_ema_slow = None


def get_trend_state():
    global last_trend_close_time, last_trend_value, last_trend_ema_fast, last_trend_ema_slow
    klines = call_with_time_sync("futures_klines(trend)", client.futures_klines, symbol=SYMBOL, interval=TREND_TIMEFRAME, limit=300)
    if len(klines) < EMA_SLOW + 2:
        return last_trend_value, last_trend_ema_fast, last_trend_ema_slow
    latest_closed = klines[-2]
    latest_closed_time = int(latest_closed[6])
    if last_trend_close_time == latest_closed_time:
        return last_trend_value, last_trend_ema_fast, last_trend_ema_slow
    closes = [float(k[4]) for k in klines[:-1]]
    ema_fast = calculate_ema(closes, EMA_FAST)
    ema_slow = calculate_ema(closes, EMA_SLOW)
    trend = "BULL" if ema_fast and ema_slow and ema_fast > ema_slow else "BEAR" if ema_fast and ema_slow and ema_fast < ema_slow else "FLAT"
    last_trend_close_time = latest_closed_time
    last_trend_value = trend
    last_trend_ema_fast = ema_fast
    last_trend_ema_slow = ema_slow
    return trend, ema_fast, ema_slow


def get_closed_candle_data():
    max_lookback = max(LOOKBACK, VOLUME_LOOKBACK, 20)
    klines = call_with_time_sync("futures_klines", client.futures_klines, symbol=SYMBOL, interval=INTERVAL, limit=CANDLE_LIMIT)
    if len(klines) < max_lookback + 2:
        return None
    closed = klines[-2]
    breakout_history = klines[-(LOOKBACK + 2):-2]
    volume_history = klines[-(VOLUME_LOOKBACK + 2):-2]
    vol20_history = klines[-22:-2]
    close_price = float(closed[4])
    close_volume = float(closed[5])
    close_time = int(closed[6])
    atr14 = calculate_atr(klines[:-1], 14)
    volume_sma20 = sum(float(k[5]) for k in vol20_history) / len(vol20_history)
    volatility_score = (atr14 / close_price) if atr14 and close_price > 0 else 0.0
    volume_score = (close_volume / volume_sma20) if volume_sma20 > 0 else 0.0
    market_state = "LOW_ACTIVITY" if (volatility_score < 0.001 or volume_score < 1.0) else "TRADEABLE"
    return {
        "close_time": close_time,
        "close_price": close_price,
        "close_volume": close_volume,
        "highest_high": max(float(c[2]) for c in breakout_history),
        "lowest_low": min(float(c[3]) for c in breakout_history),
        "avg_volume": sum(float(c[5]) for c in volume_history) / len(volume_history),
        "market_state": market_state,
    }


def get_last_price() -> float:
    ticker = call_with_time_sync("futures_symbol_ticker", client.futures_symbol_ticker, symbol=SYMBOL)
    return float(ticker["price"])


def print_heartbeat(last_closed_candle_ms, trades_this_hour, losses_in_row, active_trade=None):
    balance = get_usdt_balance()
    pos_amt = active_trade["position_size"] if PAPER_MODE and active_trade else get_position_amt()
    last_closed = datetime.fromtimestamp(last_closed_candle_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if last_closed_candle_ms else "none"
    log_event(
        f"[heartbeat] {utc_now_iso()} last_closed={last_closed} mode={get_mode_label()} margin={MARGIN_TYPE} leverage={FUTURES_LEVERAGE}x "
        f"usdt_balance={balance:.4f} position_amt={pos_amt:.6f} in_position={is_in_position(active_trade)} trades_this_hour={trades_this_hour} consecutive_losses={losses_in_row}"
    )


def place_trade(side: str):
    global paper_usdt_balance
    usdt_before = get_usdt_balance()
    close_price = get_last_price()
    is_long = side == "LONG"
    entry_price = close_price * (1 + SIMULATED_SLIPPAGE) if is_long else close_price * (1 - SIMULATED_SLIPPAGE)
    stop_price = entry_price * (1 - STOP_PCT) if is_long else entry_price * (1 + STOP_PCT)
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT) if is_long else entry_price * (1 - TAKE_PROFIT_PCT)
    position_notional_usdt = RISK_PER_TRADE_USDT / (STOP_PCT * FUTURES_LEVERAGE)
    qty_raw = position_notional_usdt / entry_price
    qty = round_down_to_step(qty_raw, qty_step)
    stop_price = round_down_to_step(stop_price, price_tick)
    take_profit_price = round_down_to_step(take_profit_price, price_tick)
    if qty < min_qty:
        qty = min_qty
    if min_notional and qty * entry_price < min_notional:
        return None
    notional = qty * entry_price
    if (notional / FUTURES_LEVERAGE) > usdt_before:
        return None
    estimated_entry_fee = notional * FUTURES_TAKER_FEE_RATE
    estimated_exit_fee = notional * FUTURES_TAKER_FEE_RATE

    if PAPER_MODE:
        if estimated_entry_fee > paper_usdt_balance:
            return None
        paper_usdt_balance -= estimated_entry_fee
        log_event(
            f"[PAPER ENTRY] close={close_price:.2f} slipped_entry={entry_price:.2f} qty={qty:.6f} side={side} notional={notional:.4f} "
            f"entry_fee={estimated_entry_fee:.4f} stop={stop_price:.2f} tp={take_profit_price:.2f} paper_usdt_balance={paper_usdt_balance:.4f}"
        )
        return {
            "entry_time_ms": int(time.time() * 1000),
            "position_size": qty,
            "position_notional": notional,
            "side": side,
            "entry_price": entry_price,
            "entry_fee": estimated_entry_fee,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "usdt_before": usdt_before,
            "estimated_exit_fee": estimated_exit_fee,
        }

    entry_order = call_with_time_sync(
        "futures_create_order(MARKET)",
        client.futures_create_order,
        symbol=SYMBOL,
        side="BUY" if is_long else "SELL",
        type="MARKET",
        quantity=qty,
    )
    call_with_time_sync(
        "futures_create_order(STOP_MARKET)",
        client.futures_create_order,
        symbol=SYMBOL,
        side="SELL" if is_long else "BUY",
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
        side="SELL" if is_long else "BUY",
        type="TAKE_PROFIT_MARKET",
        stopPrice=f"{take_profit_price:.8f}",
        quantity=qty,
        reduceOnly=True,
        workingType=WORKING_TYPE,
    )
    return {
        "entry_order": entry_order,
        "entry_time_ms": int(time.time() * 1000),
        "qty": qty,
        "side": side,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "tp_price": take_profit_price,
        "usdt_before": usdt_before,
    }


ensure_runtime_files()
log_event("Starting BTCUSDT 1m breakout scalping bot (BINANCE FUTURES USDT-M)...")
if VALIDATION_MODE:
    log_event("VALIDATION_MODE enabled: using shorter lookbacks and relaxed volume threshold.")
sync_time_offset()
ensure_futures_settings()

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
                    if active_trade["side"] == "LONG":
                        exit_reason = "LOSS"
                        exit_price = active_trade["stop_price"] * (1 - SIMULATED_SLIPPAGE)
                elif last_price >= active_trade["take_profit_price"]:
                    if active_trade["side"] == "LONG":
                        exit_reason = "WIN"
                        exit_price = active_trade["take_profit_price"] * (1 - SIMULATED_SLIPPAGE)
                if active_trade["side"] == "SHORT":
                    if last_price >= active_trade["stop_price"]:
                        exit_reason = "LOSS"
                        exit_price = active_trade["stop_price"] * (1 + SIMULATED_SLIPPAGE)
                    elif last_price <= active_trade["take_profit_price"]:
                        exit_reason = "WIN"
                        exit_price = active_trade["take_profit_price"] * (1 + SIMULATED_SLIPPAGE)

                if exit_reason:
                    gross_pnl = active_trade["position_size"] * (exit_price - active_trade["entry_price"]) if active_trade["side"] == "LONG" else active_trade["position_size"] * (active_trade["entry_price"] - exit_price)
                    exit_fee = active_trade["position_notional"] * FUTURES_TAKER_FEE_RATE
                    total_fees = active_trade["entry_fee"] + exit_fee
                    net_pnl_value = gross_pnl - total_fees
                    paper_usdt_balance += net_pnl_value
                    consecutive_losses = consecutive_losses + 1 if exit_reason == "LOSS" else 0
                    log_event(
                        f"[PAPER EXIT {exit_reason}] exit={exit_price:.2f} qty={active_trade['position_size']:.6f} side={active_trade['side']} "
                        f"gross_pnl={gross_pnl:.4f} fees={total_fees:.4f} net_pnl={net_pnl_value:.4f} balance={paper_usdt_balance:.4f}"
                    )
                    update_metrics(net_pnl_value, gross_pnl, total_fees, paper_usdt_balance)
                    log_trade_exit(
                        active_trade["side"],
                        active_trade["entry_price"],
                        exit_price,
                        active_trade["position_size"],
                        gross_pnl,
                        total_fees,
                        net_pnl_value,
                        paper_usdt_balance,
                    )
                    active_trade = None
                else:
                    log_event(
                        f"Paper trade still active. price={last_price:.2f} stop={active_trade['stop_price']:.2f} tp={active_trade['take_profit_price']:.2f}"
                    )
                    export_dashboard_state(last_processed_close_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
                    time.sleep(POLL_SECONDS)
                    continue
            else:
                pos_amt = get_position_amt()
                if abs(pos_amt) < 1e-12:
                    usdt_now = get_usdt_balance()
                    gross_pnl = usdt_now - active_trade["usdt_before"]
                    fees = 0.0
                    net_pnl_value = gross_pnl
                    result = "LOSS" if net_pnl_value < 0 else "WIN"
                    log_event(f"[FUTURES EXIT {result}] gross_pnl={gross_pnl:.4f} fees={fees:.4f} net_pnl={net_pnl_value:.4f} balance={usdt_now:.4f}")
                    consecutive_losses = consecutive_losses + 1 if net_pnl_value < 0 else 0
                    update_metrics(net_pnl_value, gross_pnl, fees, usdt_now)
                    log_trade_exit(active_trade["side"], active_trade["entry_price"], get_last_price(), active_trade["qty"], gross_pnl, fees, net_pnl_value, usdt_now)
                    active_trade = None
                else:
                    log_event("Futures trade still active. Waiting for stop/tp trigger...")
                    export_dashboard_state(last_processed_close_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
                    time.sleep(POLL_SECONDS)
                    continue

            if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                log_event("3 consecutive losses reached. Shutting down.")
                export_dashboard_state(last_processed_close_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
                break

        if len(trade_times) >= MAX_TRADES_PER_HOUR:
            last_reason = "trade_limit"
            log_event("Trade limit reached (3 per hour). Waiting...")
            export_dashboard_state(last_processed_close_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
            time.sleep(POLL_SECONDS)
            continue

        candle = get_closed_candle_data()
        if candle is None:
            last_reason = "not_enough_candles"
            log_event("Not enough candle data yet.")
            export_dashboard_state(last_processed_close_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
            time.sleep(POLL_SECONDS)
            continue

        if candle["close_time"] == last_processed_close_time:
            last_reason = "no_new_candle"
            log_event("No new closed candle.")
            export_dashboard_state(last_processed_close_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
            time.sleep(POLL_SECONDS)
            continue

        last_processed_close_time = candle["close_time"]
        last_market_state = candle["market_state"]
        log_event(f"[MARKET] {last_market_state}")

        trend, ema50, ema200 = get_trend_state()
        long_breakout = candle["close_price"] > candle["highest_high"]
        short_breakdown = candle["close_price"] < candle["lowest_low"]
        vol_ok = candle["close_volume"] >= candle["avg_volume"] * VOLUME_MULTIPLIER
        long_signal = long_breakout and vol_ok and trend == "BULL"
        short_signal = short_breakdown and vol_ok and trend == "BEAR"

        if is_in_position(active_trade):
            last_reason = "in_position"
        elif not vol_ok:
            last_reason = "low_volume"
        elif trend not in {"BULL", "BEAR"}:
            last_reason = "trend_filter_blocked"
        elif not long_breakout and not short_breakdown:
            last_reason = "no_breakout"
        elif long_breakout and trend != "BULL":
            last_reason = "trend_filter_blocked"
        elif short_breakdown and trend != "BEAR":
            last_reason = "trend_filter_blocked"
        else:
            last_reason = "breakout + volume + trend"

        if (long_signal or short_signal) and not is_in_position(active_trade):
            side = "LONG" if long_signal else "SHORT"
            last_signal = side
            log_event(f"Signal confirmed: side={side} breakout+volume+trend.")
            trade = place_trade(side)
            if trade:
                trade_times.append(int(time.time() * 1000))
                active_trade = trade
                log_event(f"Trades this hour: {len(trade_times)}/{MAX_TRADES_PER_HOUR}")
        else:
            last_signal = "NONE"
            log_event(f"No trade: {last_reason}")

        print_heartbeat(last_processed_close_time, len(trade_times), consecutive_losses, active_trade)
        export_dashboard_state(last_processed_close_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
        time.sleep(POLL_SECONDS)
    except Exception as e:
        log_event(f"Error: {e}")
        export_dashboard_state(last_processed_close_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
        time.sleep(POLL_SECONDS)
