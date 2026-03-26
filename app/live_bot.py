import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

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
MIN_VOLUME_SCORE = 0.7
MICRO_BREAKOUT_FACTOR = 0.999
MICRO_BREAKDOWN_FACTOR = 1.001
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
DEBUG_FORCE_ENTRY = True
DEBUG_DISABLE_PRICE_FILTER = False

RUNTIME_DIR = Path("runtime")
DASHBOARD_STATE_FILE = RUNTIME_DIR / "dashboard_state.json"
TRADES_LOG_FILE = RUNTIME_DIR / "trades_log.json"
CANDLES_FILE = RUNTIME_DIR / "candles.json"
PAPER_CONFIG_FILE = RUNTIME_DIR / "paper_config.json"
recent_closed_klines_cache: List[List[Any]] = []
recent_signal_markers: Deque[Dict[str, Any]] = deque(maxlen=400)


def ensure_runtime_files() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not TRADES_LOG_FILE.exists():
        safe_write_json(TRADES_LOG_FILE, [])


recent_events = deque(maxlen=50)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(message: str) -> None:
    line = f"[{utc_now_iso()}] {message}"
    print(message)
    recent_events.append(line)


def safe_float(value: Any) -> Optional[float]:
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
win_net_pnl_sum = 0.0
loss_net_pnl_sum = 0.0
total_trade_duration_minutes = 0.0
closed_trade_duration_count = 0

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


def call_api(label: str, fn, *args: Any, **kwargs: Any) -> Any:
    start = time.time()
    try:
        res = fn(*args, **kwargs)
        print(f"[api-ok] {label} {time.time() - start:.2f}s")
        return res
    except Exception as exc:
        print(f"[api-err] {label} {time.time() - start:.2f}s {type(exc).__name__}: {exc}")
        raise


def sync_time_offset() -> None:
    server_ms = call_api("get_server_time", client.get_server_time)["serverTime"]
    local_ms = int(time.time() * 1000)
    client.timestamp_offset = server_ms - local_ms
    log_event(f"Binance time offset synced: {client.timestamp_offset} ms")


def call_with_time_sync(label: str, fn, *args: Any, **kwargs: Any) -> Any:
    try:
        return call_api(label, fn, *args, **kwargs)
    except BinanceAPIException as exc:
        if exc.code == -1021:
            log_event("Received -1021 timestamp error. Resyncing time and retrying once...")
            sync_time_offset()
            return call_api(f"{label} (retry)", fn, *args, **kwargs)
        raise


def d(v: Any) -> Decimal:
    return Decimal(str(v))


def round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return float((d(value) / d(step)).to_integral_value(rounding=ROUND_DOWN) * d(step))


def round_step_size(value: float, step_size: float) -> float:
    return round_down_to_step(value, step_size)


def load_trade_log() -> List[Dict[str, Any]]:
    if not TRADES_LOG_FILE.exists():
        return []
    try:
        data = json.loads(TRADES_LOG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        log_event(f"[WARN] Could not read {TRADES_LOG_FILE}: {exc}")
        return []


def append_trade_log(trade: Dict[str, Any]) -> None:
    rows = load_trade_log()
    rows.append(trade)
    safe_write_json(TRADES_LOG_FILE, rows)


def update_metrics(net_pnl_value: float, gross_pnl: float, fees: float, current_balance: float) -> None:
    global total_trades, wins, losses, gross_profit_sum, gross_loss_sum, fees_paid, net_pnl_total, equity_peak, max_drawdown_pct, win_net_pnl_sum, loss_net_pnl_sum
    total_trades += 1
    if net_pnl_value >= 0:
        wins += 1
        win_net_pnl_sum += net_pnl_value
    else:
        losses += 1
        loss_net_pnl_sum += net_pnl_value

    if gross_pnl >= 0:
        gross_profit_sum += gross_pnl
    else:
        gross_loss_sum += gross_pnl

    fees_paid += fees
    net_pnl_total += net_pnl_value
    equity_peak = max(equity_peak, current_balance)
    dd = (equity_peak - current_balance) / equity_peak if equity_peak > 0 else 0.0
    max_drawdown_pct = max(max_drawdown_pct, dd)


def update_trade_duration_metrics(duration_minutes: Optional[float]) -> None:
    global total_trade_duration_minutes, closed_trade_duration_count
    if duration_minutes is None:
        return
    total_trade_duration_minutes += duration_minutes
    closed_trade_duration_count += 1


def metrics_snapshot() -> Any:
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else None
    profit_factor = (gross_profit_sum / abs(gross_loss_sum)) if losses > 0 and gross_loss_sum != 0 else None
    expectancy = (net_pnl_total / total_trades) if total_trades > 0 else None
    avg_win = (win_net_pnl_sum / wins) if wins > 0 else None
    avg_loss = (loss_net_pnl_sum / losses) if losses > 0 else None
    avg_trade_duration_minutes = (total_trade_duration_minutes / closed_trade_duration_count) if closed_trade_duration_count > 0 else None
    return win_rate, profit_factor, expectancy, avg_win, avg_loss, avg_trade_duration_minutes


def format_metric_value(value: Optional[float], fmt: str) -> str:
    if value is None:
        return "N/A"
    return format(value, fmt)


def print_trade_closed_summary(
    result: str,
    side: str,
    entry: float,
    exit_price: float,
    qty: float,
    gross_pnl: float,
    fees: float,
    net_pnl_value: float,
    balance: float,
) -> None:
    win_rate, profit_factor, expectancy, avg_win, avg_loss, _ = metrics_snapshot()
    print("==================================================")
    print("[TRADE CLOSED SUMMARY]")
    print(f"result={result}")
    print(f"side={side}")
    print(f"entry={entry:.2f}")
    print(f"exit={exit_price:.2f}")
    print(f"qty={qty:.6f}")
    print(f"gross_pnl={gross_pnl:.4f}")
    print(f"fees={fees:.4f}")
    print(f"net_pnl={net_pnl_value:.4f}")
    print(f"balance={balance:.4f}")
    print()
    print(f"total_trades={total_trades}")
    print(f"wins={wins}")
    print(f"losses={losses}")
    print(f"win_rate={format_metric_value(win_rate, '.2f')}%")
    print(f"gross_profit_sum={gross_profit_sum:.2f}")
    print(f"gross_loss_sum={gross_loss_sum:.2f}")
    print(f"net_pnl_total={net_pnl_total:.2f}")
    print(f"profit_factor={format_metric_value(profit_factor, '.2f')}")
    print(f"expectancy={format_metric_value(expectancy, '.4f')}")
    print(f"max_drawdown_pct={max_drawdown_pct * 100:.2f}%")
    print(f"avg_win={format_metric_value(avg_win, '.4f')}")
    print(f"avg_loss={format_metric_value(avg_loss, '.4f')}")
    print("==================================================")


def export_dashboard_state(last_closed_candle_ms: Optional[int], trades_this_hour: int, losses_in_row: int, active_trade: Optional[Dict[str, Any]], market_state: str) -> None:
    win_rate, profit_factor, expectancy, avg_win, avg_loss, avg_trade_duration_minutes = metrics_snapshot()
    balance = get_usdt_balance()
    unrealized_pnl = None
    if active_trade:
        last_price = get_last_price()
        entry_price = safe_float(active_trade.get("entry_price"))
        position_size = safe_float(active_trade.get("position_size", active_trade.get("qty")))
        side = str(active_trade.get("side"))
        if entry_price is not None and position_size is not None and side in {"LONG", "SHORT"}:
            unrealized_pnl = position_size * (last_price - entry_price) if side == "LONG" else position_size * (entry_price - last_price)
    paper_target_balance = load_paper_config_target()
    paper_config_warning = None
    if PAPER_MODE and active_trade and paper_target_balance is not None and abs(paper_target_balance - paper_balance_base) >= 1e-9:
        paper_config_warning = "Cannot change paper balance while a trade is open"
    state = {
        "timestamp_utc": utc_now_iso(),
        "mode": "paper" if PAPER_MODE else "real",
        "symbol": SYMBOL,
        "timeframe": "1m",
        "trend_timeframe": TREND_TIMEFRAME,
        "margin_type": MARGIN_TYPE,
        "leverage": FUTURES_LEVERAGE,
        "paper_balance": balance,
        "paper_balance_base": paper_balance_base if PAPER_MODE else None,
        "paper_balance_source": paper_balance_source if PAPER_MODE else None,
        "paper_target_balance": paper_target_balance if PAPER_MODE else None,
        "paper_config_warning": paper_config_warning,
        "in_position": is_in_position(active_trade),
        "position_side": active_trade["side"] if active_trade else None,
        "position_qty": float(active_trade.get("position_size", active_trade.get("qty", 0.0))) if active_trade else 0.0,
        "entry_price": safe_float(active_trade.get("entry_price")) if active_trade else None,
        "stop_price": safe_float(active_trade.get("stop_price")) if active_trade else None,
        "take_profit_price": safe_float(active_trade.get("take_profit_price", active_trade.get("tp_price"))) if active_trade else None,
        "position_unrealized_pnl": unrealized_pnl,
        "trades_this_hour": trades_this_hour,
        "consecutive_losses": losses_in_row,
        "market_state": market_state,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "gross_profit_sum": gross_profit_sum,
        "gross_loss_sum": gross_loss_sum,
        "net_pnl": net_pnl_total,
        "net_pnl_total": net_pnl_total,
        "fees_paid": fees_paid,
        "fees_paid_total": fees_paid,
        "equity_peak": equity_peak,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_trade_duration_minutes": avg_trade_duration_minutes,
        "max_drawdown_pct": max_drawdown_pct * 100,
        "last_signal": last_signal,
        "last_reason": last_reason,
        "recent_events": list(recent_events),
        "last_trade": last_trade_snapshot,
        "last_closed_candle": datetime.fromtimestamp(last_closed_candle_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if last_closed_candle_ms
        else None,
    }
    safe_write_json(DASHBOARD_STATE_FILE, state)


def log_trade_exit(
    side: str,
    entry_price: float,
    exit_price: float,
    qty: float,
    gross_pnl: float,
    fees: float,
    net_pnl_value: float,
    balance_after: float,
    duration_minutes: Optional[float] = None,
) -> None:
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
        "duration_minutes": duration_minutes,
    }
    append_trade_log(trade)
    last_trade_snapshot = trade


def bootstrap_metrics_from_trades() -> None:
    global total_trades, wins, losses, gross_profit_sum, gross_loss_sum, fees_paid, net_pnl_total, equity_peak, max_drawdown_pct, win_net_pnl_sum, loss_net_pnl_sum, total_trade_duration_minutes, closed_trade_duration_count
    trades = load_trade_log()
    if not trades:
        return

    running_balance = CAPITAL_USDT
    equity_peak_local = CAPITAL_USDT
    max_drawdown_local = 0.0

    for trade in trades:
        net = safe_float(trade.get("net_pnl")) or 0.0
        gross = safe_float(trade.get("gross_pnl")) or 0.0
        fees = safe_float(trade.get("fees")) or 0.0
        duration = safe_float(trade.get("duration_minutes"))

        total_trades += 1
        net_pnl_total += net
        fees_paid += fees
        if net >= 0:
            wins += 1
            win_net_pnl_sum += net
        else:
            losses += 1
            loss_net_pnl_sum += net

        if gross >= 0:
            gross_profit_sum += gross
        else:
            gross_loss_sum += gross

        if duration is not None:
            total_trade_duration_minutes += duration
            closed_trade_duration_count += 1

        balance_after = safe_float(trade.get("balance_after"))
        running_balance = balance_after if balance_after is not None else (running_balance + net)
        equity_peak_local = max(equity_peak_local, running_balance)
        dd = (equity_peak_local - running_balance) / equity_peak_local if equity_peak_local > 0 else 0.0
        max_drawdown_local = max(max_drawdown_local, dd)

    equity_peak = equity_peak_local
    max_drawdown_pct = max_drawdown_local


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
paper_balance_base = CAPITAL_USDT
paper_balance_source = "env"


def ensure_futures_settings() -> None:
    if PAPER_MODE:
        return
    try:
        call_with_time_sync("futures_change_margin_type", client.futures_change_margin_type, symbol=SYMBOL, marginType=MARGIN_TYPE)
    except BinanceAPIException as exc:
        if exc.code != -4046:
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


def is_in_position(active_trade: Optional[Dict[str, Any]] = None) -> bool:
    if PAPER_MODE:
        return active_trade is not None
    return abs(get_position_amt()) > 0


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_atr(klines: List[List[Any]], period: int = 14) -> Optional[float]:
    if len(klines) < period + 1:
        return None
    trs: List[float] = []
    for idx in range(1, len(klines)):
        high = float(klines[idx][2])
        low = float(klines[idx][3])
        prev_close = float(klines[idx - 1][4])
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return sum(trs[-period:]) / period


last_trend_close_time = None
last_trend_value = "FLAT"
last_trend_ema_fast = None
last_trend_ema_slow = None


def calculate_ema_series(prices: List[float], period: int) -> List[Optional[float]]:
    if len(prices) < period:
        return []
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    result: List[Optional[float]] = [None] * (period - 1) + [ema]
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
        result.append(ema)
    return result


def ms_to_utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_write_json(path: Path, payload: Any) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        log_event(f"[WARN] Could not write {path}: {exc}")


def load_paper_config_target() -> Optional[float]:
    if not PAPER_CONFIG_FILE.exists():
        return None
    try:
        payload = json.loads(PAPER_CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log_event(f"[WARN] Could not read {PAPER_CONFIG_FILE}: {exc}")
        return None
    if not isinstance(payload, dict):
        return None
    value = safe_float(payload.get("target_balance"))
    if value is None or value <= 0:
        return None
    return value


def apply_paper_balance_config_if_needed(active_trade: Optional[Dict[str, Any]]) -> None:
    global paper_usdt_balance, paper_balance_base, paper_balance_source
    if not PAPER_MODE:
        return
    target_balance = load_paper_config_target()
    if target_balance is None:
        return
    if active_trade is not None:
        return
    if abs(target_balance - paper_balance_base) < 1e-9:
        return
    paper_balance_base = target_balance
    paper_usdt_balance = target_balance
    paper_balance_source = "runtime_config"
    log_event(f"[PAPER CONFIG] Rebased paper balance to {paper_usdt_balance:.4f} from runtime/paper_config.json")


def build_signal_markers() -> List[Dict[str, Any]]:
    return list(recent_signal_markers)


def build_trade_markers(active_trade: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    for trade in load_trade_log()[-200:]:
        timestamp = trade.get("timestamp")
        side = trade.get("side")
        result = trade.get("result")
        exit_price = safe_float(trade.get("exit_price"))
        entry_price = safe_float(trade.get("entry_price"))
        if timestamp and side in {"LONG", "SHORT"} and entry_price is not None:
            markers.append({"time": timestamp, "side": side, "kind": "EXECUTED_ENTRY", "price": entry_price})
        if timestamp and side in {"LONG", "SHORT"} and result in {"WIN", "LOSS"} and exit_price is not None:
            markers.append({"time": timestamp, "side": side, "kind": f"EXIT_{result}", "price": exit_price})

    if active_trade and active_trade.get("entry_time_ms"):
        side = active_trade.get("side")
        entry_price = safe_float(active_trade.get("entry_price"))
        if side in {"LONG", "SHORT"} and entry_price is not None:
            markers.append(
                {"time": ms_to_utc_iso(int(active_trade["entry_time_ms"])), "side": side, "kind": "EXECUTED_ENTRY", "price": entry_price}
            )
    return markers


def export_candles_snapshot(active_trade: Optional[Dict[str, Any]]) -> None:
    global recent_closed_klines_cache
    closed_klines = recent_closed_klines_cache[-200:]
    if not closed_klines:
        klines = call_with_time_sync("futures_klines(chart)", client.futures_klines, symbol=SYMBOL, interval=INTERVAL, limit=201)
        if len(klines) < 2:
            return
        closed_klines = klines[:-1][-200:]
    if len(closed_klines) < 2:
        return

    closes = [float(k[4]) for k in closed_klines]
    ema_fast_series = calculate_ema_series(closes, EMA_FAST)
    ema_slow_series = calculate_ema_series(closes, EMA_SLOW)

    candles: List[Dict[str, Any]] = []
    ema_fast: List[Dict[str, Any]] = []
    ema_slow: List[Dict[str, Any]] = []
    for idx, kline in enumerate(closed_klines):
        timestamp = ms_to_utc_iso(int(kline[0]))
        candles.append(
            {
                "time": timestamp,
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
            }
        )
        fast_value = ema_fast_series[idx] if idx < len(ema_fast_series) else None
        slow_value = ema_slow_series[idx] if idx < len(ema_slow_series) else None
        if fast_value is not None:
            ema_fast.append({"time": timestamp, "value": fast_value})
        if slow_value is not None:
            ema_slow.append({"time": timestamp, "value": slow_value})

    levels: Dict[str, Optional[float]] = {"entry_price": None, "stop_price": None, "take_profit_price": None}
    if active_trade:
        levels["entry_price"] = safe_float(active_trade.get("entry_price"))
        levels["stop_price"] = safe_float(active_trade.get("stop_price"))
        levels["take_profit_price"] = safe_float(active_trade.get("take_profit_price", active_trade.get("tp_price")))

    payload = {
        "symbol": SYMBOL,
        "timeframe": "1m",
        "updated_at_utc": utc_now_iso(),
        "candles": candles,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "levels": levels,
        "signal_markers": build_signal_markers(),
        "trade_markers": build_trade_markers(active_trade),
    }
    safe_write_json(CANDLES_FILE, payload)


def export_runtime_state(last_closed_candle_ms: Optional[int], trades_this_hour: int, losses_in_row: int, active_trade: Optional[Dict[str, Any]], market_state: str) -> None:
    export_dashboard_state(last_closed_candle_ms, trades_this_hour, losses_in_row, active_trade, market_state)
    export_candles_snapshot(active_trade)


def get_trend_state() -> Any:
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


def get_closed_candle_data() -> Optional[Dict[str, Any]]:
    global recent_closed_klines_cache
    max_lookback = max(LOOKBACK, VOLUME_LOOKBACK, 20)
    klines = call_with_time_sync("futures_klines", client.futures_klines, symbol=SYMBOL, interval=INTERVAL, limit=250)
    if len(klines) < max_lookback + 2:
        return None

    recent_closed_klines_cache = klines[:-1]
    closed = klines[-2]
    breakout_history = klines[-(LOOKBACK + 2) : -2]
    volume_history = klines[-(VOLUME_LOOKBACK + 2) : -2]
    vol20_history = klines[-22:-2]
    close_price = float(closed[4])
    close_volume = float(closed[5])
    close_time = int(closed[6])
    atr14 = calculate_atr(klines[:-1], 14)
    volume_sma20 = sum(float(k[5]) for k in vol20_history) / len(vol20_history)
    volatility_score = (atr14 / close_price) if atr14 and close_price > 0 else 0.0
    volume_score = (close_volume / volume_sma20) if volume_sma20 > 0 else 0.0
    market_state = "LOW_ACTIVITY" if (volatility_score < 0.001 or volume_score < MIN_VOLUME_SCORE) else "TRADEABLE"

    return {
        "close_time": close_time,
        "close_price": close_price,
        "close_volume": close_volume,
        "high_price": float(closed[2]),
        "low_price": float(closed[3]),
        "volume_score": volume_score,
        "highest_high": max(float(c[2]) for c in breakout_history),
        "lowest_low": min(float(c[3]) for c in breakout_history),
        "avg_volume": sum(float(c[5]) for c in volume_history) / len(volume_history),
        "market_state": market_state,
    }


def get_last_price() -> float:
    ticker = call_with_time_sync("futures_symbol_ticker", client.futures_symbol_ticker, symbol=SYMBOL)
    return float(ticker["price"])


def print_heartbeat(last_closed_candle_ms: Optional[int], trades_this_hour: int, losses_in_row: int, active_trade: Optional[Dict[str, Any]] = None) -> None:
    balance = get_usdt_balance()
    pos_amt = active_trade["position_size"] if PAPER_MODE and active_trade else get_position_amt()
    last_closed = datetime.fromtimestamp(last_closed_candle_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if last_closed_candle_ms else "none"
    log_event(
        f"[heartbeat] {utc_now_iso()} last_closed={last_closed} mode={get_mode_label()} margin={MARGIN_TYPE} leverage={FUTURES_LEVERAGE}x "
        f"usdt_balance={balance:.4f} position_amt={pos_amt:.6f} in_position={is_in_position(active_trade)} trades_this_hour={trades_this_hour} consecutive_losses={losses_in_row}"
    )


def build_trade_plan(side: str, current_price: float, usdt_before: float) -> Tuple[Optional[Dict[str, Any]], str]:
    if current_price <= 0:
        return None, "skipped_price_invalid"
    is_long = side == "LONG"
    entry_price = current_price * (1 + SIMULATED_SLIPPAGE) if is_long else current_price * (1 - SIMULATED_SLIPPAGE)
    if entry_price <= 0:
        return None, "skipped_price_invalid"
    stop_price = entry_price * (1 - STOP_PCT) if is_long else entry_price * (1 + STOP_PCT)
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT) if is_long else entry_price * (1 - TAKE_PROFIT_PCT)
    usdt_balance = usdt_before
    risk_per_trade_usdt = RISK_PER_TRADE_USDT
    stop_distance_abs = abs(entry_price - stop_price)
    stop_distance_pct = (stop_distance_abs / entry_price) if entry_price > 0 else 0.0
    leverage = FUTURES_LEVERAGE
    step_size = qty_step

    if stop_distance_abs <= 0:
        return None, "skipped_stop_distance_invalid"

    raw_qty = risk_per_trade_usdt / stop_distance_abs
    raw_notional = raw_qty * entry_price
    rounded_qty = round_step_size(raw_qty, step_size)

    print("[DEBUG SIZING]")
    print(f"balance={usdt_balance}")
    print(f"risk_per_trade_usdt={risk_per_trade_usdt}")
    print(f"entry_price={entry_price}")
    print(f"stop_price={stop_price}")
    print(f"stop_distance_pct={stop_distance_pct}")
    print(f"leverage={leverage}")
    print(f"raw_notional={raw_notional}")
    print(f"raw_qty={raw_qty}")
    print(f"step_size={step_size}")
    print(f"min_qty={min_qty}")
    print(f"min_notional={min_notional}")
    print(f"rounded_qty={rounded_qty}")

    if rounded_qty <= 0:
        if PAPER_MODE and min_qty > 0 and raw_qty > 0:
            print("[PAPER DEBUG] forcing minQty for paper simulation")
            rounded_qty = max(min_qty, rounded_qty)
        else:
            return None, "skipped_rounding_to_zero"

    if rounded_qty < min_qty:
        if PAPER_MODE:
            print("[PAPER DEBUG] forcing minQty for paper simulation")
            rounded_qty = max(min_qty, rounded_qty)
        else:
            rounded_qty = min_qty

    qty = rounded_qty
    if qty <= 0:
        return None, "skipped_qty_invalid"

    effective_risk_usdt = qty * stop_distance_abs
    print(f"[DEBUG SIZING] effective_risk_usdt={effective_risk_usdt}")

    stop_price = round_down_to_step(stop_price, price_tick)
    take_profit_price = round_down_to_step(take_profit_price, price_tick)

    if min_notional and qty * entry_price < min_notional:
        return None, "skipped_min_notional"

    notional = qty * entry_price
    if (notional / FUTURES_LEVERAGE) > usdt_before:
        return None, "skipped_balance_insufficient"
    estimated_entry_fee = notional * FUTURES_TAKER_FEE_RATE
    estimated_exit_fee = notional * FUTURES_TAKER_FEE_RATE
    return {
        "current_price": current_price,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "qty": qty,
        "notional": notional,
        "estimated_entry_fee": estimated_entry_fee,
        "estimated_exit_fee": estimated_exit_fee,
    }, ""


def place_trade(side: str, current_price: float) -> Tuple[Optional[Dict[str, Any]], str, float]:
    global paper_usdt_balance
    usdt_before = get_usdt_balance()
    plan, plan_error = build_trade_plan(side, current_price, usdt_before)
    if plan is None:
        return None, plan_error, 0.0

    is_long = side == "LONG"
    entry_price = float(plan["entry_price"])
    stop_price = float(plan["stop_price"])
    take_profit_price = float(plan["take_profit_price"])
    qty = float(plan["qty"])
    notional = float(plan["notional"])
    estimated_entry_fee = float(plan["estimated_entry_fee"])
    estimated_exit_fee = float(plan["estimated_exit_fee"])

    if PAPER_MODE:
        if estimated_entry_fee > paper_usdt_balance:
            return None, "skipped_balance_insufficient", qty
        paper_usdt_balance -= estimated_entry_fee
        log_event(
            f"[PAPER ENTRY] side={side} entry={entry_price:.2f} qty={qty:.6f} stop={stop_price:.2f} tp={take_profit_price:.2f} "
            f"notional={notional:.4f} entry_fee={estimated_entry_fee:.4f} paper_usdt_balance={paper_usdt_balance:.4f}"
        )
        return {
            "entry_time_ms": int(time.time() * 1000),
            "position_size": qty,
            "position_qty": qty,
            "position_notional": notional,
            "side": side,
            "entry_price": entry_price,
            "entry_fee": estimated_entry_fee,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "usdt_before": usdt_before,
            "estimated_exit_fee": estimated_exit_fee,
        }, "", qty

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
    log_event(
        f"[REAL ENTRY] close={current_price:.2f} entry={entry_price:.2f} qty={qty:.6f} side={side} stop={stop_price:.2f} tp={take_profit_price:.2f}"
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
    }, "", qty


in_position = False
executing_trade = False
position_entry_price: Optional[float] = None
position_stop_price: Optional[float] = None
position_tp_price: Optional[float] = None
position_side: Optional[str] = None


def execute_trade(side: str, entry_price: float) -> Tuple[Optional[Dict[str, Any]], str]:
    global in_position, executing_trade, position_entry_price, position_stop_price, position_tp_price, position_side
    log_event("[DEBUG] entering execution path")
    log_event(f"[DEBUG] in_position before = {in_position}")
    log_event(f"[DEBUG] requested_entry_price={entry_price:.2f}")
    executing_trade = True
    log_event("[DEBUG] executing_trade=True")
    try:
        trade, skip_reason, qty = place_trade(side, entry_price)
        log_event(f"[DEBUG] qty calculated = {qty:.6f}")
    except Exception as exc:
        executing_trade = False
        log_event("[DEBUG] executing_trade=False")
        log_event(f"[DEBUG] skipped_execution_exception: {type(exc).__name__}: {exc}")
        log_event(f"[DEBUG] in_position after = {in_position}")
        log_event("[DEBUG] execution success = False")
        return None, "skipped_execution_exception"

    if not trade:
        executing_trade = False
        log_event("[DEBUG] executing_trade=False")
        reason = skip_reason or "skipped_qty_invalid"
        log_event(f"[DEBUG] {reason}")
        log_event(f"[DEBUG] in_position after = {in_position}")
        log_event("[DEBUG] execution success = False")
        return None, reason

    in_position = True
    position_entry_price = float(trade["entry_price"])
    position_stop_price = safe_float(trade.get("stop_price"))
    position_tp_price = safe_float(trade.get("take_profit_price", trade.get("tp_price")))
    position_side = side
    executing_trade = False
    log_event("[DEBUG] executing_trade=False")
    log_event(f"[DEBUG] in_position after = {in_position}")
    log_event("[DEBUG] execution success = True")
    log_event(f"[EXECUTION] OPEN {side} at {position_entry_price:.2f}")
    return trade, ""


def add_signal_marker(side: str, price: float, candle_time_ms: int) -> None:
    if side not in {"LONG", "SHORT"}:
        return
    recent_signal_markers.append(
        {
            "time": ms_to_utc_iso(candle_time_ms),
            "side": side,
            "kind": f"SIGNAL_{side}",
            "price": price,
        }
    )


ensure_runtime_files()
bootstrap_metrics_from_trades()
existing_trades = load_trade_log()
if existing_trades:
    last_trade_snapshot = existing_trades[-1]
if PAPER_MODE and total_trades > 0:
    paper_usdt_balance = CAPITAL_USDT + net_pnl_total
    paper_balance_base = paper_usdt_balance
    paper_balance_source = "trade_log_reconstructed"
elif PAPER_MODE:
    paper_balance_base = paper_usdt_balance
    paper_balance_source = "env"

if PAPER_MODE:
    apply_paper_balance_config_if_needed(active_trade=None)
log_event("Starting BTCUSDT 1m breakout scalping bot (BINANCE FUTURES USDT-M)...")
if VALIDATION_MODE:
    log_event("VALIDATION_MODE enabled: using shorter lookbacks and relaxed volume threshold.")
sync_time_offset()
ensure_futures_settings()

last_processed_candle_time: Optional[int] = None
trade_times: Deque[int] = deque()
consecutive_losses = 0
active_trade: Optional[Dict[str, Any]] = None
last_market_state = "TRADEABLE"

while True:
    try:
        now_ms = int(time.time() * 1000)
        while trade_times and now_ms - trade_times[0] > 3600 * 1000:
            trade_times.popleft()
        apply_paper_balance_config_if_needed(active_trade)

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

                if exit_reason and exit_price is not None:
                    gross_pnl = active_trade["position_size"] * (exit_price - active_trade["entry_price"]) if active_trade["side"] == "LONG" else active_trade["position_size"] * (active_trade["entry_price"] - exit_price)
                    exit_fee = active_trade["position_notional"] * FUTURES_TAKER_FEE_RATE
                    total_fees = active_trade["entry_fee"] + exit_fee
                    net_pnl_value = gross_pnl - total_fees
                    duration_minutes = max(0.0, (int(time.time() * 1000) - int(active_trade.get("entry_time_ms", int(time.time() * 1000)))) / 60000.0)
                    paper_usdt_balance += net_pnl_value
                    consecutive_losses = consecutive_losses + 1 if exit_reason == "LOSS" else 0
                    log_event(
                        f"[PAPER EXIT {exit_reason}] exit={exit_price:.2f} qty={active_trade['position_size']:.6f} side={active_trade['side']} "
                        f"gross_pnl={gross_pnl:.4f} fees={total_fees:.4f} net_pnl={net_pnl_value:.4f} balance={paper_usdt_balance:.4f}"
                    )
                    update_metrics(net_pnl_value, gross_pnl, total_fees, paper_usdt_balance)
                    update_trade_duration_metrics(duration_minutes)
                    log_trade_exit(
                        active_trade["side"],
                        active_trade["entry_price"],
                        exit_price,
                        active_trade["position_size"],
                        gross_pnl,
                        total_fees,
                        net_pnl_value,
                        paper_usdt_balance,
                        duration_minutes,
                    )
                    print_trade_closed_summary(
                        exit_reason,
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
                    in_position = False
                else:
                    log_event(
                        f"Paper trade still active. price={last_price:.2f} stop={active_trade['stop_price']:.2f} tp={active_trade['take_profit_price']:.2f}"
                    )
                    export_runtime_state(last_processed_candle_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
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
                    exit_price = get_last_price()
                    duration_minutes = max(0.0, (int(time.time() * 1000) - int(active_trade.get("entry_time_ms", int(time.time() * 1000)))) / 60000.0)
                    log_event(f"[FUTURES EXIT {result}] gross_pnl={gross_pnl:.4f} fees={fees:.4f} net_pnl={net_pnl_value:.4f} balance={usdt_now:.4f}")
                    consecutive_losses = consecutive_losses + 1 if net_pnl_value < 0 else 0
                    update_metrics(net_pnl_value, gross_pnl, fees, usdt_now)
                    update_trade_duration_metrics(duration_minutes)
                    log_trade_exit(
                        active_trade["side"],
                        active_trade["entry_price"],
                        exit_price,
                        active_trade["qty"],
                        gross_pnl,
                        fees,
                        net_pnl_value,
                        usdt_now,
                        duration_minutes,
                    )
                    print_trade_closed_summary(
                        result,
                        active_trade["side"],
                        active_trade["entry_price"],
                        exit_price,
                        active_trade["qty"],
                        gross_pnl,
                        fees,
                        net_pnl_value,
                        usdt_now,
                    )
                    active_trade = None
                    in_position = False
                else:
                    log_event("Futures trade still active. Waiting for stop/tp trigger...")
                    export_runtime_state(last_processed_candle_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
                    time.sleep(POLL_SECONDS)
                    continue

            if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                log_event("3 consecutive losses reached. Shutting down.")
                export_runtime_state(last_processed_candle_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
                break

        candle = get_closed_candle_data()
        if candle is None:
            last_reason = "not_enough_candles"
            log_event("Not enough candle data yet.")
            export_runtime_state(last_processed_candle_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
            time.sleep(POLL_SECONDS)
            continue

        candle_time = int(candle["close_time"])
        skipped_old_candle = last_processed_candle_time is not None and candle_time <= last_processed_candle_time
        log_event(f"[DEBUG] newest_closed_candle={candle_time}")
        log_event(f"[DEBUG] last_processed_candle_before={last_processed_candle_time}")
        log_event(f"[DEBUG] skipped_old_candle={skipped_old_candle}")

        if skipped_old_candle:
            last_reason = "no_new_candle"
            log_event("[DEBUG] skipped_old_candle")
            log_event("No new closed candle.")
            log_event(f"[DEBUG] last_processed_candle_after={last_processed_candle_time}")
            export_runtime_state(last_processed_candle_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
            time.sleep(POLL_SECONDS)
            continue

        last_market_state = candle["market_state"]
        log_event(f"[MARKET] {last_market_state}")

        trend, ema50, ema200 = get_trend_state()
        long_breakout = candle["close_price"] > candle["highest_high"] * MICRO_BREAKOUT_FACTOR
        short_breakdown = candle["close_price"] < candle["lowest_low"] * MICRO_BREAKDOWN_FACTOR
        volume_ok = candle["volume_score"] >= MIN_VOLUME_SCORE
        breakout_ok = long_breakout or short_breakdown
        candle_valid = candle["close_time"] is not None
        current_price = candle["close_price"]
        candle_high = candle["high_price"]
        candle_low = candle["low_price"]
        long_signal = long_breakout and volume_ok
        short_signal = short_breakdown and volume_ok
        trigger_price = candle["highest_high"] * MICRO_BREAKOUT_FACTOR if long_signal else candle["lowest_low"] * MICRO_BREAKDOWN_FACTOR
        if DEBUG_DISABLE_PRICE_FILTER:
            price_valid = True
            print("[DEBUG] price filter disabled")
        elif long_signal:
            price_valid = candle_high >= trigger_price * 0.999
        elif short_signal:
            price_valid = candle_low <= trigger_price * 1.001
        else:
            price_valid = current_price > 0
        trend_aligned = (long_signal and trend == "BULL") or (short_signal and trend == "BEAR")
        signal_detected = long_signal or short_signal
        log_event(f"[DEBUG] signal_detected={signal_detected}")

        if long_signal:
            last_signal = "LONG"
            add_signal_marker("LONG", candle["close_price"], candle_time)
            trend_note = " trend_aligned" if trend_aligned else ""
            log_event(f"[SIGNAL] LONG close={candle['close_price']:.2f} volume_score={candle['volume_score']:.4f}{trend_note}")
        elif short_signal:
            last_signal = "SHORT"
            add_signal_marker("SHORT", candle["close_price"], candle_time)
            trend_note = " trend_aligned" if trend_aligned else ""
            log_event(f"[SIGNAL] SHORT close={candle['close_price']:.2f} volume_score={candle['volume_score']:.4f}{trend_note}")
        else:
            last_signal = "NONE"

        execution_skip_reason = None
        trade = None
        trades_this_hour = len(trade_times)
        in_position = is_in_position(active_trade)

        print("[DEBUG EXECUTION CONDITIONS]")
        print(f"signal_detected={signal_detected}")
        print(f"in_position={in_position}")
        print(f"trades_this_hour={trades_this_hour}")
        print(f"max_trades={MAX_TRADES_PER_HOUR}")
        print(f"volume_ok={volume_ok}")
        print(f"breakout_ok={breakout_ok}")
        print(f"candle_valid={candle_valid}")
        print(f"[DEBUG] trigger_price={trigger_price:.2f}")
        print(f"[DEBUG] current_price={current_price:.2f}")
        print(f"[DEBUG] candle_high={candle_high:.2f}")
        print(f"[DEBUG] candle_low={candle_low:.2f}")
        print(f"[DEBUG] price_valid={price_valid}")
        print(f"price_valid={price_valid}")

        executing_trade = (
            signal_detected
            and not in_position
            and trades_this_hour < MAX_TRADES_PER_HOUR
            and volume_ok
            and breakout_ok
            and candle_valid
            and price_valid
        )
        print(f"[DEBUG] executing_trade={executing_trade}")

        if not executing_trade:
            if not signal_detected:
                print("skip_reason=signal_false")
            elif in_position:
                print("skip_reason=already_in_position")
            elif trades_this_hour >= MAX_TRADES_PER_HOUR:
                print("skip_reason=trade_limit")
            elif not volume_ok:
                print("skip_reason=volume_filter")
            elif not breakout_ok:
                print("skip_reason=breakout_filter")
            elif not candle_valid:
                print("skip_reason=candle_filter")
            elif not price_valid:
                print("skip_reason=price_filter")

        if DEBUG_FORCE_ENTRY and signal_detected and not in_position:
            print("[DEBUG] FORCING TRADE EXECUTION")
            executing_trade = True
            print(f"[DEBUG] executing_trade={executing_trade}")

        if not signal_detected:
            if not volume_ok:
                last_reason = "low_volume"
            elif not long_breakout and not short_breakdown:
                last_reason = "no_breakout"
            else:
                last_reason = "no_signal"
        elif in_position:
            execution_skip_reason = "skipped_in_position"
            last_reason = "in_position"
            in_position = True
        elif trades_this_hour >= MAX_TRADES_PER_HOUR:
            execution_skip_reason = "skipped_trade_limit"
            last_reason = "trade_limit"
        elif not candle_valid:
            execution_skip_reason = "skipped_candle_invalid"
            last_reason = "candle_invalid"
        elif not price_valid:
            execution_skip_reason = "skipped_price_invalid"
            last_reason = "price_invalid"
        else:
            side = "LONG" if long_signal else "SHORT"
            last_reason = "breakout + volume"
            log_event("[DEBUG] signal accepted")
            if executing_trade:
                trade, execution_error = execute_trade(side, candle["close_price"])
                if trade:
                    trade_times.append(int(time.time() * 1000))
                    active_trade = trade
                    log_event(f"Trades this hour: {len(trade_times)}/{MAX_TRADES_PER_HOUR}")
                else:
                    execution_skip_reason = execution_error or "skipped_qty_invalid"
            else:
                execution_skip_reason = "skipped_execution_gate"

        if execution_skip_reason:
            log_event(f"[DEBUG] {execution_skip_reason}")
            log_event("[DEBUG] executing_trade=False")
            log_event(f"No trade: {last_reason}")

        if last_processed_candle_time is None or candle_time > last_processed_candle_time:
            last_processed_candle_time = candle_time

        log_event(f"[DEBUG] last_processed_candle_after={last_processed_candle_time}")
        print_heartbeat(last_processed_candle_time, len(trade_times), consecutive_losses, active_trade)
        export_runtime_state(last_processed_candle_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
        time.sleep(POLL_SECONDS)
    except Exception as exc:
        if executing_trade:
            log_event("[DEBUG] skipped_execution_exception")
            executing_trade = False
            log_event("[DEBUG] executing_trade=False")
        log_event(f"Error: {exc}")
        export_runtime_state(last_processed_candle_time, len(trade_times), consecutive_losses, active_trade, last_market_state)
        time.sleep(POLL_SECONDS)
