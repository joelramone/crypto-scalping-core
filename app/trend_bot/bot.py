"""Daily BTC trend bot: the frozen rule of HYP-DAILY-TREND-BTC-001, executed on Binance spot.

Once per day, after the 00:00 UTC daily close, the bot:
1. downloads the last closed daily candles (public data, no key needed);
2. computes the target BTC fraction with the frozen rule (0, 1/3, 2/3 or 1);
3. rebalances its own capped "sleeve" (USDT + BTC) toward the target.

Modes:
- paper (default): simulated fills at the current price with a 0.1% fee. No keys.
- live: real market orders on Binance spot. Requires an API key WITHOUT
  withdrawal permission, refuses to run otherwise, and never trades more than
  the sleeve it tracks (capped by TREND_BOT_CAPITAL_USDT).

Safety: long/flat spot only, no leverage, a kill-switch file, one run per
daily candle (idempotent), and an append-only JSONL ledger.

    python -m app.trend_bot.bot            # uses environment variables, see README.md
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlencode

import pandas as pd
import requests

from app.research.daily_trend import FEE_RATE, positions

SYMBOL = "BTCUSDT"
PUBLIC_URLS = ("https://data-api.binance.vision", "https://api.binance.com")
TRADE_URL = "https://api.binance.com"
DAY_MS = 86_400_000
HISTORY_DAYS = 260
MIN_ORDER_USDT = 10.0
DRIFT_TOLERANCE = 0.10
TIMEOUT = 20


@dataclass
class Config:
    mode: str = "paper"
    capital_usdt: float = 1000.0
    state_dir: Path = Path("research/paper/trend_bot_v1")
    kill_switch: Path | None = None
    api_key: str | None = None
    api_secret: str | None = None
    telegram_token: str | None = None
    telegram_chat_id: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        mode = os.getenv("TREND_BOT_MODE", "paper").strip().lower()
        if mode not in {"paper", "live"}:
            raise SystemExit("TREND_BOT_MODE must be 'paper' or 'live'")
        default_dir = "research/paper/trend_bot_v1" if mode == "paper" else str(Path.home() / ".trend_bot")
        state_dir = Path(os.getenv("TREND_BOT_STATE_DIR", default_dir))
        capital = float(os.getenv("TREND_BOT_CAPITAL_USDT", "1000"))
        if not (0 < capital <= 1_000_000):
            raise SystemExit("TREND_BOT_CAPITAL_USDT must be positive")
        return cls(mode=mode, capital_usdt=capital, state_dir=state_dir,
                   kill_switch=Path(os.getenv("TREND_BOT_KILL_SWITCH", str(state_dir / "STOP"))),
                   api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"),
                   telegram_token=os.getenv("TELEGRAM_BOT_TOKEN"), telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"))


# ---------------------------------------------------------------- market data

def public_get(path: str, params: dict[str, Any]) -> Any:
    errors = []
    for base in PUBLIC_URLS:
        for attempt in range(3):
            try:
                response = requests.get(base + path, params=params, timeout=TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                errors.append(f"{base}: {exc}")
                time.sleep(2 ** attempt)
    raise RuntimeError("Public market data unavailable: " + " | ".join(errors[-2:]))


def closed_daily_candles(now_ms: int, get: Callable[[str, dict[str, Any]], Any] = public_get) -> pd.DataFrame:
    rows = get("/api/v3/klines", {"symbol": SYMBOL, "interval": "1d", "limit": HISTORY_DAYS + 2})
    frame = pd.DataFrame([r[:7] for r in rows], columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    frame = frame[frame["close_time"].astype("int64") < now_ms].reset_index(drop=True)
    frame["close"] = frame["close"].astype(float)
    times = frame["open_time"].astype("int64")
    if len(frame) < 200 or not (times.diff().iloc[1:] == DAY_MS).all():
        raise ValueError("Daily history is incomplete or not contiguous")
    if not frame["close"].map(math.isfinite).all() or (frame["close"] <= 0).any():
        raise ValueError("Invalid close prices")
    frame["date"] = pd.to_datetime(times, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return frame


def target_fraction(candles: pd.DataFrame) -> float:
    value = positions(candles["close"]).iloc[-1]
    if pd.isna(value):
        raise ValueError("Not enough history for the 200-day SMA")
    return float(value)


def last_price(get: Callable[[str, dict[str, Any]], Any] = public_get) -> float:
    price = float(get("/api/v3/ticker/price", {"symbol": SYMBOL})["price"])
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Invalid ticker price")
    return price


# ---------------------------------------------------------------- brokers

@dataclass
class Fill:
    side: str
    btc: float
    usdt: float
    fee_usdt: float
    price: float
    order_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Broker(Protocol):
    def preflight(self, sleeve: dict[str, float]) -> None: ...
    def buy(self, usdt: float, price: float, client_id: str) -> Fill: ...
    def sell(self, btc: float, price: float, client_id: str) -> Fill: ...


class PaperBroker:
    def preflight(self, sleeve: dict[str, float]) -> None:
        return None

    def buy(self, usdt: float, price: float, client_id: str) -> Fill:
        fee = usdt * FEE_RATE
        return Fill("BUY", (usdt - fee) / price, usdt, fee, price, client_id)

    def sell(self, btc: float, price: float, client_id: str) -> Fill:
        gross = btc * price
        fee = gross * FEE_RATE
        return Fill("SELL", btc, gross - fee, fee, price, client_id)


def floor_step(value: float, step: str) -> float:
    step_d = Decimal(step)
    return float((Decimal(str(value)) / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d)


class LiveBroker:
    """Real Binance spot orders. Refuses keys that can withdraw."""

    def __init__(self, api_key: str, api_secret: str, session: requests.Session | None = None) -> None:
        if not api_key or not api_secret:
            raise SystemExit("Live mode needs BINANCE_API_KEY and BINANCE_API_SECRET")
        self.key, self.secret = api_key, api_secret.encode()
        self.session = session or requests.Session()
        self.step = "0.00001"

    def _signed(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {}, timestamp=int(time.time() * 1000), recvWindow=10_000)
        query = urlencode(params)
        signature = hmac.new(self.secret, query.encode(), hashlib.sha256).hexdigest()
        response = self.session.request(method, f"{TRADE_URL}{path}?{query}&signature={signature}",
                                        headers={"X-MBX-APIKEY": self.key}, timeout=TIMEOUT)
        if response.status_code >= 400:
            raise RuntimeError(f"Binance {path} failed: HTTP {response.status_code} {response.text[:300]}")
        return response.json()

    def preflight(self, sleeve: dict[str, float]) -> None:
        restrictions = self._signed("GET", "/sapi/v1/account/apiRestrictions")
        if restrictions.get("enableWithdrawals"):
            raise SystemExit("Refusing to run: this API key can withdraw funds. Create a key without withdrawals.")
        if not restrictions.get("enableSpotAndMarginTrading"):
            raise SystemExit("This API key cannot trade spot. Enable 'Spot & Margin Trading' on the key.")
        info = self.session.get(f"{TRADE_URL}/api/v3/exchangeInfo", params={"symbol": SYMBOL}, timeout=TIMEOUT).json()
        lot = next(f for f in info["symbols"][0]["filters"] if f["filterType"] == "LOT_SIZE")
        self.step = lot["stepSize"]
        balances = {b["asset"]: float(b["free"]) for b in self._signed("GET", "/api/v3/account")["balances"]}
        if balances.get("USDT", 0.0) + 1e-9 < sleeve["usdt"] or balances.get("BTC", 0.0) + 1e-12 < sleeve["btc"]:
            raise SystemExit(f"Account balances {balances.get('USDT', 0):.2f} USDT / {balances.get('BTC', 0):.8f} BTC "
                             f"do not cover the bot's sleeve {sleeve['usdt']:.2f} USDT / {sleeve['btc']:.8f} BTC")

    def _fill(self, side: str, order: dict[str, Any], price: float) -> Fill:
        btc = float(order["executedQty"])
        usdt = float(order["cummulativeQuoteQty"])
        fee_usdt = 0.0
        for f in order.get("fills", []):
            commission = float(f["commission"])
            if f["commissionAsset"] == "USDT":
                fee_usdt += commission
            elif f["commissionAsset"] == "BTC":
                fee_usdt += commission * float(f["price"])
                btc = btc - commission if side == "BUY" else btc
        if side == "SELL":
            usdt -= sum(float(f["commission"]) for f in order.get("fills", []) if f["commissionAsset"] == "USDT")
        avg = float(order["cummulativeQuoteQty"]) / float(order["executedQty"]) if float(order["executedQty"]) else price
        return Fill(side, btc, usdt, fee_usdt, avg, str(order.get("orderId")), order)

    def buy(self, usdt: float, price: float, client_id: str) -> Fill:
        order = self._signed("POST", "/api/v3/order", {"symbol": SYMBOL, "side": "BUY", "type": "MARKET",
                                                       "quoteOrderQty": f"{math.floor(usdt * 100) / 100:.2f}",
                                                       "newClientOrderId": client_id, "newOrderRespType": "FULL"})
        return self._fill("BUY", order, price)

    def sell(self, btc: float, price: float, client_id: str) -> Fill:
        qty = floor_step(btc, self.step)
        order = self._signed("POST", "/api/v3/order", {"symbol": SYMBOL, "side": "SELL", "type": "MARKET",
                                                       "quantity": f"{qty:.8f}".rstrip("0").rstrip("."),
                                                       "newClientOrderId": client_id, "newOrderRespType": "FULL"})
        return self._fill("SELL", order, price)


# ---------------------------------------------------------------- state and run

def load_state(config: Config) -> dict[str, Any]:
    path = config.state_dir / "state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["mode"] != config.mode:
            raise SystemExit(f"State in {path} belongs to {state['mode']} mode")
        return state
    return {"mode": config.mode, "symbol": SYMBOL, "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "initial_capital_usdt": config.capital_usdt, "sleeve": {"usdt": config.capital_usdt, "btc": 0.0},
            "last_candle_date": None, "last_target": None, "total_fees_usdt": 0.0, "trades": 0}


def save_state(config: Config, state: dict[str, Any], event: dict[str, Any]) -> None:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    tmp = config.state_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(config.state_dir / "state.json")
    with (config.state_dir / "ledger.jsonl").open("a") as ledger:
        ledger.write(json.dumps(event, sort_keys=True) + "\n")


def notify(config: Config, text: str) -> None:
    if not (config.telegram_token and config.telegram_chat_id):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{config.telegram_token}/sendMessage",
                      json={"chat_id": config.telegram_chat_id, "text": text}, timeout=TIMEOUT)
    except requests.RequestException:
        pass


def plan_trade(sleeve: dict[str, float], price: float, target: float, last_target: float | None) -> tuple[str, float]:
    """Return (side, amount): amount in USDT for BUY, BTC for SELL; ('HOLD', 0) if no trade."""
    equity = sleeve["usdt"] + sleeve["btc"] * price
    current = sleeve["btc"] * price / equity if equity > 0 else 0.0
    if target == last_target and abs(current - target) <= DRIFT_TOLERANCE:
        return "HOLD", 0.0
    delta_usdt = (target - current) * equity
    if abs(delta_usdt) < MIN_ORDER_USDT:
        return "HOLD", 0.0
    if delta_usdt > 0:
        return "BUY", min(delta_usdt, sleeve["usdt"])
    btc = sleeve["btc"] if target == 0.0 else min(-delta_usdt / price, sleeve["btc"])
    return "SELL", btc


def run_once(config: Config, broker: Broker | None = None, now_ms: int | None = None,
             get: Callable[[str, dict[str, Any]], Any] = public_get) -> dict[str, Any]:
    if config.kill_switch and config.kill_switch.exists():
        return {"status": "STOPPED", "reason": f"kill switch present: {config.kill_switch}"}
    now_ms = now_ms or int(time.time() * 1000)
    state = load_state(config)
    candles = closed_daily_candles(now_ms, get)
    candle_date = candles["date"].iloc[-1]
    if state["last_candle_date"] == candle_date:
        return {"status": "ALREADY_RAN", "candle_date": candle_date}
    target = target_fraction(candles)
    price = last_price(get)
    broker = broker or (PaperBroker() if config.mode == "paper" else LiveBroker(config.api_key or "", config.api_secret or ""))
    broker.preflight(state["sleeve"])
    side, amount = plan_trade(state["sleeve"], price, target, state["last_target"])
    fill = None
    client_id = f"trend-{candle_date}-{side.lower()}"
    if side == "BUY":
        fill = broker.buy(amount, price, client_id)
        state["sleeve"]["usdt"] -= fill.usdt
        state["sleeve"]["btc"] += fill.btc
    elif side == "SELL":
        fill = broker.sell(amount, price, client_id)
        state["sleeve"]["btc"] -= fill.btc
        state["sleeve"]["usdt"] += fill.usdt
    state["sleeve"] = {k: max(v, 0.0) for k, v in state["sleeve"].items()}
    if fill:
        state["total_fees_usdt"] += fill.fee_usdt
        state["trades"] += 1
    state["last_candle_date"], state["last_target"] = candle_date, target
    equity = state["sleeve"]["usdt"] + state["sleeve"]["btc"] * price
    event = {"run_at_utc": datetime.fromtimestamp(now_ms / 1000, timezone.utc).isoformat(timespec="seconds"),
             "mode": config.mode, "candle_date": candle_date, "close": float(candles["close"].iloc[-1]),
             "price": price, "target_fraction": target, "action": side,
             "fill": None if fill is None else {k: v for k, v in fill.__dict__.items() if k != "raw"},
             "sleeve": dict(state["sleeve"]), "equity_usdt": equity,
             "return_since_start": equity / state["initial_capital_usdt"] - 1.0}
    save_state(config, state, event)
    notify(config, f"[trend bot {config.mode}] {candle_date}: target {target:.0%} BTC, action {side}, "
                   f"equity {equity:.2f} USDT ({event['return_since_start']:+.2%})")
    return {"status": "OK", **event}


def main() -> None:
    config = Config.from_env()
    try:
        result = run_once(config)
    except BaseException as exc:
        notify(config, f"[trend bot {config.mode}] ERROR: {exc}")
        raise
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
