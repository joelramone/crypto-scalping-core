import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, redirect, render_template_string, request, url_for

app = Flask(__name__)
RUNTIME_DIR = Path("runtime")
STATE_FILE = RUNTIME_DIR / "dashboard_state.json"
TRADES_FILE = RUNTIME_DIR / "trades_log.json"
CANDLES_FILE = RUNTIME_DIR / "candles.json"
PAPER_CONFIG_FILE = RUNTIME_DIR / "paper_config.json"

UTC_TZ = timezone.utc
ARG_TZ = ZoneInfo("America/Argentina/Buenos_Aires")


EMPTY_CANDLES_PAYLOAD: Dict[str, Any] = {
    "symbol": "BTCUSDT",
    "timeframe": "1m",
    "updated_at_utc": None,
    "candles": [],
    "ema_fast": [],
    "ema_slow": [],
    "levels": {
        "entry_price": None,
        "stop_price": None,
        "take_profit_price": None,
    },
    "signal_markers": [],
    "trade_markers": [],
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_num(value: Any, ndigits: int = 4) -> str:
    num = to_float(value)
    if num is None:
        return "N/A"
    return f"{num:.{ndigits}f}"


def fmt_signed_num(value: Any, ndigits: int = 4) -> str:
    num = to_float(value)
    if num is None:
        return "N/A"
    return f"{num:+.{ndigits}f}"


def pnl_class(value: Any) -> str:
    num = to_float(value)
    if num is None:
        return "neutral"
    if num > 0:
        return "positive"
    if num < 0:
        return "negative"
    return "neutral"


def bot_status(state: Dict[str, Any], updated_at: str) -> str:
    if bool(state.get("in_position", False)):
        return "IN POSITION"
    if isinstance(updated_at, str) and updated_at != "N/A":
        return "RUNNING"
    return "IDLE"


def parse_tz_param(value: Optional[str]) -> str:
    return "arg" if str(value or "").lower() == "arg" else "utc"


def get_timezone_from_param(value: str) -> timezone:
    return ARG_TZ if value == "arg" else UTC_TZ


def parse_timestamp_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC_TZ)
        return dt.astimezone(UTC_TZ)
    except (TypeError, ValueError):
        return None


def format_timestamp(value: Any, selected_tz: str) -> str:
    dt = parse_timestamp_utc(value)
    if dt is None:
        return str(value or "N/A")
    tz_obj = get_timezone_from_param(selected_tz)
    return dt.astimezone(tz_obj).strftime("%Y-%m-%d %H:%M:%S")


def read_paper_config() -> Dict[str, Any]:
    data = load_json(PAPER_CONFIG_FILE, {})
    return data if isinstance(data, dict) else {}


def write_paper_config(balance: float) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_balance": balance,
        "updated_at_utc": datetime.now(tz=UTC_TZ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    PAPER_CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _add_marker_traces(fig: go.Figure, markers: List[Dict[str, Any]], styles: Dict[str, Dict[str, Any]], default_name: str) -> None:
    if not markers:
        return

    marker_frame = pd.DataFrame(markers)
    if marker_frame.empty:
        return

    marker_frame["time"] = pd.to_datetime(marker_frame["time"], utc=True, errors="coerce")
    marker_frame["price"] = marker_frame["price"].map(to_float)
    marker_frame = marker_frame.dropna(subset=["time", "price"])
    if marker_frame.empty:
        return

    for kind, group in marker_frame.groupby("kind"):
        style = styles.get(kind)
        if not style:
            continue
        name = style.get("name", default_name)
        fig.add_trace(
            go.Scatter(
                x=group["time"],
                y=group["price"],
                mode="markers",
                marker={
                    "color": style["color"],
                    "size": style["size"],
                    "symbol": style["symbol"],
                    "opacity": style["opacity"],
                    "line": {"color": style.get("line_color", style["color"]), "width": style.get("line_width", 1)},
                },
                name=name,
            )
        )


def build_chart_html(candles_payload: Dict[str, Any], selected_tz: str, window_size: str, active_trade: Optional[Dict[str, Any]]) -> str:
    candles = candles_payload.get("candles", [])
    if not candles:
        return "<div class='empty-state'>Candlestick data is not available yet. Start live_bot.py to begin streaming.</div>"

    frame = pd.DataFrame(candles)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["time"])
    if frame.empty:
        return "<div class='empty-state'>Candlestick data is not available yet. Start live_bot.py to begin streaming.</div>"

    display_tz = get_timezone_from_param(selected_tz)
    frame["time_local"] = frame["time"].dt.tz_convert(display_tz)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame["time_local"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Candles",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            increasing_line_width=1.5,
            decreasing_line_width=1.5,
            customdata=frame[["volume"]],
            hovertemplate=(
                "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                "Open: %{open:.4f}<br>"
                "High: %{high:.4f}<br>"
                "Low: %{low:.4f}<br>"
                "Close: %{close:.4f}<br>"
                "Volume: %{customdata[0]:.4f}<extra></extra>"
            ),
        )
    )

    ema_fast = pd.DataFrame(candles_payload.get("ema_fast", []))
    if not ema_fast.empty:
        ema_fast["time"] = pd.to_datetime(ema_fast["time"], utc=True, errors="coerce")
        ema_fast = ema_fast.dropna(subset=["time"])
        ema_fast["time_local"] = ema_fast["time"].dt.tz_convert(display_tz)
        fig.add_trace(
            go.Scatter(
                x=ema_fast["time_local"],
                y=ema_fast["value"],
                mode="lines",
                name="EMA Fast",
                line={"color": "#eab308", "width": 2},
            )
        )

    ema_slow = pd.DataFrame(candles_payload.get("ema_slow", []))
    if not ema_slow.empty:
        ema_slow["time"] = pd.to_datetime(ema_slow["time"], utc=True, errors="coerce")
        ema_slow = ema_slow.dropna(subset=["time"])
        ema_slow["time_local"] = ema_slow["time"].dt.tz_convert(display_tz)
        fig.add_trace(
            go.Scatter(
                x=ema_slow["time_local"],
                y=ema_slow["value"],
                mode="lines",
                name="EMA Slow",
                line={"color": "#38bdf8", "width": 2},
            )
        )

    signal_styles: Dict[str, Dict[str, Any]] = {
        "SIGNAL_LONG": {"name": "Signal LONG", "color": "rgba(34, 197, 94, 0.45)", "symbol": "circle", "size": 8, "opacity": 0.65},
        "SIGNAL_SHORT": {"name": "Signal SHORT", "color": "rgba(239, 68, 68, 0.45)", "symbol": "circle", "size": 8, "opacity": 0.65},
    }

    trade_styles: Dict[str, Dict[str, Any]] = {
        "EXECUTED_ENTRY_LONG": {"name": "LONG Entry", "color": "#22c55e", "symbol": "diamond", "size": 11, "opacity": 1.0},
        "EXECUTED_ENTRY_SHORT": {"name": "SHORT Entry", "color": "#ef4444", "symbol": "diamond", "size": 11, "opacity": 1.0},
        "EXIT_WIN": {"name": "WIN Exit", "color": "#22d3ee", "symbol": "triangle-up", "size": 12, "opacity": 1.0},
        "EXIT_LOSS": {"name": "LOSS Exit", "color": "#fb923c", "symbol": "triangle-down", "size": 12, "opacity": 1.0},
    }

    signal_markers = candles_payload.get("signal_markers", [])
    normalized_signal_markers: List[Dict[str, Any]] = []
    for marker in signal_markers:
        side = str(marker.get("side", "")).upper()
        normalized_signal_markers.append({"time": marker.get("time"), "price": marker.get("price"), "kind": f"SIGNAL_{side}"})
    for marker in normalized_signal_markers:
        marker_time = parse_timestamp_utc(marker.get("time"))
        marker["time"] = marker_time.astimezone(display_tz) if marker_time else None
    _add_marker_traces(fig, normalized_signal_markers, signal_styles, "Signal")

    trade_markers = candles_payload.get("trade_markers", [])
    normalized_trade_markers: List[Dict[str, Any]] = []
    for marker in trade_markers:
        side = str(marker.get("side", "")).upper()
        kind = str(marker.get("kind", "")).upper()
        normalized_kind = f"EXECUTED_ENTRY_{side}" if kind == "EXECUTED_ENTRY" else kind
        normalized_trade_markers.append({"time": marker.get("time"), "price": marker.get("price"), "kind": normalized_kind})
    for marker in normalized_trade_markers:
        marker_time = parse_timestamp_utc(marker.get("time"))
        marker["time"] = marker_time.astimezone(display_tz) if marker_time else None
    _add_marker_traces(fig, normalized_trade_markers, trade_styles, "Trade")

    levels = candles_payload.get("levels", {}) or {}
    entry_price = to_float(levels.get("entry_price"))
    stop_price = to_float(levels.get("stop_price"))
    tp_price = to_float(levels.get("take_profit_price"))

    for key, color, label in [
        ("entry_price", "#a78bfa", "Entry"),
        ("stop_price", "#ef4444", "Stop"),
        ("take_profit_price", "#22c55e", "Take Profit"),
    ]:
        level_price = to_float(levels.get(key))
        if level_price is not None:
            fig.add_hline(y=level_price, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top left")

    if stop_price is not None and tp_price is not None:
        fig.add_hrect(
            y0=min(stop_price, tp_price),
            y1=max(stop_price, tp_price),
            fillcolor="rgba(148, 163, 184, 0.12)",
            line_width=0,
            annotation_text="Open Position Zone",
            annotation_position="top right",
        )

    if window_size == "all":
        window_count = len(frame)
    elif window_size == "50":
        window_count = 50
    elif window_size == "200":
        window_count = 200
    else:
        window_count = 100

    latest_index = len(frame) - 1
    anchor_index = latest_index
    if active_trade:
        entry_price = to_float(active_trade.get("entry_price"))
        if entry_price is not None:
            idx = (frame["close"].astype(float) - entry_price).abs().idxmin()
            anchor_index = int(idx)

    if window_size != "all" and len(frame) > window_count:
        half = max(1, window_count // 2)
        start_idx = max(0, anchor_index - half)
        end_idx = min(latest_index, start_idx + window_count - 1)
        start_idx = max(0, end_idx - window_count + 1)
        x_start = frame.iloc[start_idx]["time_local"]
        x_end = frame.iloc[end_idx]["time_local"]
        fig.update_xaxes(range=[x_start, x_end])

    fig.update_layout(
        template="plotly_dark",
        height=640,
        margin={"l": 20, "r": 20, "t": 24, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        hovermode="x unified",
        dragmode="zoom",
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1)

    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "doubleClick": "reset+autosize",
            "modeBarButtonsToRemove": [],
        },
    )


def normalize_trade_row(trade: Dict[str, Any], selected_tz: str) -> Dict[str, Any]:
    row = dict(trade)
    row["result"] = str(row.get("result") or "N/A").upper()
    row["side"] = str(row.get("side") or "N/A").upper()
    row["net_pnl_class"] = pnl_class(row.get("net_pnl"))
    row["net_pnl_fmt"] = fmt_signed_num(row.get("net_pnl"), 4)
    row["entry_price_fmt"] = fmt_num(row.get("entry_price"), 2)
    row["exit_price_fmt"] = fmt_num(row.get("exit_price"), 2)
    row["qty_fmt"] = fmt_num(row.get("qty"), 6)
    row["gross_pnl_fmt"] = fmt_signed_num(row.get("gross_pnl"), 4)
    row["fees_fmt"] = fmt_num(row.get("fees"), 4)
    row["balance_after_fmt"] = fmt_num(row.get("balance_after"), 4)
    row["duration_minutes_fmt"] = fmt_num(row.get("duration_minutes"), 2)
    row["timestamp_fmt"] = format_timestamp(row.get("timestamp"), selected_tz)
    return row


def parse_history_filter(value: Optional[str]) -> str:
    normalized = str(value or "all").lower()
    if normalized in {"all", "long", "short", "win", "loss"}:
        return normalized
    return "all"


def apply_trade_filter(trades: List[Dict[str, Any]], filter_key: str) -> List[Dict[str, Any]]:
    if filter_key == "long":
        return [t for t in trades if str(t.get("side")).upper() == "LONG"]
    if filter_key == "short":
        return [t for t in trades if str(t.get("side")).upper() == "SHORT"]
    if filter_key == "win":
        return [t for t in trades if str(t.get("result")).upper() == "WIN"]
    if filter_key == "loss":
        return [t for t in trades if str(t.get("result")).upper() == "LOSS"]
    return trades


def compute_status_pills(state: Dict[str, Any], symbol: str, timeframe: str, updated_at: str) -> List[Tuple[str, str]]:
    return [
        ("Bot Status", bot_status(state, updated_at)),
        ("Mode", str(state.get("mode", "N/A")).upper()),
        ("Symbol", symbol),
        ("Timeframe", timeframe),
        ("Last Update", updated_at),
    ]


TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="5">
  <title>Crypto Scalping Core Dashboard</title>
  <style>
    body { background: #020617; color: #e2e8f0; font-family: Inter, Arial, sans-serif; margin: 18px; }
    h1, h2, h3 { margin: 0; }
    .title-row { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:12px; }
    .muted { color:#94a3b8; font-size:13px; }
    .status-bar { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:8px; margin: 10px 0 16px; }
    .status-pill { background:#0b1220; border:1px solid #334155; border-radius:8px; padding:8px 10px; }
    .status-pill .label { font-size:11px; color:#94a3b8; text-transform:uppercase; }
    .status-pill .value { margin-top:4px; font-size:14px; font-weight:600; }
    .grid-row { display:grid; gap:12px; margin: 12px 0; }
    .metrics-row { grid-template-columns: minmax(260px, 1.2fr) repeat(auto-fit, minmax(160px, 1fr)); }
    .dual-row { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .card { background:#111827; border:1px solid #334155; padding:12px; border-radius:10px; }
    .card .label { color:#94a3b8; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; }
    .card .value { font-size:17px; margin-top:5px; }
    .big-pnl .value { font-size:36px; font-weight:700; line-height:1.15; }
    .positive { color:#22c55e; }
    .negative { color:#ef4444; }
    .neutral { color:#e2e8f0; }
    .chart-wrap { background:#111827; border:1px solid #334155; border-radius:10px; padding:10px; }
    .controls-row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin: 8px 0 10px; }
    .ctrl-group { display:flex; align-items:center; gap:6px; background:#0b1220; border:1px solid #334155; border-radius:8px; padding:6px 8px; }
    .ctrl-group label { font-size:12px; color:#94a3b8; }
    .ctrl-group select, .ctrl-group input { background:#111827; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:5px 7px; font-size:12px; }
    .btn { background:#1e293b; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:5px 10px; font-size:12px; text-decoration:none; cursor:pointer; }
    .btn.active { background:#334155; }
    .filter-row { display:flex; gap:6px; align-items:center; margin: 10px 0; flex-wrap:wrap; }
    .warn { color:#f59e0b; font-size:12px; }
    .ok { color:#22c55e; font-size:12px; }
    .highlight { border-color:#475569; box-shadow: inset 0 0 0 1px rgba(148,163,184,0.2); }
    .panel-title { font-size:14px; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.04em; color:#cbd5e1; }
    .kv-grid { display:grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap:8px; }
    .kv { background:#0b1220; border:1px solid #1f2937; border-radius:8px; padding:7px 8px; }
    .kv .k { color:#94a3b8; font-size:11px; text-transform:uppercase; }
    .kv .v { margin-top:3px; font-size:14px; }
    .last-trade.win { background: rgba(34, 197, 94, 0.13); border-color: rgba(34, 197, 94, 0.35); }
    .last-trade.loss { background: rgba(239, 68, 68, 0.13); border-color: rgba(239, 68, 68, 0.35); }
    .table-wrap { background:#111827; border:1px solid #334155; border-radius:10px; overflow:auto; max-height:380px; }
    table { width:100%; border-collapse: collapse; }
    th, td { padding: 9px 10px; border-bottom: 1px solid #1f2937; font-size: 13px; white-space: nowrap; }
    th { background:#0b1220; color:#cbd5e1; text-align:left; position: sticky; top: 0; z-index: 2; }
    tbody tr:nth-child(odd) { background: rgba(15, 23, 42, 0.75); }
    tbody tr:nth-child(even) { background: rgba(17, 24, 39, 0.75); }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700; letter-spacing:0.03em; }
    .badge.win { background: rgba(34,197,94,0.2); color:#4ade80; }
    .badge.loss { background: rgba(239,68,68,0.2); color:#f87171; }
    .badge.long { background: rgba(34,197,94,0.18); color:#86efac; }
    .badge.short { background: rgba(239,68,68,0.18); color:#fca5a5; }
    .events { background:#111827; border:1px solid #334155; border-radius:10px; padding:12px; max-height:240px; overflow:auto; }
    .event-item { padding:5px 0; border-bottom:1px dashed #1f2937; font-size: 13px; }
    .event-item:last-child { border-bottom:none; }
    .empty-state { background:#111827; border:1px dashed #334155; color:#94a3b8; padding:18px; border-radius:10px; }
    .legend { margin: 6px 0 8px; display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:#cbd5e1; }
    .legend-item { display:flex; align-items:center; gap:6px; }
    .legend-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  </style>
</head>
<body>
  <div class="title-row">
    <div>
      <h1>Crypto Scalping Core Dashboard</h1>
      <div class="muted">Trader-friendly runtime view</div>
    </div>
  </div>

  <div class="status-bar">
    {% for label, value in status_pills %}
      <div class="status-pill"><div class="label">{{ label }}</div><div class="value">{{ value }}</div></div>
    {% endfor %}
  </div>

  <div class="grid-row metrics-row">
    <div class="card big-pnl">
      <div class="label">Net PnL</div>
      <div class="value {{ net_pnl_class }}">{{ net_pnl_value }}</div>
      <div class="muted">Balance: {{ balance_value }}</div>
    </div>
    {% for label, value in key_metric_cards %}
      <div class="card"><div class="label">{{ label }}</div><div class="value">{{ value }}</div></div>
    {% endfor %}
  </div>

  <div class="legend">
    <span class="legend-item"><span class="legend-dot" style="background:#22c55e;"></span>LONG entry</span>
    <span class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span>SHORT entry</span>
    <span class="legend-item"><span class="legend-dot" style="background:#22d3ee;"></span>WIN exit</span>
    <span class="legend-item"><span class="legend-dot" style="background:#fb923c;"></span>LOSS exit</span>
  </div>
  <div class="controls-row">
    <div class="ctrl-group">
      <label>Timezone</label>
      <a class="btn {{ 'active' if selected_tz == 'utc' else '' }}" href="{{ url_for('index', tz='utc', cw=candle_window, tf=trade_filter) }}">UTC</a>
      <a class="btn {{ 'active' if selected_tz == 'arg' else '' }}" href="{{ url_for('index', tz='arg', cw=candle_window, tf=trade_filter) }}">ARG</a>
    </div>
    <div class="ctrl-group">
      <label>Candles</label>
      <a class="btn {{ 'active' if candle_window == '50' else '' }}" href="{{ url_for('index', tz=selected_tz, cw='50', tf=trade_filter) }}">50</a>
      <a class="btn {{ 'active' if candle_window == '100' else '' }}" href="{{ url_for('index', tz=selected_tz, cw='100', tf=trade_filter) }}">100</a>
      <a class="btn {{ 'active' if candle_window == '200' else '' }}" href="{{ url_for('index', tz=selected_tz, cw='200', tf=trade_filter) }}">200</a>
      <a class="btn {{ 'active' if candle_window == 'all' else '' }}" href="{{ url_for('index', tz=selected_tz, cw='all', tf=trade_filter) }}">All</a>
    </div>
    <form class="ctrl-group" method="post" action="{{ url_for('set_paper_balance') }}">
      <label>Paper Balance</label>
      <select name="preset">
        <option value="100">100</option>
        <option value="200">200</option>
        <option value="500">500</option>
        <option value="custom">Custom</option>
      </select>
      <input type="number" name="custom_value" step="0.01" min="1" placeholder="Custom" />
      <input type="hidden" name="tz" value="{{ selected_tz }}" />
      <input type="hidden" name="cw" value="{{ candle_window }}" />
      <input type="hidden" name="tf" value="{{ trade_filter }}" />
      <button class="btn" type="submit">Apply</button>
    </form>
    <div class="ctrl-group"><span class="muted">Auto refresh: 5s</span><a class="btn" href="{{ url_for('index', tz=selected_tz, cw=candle_window, tf=trade_filter) }}">Refresh</a></div>
  </div>
  {% if paper_warning %}
    <div class="warn">{{ paper_warning }}</div>
  {% endif %}
  {% if paper_info %}
    <div class="ok">{{ paper_info }}</div>
  {% endif %}
  <div class="muted">Paper config: {{ paper_config_display }} | Future expansion candidate: ETHUSDT</div>
  <div class="chart-wrap">{{ chart_html|safe }}</div>

  <div class="grid-row dual-row">
    <div class="card highlight">
      <div class="panel-title">Current Position</div>
      {% if in_position %}
      <div class="kv-grid">
        <div class="kv"><div class="k">Side</div><div class="v">{{ current_position.side }}</div></div>
        <div class="kv"><div class="k">Qty</div><div class="v">{{ current_position.qty }}</div></div>
        <div class="kv"><div class="k">Entry</div><div class="v">{{ current_position.entry }}</div></div>
        <div class="kv"><div class="k">Stop</div><div class="v">{{ current_position.stop }}</div></div>
        <div class="kv"><div class="k">Take Profit</div><div class="v">{{ current_position.tp }}</div></div>
        <div class="kv"><div class="k">Unrealized PnL</div><div class="v {{ current_position.unrealized_class }}">{{ current_position.unrealized }}</div></div>
        <div class="kv"><div class="k">Risk @ Stop</div><div class="v">{{ current_position.risk_amount }}</div></div>
        <div class="kv"><div class="k">Reward @ TP</div><div class="v">{{ current_position.reward_amount }}</div></div>
        <div class="kv"><div class="k">RR Ratio</div><div class="v">{{ current_position.rr_ratio }}</div></div>
      </div>
      {% else %}
      <div class="muted">No open position.</div>
      {% endif %}
    </div>

    <div class="card last-trade {{ last_trade_css }}">
      <div class="panel-title">Last Closed Trade</div>
      {% if last_trade %}
      <div class="kv-grid">
        <div class="kv"><div class="k">Result</div><div class="v">{{ last_trade.result }}</div></div>
        <div class="kv"><div class="k">Side</div><div class="v">{{ last_trade.side }}</div></div>
        <div class="kv"><div class="k">Entry</div><div class="v">{{ last_trade.entry_price_fmt }}</div></div>
        <div class="kv"><div class="k">Exit</div><div class="v">{{ last_trade.exit_price_fmt }}</div></div>
        <div class="kv"><div class="k">Net PnL</div><div class="v {{ last_trade.net_pnl_class }}">{{ last_trade.net_pnl_fmt }}</div></div>
        <div class="kv"><div class="k">Duration (min)</div><div class="v">{{ last_trade.duration_minutes_fmt }}</div></div>
        <div class="kv"><div class="k">Closed At</div><div class="v">{{ last_trade.timestamp_fmt }}</div></div>
        <div class="kv"><div class="k">Fees</div><div class="v">{{ last_trade.fees_fmt }}</div></div>
      </div>
      {% else %}
      <div class="muted">No closed trades yet.</div>
      {% endif %}
    </div>
  </div>

  <h2>Trade History (Last 20)</h2>
  <div class="filter-row">
    <a class="btn {{ 'active' if trade_filter == 'all' else '' }}" href="{{ url_for('index', tz=selected_tz, cw=candle_window, tf='all') }}">All</a>
    <a class="btn {{ 'active' if trade_filter == 'long' else '' }}" href="{{ url_for('index', tz=selected_tz, cw=candle_window, tf='long') }}">LONG</a>
    <a class="btn {{ 'active' if trade_filter == 'short' else '' }}" href="{{ url_for('index', tz=selected_tz, cw=candle_window, tf='short') }}">SHORT</a>
    <a class="btn {{ 'active' if trade_filter == 'win' else '' }}" href="{{ url_for('index', tz=selected_tz, cw=candle_window, tf='win') }}">WIN</a>
    <a class="btn {{ 'active' if trade_filter == 'loss' else '' }}" href="{{ url_for('index', tz=selected_tz, cw=candle_window, tf='loss') }}">LOSS</a>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Result</th><th>Side</th><th>Entry</th><th>Exit</th><th>Qty</th><th>Gross PnL</th><th>Fees</th><th>Net PnL</th><th>Duration (min)</th><th>Balance After</th>
        </tr>
      </thead>
      <tbody>
        {% if trades %}
          {% for t in trades %}
          <tr>
            <td>{{ t.timestamp_fmt }}</td>
            <td>
              <span class="badge {{ 'win' if t.result == 'WIN' else 'loss' if t.result == 'LOSS' else '' }}">{{ t.result }}</span>
            </td>
            <td>
              <span class="badge {{ 'long' if t.side == 'LONG' else 'short' if t.side == 'SHORT' else '' }}">{{ t.side }}</span>
            </td>
            <td>{{ t.entry_price_fmt }}</td>
            <td>{{ t.exit_price_fmt }}</td>
            <td>{{ t.qty_fmt }}</td>
            <td>{{ t.gross_pnl_fmt }}</td>
            <td>{{ t.fees_fmt }}</td>
            <td class="{{ t.net_pnl_class }}">{{ t.net_pnl_fmt }}</td>
            <td>{{ t.duration_minutes_fmt }}</td>
            <td>{{ t.balance_after_fmt }}</td>
          </tr>
          {% endfor %}
        {% else %}
          <tr><td colspan="11">No trade history yet.</td></tr>
        {% endif %}
      </tbody>
    </table>
  </div>

  <h2>Recent Events</h2>
  <div class="events">
    {% if recent_events %}
      {% for e in recent_events %}
        <div class="event-item">{{ e }}</div>
      {% endfor %}
    {% else %}
      <div class="event-item">No events yet.</div>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/")
def index() -> str:
    selected_tz = parse_tz_param(request.args.get("tz"))
    candle_window = str(request.args.get("cw") or "100").lower()
    if candle_window not in {"50", "100", "200", "all"}:
        candle_window = "100"
    trade_filter = parse_history_filter(request.args.get("tf"))
    paper_warning = str(request.args.get("paper_warn") or "")
    paper_info = str(request.args.get("paper_info") or "")

    state = load_json(STATE_FILE, {})
    trades = load_json(TRADES_FILE, [])
    candles_payload = load_json(CANDLES_FILE, EMPTY_CANDLES_PAYLOAD)
    paper_config = read_paper_config()
    if not paper_warning:
        paper_warning = str((state if isinstance(state, dict) else {}).get("paper_config_warning") or "")

    state = state if isinstance(state, dict) else {}
    trades = trades if isinstance(trades, list) else []
    candles_payload = candles_payload if isinstance(candles_payload, dict) else EMPTY_CANDLES_PAYLOAD

    symbol = str(state.get("symbol") or candles_payload.get("symbol") or "BTCUSDT")
    timeframe = str(state.get("timeframe") or candles_payload.get("timeframe") or "1m")
    updated_at = format_timestamp(state.get("timestamp_utc") or candles_payload.get("updated_at_utc"), selected_tz)

    net_pnl_value = fmt_signed_num(state.get("net_pnl_total", state.get("net_pnl")), 4)
    balance_value = fmt_num(state.get("paper_balance"), 4)
    net_pnl_css = pnl_class(state.get("net_pnl_total", state.get("net_pnl")))

    key_metric_cards = [
        ("Total Trades", str(state.get("total_trades", 0))),
        ("Win Rate", f"{fmt_num(state.get('win_rate'), 2)}%" if to_float(state.get("win_rate")) is not None else "N/A"),
        ("Profit Factor", fmt_num(state.get("profit_factor"), 2)),
        ("Expectancy", fmt_num(state.get("expectancy"), 4)),
        ("Avg Win", fmt_signed_num(state.get("avg_win"), 4)),
        ("Avg Loss", fmt_signed_num(state.get("avg_loss"), 4)),
        ("Avg Duration (min)", fmt_num(state.get("avg_trade_duration_minutes"), 2)),
        ("Max Drawdown %", fmt_num(state.get("max_drawdown_pct"), 2)),
    ]

    in_position = bool(state.get("in_position", False))
    entry_price = to_float(state.get("entry_price"))
    stop_price = to_float(state.get("stop_price"))
    take_profit = to_float(state.get("take_profit_price"))
    position_qty = to_float(state.get("position_qty"))
    pos_side = str(state.get("position_side") or "N/A").upper()

    risk_amount = None
    reward_amount = None
    rr_ratio = None
    if entry_price is not None and stop_price is not None and take_profit is not None and position_qty is not None:
        if pos_side == "LONG":
            risk_amount = position_qty * max(0.0, entry_price - stop_price)
            reward_amount = position_qty * max(0.0, take_profit - entry_price)
        elif pos_side == "SHORT":
            risk_amount = position_qty * max(0.0, stop_price - entry_price)
            reward_amount = position_qty * max(0.0, entry_price - take_profit)
        if risk_amount and risk_amount > 0:
            rr_ratio = reward_amount / risk_amount if reward_amount is not None else None

    current_position = {
        "side": str(state.get("position_side") or "N/A"),
        "entry": fmt_num(entry_price, 2),
        "stop": fmt_num(stop_price, 2),
        "tp": fmt_num(take_profit, 2),
        "qty": fmt_num(position_qty, 6),
        "unrealized": fmt_signed_num(state.get("position_unrealized_pnl"), 4),
        "unrealized_class": pnl_class(state.get("position_unrealized_pnl")),
        "risk_amount": fmt_num(risk_amount, 4),
        "reward_amount": fmt_num(reward_amount, 4),
        "rr_ratio": fmt_num(rr_ratio, 2),
    }

    normalized_trades = [normalize_trade_row(t, selected_tz) for t in trades if isinstance(t, dict)]
    filtered_trades = apply_trade_filter(normalized_trades, trade_filter)
    trades_last_20 = list(reversed(filtered_trades[-20:]))
    last_trade = trades_last_20[0] if trades_last_20 else None
    last_trade_css = "win" if last_trade and last_trade.get("result") == "WIN" else "loss" if last_trade and last_trade.get("result") == "LOSS" else ""

    status_pills = compute_status_pills(state, symbol, timeframe, updated_at)
    chart_html = build_chart_html(candles_payload, selected_tz, candle_window, state if in_position else None)

    recent_events_raw = state.get("recent_events", [])
    recent_events: List[str] = []
    if isinstance(recent_events_raw, list):
        for event in reversed(recent_events_raw):
            text = str(event)
            if text.startswith("[") and "]" in text:
                ts_candidate = text[1 : text.index("]")]
                converted = format_timestamp(ts_candidate, selected_tz)
                text = f"[{converted}]{text[text.index(']') + 1:]}"
            recent_events.append(text)

    current_base = to_float(state.get("paper_balance_base"))
    target_base = to_float(paper_config.get("target_balance"))
    source = str(state.get("paper_balance_source") or "default")
    if current_base is not None:
        paper_config_display = f"base={current_base:.2f} source={source}"
    elif target_base is not None:
        paper_config_display = f"target={target_base:.2f}"
    else:
        paper_config_display = "default"

    return render_template_string(
        TEMPLATE,
        status_pills=status_pills,
        net_pnl_value=net_pnl_value,
        net_pnl_class=net_pnl_css,
        balance_value=balance_value,
        key_metric_cards=key_metric_cards,
        chart_html=chart_html,
        in_position=in_position,
        current_position=current_position,
        last_trade=last_trade,
        last_trade_css=last_trade_css,
        trades=trades_last_20,
        recent_events=recent_events,
        selected_tz=selected_tz,
        candle_window=candle_window,
        trade_filter=trade_filter,
        paper_warning=paper_warning,
        paper_info=paper_info,
        paper_config_display=paper_config_display,
    )


@app.route("/set-paper-balance", methods=["POST"])
def set_paper_balance() -> Any:
    tz = parse_tz_param(request.form.get("tz"))
    cw = str(request.form.get("cw") or "100")
    tf = parse_history_filter(request.form.get("tf"))
    state = load_json(STATE_FILE, {})
    if not isinstance(state, dict):
        state = {}
    if str(state.get("mode", "")).lower() != "paper":
        return redirect(url_for("index", tz=tz, cw=cw, tf=tf, paper_warn="Paper balance control only works in PAPER mode"))
    if bool(state.get("in_position", False)):
        return redirect(url_for("index", tz=tz, cw=cw, tf=tf, paper_warn="Cannot change paper balance while a trade is open"))

    preset = str(request.form.get("preset") or "100")
    custom_value = to_float(request.form.get("custom_value"))
    if preset == "custom":
        new_value = custom_value
    else:
        new_value = to_float(preset)

    if new_value is None or new_value <= 0:
        return redirect(url_for("index", tz=tz, cw=cw, tf=tf, paper_warn="Invalid paper balance value"))

    try:
        write_paper_config(float(new_value))
    except OSError:
        return redirect(url_for("index", tz=tz, cw=cw, tf=tf, paper_warn="Could not persist paper balance config"))
    return redirect(url_for("index", tz=tz, cw=cw, tf=tf, paper_info=f"Paper balance target updated to {new_value:.2f}"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
