import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, render_template_string

app = Flask(__name__)
RUNTIME_DIR = Path("runtime")
STATE_FILE = RUNTIME_DIR / "dashboard_state.json"
TRADES_FILE = RUNTIME_DIR / "trades_log.json"
CANDLES_FILE = RUNTIME_DIR / "candles.json"


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
                mode="markers+text",
                text=group.get("side", pd.Series([""] * len(group))).fillna("").astype(str).radd(name.split()[0] + " "),
                textposition="top center",
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


def build_chart_html(candles_payload: Dict[str, Any]) -> str:
    candles = candles_payload.get("candles", [])
    if not candles:
        return "<div class='empty-state'>Candlestick data is not available yet. Start live_bot.py to begin streaming.</div>"

    frame = pd.DataFrame(candles)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame["time"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Candles",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )
    )

    ema_fast = pd.DataFrame(candles_payload.get("ema_fast", []))
    if not ema_fast.empty:
        ema_fast["time"] = pd.to_datetime(ema_fast["time"], utc=True)
        fig.add_trace(
            go.Scatter(
                x=ema_fast["time"],
                y=ema_fast["value"],
                mode="lines",
                name="EMA Fast",
                line={"color": "#eab308", "width": 2},
            )
        )

    ema_slow = pd.DataFrame(candles_payload.get("ema_slow", []))
    if not ema_slow.empty:
        ema_slow["time"] = pd.to_datetime(ema_slow["time"], utc=True)
        fig.add_trace(
            go.Scatter(
                x=ema_slow["time"],
                y=ema_slow["value"],
                mode="lines",
                name="EMA Slow",
                line={"color": "#38bdf8", "width": 2},
            )
        )

    signal_styles: Dict[str, Dict[str, Any]] = {
        "SIGNAL_LONG": {
            "name": "Signal LONG",
            "color": "rgba(34, 197, 94, 0.35)",
            "symbol": "circle",
            "size": 9,
            "opacity": 0.45,
            "line_color": "rgba(34, 197, 94, 0.45)",
            "line_width": 1,
        },
        "SIGNAL_SHORT": {
            "name": "Signal SHORT",
            "color": "rgba(239, 68, 68, 0.35)",
            "symbol": "circle",
            "size": 9,
            "opacity": 0.45,
            "line_color": "rgba(239, 68, 68, 0.45)",
            "line_width": 1,
        },
    }

    trade_styles: Dict[str, Dict[str, Any]] = {
        "EXECUTED_ENTRY_LONG": {
            "name": "Executed LONG",
            "color": "rgba(34, 197, 94, 1.0)",
            "symbol": "diamond",
            "size": 12,
            "opacity": 1.0,
            "line_color": "rgba(15, 23, 42, 1.0)",
            "line_width": 1,
        },
        "EXECUTED_ENTRY_SHORT": {
            "name": "Executed SHORT",
            "color": "rgba(239, 68, 68, 1.0)",
            "symbol": "diamond",
            "size": 12,
            "opacity": 1.0,
            "line_color": "rgba(15, 23, 42, 1.0)",
            "line_width": 1,
        },
        "EXIT_WIN": {
            "name": "Exit WIN",
            "color": "rgba(34, 197, 94, 1.0)",
            "symbol": "triangle-up",
            "size": 13,
            "opacity": 1.0,
            "line_color": "rgba(15, 23, 42, 1.0)",
            "line_width": 1,
        },
        "EXIT_LOSS": {
            "name": "Exit LOSS",
            "color": "rgba(239, 68, 68, 1.0)",
            "symbol": "triangle-down",
            "size": 13,
            "opacity": 1.0,
            "line_color": "rgba(15, 23, 42, 1.0)",
            "line_width": 1,
        },
    }

    signal_markers = candles_payload.get("signal_markers", [])
    normalized_signal_markers: List[Dict[str, Any]] = []
    for marker in signal_markers:
        side = str(marker.get("side", "")).upper()
        normalized_signal_markers.append(
            {
                "time": marker.get("time"),
                "price": marker.get("price"),
                "side": side,
                "kind": f"SIGNAL_{side}",
            }
        )
    _add_marker_traces(fig, normalized_signal_markers, signal_styles, "Signal")

    trade_markers = candles_payload.get("trade_markers", [])
    normalized_trade_markers: List[Dict[str, Any]] = []
    for marker in trade_markers:
        side = str(marker.get("side", "")).upper()
        kind = str(marker.get("kind", "")).upper()
        normalized_kind = f"EXECUTED_ENTRY_{side}" if kind == "EXECUTED_ENTRY" else kind
        normalized_trade_markers.append(
            {
                "time": marker.get("time"),
                "price": marker.get("price"),
                "side": side,
                "kind": normalized_kind,
            }
        )
    _add_marker_traces(fig, normalized_trade_markers, trade_styles, "Trade")

    levels = candles_payload.get("levels", {})
    for key, color, label in [
        ("entry_price", "#a78bfa", "Entry"),
        ("stop_price", "#ef4444", "Stop"),
        ("take_profit_price", "#22c55e", "Take Profit"),
    ]:
        level_price = to_float(levels.get(key))
        if level_price is not None:
            fig.add_hline(y=level_price, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top left")

    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=True, config={"responsive": True, "displaylogo": False})


def fmt_num(value: Any, ndigits: int = 4) -> str:
    num = to_float(value)
    if num is None:
        return "-"
    return f"{num:.{ndigits}f}"


TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="5">
  <title>Crypto Scalping Core Dashboard</title>
  <style>
    body { background: #020617; color: #e2e8f0; font-family: Inter, Arial, sans-serif; margin: 18px; }
    h1, h2 { margin: 0; }
    .header { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:16px; }
    .muted { color:#94a3b8; font-size:13px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 12px 0 18px; }
    .card { background:#111827; border:1px solid #334155; padding:12px; border-radius:10px; }
    .card .label { color:#94a3b8; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; }
    .card .value { font-size:17px; margin-top:5px; }
    .section { margin-top: 12px; }
    .table-wrap { background:#111827; border:1px solid #334155; border-radius:10px; overflow:hidden; }
    table { width:100%; border-collapse: collapse; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #1f2937; font-size: 13px; }
    th { background:#0b1220; color:#cbd5e1; text-align:left; }
    tr.win { background: rgba(34, 197, 94, 0.14); }
    tr.loss { background: rgba(239, 68, 68, 0.14); }
    .events { background:#111827; border:1px solid #334155; border-radius:10px; padding:12px; max-height:260px; overflow:auto; }
    .event-item { padding:4px 0; border-bottom:1px dashed #1f2937; font-size: 13px; }
    .event-item:last-child { border-bottom:none; }
    .empty-state { background:#111827; border:1px dashed #334155; color:#94a3b8; padding:18px; border-radius:10px; }
    .legend { margin: 12px 0 2px; display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:#cbd5e1; }
    .legend-item { display:flex; align-items:center; gap:6px; }
    .legend-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Crypto Scalping Core Dashboard</h1>
      <div class="muted">Mode: {{ mode }} | Symbol: {{ symbol }} | Timeframe: {{ timeframe }}</div>
    </div>
    <div class="muted">Updated: {{ updated_at }}</div>
  </div>

  <div class="legend">
    <span class="legend-item"><span class="legend-dot" style="background: rgba(34, 197, 94, 0.45);"></span>Signal LONG</span>
    <span class="legend-item"><span class="legend-dot" style="background: rgba(239, 68, 68, 0.45);"></span>Signal SHORT</span>
    <span class="legend-item"><span class="legend-dot" style="background: rgba(34, 197, 94, 1.0);"></span>Executed LONG</span>
    <span class="legend-item"><span class="legend-dot" style="background: rgba(239, 68, 68, 1.0);"></span>Executed SHORT</span>
    <span class="legend-item"><span class="legend-dot" style="background: rgba(34, 197, 94, 1.0);"></span>Exit WIN</span>
    <span class="legend-item"><span class="legend-dot" style="background: rgba(239, 68, 68, 1.0);"></span>Exit LOSS</span>
  </div>

  <div class="section">{{ chart_html|safe }}</div>

  <h2>Summary Metrics</h2>
  <div class="grid">
    {% for label, value in summary_cards %}
    <div class="card"><div class="label">{{ label }}</div><div class="value">{{ value }}</div></div>
    {% endfor %}
  </div>

  <h2>Bot State</h2>
  <div class="grid">
    {% for label, value in bot_cards %}
    <div class="card"><div class="label">{{ label }}</div><div class="value">{{ value }}</div></div>
    {% endfor %}
  </div>

  <h2>Trade History (Last 20)</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Side</th><th>Entry</th><th>Exit</th><th>Qty</th><th>Gross PnL</th><th>Fees</th><th>Net PnL</th><th>Result</th><th>Balance After</th>
        </tr>
      </thead>
      <tbody>
        {% if trades %}
          {% for t in trades %}
          <tr class="{{ 'win' if t.result == 'WIN' else 'loss' if t.result == 'LOSS' else '' }}">
            <td>{{ t.timestamp }}</td>
            <td>{{ t.side }}</td>
            <td>{{ t.entry_price }}</td>
            <td>{{ t.exit_price }}</td>
            <td>{{ t.qty }}</td>
            <td>{{ t.gross_pnl }}</td>
            <td>{{ t.fees }}</td>
            <td>{{ t.net_pnl }}</td>
            <td>{{ t.result }}</td>
            <td>{{ t.balance_after }}</td>
          </tr>
          {% endfor %}
        {% else %}
          <tr><td colspan="10">No trade history yet.</td></tr>
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
    state = load_json(STATE_FILE, {})
    trades = load_json(TRADES_FILE, [])
    candles_payload = load_json(CANDLES_FILE, EMPTY_CANDLES_PAYLOAD)

    mode = state.get("mode", "-")
    symbol = state.get("symbol", candles_payload.get("symbol", "BTCUSDT"))
    timeframe = state.get("timeframe", candles_payload.get("timeframe", "1m"))
    updated_at = state.get("timestamp_utc") or candles_payload.get("updated_at_utc") or "no data yet"

    summary_cards = [
        ("Balance", fmt_num(state.get("paper_balance"), 4)),
        ("Total Trades", str(state.get("total_trades", 0))),
        ("Wins", str(state.get("wins", 0))),
        ("Losses", str(state.get("losses", 0))),
        ("Win Rate %", fmt_num(state.get("win_rate"), 2)),
        ("Net PnL", fmt_num(state.get("net_pnl"), 4)),
        ("Expectancy", fmt_num(state.get("expectancy"), 4)),
        ("Profit Factor", fmt_num(state.get("profit_factor"), 4)),
        ("Max Drawdown %", fmt_num(state.get("max_drawdown_pct"), 2)),
    ]

    bot_cards = [
        ("In Position", str(state.get("in_position", False))),
        ("Side", str(state.get("position_side") or "-")),
        ("Qty", fmt_num(state.get("position_qty"), 6)),
        ("Entry", fmt_num(state.get("entry_price"), 2)),
        ("Stop", fmt_num(state.get("stop_price"), 2)),
        ("Take Profit", fmt_num(state.get("take_profit_price"), 2)),
        ("Trades This Hour", str(state.get("trades_this_hour", 0))),
        ("Consecutive Losses", str(state.get("consecutive_losses", 0))),
        ("Market State", str(state.get("market_state") or "-")),
    ]

    trades_last_20 = list(reversed(trades[-20:])) if isinstance(trades, list) else []
    chart_html = build_chart_html(candles_payload if isinstance(candles_payload, dict) else EMPTY_CANDLES_PAYLOAD)

    return render_template_string(
        TEMPLATE,
        mode=mode,
        symbol=symbol,
        timeframe=timeframe,
        updated_at=updated_at,
        summary_cards=summary_cards,
        bot_cards=bot_cards,
        chart_html=chart_html,
        trades=trades_last_20,
        recent_events=list(reversed(state.get("recent_events", []))) if isinstance(state, dict) else [],
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
