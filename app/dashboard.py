import json
from pathlib import Path

from flask import Flask, render_template_string

app = Flask(__name__)
RUNTIME_DIR = Path("runtime")
STATE_FILE = RUNTIME_DIR / "dashboard_state.json"
TRADES_FILE = RUNTIME_DIR / "trades_log.json"


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="5">
  <title>Crypto Scalping Core Dashboard</title>
  <style>
    body { background: #0f172a; color: #e2e8f0; font-family: Arial, sans-serif; margin: 20px; }
    h1, h2 { color: #f8fafc; }
    .grid { display:grid; grid-template-columns: repeat(3, minmax(200px, 1fr)); gap: 12px; }
    .card { background:#1e293b; padding:12px; border-radius:8px; }
    table { width:100%; border-collapse: collapse; margin-top:10px; }
    th, td { border: 1px solid #334155; padding: 8px; text-align: left; }
    th { background:#1e293b; }
    .win { color:#22c55e; font-weight:bold; }
    .loss { color:#ef4444; font-weight:bold; }
    .events { background:#111827; padding:12px; border-radius:8px; max-height:300px; overflow:auto; }
  </style>
</head>
<body>
  <h1>Crypto Scalping Core Dashboard</h1>
  <div class="grid">
    {% for label, value in summary.items() %}
      <div class="card"><strong>{{ label }}</strong><br>{{ value }}</div>
    {% endfor %}
  </div>

  <h2>Bot State</h2>
  <div class="grid">
    {% for label, value in bot_state.items() %}
      <div class="card"><strong>{{ label }}</strong><br>{{ value }}</div>
    {% endfor %}
  </div>

  <h2>Last Trade</h2>
  <div class="grid">
    {% for label, value in last_trade.items() %}
      <div class="card"><strong>{{ label }}</strong><br>{{ value }}</div>
    {% endfor %}
  </div>

  <h2>Trade History (Last 20)</h2>
  <table>
    <thead>
      <tr>
        <th>Time</th><th>Side</th><th>Entry</th><th>Exit</th><th>Qty</th><th>Gross PnL</th><th>Fees</th><th>Net PnL</th><th>Result</th><th>Balance</th>
      </tr>
    </thead>
    <tbody>
      {% for t in trades %}
      <tr>
        <td>{{ t.timestamp }}</td>
        <td>{{ t.side }}</td>
        <td>{{ t.entry_price }}</td>
        <td>{{ t.exit_price }}</td>
        <td>{{ t.qty }}</td>
        <td>{{ t.gross_pnl }}</td>
        <td>{{ t.fees }}</td>
        <td>{{ t.net_pnl }}</td>
        <td class="{{ 'win' if t.result == 'WIN' else 'loss' }}">{{ t.result }}</td>
        <td>{{ t.balance_after }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>Recent Events</h2>
  <div class="events">
    {% for e in events %}<div>{{ e }}</div>{% endfor %}
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    state = load_json(STATE_FILE, {})
    trades = load_json(TRADES_FILE, [])
    trades_last_20 = list(reversed(trades[-20:]))

    summary = {
        "Balance": state.get("paper_balance"),
        "Total Trades": state.get("total_trades"),
        "Wins": state.get("wins"),
        "Losses": state.get("losses"),
        "Winrate": state.get("win_rate"),
        "Net PnL": state.get("net_pnl"),
        "Expectancy": state.get("expectancy"),
        "Profit Factor": state.get("profit_factor"),
        "Max Drawdown": state.get("max_drawdown_pct"),
    }

    bot_state = {
        "In position": state.get("in_position"),
        "Side": state.get("position_side"),
        "Qty": state.get("position_qty"),
        "Entry": state.get("entry_price"),
        "Stop": state.get("stop_price"),
        "TP": state.get("take_profit_price"),
        "Trades this hour": state.get("trades_this_hour"),
        "Consecutive losses": state.get("consecutive_losses"),
        "Market state": state.get("market_state"),
    }

    lt = state.get("last_trade") or {}
    last_trade = {
        "Side": lt.get("side"),
        "Result": lt.get("result"),
        "Entry": lt.get("entry_price"),
        "Exit": lt.get("exit_price"),
        "Gross pnl": lt.get("gross_pnl"),
        "Fees": lt.get("fees"),
        "Net pnl": lt.get("net_pnl"),
    }

    return render_template_string(
        TEMPLATE,
        summary=summary,
        bot_state=bot_state,
        last_trade=last_trade,
        trades=trades_last_20,
        events=list(reversed(state.get("recent_events", []))),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
