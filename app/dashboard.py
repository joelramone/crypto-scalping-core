import json
from pathlib import Path

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
RUNTIME_DIR = Path("runtime")
STATE_FILE = RUNTIME_DIR / "dashboard_state.json"
TRADES_FILE = RUNTIME_DIR / "trades_log.json"
CANDLES_FILE = RUNTIME_DIR / "candles.json"


EMPTY_CANDLES_PAYLOAD = {
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
    "markers": [],
}


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
    .chart-card { background:#111827; border: 1px solid #334155; border-radius:10px; padding:12px; margin-bottom:18px; }
    .chart-meta { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; color:#94a3b8; font-size:13px; }
    #candlestickChart { width:100%; height:500px; }
  </style>
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
  <h1>Crypto Scalping Core Dashboard</h1>

  <h2>Live Candlestick Chart</h2>
  <div class="chart-card">
    <div class="chart-meta">
      <div id="chartSymbol">BTCUSDT · 1m</div>
      <div id="chartUpdated">Waiting for data...</div>
    </div>
    <div id="candlestickChart"></div>
  </div>

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

  <script>
    const chartContainer = document.getElementById('candlestickChart');
    const chart = LightweightCharts.createChart(chartContainer, {
      layout: {
        background: { color: '#111827' },
        textColor: '#cbd5e1',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      rightPriceScale: {
        borderColor: '#334155',
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
      },
      width: chartContainer.clientWidth,
      height: 500,
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });

    const emaFastSeries = chart.addLineSeries({ color: '#eab308', lineWidth: 2, priceLineVisible: false });
    const emaSlowSeries = chart.addLineSeries({ color: '#38bdf8', lineWidth: 2, priceLineVisible: false });

    let levelLines = [];

    function clearLevelLines() {
      for (const line of levelLines) {
        candleSeries.removePriceLine(line);
      }
      levelLines = [];
    }

    function addPriceLine(value, title, color) {
      if (value === null || value === undefined) return;
      levelLines.push(
        candleSeries.createPriceLine({
          price: Number(value),
          color,
          lineWidth: 1,
          lineStyle: LightweightCharts.LineStyle.Dashed,
          axisLabelVisible: true,
          title,
        })
      );
    }

    async function refreshChart() {
      try {
        const response = await fetch('/api/candles', { cache: 'no-store' });
        if (!response.ok) return;
        const payload = await response.json();

        const candles = (payload.candles || []).map(c => ({
          time: Math.floor(new Date(c.time).getTime() / 1000),
          open: Number(c.open),
          high: Number(c.high),
          low: Number(c.low),
          close: Number(c.close),
        }));

        const emaFast = (payload.ema_fast || []).map(v => ({
          time: Math.floor(new Date(v.time).getTime() / 1000),
          value: Number(v.value),
        }));

        const emaSlow = (payload.ema_slow || []).map(v => ({
          time: Math.floor(new Date(v.time).getTime() / 1000),
          value: Number(v.value),
        }));

        const markers = (payload.markers || []).map(m => ({
          time: Math.floor(new Date(m.time).getTime() / 1000),
          position: m.position,
          color: m.color,
          shape: m.shape,
          text: m.text,
        }));

        candleSeries.setData(candles);
        emaFastSeries.setData(emaFast);
        emaSlowSeries.setData(emaSlow);
        candleSeries.setMarkers(markers);

        clearLevelLines();
        const levels = payload.levels || {};
        addPriceLine(levels.entry_price, 'Entry', '#a78bfa');
        addPriceLine(levels.stop_price, 'Stop', '#ef4444');
        addPriceLine(levels.take_profit_price, 'Take Profit', '#22c55e');

        chart.timeScale().fitContent();

        document.getElementById('chartSymbol').textContent = `${payload.symbol || 'BTCUSDT'} · ${payload.timeframe || '1m'}`;
        document.getElementById('chartUpdated').textContent = payload.updated_at_utc
          ? `Updated: ${payload.updated_at_utc}`
          : 'Waiting for data...';
      } catch (error) {
        console.error('Chart refresh failed:', error);
      }
    }

    window.addEventListener('resize', () => {
      chart.applyOptions({ width: chartContainer.clientWidth, height: 500 });
    });

    refreshChart();
    setInterval(refreshChart, 5000);
  </script>
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


@app.route("/api/candles")
def api_candles():
    payload = load_json(CANDLES_FILE, EMPTY_CANDLES_PAYLOAD)
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
