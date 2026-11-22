"""Build notification text from the trend bot ledger.

    python -m app.trend_bot.notify --ledger research/paper/trend_bot_v1/ledger.jsonl --since-line N [--force]

Prints a Markdown message when the newest ledger event (after line N) is worth a
notification: a trade, or a target that differs from the previous event. Prints
nothing otherwise. Used by the GitHub Actions workflow to comment on the
activity-log issue, which emails watchers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def should_notify(events: list[dict[str, Any]], since_line: int) -> bool:
    if len(events) <= since_line:
        return False
    last = events[-1]
    if last["action"] != "HOLD":
        return True
    previous = events[-2] if len(events) >= 2 else None
    return previous is None or previous["target_fraction"] != last["target_fraction"]


def render(event: dict[str, Any]) -> str:
    action = {"BUY": "Compra", "SELL": "Venta", "HOLD": "Sin operación"}[event["action"]]
    fill = event.get("fill") or {}
    lines = [
        f"**Trend bot ({event['mode']}) — {event['candle_date']}**",
        "",
        f"- Posición objetivo: **{event['target_fraction']:.0%} en BTC**",
        f"- Acción: **{action}**",
    ]
    if fill:
        lines.append(f"- Ejecutado: {fill['btc']:.6f} BTC a {fill['price']:,.2f} USDT (comisión {fill['fee_usdt']:.2f} USDT)")
    lines += [
        f"- Precio BTC: {event['price']:,.2f} USDT",
        f"- Capital: {event['equity_usdt']:,.2f} USDT ({event['return_since_start']:+.2%} desde el inicio)",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--since-line", required=True, type=int)
    parser.add_argument("--force", action="store_true", help="Render the latest event even without a change")
    args = parser.parse_args()
    events = read_events(args.ledger)
    if events and (args.force or should_notify(events, args.since_line)):
        prefix = "_Mensaje de prueba de notificaciones._\n\n" if args.force else ""
        print(prefix + render(events[-1]))


if __name__ == "__main__":
    main()
