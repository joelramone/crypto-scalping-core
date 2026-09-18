"""One-shot orchestration for preregistered frozen research baselines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from app.research.automation.diagnostics import diagnose_leaderboard
from app.research.automation.report import render_report
from app.research.automation.validator import validate_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen research baseline once.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--report",
        default="research/reports/latest_baseline.md",
        help="Human-review report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = validate_baseline(args.config)

    command = [
        sys.executable,
        "-m",
        "app.research.optimizer.grid_search",
        "--config",
        str(validation.config_path),
    ]
    subprocess.run(command, check=True)

    if not validation.output_path.is_file():
        raise RuntimeError(f"Expected leaderboard was not created: {validation.output_path}")

    diagnostics = diagnose_leaderboard(validation.output_path)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(diagnostics), encoding="utf-8")

    print(f"Baseline complete: {validation.strategy} {validation.timeframe}")
    print(f"Diagnostic class: {diagnostics['failure_mode']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
