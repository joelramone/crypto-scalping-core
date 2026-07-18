"""Analyze exported Donchian candidates and summarize feature usefulness."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "quality_score",
    "body_to_range",
    "close_location_value",
    "range_expansion_ratio",
    "atr_expansion_ratio",
    "ema20_slope_pct",
    "ema_alignment_strength",
    "breakout_distance_pct",
    "volume_ratio",
    "rsi14",
    "atr14",
]


def _parse_args() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(description="Analyze exported Donchian base candidates.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return Path(args.input), Path(args.output)


def _profit_factor(values: pd.Series) -> float:
    positive = values[values > 0].sum()
    negative = values[values < 0].sum()
    if negative == 0:
        return float("inf") if positive > 0 else 0.0
    return float(positive / abs(negative))


def _quartile_table(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    working = df[[feature, "net_pnl"]].dropna()
    if len(working) < 4 or working[feature].nunique() < 4:
        return pd.DataFrame()
    quartiles = pd.qcut(working[feature], q=4, duplicates="drop")
    grouped = working.groupby(quartiles, observed=False)
    return grouped["net_pnl"].agg(
        trade_count="count",
        expectancy="mean",
        median="median",
        profit_factor=_profit_factor,
    ).reset_index(names="bucket")


def _winner_loser_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    winners = df[df["net_pnl"] > 0]
    losers = df[df["net_pnl"] < 0]
    for feature in FEATURE_COLUMNS:
        rows.append(
            {
                "feature": feature,
                "winner_avg": winners[feature].mean(),
                "winner_median": winners[feature].median(),
                "loser_avg": losers[feature].mean(),
                "loser_median": losers[feature].median(),
                "corr_net_pnl": df[feature].corr(df["net_pnl"]),
            }
        )
    return pd.DataFrame(rows).sort_values("corr_net_pnl", ascending=False, na_position="last")


def _top_feature_candidates(summary: pd.DataFrame, quartile_summaries: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    scored: list[tuple[float, str]] = []
    weak: list[str] = []
    for _, row in summary.iterrows():
        feature = str(row["feature"])
        quartiles = quartile_summaries.get(feature, pd.DataFrame())
        expectancy_spread = 0.0
        pf_spread = 0.0
        if not quartiles.empty:
            expectancy_spread = float(quartiles["expectancy"].max() - quartiles["expectancy"].min())
            finite_pf = quartiles["profit_factor"].replace([float("inf")], pd.NA).dropna()
            if not finite_pf.empty:
                pf_spread = float(finite_pf.max() - finite_pf.min())
        corr_value = abs(float(row["corr_net_pnl"])) if pd.notna(row["corr_net_pnl"]) else 0.0
        score = corr_value + expectancy_spread + pf_spread
        if score < 0.05:
            weak.append(feature)
        scored.append((score, feature))
    scored.sort(reverse=True)
    return [feature for _, feature in scored[:3]], weak[:5]


def build_report(df: pd.DataFrame) -> str:
    opened = df[df["trade_opened"] == True].copy()  # noqa: E712
    summary = _winner_loser_table(opened)
    quartile_summaries = {feature: _quartile_table(opened, feature) for feature in FEATURE_COLUMNS}
    top_features, weak_features = _top_feature_candidates(summary, quartile_summaries)

    lines = [
        "# Donchian Breakout 15m Quality Analysis",
        "",
        f"Opened trades analyzed: {len(opened)}",
        f"Winners: {(opened['net_pnl'] > 0).sum()}",
        f"Losers: {(opened['net_pnl'] < 0).sum()}",
        f"Flat: {(opened['net_pnl'] == 0).sum()}",
        "",
        "## Winner vs Loser Feature Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Feature Quartiles",
        "",
    ]

    for feature, quartiles in quartile_summaries.items():
        lines.append(f"### {feature}")
        lines.append("")
        if quartiles.empty:
            lines.append("Not enough distinct values for quartile analysis.")
        else:
            lines.append(quartiles.to_markdown(index=False))
        lines.append("")

    lines.extend(
        [
            "## Quality Score Performance",
            "",
            opened.groupby("quality_score", observed=False)["net_pnl"].agg(
                trade_count="count",
                expectancy="mean",
                median="median",
                profit_factor=_profit_factor,
            ).reset_index().to_markdown(index=False),
            "",
            "## Potentially Useful Features",
            "",
        ]
    )
    for feature in top_features:
        lines.append(f"- {feature}")
    lines.extend(["", "## Weak Or Contradictory Evidence", ""])
    if weak_features:
        for feature in weak_features:
            lines.append(f"- {feature}")
    else:
        lines.append("- None identified by the simple heuristic scan.")
    lines.append("")
    return "\n".join(lines)


def write_report(input_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(input_path)
    report = build_report(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main() -> None:
    input_path, output_path = _parse_args()
    final_path = write_report(input_path, output_path)
    print(f"Wrote {final_path}")


if __name__ == "__main__":
    main()

