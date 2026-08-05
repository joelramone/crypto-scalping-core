from __future__ import annotations

import pandas as pd

from app.research.simulation import simulate_strategy
from app.research.strategies import DonchianBreakoutStrategy


def _featured_candles() -> pd.DataFrame:
    rows = 60
    close = pd.Series([100.0 + i for i in range(rows)])
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="15min"),
            "open": close - 0.8,
            "high": close + 0.1,
            "low": close - 1.0,
            "close": close,
            "volume_ratio": 1.0,
            "atr14": [1.0] * 30 + [2.0] * 30,
            "ema200": close - 10.0,
            "ema20_slope": 1.0,
            "body_to_range": 0.7,
            "close_location_value": [0.85, 0.91, 0.93, 0.95] * 15,
            "range_expansion_ratio": 1.0,
            "atr_expansion_ratio": 1.0,
            "ema20_slope_pct": 0.01,
            "ema_alignment_strength": 0.01,
        }
    )


def _strategy(threshold: float | None = None) -> DonchianBreakoutStrategy:
    kwargs = {
        "lookback": 3,
        "volume_ratio": 0.4,
        "take_profit_pct": 0.5,
        "stop_loss_pct": 0.5,
        "max_holding_candles": 5,
    }
    if threshold is not None:
        kwargs["min_close_location_filter"] = threshold
    return DonchianBreakoutStrategy(**kwargs)


def test_zero_close_location_filter_is_exact_baseline() -> None:
    df = _featured_candles()
    baseline = _strategy()
    zero_filter = _strategy(0.0)

    pd.testing.assert_series_equal(
        baseline.generate_entries(df),
        zero_filter.generate_entries(df),
    )
    assert simulate_strategy(df, baseline).model_dump() == simulate_strategy(
        df,
        zero_filter,
    ).model_dump()


def test_close_location_filter_only_removes_baseline_entries() -> None:
    df = _featured_candles()
    baseline_entries = _strategy(0.0).generate_entries(df)
    filtered_entries = _strategy(0.92).generate_entries(df)

    assert filtered_entries.sum() < baseline_entries.sum()
    assert filtered_entries.sum() > 0
    assert not (filtered_entries & ~baseline_entries).any()
    assert filtered_entries[baseline_entries].equals(
        df.loc[baseline_entries, "close_location_value"].ge(0.92)
    )
