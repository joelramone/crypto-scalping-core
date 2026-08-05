import pandas as pd

from app.research.features import compute_features
from app.research.optimizer.grid_search import load_grid_search_config
from app.research.signal_quality.features import add_signal_quality_features
from app.research.signal_quality.scoring import calculate_long_breakout_quality
from app.research.strategies.donchian_breakout import DonchianBreakoutStrategy


def _feature_input() -> pd.DataFrame:
    rows = 240
    close = [100.0 + (index * 0.02) for index in range(rows)]
    for index in range(220, rows):
        close[index] = close[219] + 1.0 + (index - 220)

    breakout = [index >= 220 for index in range(rows)]
    base = pd.DataFrame(
        {
            "timestamp": list(range(rows)),
            "open": [value - 0.05 for value in close],
            "high": [
                value + (0.25 if is_breakout else 0.15)
                for value, is_breakout in zip(close, breakout, strict=True)
            ],
            "low": [
                value - (0.25 if is_breakout else 0.15)
                for value, is_breakout in zip(close, breakout, strict=True)
            ],
            "close": close,
            "volume": [2500.0 if is_breakout else 1000.0 for is_breakout in breakout],
        }
    )
    return compute_features(base)


def test_zero_range_candle_handling():
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "atr14": [1.0],
            "ema20": [100.0],
            "ema20_slope": [0.0],
            "ema50": [100.0],
            "ema200": [100.0],
        }
    )

    featured = add_signal_quality_features(df)

    assert featured.loc[0, "body_to_range"] == 0.0
    assert featured.loc[0, "close_location_value"] == 0.0
    assert featured.loc[0, "upper_wick_ratio"] == 0.0
    assert featured.loc[0, "lower_wick_ratio"] == 0.0


def test_body_to_range_and_close_location_calculation():
    df = pd.DataFrame(
        {
            "open": [10.0],
            "high": [14.0],
            "low": [8.0],
            "close": [13.0],
            "atr14": [2.0],
            "ema20": [12.0],
            "ema20_slope": [0.5],
            "ema50": [12.0],
            "ema200": [10.0],
        }
    )

    featured = add_signal_quality_features(df)

    assert featured.loc[0, "body_abs"] == 3.0
    assert featured.loc[0, "body_to_range"] == 0.5
    assert featured.loc[0, "close_location_value"] == 5.0 / 6.0


def test_range_expansion_calculation():
    df = _feature_input()

    assert df["range_sma20"].iloc[-1] > 0.0
    assert df["range_expansion_ratio"].iloc[-1] > 0.0


def test_quality_score_components():
    df = pd.DataFrame(
        {
            "body_to_range": [0.7],
            "close_location_value": [0.9],
            "range_expansion_ratio": [1.3],
            "atr_expansion_ratio": [1.1],
            "ema20_slope_pct": [0.001],
            "ema_alignment_strength": [0.01],
        }
    )
    breakout_distance_pct = pd.Series([0.002], index=df.index)

    score, components = calculate_long_breakout_quality(
        df,
        breakout_distance_pct,
        min_body_to_range=0.6,
        min_close_location=0.85,
        min_range_expansion=1.25,
        min_atr_expansion=1.0,
        min_ema20_slope_pct=0.0,
        min_ema_alignment_strength=0.0,
        min_breakout_distance_pct=0.0005,
    )

    assert int(score.iloc[0]) == 7
    assert components.iloc[0].all()


def test_donchian_backward_compatibility():
    df = _feature_input()
    strategy = DonchianBreakoutStrategy(lookback=3, volume_ratio=0.0)

    entries = strategy.generate_entries(df)

    assert isinstance(entries, pd.Series)
    assert strategy.quality_filter_active() is False


def test_quality_filter_rejects_scores_below_threshold():
    df = _feature_input()
    strategy = DonchianBreakoutStrategy(
        lookback=3,
        volume_ratio=0.0,
        min_quality_score=7,
        min_body_to_range=0.99,
        min_close_location=0.99,
        min_range_expansion=9.0,
        min_atr_expansion=9.0,
        min_ema20_slope_pct=1.0,
        min_ema_alignment_strength=1.0,
        min_breakout_distance_pct=1.0,
    )

    entries = strategy.generate_entries(df)

    assert not entries.any()


def test_quality_filter_accepts_scores_at_or_above_threshold():
    df = _feature_input()
    strategy = DonchianBreakoutStrategy(
        lookback=3,
        volume_ratio=0.0,
        min_quality_score=1,
        min_body_to_range=0.0,
        min_close_location=0.0,
        min_range_expansion=0.0,
        min_atr_expansion=0.0,
        min_ema20_slope_pct=-1.0,
        min_ema_alignment_strength=-1.0,
        min_breakout_distance_pct=-1.0,
    )

    entries = strategy.generate_entries(df)

    assert entries.any()


def test_optimizer_parameter_propagation(tmp_path):
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        "\n".join(
            [
                "strategy: donchian_breakout",
                "data: data/BTCUSDT_1m.csv",
                "timeframe: 15m",
                "output: research/leaderboards/test.csv",
                "parameters:",
                "  lookback:",
                "    - 3",
                "  volume_ratio:",
                "    - 0.4",
                "  take_profit_pct:",
                "    - 0.012",
                "  stop_loss_pct:",
                "    - 0.008",
                "  max_holding_candles:",
                "    - 24",
                "  min_quality_score:",
                "    - 2",
                "  min_body_to_range:",
                "    - 0.4",
                "  min_close_location:",
                "    - 0.7",
                "  min_range_expansion:",
                "    - 1.0",
                "  min_atr_expansion:",
                "    - 0.9",
                "  min_ema20_slope_pct:",
                "    - 0.0",
                "  min_ema_alignment_strength:",
                "    - 0.0",
                "  min_breakout_distance_pct:",
                "    - 0.0005",
            ]
        ),
        encoding="utf-8",
    )

    config = load_grid_search_config(config_path)

    assert config.parameters["min_quality_score"] == [2]
    assert config.parameters["min_body_to_range"] == [0.4]
    assert config.parameters["min_breakout_distance_pct"] == [0.0005]
