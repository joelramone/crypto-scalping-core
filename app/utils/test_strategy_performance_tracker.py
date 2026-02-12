from app.utils.strategy_performance_tracker import StrategyPerformanceTracker


def test_tracker_accumulates_strategy_metrics():
    tracker = StrategyPerformanceTracker()

    tracker.record_trade("RSI", 10.0)
    tracker.record_trade("RSI", -5.0)
    tracker.record_trade("Breakout", -2.0)

    exported = tracker.export()

    assert exported["RSI"] == {
        "trades_count": 2,
        "wins": 1,
        "losses": 1,
        "gross_profit": 10.0,
        "gross_loss": 5.0,
        "net_profit": 5.0,
    }
    assert exported["Breakout"] == {
        "trades_count": 1,
        "wins": 0,
        "losses": 1,
        "gross_profit": 0.0,
        "gross_loss": 2.0,
        "net_profit": -2.0,
    }


def test_tracker_computed_metrics_helpers():
    stats = {
        "trades_count": 4,
        "wins": 2,
        "gross_profit": 20.0,
        "gross_loss": 5.0,
        "net_profit": 12.0,
    }

    assert StrategyPerformanceTracker.compute_win_rate(stats) == 50.0
    assert StrategyPerformanceTracker.compute_profit_factor(stats) == 4.0
    assert StrategyPerformanceTracker.compute_expectancy(stats) == 3.0


def test_profit_factor_handles_zero_loss_and_zero_profit_edge_cases():
    only_wins = {"gross_profit": 12.0, "gross_loss": 0.0}
    only_losses = {"gross_profit": 0.0, "gross_loss": 3.0}
    no_activity = {"gross_profit": 0.0, "gross_loss": 0.0}

    assert StrategyPerformanceTracker.compute_profit_factor(only_wins) == float("inf")
    assert StrategyPerformanceTracker.compute_profit_factor(only_losses) == 0.0
    assert StrategyPerformanceTracker.compute_profit_factor(no_activity) == 0.0
