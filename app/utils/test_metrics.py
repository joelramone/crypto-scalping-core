import math

from app.utils.metrics import compute_trade_metrics


def test_compute_trade_metrics_typical_values():
    r_values = [0.10, -0.05, 0.20, -0.10, 0.05]

    result = compute_trade_metrics(r_values)

    assert result["trades"] == 5.0
    assert math.isclose(result["expectancy"], 0.04)
    assert math.isclose(result["profit_factor"], 2.3333333333333335)
    assert math.isclose(result["winrate"], 0.6)
    assert result["sharpe_annualized"] > 0.0
    assert result["sortino"] > 0.0
    assert math.isclose(result["max_drawdown"], 0.1)
    assert result["ulcer_index"] >= 0.0
    assert result["kelly_fraction"] > 0.0
    assert "percentiles" in result
    assert result["percentiles"]["p01"] <= result["percentiles"]["p99"]


def test_compute_trade_metrics_no_trades_edge_case():
    result = compute_trade_metrics([])

    assert result["trades"] == 0.0
    assert result["expectancy"] == 0.0
    assert result["profit_factor"] == 0.0
    assert result["winrate"] == 0.0
    assert result["sharpe_annualized"] == 0.0
    assert result["sortino"] == 0.0
    assert result["max_drawdown"] == 0.0
    assert result["ulcer_index"] == 0.0
    assert result["kelly_fraction"] == 0.0
    assert result["skewness"] == 0.0
    assert result["kurtosis"] == 0.0
    assert result["percentiles"] == {"p01": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_compute_trade_metrics_only_wins_edge_case():
    result = compute_trade_metrics([0.05, 0.10, 0.20])

    assert math.isinf(result["profit_factor"])
    assert result["winrate"] == 1.0
    assert result["sharpe_annualized"] > 0.0
    assert math.isinf(result["sortino"])
    assert result["max_drawdown"] == 0.0
    assert result["ulcer_index"] == 0.0
    assert result["kelly_fraction"] == 1.0


def test_compute_trade_metrics_only_losses_edge_case():
    result = compute_trade_metrics([-0.05, -0.10, -0.20])

    assert result["profit_factor"] == 0.0
    assert result["winrate"] == 0.0
    assert result["sharpe_annualized"] < 0.0
    assert result["sortino"] < 0.0
    assert result["max_drawdown"] > 0.0
    assert result["ulcer_index"] > 0.0
    assert result["kelly_fraction"] == 0.0


def test_compute_trade_metrics_respects_custom_equity_curve():
    result = compute_trade_metrics([0.1, -0.1], equity_curve=[100.0, 120.0, 90.0])

    assert math.isclose(result["max_drawdown"], 0.25)
