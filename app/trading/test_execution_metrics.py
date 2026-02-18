import pytest

from app.trading.execution_metrics import compute_r_multiple


def test_compute_r_multiple_long_formula() -> None:
    assert compute_r_multiple("LONG", entry_price=100.0, exit_price=110.0, stop_price=95.0) == pytest.approx(2.0)


def test_compute_r_multiple_short_formula() -> None:
    assert compute_r_multiple("SHORT", entry_price=100.0, exit_price=95.0, stop_price=102.5) == pytest.approx(2.0)


def test_stop_loss_is_exactly_minus_one_r_long() -> None:
    assert compute_r_multiple("LONG", entry_price=100.0, exit_price=95.0, stop_price=95.0) == pytest.approx(-1.0)


def test_stop_loss_is_exactly_minus_one_r_short() -> None:
    assert compute_r_multiple("SHORT", entry_price=100.0, exit_price=105.0, stop_price=105.0) == pytest.approx(-1.0)


def test_compute_r_multiple_rejects_invalid_risk_definition() -> None:
    with pytest.raises(ValueError):
        compute_r_multiple("LONG", entry_price=100.0, exit_price=99.0, stop_price=100.0)

    with pytest.raises(ValueError):
        compute_r_multiple("SHORT", entry_price=100.0, exit_price=101.0, stop_price=100.0)
