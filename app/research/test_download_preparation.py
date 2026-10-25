"""Offline preparation and bounded downloader checks; no historical data."""
import json
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from app.research import download_data as subject
from app.research.backtester import load_ohlcv_csv


def config(tmp_path, **changes):
    # Synthetic fixture dates only: neither a selected nor an authorized research period.
    values = dict(symbol="BTCUSDT", interval="1m", start="2024-01-01T00:00:00Z",
                  end="2024-01-01T00:03:00Z", output=tmp_path / "fixture.csv")
    return subject.DownloadConfig(**(values | changes))


def rows(start, offsets=(0, 1, 2)):
    return [[start + n * 60000, "100", "102", "99", "101", "1",
             start + (n + 1) * 60000 - 1, "100", 1, "0.5", "50", "0"] for n in offsets]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(requests.Session, "request", Mock(side_effect=AssertionError("Network prohibited")))


@pytest.mark.parametrize("changes", [dict(start="bad"), dict(end="2024-01-01"),
    dict(start="2024-01-01T00:00:00"), dict(start="2024-01-01T00:00:01Z"),
    dict(symbol="ETHUSDT"), dict(interval="5m")])
def test_bad_configuration(tmp_path, changes):
    with pytest.raises(ValueError):
        subject.preparation_plan(config(tmp_path, **changes))


@pytest.mark.parametrize("start,end", [("2024-12-31", "2025-01-02"),
    ("2025-01-01", "2025-01-02"), ("2026-01-01", "2026-01-02"),
    ("2026-08-06", "2026-08-07"), ("2030-01-01", "2030-01-02")])
def test_forbidden_before_session(tmp_path, monkeypatch, start, end):
    session = Mock(side_effect=AssertionError("No session for forbidden dates"))
    monkeypatch.setattr(subject.requests, "Session", session)
    with pytest.raises(ValueError, match="consumed/reserved"):
        subject.download_binance_futures_klines(config(tmp_path, start=start, end=end))
    session.assert_not_called()


def test_dry_run_is_read_only(tmp_path, monkeypatch, capsys):
    c = config(tmp_path)
    monkeypatch.setattr("sys.argv", ["download_data", "--exchange", "binance", "--product",
        "usdm-futures", "--symbol", c.symbol, "--interval", c.interval,
        "--start", c.start, "--end", c.end, "--output", str(c.output), "--dry-run"])
    subject.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["download_authorized"] is False
    assert plan["expected_candles"] == 3
    assert plan == subject.preparation_plan(c)
    assert not c.output.exists()


def test_dry_run_mandatory():
    with pytest.raises(SystemExit):
        subject.build_arg_parser().parse_args([])


def test_utc_normalization_and_exclusive_boundary(tmp_path):
    c = config(tmp_path, start="2024-12-31T23:58:00Z", end="2025-01-01")
    plan = subject.preparation_plan(c)
    assert plan["expected_candles"] == 2
    assert subject.parse_datetime_to_utc_ms("2024-01-01T01:00:00+01:00") == subject.parse_datetime_to_utc_ms("2024-01-01")


@pytest.mark.parametrize("offsets", [(0, 0, 1, 2), (0, 2), (1, 2), (0, 1), (0, 1, 3), (2, 1, 0), ()])
def test_duplicate_gaps_and_bounds(tmp_path, offsets):
    c = config(tmp_path); start, end = subject.validate_preparation(c)
    with pytest.raises(ValueError):
        subject.validate_download_frame(subject.rows_to_dataframe(rows(start, offsets)), start, end)


def test_pagination_and_compatible_csv(tmp_path, monkeypatch):
    c = config(tmp_path); start, end = subject.validate_preparation(c)
    # Short pages must not silently truncate the requested range.
    fetch = Mock(side_effect=[rows(start, (0, 1)), rows(start, (2,))])
    monkeypatch.setattr(subject, "fetch_klines_page", fetch)
    monkeypatch.setattr(subject.time, "sleep", lambda _: None)
    frame = subject.download_binance_futures_klines(c)
    assert [call.args[2:] for call in fetch.call_args_list] == [(start, end - 1), (start + 120000, end - 1)]
    loaded = load_ohlcv_csv(c.output)
    assert len(loaded) == len(frame) == 3
    assert list(loaded.columns)[:6] == ["timestamp", "open", "high", "low", "close", "volume"]
    with pytest.raises(FileExistsError):
        subject.download_binance_futures_klines(c)


def test_retry_parameters(tmp_path, monkeypatch):
    c = config(tmp_path); start, end = subject.validate_preparation(c)
    response = Mock(); response.json.return_value = rows(start)
    session = Mock(); session.get.side_effect = [requests.Timeout(), response]
    monkeypatch.setattr(subject.time, "sleep", lambda _: None)
    assert subject.fetch_klines_page(session, c, start, end - 1) == rows(start)
    assert session.get.call_count == 2
    assert session.get.call_args.kwargs["params"]["endTime"] == end - 1


def test_invalid_data_never_written(tmp_path, monkeypatch):
    c = config(tmp_path)
    monkeypatch.setattr(subject, "fetch_klines_page", Mock(return_value=[]))
    with pytest.raises(ValueError):
        subject.download_binance_futures_klines(c)
    assert not c.output.exists()
