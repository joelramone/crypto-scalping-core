"""Authorized CLI and publication tests: synthetic prices, no Binance traffic."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from app.research import download_data as subject
from app.research.test_download_preparation import config, rows


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(requests.Session, "request", Mock(side_effect=AssertionError("Network forbidden")))
    monkeypatch.setattr(subject.time, "sleep", lambda _: None)


def authorized():
    return subject.DownloadConfig(symbol="BTCUSDT", interval="1m", start=subject.AUTHORIZED_START,
                                  end=subject.AUTHORIZED_END, output=subject.AUTHORIZED_OUTPUT)


@pytest.mark.parametrize("changes", [dict(start="2024-09-30"), dict(end="2024-10-09"),
    dict(symbol="ETHUSDT"), dict(interval="5m"), dict(output=Path("data/other.csv")),
    dict(output=Path("elsewhere") / subject.AUTHORIZED_OUTPUT.name),
    dict(start="2024-10-01T00:00:00.000001Z")])
def test_scope_rejections(tmp_path, monkeypatch, changes):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        c = subject.DownloadConfig(**(authorized().model_dump() | changes))
        subject.validate_execution(c, "binance", "usdm-futures")


@pytest.mark.parametrize("exchange,product", [("other", "usdm-futures"), ("binance", "spot")])
def test_product_rejections(tmp_path, monkeypatch, exchange, product):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        subject.validate_execution(authorized(), exchange, product)


def test_exact_scope_and_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subject.validate_execution(authorized(), "binance", "usdm-futures")
    args = ["--exchange", "binance", "--product", "usdm-futures", "--symbol", "BTCUSDT",
            "--interval", "1m", "--start", subject.AUTHORIZED_START, "--end", subject.AUTHORIZED_END,
            "--output", str(subject.AUTHORIZED_OUTPUT)]
    for modes in ([], ["--dry-run", "--execute-authorized-download"]):
        with pytest.raises(SystemExit):
            subject.build_arg_parser().parse_args(args + modes)


def test_complete_authorized_mock_download(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    c = authorized(); start, end = subject.validate_preparation(c)
    fixture = rows(start, range(10080))
    def page(session, cfg, offset, inclusive_end):
        assert inclusive_end == end - 1
        i = (offset - start) // 60000
        return fixture[i:i + subject.MAX_LIMIT]
    fetch = Mock(side_effect=page)
    monkeypatch.setattr(subject, "fetch_klines_page", fetch)
    # Reject imports of execution/research-writing modules during acquisition.
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith(("app.research.backtester", "app.research.optimizer", "app.research.memory", "app.execution")):
            raise AssertionError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr("sys.argv", ["download_data", "--exchange", "binance", "--product", "usdm-futures",
        "--symbol", "BTCUSDT", "--interval", "1m", "--start", c.start, "--end", c.end,
        "--output", str(c.output), "--execute-authorized-download"])
    subject.main()
    metadata = json.loads(capsys.readouterr().out)
    assert metadata == json.loads(subject.manifest_path(c.output).read_text())
    assert metadata["row_count"] == 10080 and fetch.call_count == 7
    assert metadata["sha256"] == hashlib.sha256(c.output.read_bytes()).hexdigest()
    assert metadata["file_size_bytes"] == c.output.stat().st_size
    assert metadata["first_timestamp_utc"] == "2024-10-01T00:00:00+00:00"
    assert metadata["last_timestamp_utc"] == "2024-10-07T23:59:00+00:00"
    assert metadata["acquisition_duration_seconds"] >= 0
    assert not list(c.output.parent.glob(".ohlcv-*.tmp"))


@pytest.mark.parametrize("which", ["csv", "manifest"])
def test_existing_destination_blocks_before_network(tmp_path, monkeypatch, which):
    c = config(tmp_path)
    target = c.output if which == "csv" else subject.manifest_path(c.output)
    target.write_text("preserve")
    fetch = Mock(side_effect=AssertionError("No request"))
    monkeypatch.setattr(subject, "fetch_klines_page", fetch)
    with pytest.raises(FileExistsError):
        subject.download_binance_futures_klines(c)
    assert target.read_text() == "preserve"
    fetch.assert_not_called()


@pytest.mark.parametrize("failure", ["serialization", "csv_link", "manifest_link", "unsupported", "race"])
def test_atomic_failures_leave_no_own_final_files(tmp_path, monkeypatch, failure):
    c = config(tmp_path); start, _ = subject.validate_preparation(c)
    frame = subject.rows_to_dataframe(rows(start))
    real_link = os.link
    def link(src, dest):
        if failure == "race" and Path(dest) == c.output:
            c.output.write_text("another writer")
        if failure in ("csv_link", "unsupported") or (failure == "manifest_link" and Path(dest) == subject.manifest_path(c.output)):
            raise OSError("simulated publication failure")
        real_link(src, dest)
    monkeypatch.setattr(subject.os, "link", link)
    if failure == "serialization":
        monkeypatch.setattr(pd.DataFrame, "to_csv", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        subject.publish_validated_dataset(c, frame, subject.time.perf_counter())
    if failure == "race":
        assert c.output.read_text() == "another writer"
    else:
        assert not c.output.exists()
    assert not subject.manifest_path(c.output).exists()
    assert not list(tmp_path.glob(".ohlcv-*.tmp"))


def test_validated_csv_is_published_atomically_before_manifest(tmp_path, monkeypatch):
    c = config(tmp_path); start, _ = subject.validate_preparation(c)
    real_link = os.link; events = []
    def link(src, dest):
        assert not Path(dest).exists()
        if Path(dest) == c.output:
            assert len(pd.read_csv(src)) == 3
        else:
            assert c.output.exists() and json.loads(Path(src).read_text())["row_count"] == 3
        events.append(Path(dest)); real_link(src, dest)
    monkeypatch.setattr(subject.os, "link", link)
    subject.publish_validated_dataset(c, subject.rows_to_dataframe(rows(start)), subject.time.perf_counter())
    assert events == [c.output, subject.manifest_path(c.output)]


@pytest.mark.parametrize("failure", ["request", "duplicate", "missing", "shifted", "infinite"])
def test_failed_acquisition_no_csv_or_manifest(tmp_path, monkeypatch, failure):
    c = config(tmp_path); start, _ = subject.validate_preparation(c)
    fixture = rows(start, (0, 0, 2) if failure == "duplicate" else (0, 2) if failure == "missing" else (1, 2, 3) if failure == "shifted" else (0, 1, 2))
    if failure == "infinite":
        fixture[0][2] = "inf"
    fetch = Mock(side_effect=RuntimeError("request failed")) if failure == "request" else Mock(side_effect=[fixture, []])
    monkeypatch.setattr(subject, "fetch_klines_page", fetch)
    with pytest.raises((ValueError, RuntimeError)):
        subject.download_binance_futures_klines(c)
    assert not c.output.exists() and not subject.manifest_path(c.output).exists()
    assert not list(tmp_path.glob(".ohlcv-*.tmp"))
