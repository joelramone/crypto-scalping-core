import json

import pandas as pd
import pytest

from app.research import download_funding as subject

H8 = subject.EIGHT_HOURS_MS


def fake_rows(start_ms, n, rate="0.0001", symbol="BTCUSDT"):
    return [{"symbol": symbol, "fundingTime": start_ms + i * H8, "fundingRate": rate, "markPrice": "100.0"}
            for i in range(n)]


def paged_fetch(all_rows):
    calls = []

    def fetch(session, symbol, start_ms, end_ms):
        calls.append(start_ms)
        eligible = [r for r in all_rows if start_ms <= r["fundingTime"] <= end_ms]
        return eligible[: subject.MAX_LIMIT]
    fetch.calls = calls
    return fetch


@pytest.fixture
def small_scope(monkeypatch):
    scope = ("BTCUSDT", "2020-01-01T00:00:00Z", "2021-12-01T00:00:00Z", subject.Path("data/f.csv"))
    monkeypatch.setitem(subject.AUTHORIZED_SCOPES, "TEST", scope)
    monkeypatch.setattr(subject, "REQUEST_SLEEP_SECONDS", 0)
    return scope


def test_paginates_and_publishes_manifest(tmp_path, monkeypatch, small_scope):
    monkeypatch.chdir(tmp_path)
    start = subject.to_ms(small_scope[1])
    fetch = paged_fetch(fake_rows(start, 2100))
    manifest = subject.execute("TEST", fetch)
    assert len(fetch.calls) == 3
    assert manifest["integrity_checks"]["row_count"] == 2100
    assert manifest["status"] == "DATASET_VALIDATED" and manifest["authorization_scope"] == "TEST"
    frame = pd.read_csv(tmp_path / "data/f.csv")
    assert list(frame.columns) == subject.CSV_COLUMNS and len(frame) == 2100


def test_refuses_overwrite(tmp_path, monkeypatch, small_scope):
    monkeypatch.chdir(tmp_path)
    start = subject.to_ms(small_scope[1])
    subject.execute("TEST", paged_fetch(fake_rows(start, 10)))
    with pytest.raises(FileExistsError):
        subject.execute("TEST", paged_fetch(fake_rows(start, 10)))


@pytest.mark.parametrize("mutate", ["gap", "duplicate", "nan", "huge", "symbol"])
def test_invalid_data_blocks_publication(tmp_path, monkeypatch, small_scope, mutate):
    monkeypatch.chdir(tmp_path)
    start = subject.to_ms(small_scope[1])
    rows = fake_rows(start, 20)
    if mutate == "gap":
        del rows[5]
    elif mutate == "duplicate":
        rows.insert(5, dict(rows[4]))
    elif mutate == "nan":
        rows[3]["fundingRate"] = "nan"
    elif mutate == "huge":
        rows[3]["fundingRate"] = "0.06"
    else:
        rows[3]["symbol"] = "ETHUSDT"
    with pytest.raises(ValueError):
        subject.execute("TEST", lambda *a: rows if a[2] == start else [])
    assert not (tmp_path / "data/f.csv").exists()
    assert not (tmp_path / "data/f.csv.manifest.json").exists()


def test_only_registered_scope_and_dry_run_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = subject.plan()
    assert plan["scope"] == "TASK-034B" and plan["symbol"] == "BTCUSDT"
    assert not (tmp_path / "data").exists()
    with pytest.raises(KeyError):
        subject.execute("TASK-999X", lambda *a: [])
