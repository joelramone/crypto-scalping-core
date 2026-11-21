import pandas as pd
import pytest

from app.research import download_daily as subject

DAY = subject.DAY_MS


def kline(t, price=100.0):
    return [t, str(price), str(price + 1), str(price - 1), str(price), "10", t + DAY - 1, "1000", 5, "0", "0", "0"]


def make_fetch(start_ms, n):
    rows = [kline(start_ms + i * DAY) for i in range(n)]

    def fetch(session, symbol, s, e):
        return [r for r in rows if s <= r[0] <= e][: subject.MAX_LIMIT]
    return fetch


@pytest.fixture
def scope(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subject, "REQUEST_SLEEP_SECONDS", 0)
    monkeypatch.setitem(subject.AUTHORIZED_SCOPES, "TEST",
                        [("BTCUSDT", subject.Path("data/b.csv")), ("ETHUSDT", subject.Path("data/e.csv"))])
    return "2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"


def test_downloads_all_symbols_with_manifests(tmp_path, scope):
    start, end = scope
    n = (subject.to_ms(end) - subject.to_ms(start)) // DAY
    manifests = subject.execute("TEST", make_fetch(subject.to_ms(start), n), start, end)
    assert [m["symbol"] for m in manifests] == ["BTCUSDT", "ETHUSDT"]
    assert all(m["integrity_checks"]["row_count"] == n for m in manifests)
    assert list(pd.read_csv(tmp_path / "data/b.csv").columns) == subject.CSV_COLUMNS


def test_missing_day_or_short_coverage_blocks_publication(tmp_path, scope):
    start, end = scope
    n = (subject.to_ms(end) - subject.to_ms(start)) // DAY
    rows = [kline(subject.to_ms(start) + i * DAY) for i in range(n)]
    del rows[100]
    with pytest.raises(ValueError):
        subject.execute("TEST", lambda *a: [r for r in rows if a[2] <= r[0] <= a[3]][:1000], start, end)
    with pytest.raises(ValueError):
        subject.execute("TEST", make_fetch(subject.to_ms(start), n - 5), start, end)
    assert not (tmp_path / "data/b.csv").exists()


def test_bad_prices_rejected():
    start = subject.to_ms("2020-01-01T00:00:00Z")
    frame = subject.rows_to_frame([kline(start + i * DAY) for i in range(3)])
    frame.loc[1, "high"] = 50.0
    with pytest.raises(ValueError):
        subject.validate_frame(frame, start, start + 3 * DAY)


def test_refuses_overwrite_and_unknown_scope(tmp_path, scope):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/e.csv").write_text("x")
    with pytest.raises(FileExistsError):
        subject.execute("TEST", lambda *a: [], *scope)
    with pytest.raises(KeyError):
        subject.execute("TASK-999X", lambda *a: [])
    assert subject.plan()["expected_rows_per_symbol"] == 3363
