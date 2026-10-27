"""Synthetic-only loader/resampling integration checks."""
import hashlib
import json
from unittest.mock import Mock

import pandas as pd
import pytest

from app.research.data_validation import check_integration as subject
from app.research.data_utils import resample_ohlcv


@pytest.fixture
def local_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame({"timestamp": pd.date_range(subject.START, periods=10080, freq="1min"),
                          "open": 100., "high": 102., "low": 99., "close": 101., "volume": 1.,
                          "trades": 2})
    path = tmp_path / subject.EXPECTED_PATH
    path.parent.mkdir()
    manifest = path.with_suffix(".csv.manifest.json")
    def write(data=frame, **changes):
        data.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Only tests substitute the known real-data hash with synthetic bytes.
        monkeypatch.setattr(subject, "EXPECTED_SHA256", digest)
        metadata = {"status": "DATASET_VALIDATED", "source": "binance", "product": "usdm-futures",
                    "symbol": "BTCUSDT", "timeframe": "1m", "row_count": 10080,
                    "sha256": digest, "file_size_bytes": path.stat().st_size,
                    "start_inclusive_utc": subject.START.isoformat(), "end_exclusive_utc": subject.END.isoformat(),
                    "first_timestamp_utc": subject.START.isoformat(),
                    "last_timestamp_utc": (subject.END - pd.Timedelta(minutes=1)).isoformat()}
        manifest.write_text(json.dumps(metadata | changes))
    write()
    return path, manifest, frame, write


def test_valid_read_only_summary(local_fixture, monkeypatch):
    path, manifest, _, _ = local_fixture
    before = (path.read_bytes(), manifest.read_bytes())
    import app.research.backtester as backtester
    monkeypatch.setattr(backtester, "simulate_strategy", Mock(side_effect=AssertionError("No strategy")))
    result = subject.check_integration(path, manifest)
    assert result["status"] == "INTEGRATION_VALIDATED"
    assert result["strategy_executed"] is False
    assert {k:v["row_count"] for k,v in result["resampled"].items()} == {"1m":10080,"5m":2016,"15m":672}
    assert result["resampled"]["15m"]["last_open_utc"] == "2024-10-07T23:45:00+00:00"
    assert result["resampled"]["15m"]["last_available_at_utc"] == subject.END.isoformat()
    assert (path.read_bytes(), manifest.read_bytes()) == before


@pytest.mark.parametrize("field,value", [("sha256","bad"),("row_count",3),("source","other"),
    ("product","spot"),("symbol","ETHUSDT"),("timeframe","5m"),("file_size_bytes",0),
    ("start_inclusive_utc","wrong"),("end_exclusive_utc","wrong"),("status","INCOMPLETE")])
def test_manifest_mismatches(local_fixture, field, value):
    path, manifest, _, write = local_fixture
    write(**{field:value})
    with pytest.raises(ValueError):subject.check_integration(path, manifest)


def test_modified_bytes(local_fixture):
    path, manifest, _, _ = local_fixture
    with path.open("a") as handle:handle.write("\n")
    with pytest.raises(ValueError,match="SHA-256"):subject.check_integration(path,manifest)


@pytest.mark.parametrize("kind",["missing","duplicate","unsorted","shifted","null","infinite","bad_ohlc","negative_volume","schema","timestamp"])
def test_source_defects(local_fixture, kind):
    path, manifest, frame, write = local_fixture
    frame = frame.copy()
    if kind == "missing":frame=frame.drop(index=100)
    elif kind == "duplicate":frame=pd.concat([frame,frame.iloc[[1]]],ignore_index=True)
    elif kind == "unsorted":frame=frame.iloc[::-1]
    elif kind == "shifted":frame["timestamp"]+=pd.Timedelta(minutes=1)
    elif kind == "null":frame.loc[2,"close"]=None
    elif kind == "infinite":frame.loc[2,"high"]=float("inf")
    elif kind == "bad_ohlc":frame.loc[2,"high"]=90
    elif kind == "negative_volume":frame.loc[2,"volume"]=-1
    elif kind == "schema":frame=frame.drop(columns="open")
    else:frame["timestamp"]=frame["timestamp"].astype(str);frame.loc[2,"timestamp"]="bad"
    write(frame)
    with pytest.raises(ValueError):subject.check_integration(path,manifest)


def test_missing_files_and_wrong_path(local_fixture):
    path,manifest,_,_=local_fixture
    with pytest.raises(ValueError):subject.check_integration(path.with_name("other.csv"),manifest)
    manifest.unlink()
    with pytest.raises(OSError):subject.check_integration(path,manifest)


def test_cli_json_error(local_fixture,monkeypatch,capsys):
    path,manifest,_,_=local_fixture
    monkeypatch.setattr("sys.argv",["check_integration","--input",str(path),"--manifest",str(manifest)])
    subject.main()
    assert json.loads(capsys.readouterr().out)["status"] == "INTEGRATION_VALIDATED"
    path.unlink()
    with pytest.raises(SystemExit) as error:subject.main()
    assert error.value.code==2
    assert json.loads(capsys.readouterr().out)["status"] == "INTEGRATION_BLOCKED"


def test_resampling_boundaries_and_no_future_bucket_influence(local_fixture):
    _,_,frame,_=local_fixture
    for interval,minutes in (("5m",5),("15m",15)):
        before=resample_ohlcv(frame,interval)
        changed=frame.copy()
        changed.loc[minutes:, ["open","high","low","close","volume"]]=999.
        after=resample_ohlcv(changed,interval)
        pd.testing.assert_series_equal(before.iloc[0],after.iloc[0])
        assert before.iloc[0]["volume"]==minutes
        assert before.iloc[0]["trades"]==minutes*2
        assert before.iloc[1]["timestamp"]==subject.START+pd.Timedelta(minutes=minutes)
