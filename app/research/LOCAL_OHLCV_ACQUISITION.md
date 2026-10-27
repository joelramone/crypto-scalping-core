# Authorized local acquisition (Windows CMD)

Purpose: technical ingestion/integrity only, not strategy evaluation. Run from
repository root on persistent local storage. No download was run in Codex Cloud.
Python 3.12 is recommended; offline tests ran on Python 3.12.14/Linux with the
versions below. Windows itself was not available for testing.

After human review and checkout of `feat/local-authorized-ohlcv-acquisition-v1`
(or the merged main containing it), use Windows CMD; no PowerShell required:

```bat
py -3.12 -m venv .venv-local
.venv-local\Scripts\python.exe -m pip install pandas==2.2.3 requests==2.34.2 "pydantic==2.13.4"
.venv-local\Scripts\python.exe -m app.research.download_data --exchange binance --product usdm-futures --symbol BTCUSDT --interval 1m --start 2024-10-01T00:00:00Z --end 2024-10-08T00:00:00Z --output data/binance_usdm_BTCUSDT_1m_20241001_20241008.csv --dry-run
.venv-local\Scripts\python.exe -m app.research.download_data --exchange binance --product usdm-futures --symbol BTCUSDT --interval 1m --start 2024-10-01T00:00:00Z --end 2024-10-08T00:00:00Z --output data/binance_usdm_BTCUSDT_1m_20241001_20241008.csv --execute-authorized-download
```

Only the final command contacts Binance's public USD-M klines endpoint. No keys,
paid services, backtest, optimizer or Research Memory writer are involved. No
source fallback is performed. End is exclusive. Dates from 2025 onward remain
rejected; dry-run of other technically allowed dates does not authorize use.

Expected persistent outputs (both gitignored):

- `data\binance_usdm_BTCUSDT_1m_20241001_20241008.csv`
- `data\binance_usdm_BTCUSDT_1m_20241001_20241008.csv.manifest.json`

Verify the local bytes, schema, interval, count and manifest using CMD:

```bat
type data\binance_usdm_BTCUSDT_1m_20241001_20241008.csv.manifest.json
certutil -hashfile data\binance_usdm_BTCUSDT_1m_20241001_20241008.csv SHA256
.venv-local\Scripts\python.exe -c "import hashlib,json,pathlib,pandas as pd; from app.research.download_data import validate_download_frame,parse_datetime_to_utc_ms,CSV_COLUMNS; p=pathlib.Path('data/binance_usdm_BTCUSDT_1m_20241001_20241008.csv'); m=json.loads(p.with_suffix('.csv.manifest.json').read_text()); d=pd.read_csv(p); assert list(d.columns)==CSV_COLUMNS; d['timestamp']=pd.to_datetime(d['timestamp'],utc=True); validate_download_frame(d,parse_datetime_to_utc_ms('2024-10-01T00:00:00Z'),parse_datetime_to_utc_ms('2024-10-08T00:00:00Z')); assert len(d)==m['row_count']==10080; assert p.stat().st_size==m['file_size_bytes']; assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']; print('LOCAL_DATASET_VALIDATED')"
```

The manifest records source/product, UTC bounds, first/last timestamps, row count,
SHA-256 of CSV bytes, byte size, acquisition time and duration. Save both files
in your normal local backup. Do not commit them or assume a Git checkout backs
them up. No external upload is performed.

Publication uses same-directory temporary files and atomic, no-replacement hard
links supported by NTFS. Other filesystems may not support this: failure is
explicit, with no unsafe overwrite fallback. Handled errors remove temporary
files and this run's published files. Two files are not one atomic transaction:
a process kill/power loss can leave temporary files or a CSV without a manifest.
Such an interrupted pair is not success; inspect it manually before recovery.
Never delete or overwrite an existing valid dataset to rerun. Either existing
CSV or manifest blocks acquisition before any request. Binance availability,
regional access and actual historical completeness remain untested in Cloud.

## TASK-020: read-only loader/resampling integration

After checking out the reviewed integration change, use the same environment
and run this command from repository root in Windows CMD:

```bat
.venv-local\Scripts\python.exe -m app.research.data_validation.check_integration --input data/binance_usdm_BTCUSDT_1m_20241001_20241008.csv --manifest data/binance_usdm_BTCUSDT_1m_20241001_20241008.csv.manifest.json
```

Success prints JSON with `status: INTEGRATION_VALIDATED`, verified SHA-256,
schema/UTC/continuity flags and `resampled` counts 10080 (1m), 2016 (5m), 672
(15m). Failure prints `INTEGRATION_BLOCKED` and exits with code 2. The command
writes no files, downloads nothing and executes no strategy or performance
calculation. It reads only the authorized local dataset and adjacent manifest;
the expected checksum is pinned to the operator's independently verified file.

Existing resampling is UTC, left-closed and labeled by opening time. Each
aggregate is available only at its bucket end, not at its label. The check
requires all constituent minutes in every bucket; it does not fill gaps,
interpolate, calculate indicators or change simulator semantics. Passing is
technical compatibility only, not authorization for a trading experiment.

## TASK-021B: future fixed technical diagnostic (not authorized to run yet)

After separate human authorization for this exact configuration and dataset use,
run from repository root in Windows CMD with the same Python 3.12 environment:

```bat
.venv-local\Scripts\python.exe -m app.research.backtester --data data/binance_usdm_BTCUSDT_1m_20241001_20241008.csv --strategy baseline_trend --timeframe 1m --execution-mode next_bar_open --expected-sha256 f04bbaf8b8fe5a333bd5e67891cec081fea4b9dba8850a3166bbbfb9e3893c5f --json
```

Do not execute this during preparation. Cloud tests use synthetic fixtures only.
The JSON diagnostic is restricted to baseline_trend/1m with a mandatory expected
SHA-256; the hashed bytes are exactly those passed to the existing loader. Hash
mismatch, invalid/nonfinite OHLCV or discontinuous timestamps abort before strategy
execution. The command writes nothing and invokes neither optimizer nor Research
Memory writers. It reports actual fixed defaults (long, TP 0.004, SL 0.0025,
holding 30, notional 100 USDT, fee 0.0004 per side), source hash/coverage, Git
revision and dirty-tree flag, duration, gross/net PnL, fees, trades, net PF and
closed-trade drawdown. Infinite PF is JSON null with an explicit status; a
zero-trade run reports zero metrics. This is not a performance acceptance gate.

`next_bar_open` observes the signal at completed candle t's close and enters at
open[t+1], unconditionally with respect to that next candle's high/low/close.
Without the next candle no trade opens. SL/TP may trigger during the entry candle;
SL precedes TP, then indicator exit, then max-holding close. The holding ceiling
counts the entry candle: 30 bars allow exits through entry index + 29. Existing
trade `holding_candles` remains the index difference for compatibility (29 at
that ceiling), not a new inclusive-bar count. Near dataset end the holding period
is truncated. One position is allowed; scanning resumes at exit index + 1, so
signals on the exit candle remain suppressed. Features are computed causally
then warm-up rows dropped, as in the existing pipeline.

The default remains `signal_close` for all legacy callers: same-candle close
entry, exits starting on the next candle, and unchanged historical holding/index
semantics. New mode is opt-in; no old results/configurations are rewritten.
This changes entry/holding semantics only for callers explicitly selecting it.

Neither convention models spread, slippage, transport latency, partial fills,
market impact, funding or opening-gap barrier fills. Candle opens are assumed
fully executable, not verified bid/ask fills. Drawdown uses closed trades, not
intratrade equity; seven days, possible zero trades and indicator warm-up cannot
establish profitability. No historical diagnostic has been run or authorized by
this code delivery. Windows execution itself remains untested in Cloud.
