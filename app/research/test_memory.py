import argparse
import csv
import json
from pathlib import Path

import pytest

from app.research.memory.experiment_store import MEMORY_INDEX_COLUMNS, load_memory_index
from app.research.memory.experiment_writer import (
    write_experiment_memory,
    write_imported_experiment_memory,
)
from app.research.memory.import_leaderboard import import_historical_leaderboard
from app.research.memory.knowledge_base import build_memory_summary
from app.research.memory.journal import build_experiment_journal
from app.research.memory.report_index import render_memory_summary
from app.research.optimizer.grid_search import (
    GridSearchConfig,
    GridSearchResult,
    GridSearchSummary,
    main as optimizer_main,
)
from app.research.optimizer.leaderboard import LeaderboardRow
from app.research.simulation import BacktestMetrics


def _sample_summary() -> GridSearchSummary:
    metrics = BacktestMetrics(
        total_trades=123,
        wins=60,
        losses=63,
        win_rate=60 / 123,
        gross_pnl=12.5,
        estimated_fees=5.0,
        net_pnl=7.5,
        profit_factor=1.42,
        expectancy=0.08,
        average_win=1.1,
        average_loss=-0.9,
        max_drawdown=3.4,
    )
    result = GridSearchResult(
        strategy="mean_reversion",
        parameters={
            "rsi_threshold": 25,
            "volume_ratio": 1.0,
            "take_profit_pct": 0.003,
            "stop_loss_pct": 0.002,
            "max_holding_candles": 20,
            "distance_from_ema20": -0.002,
        },
        metrics=metrics,
        average_holding_candles=14.5,
    )
    leaderboard_row = LeaderboardRow(
        strategy="mean_reversion",
        timeframe="1m",
        rank=1,
        total_trades=metrics.total_trades,
        wins=metrics.wins,
        losses=metrics.losses,
        win_rate=metrics.win_rate,
        gross_profit=metrics.average_win * metrics.wins,
        gross_loss=abs(metrics.average_loss * metrics.losses),
        profit_factor=metrics.profit_factor,
        expectancy=metrics.expectancy,
        max_drawdown=metrics.max_drawdown,
        gross_pnl=metrics.gross_pnl,
        net_pnl=metrics.net_pnl,
        average_holding_candles=result.average_holding_candles,
        parameters=result.parameters,
    )
    return GridSearchSummary(
        strategy="mean_reversion",
        timeframe="1m",
        evaluated_configurations=48,
        ranked_results=[result],
        leaderboard_rows=[leaderboard_row],
    )


def _sample_config(tmp_path: Path) -> GridSearchConfig:
    return GridSearchConfig(
        strategy="mean_reversion",
        data=Path("data/BTCUSDT_1m.csv"),
        timeframe="1m",
        output=Path("research/leaderboards/mean_reversion_test.csv"),
        config_file=tmp_path / "mean_reversion.yaml",
        min_trades=50,
        parameters={
            "rsi_threshold": [25, 30],
            "volume_ratio": [0.8, 1.0],
            "take_profit_pct": [0.003],
            "stop_loss_pct": [0.002],
            "max_holding_candles": [20],
            "distance_from_ema20": [-0.002],
        },
    )


def _patch_memory_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    index_path = tmp_path / "research" / "memory" / "index.csv"
    journal_dir = tmp_path / "research" / "journal"
    memory_dir = tmp_path / "research" / "memory"

    import app.research.memory.experiment_id as experiment_id_module
    import app.research.memory.experiment_store as store_module
    import app.research.memory.experiment_writer as writer_module

    monkeypatch.setattr(store_module, "JOURNAL_DIR", journal_dir)
    monkeypatch.setattr(store_module, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(store_module, "MEMORY_INDEX_PATH", index_path)
    monkeypatch.setattr(experiment_id_module, "MEMORY_INDEX_PATH", index_path)
    monkeypatch.setattr(writer_module, "JOURNAL_DIR", journal_dir)
    return index_path


def test_legacy_index_schema_migrates_and_preserves_exp_000001(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        (
            "experiment_id,date,strategy,timeframe,dataset,best_profit_factor,"
            "best_expectancy,best_configuration,status\n"
            "EXP-000001,2026-07-14,mean_reversion,1m,data/BTCUSDT_1m.csv,0.51,-0.07,"
            "\"{\"\"rsi_threshold\"\":25}\",completed\n"
        ),
        encoding="utf-8",
    )

    migrated = load_memory_index(index_path)
    migrated_again = load_memory_index(index_path)

    assert migrated.columns.tolist() == MEMORY_INDEX_COLUMNS
    assert len(migrated) == 1
    assert len(migrated_again) == 1
    assert migrated.loc[0, "experiment_id"] == "EXP-000001"
    assert migrated.loc[0, "created_at_utc"] == ""
    assert migrated.loc[0, "config_file"] == ""
    assert migrated.loc[0, "leaderboard_file"] == ""
    assert migrated.loc[0, "status"] == "completed"
    assert json.loads(migrated.loc[0, "best_configuration"]) == {"rsi_threshold": 25}
    assert index_path.read_text(encoding="utf-8").count("EXP-000001") == 1


def test_write_experiment_memory_creates_complete_row_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    index_path = _patch_memory_paths(monkeypatch, tmp_path)
    config = _sample_config(tmp_path)
    summary = _sample_summary()

    artifacts = write_experiment_memory(config, summary)
    index_df = load_memory_index(index_path)

    assert artifacts.experiment_id == "EXP-000001"
    assert artifacts.journal_path == tmp_path / "research" / "journal" / "EXP-000001.md"
    assert artifacts.index_path == index_path
    assert len(index_df) == 1

    row = index_df.loc[0].to_dict()
    assert row["experiment_id"] == "EXP-000001"
    assert row["strategy"] == "mean_reversion"
    assert row["timeframe"] == "1m"
    assert row["dataset"] == "data/BTCUSDT_1m.csv"
    assert row["created_at_utc"].endswith("Z")
    assert row["config_file"].endswith("mean_reversion.yaml")
    assert row["leaderboard_file"] == "research/leaderboards/mean_reversion_test.csv"
    assert int(row["total_configurations"]) == 48
    assert int(row["eligible_configurations"]) == 1
    assert float(row["best_profit_factor"]) == pytest.approx(1.42)
    assert float(row["best_expectancy"]) == pytest.approx(0.08)
    assert float(row["best_max_drawdown"]) == pytest.approx(3.4)
    assert int(row["best_total_trades"]) == 123
    assert row["status"] == "completed"
    assert json.loads(row["best_configuration"])["rsi_threshold"] == 25

    journal_text = artifacts.journal_path.read_text(encoding="utf-8")
    assert "Experiment ID: EXP-000001" in journal_text
    assert "Date:" in journal_text
    assert "Strategy: mean_reversion" in journal_text
    assert "Timeframe: 1m" in journal_text
    assert "Dataset: data/BTCUSDT_1m.csv" in journal_text
    assert "Config File:" in journal_text
    assert "Leaderboard File: research/leaderboards/mean_reversion_test.csv" in journal_text
    assert "Total Configurations: 48" in journal_text
    assert "Eligible Configurations: 1" in journal_text
    assert "Best PF: 1.4200" in journal_text
    assert "Best Expectancy: 0.0800" in journal_text
    assert "Best Max Drawdown: 3.4000" in journal_text
    assert "Best Total Trades: 123" in journal_text
    assert "Best Configuration:" in journal_text
    assert "Deterministic Interpretation" in journal_text
    assert "Deterministic Boundary Recommendations" in journal_text


def test_exploratory_journal_retains_generic_follow_up_recommendations(tmp_path: Path):
    journal_text = build_experiment_journal(
        "EXP-EXPLORATORY",
        "2026-09-02T00:00:00Z",
        _sample_config(tmp_path),
        _sample_summary(),
    )

    assert "Center the next grid around the current best parameters" in journal_text
    assert "another timeframe" in journal_text


def test_closed_preregistered_journal_emits_no_rescue_closure(tmp_path: Path):
    config = _sample_config(tmp_path).model_copy(
        update={
            "hypothesis_id": "HYP-CLOSED-001",
            "preregistered": True,
            "anti_tuning": True,
            "final_status": "CLOSED_REJECTED",
            "verdict": "BASELINE_REJECT",
            "failure_classification": "FEE_DOMINATED",
        }
    )

    journal_text = build_experiment_journal(
        "EXP-CLOSED",
        "2026-09-02T00:00:00Z",
        config,
        _sample_summary(),
    )

    assert "one frozen baseline configuration" in journal_text
    assert "Center the next grid" not in journal_text
    assert "another timeframe" not in journal_text
    assert "No rescue or parameter optimization is authorized" in journal_text
    assert "Final Status: CLOSED_REJECTED" in journal_text
    assert "Failure Classification: FEE_DOMINATED" in journal_text


def test_legacy_grid_config_defaults_to_exploratory_governance(tmp_path: Path):
    config = _sample_config(tmp_path)

    assert config.hypothesis_id is None
    assert config.preregistered is False
    assert config.anti_tuning is False
    assert config.final_status is None
    assert config.verdict is None


def test_report_index_aggregates_canonical_and_legacy_rows(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEMORY_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "EXP-000001",
                "created_at_utc": "",
                "strategy": "mean_reversion",
                "timeframe": "1m",
                "dataset": "data/BTCUSDT_1m.csv",
                "config_file": "",
                "leaderboard_file": "",
                "total_configurations": "",
                "eligible_configurations": "",
                "best_profit_factor": "0.51",
                "best_expectancy": "-0.07",
                "best_max_drawdown": "",
                "best_total_trades": "",
                "best_configuration": json.dumps({"rsi_threshold": 25}),
                "status": "completed",
            }
        )
        writer.writerow(
            {
                "experiment_id": "EXP-000002",
                "created_at_utc": "2026-07-14T12:00:00Z",
                "strategy": "donchian_breakout",
                "timeframe": "5m",
                "dataset": "data/BTCUSDT_1m.csv",
                "config_file": "config.yaml",
                "leaderboard_file": "leaderboard.csv",
                "total_configurations": "324",
                "eligible_configurations": "12",
                "best_profit_factor": "0.77",
                "best_expectancy": "-0.05",
                "best_max_drawdown": "64.5",
                "best_total_trades": "1057",
                "best_configuration": json.dumps({"lookback": 10}),
                "status": "completed",
            }
        )

    summary = build_memory_summary(index_path)

    assert summary.total_experiments == 2
    assert summary.completed_experiments == 2
    assert summary.strategies_tested == ["donchian_breakout", "mean_reversion"]
    assert summary.timeframes_tested == ["1m", "5m"]
    assert summary.best_experiment_id == "EXP-000002"
    assert summary.best_strategy == "donchian_breakout"
    assert summary.best_timeframe == "5m"
    assert summary.best_profit_factor == 0.77
    assert summary.best_expectancy == -0.05
    assert summary.average_best_profit_factor == pytest.approx(0.64)
    assert summary.average_best_expectancy == pytest.approx(-0.06)


def test_render_memory_summary_outputs_expected_lines(monkeypatch: pytest.MonkeyPatch):
    import app.research.memory.report_index as report_index_module
    from app.research.memory.knowledge_base import MemorySummary

    monkeypatch.setattr(
        report_index_module,
        "build_memory_summary",
        lambda: MemorySummary(
            total_experiments=2,
            completed_experiments=2,
            strategies_tested=["donchian_breakout", "mean_reversion"],
            timeframes_tested=["1m", "5m"],
            best_experiment_id="EXP-000002",
            best_strategy="donchian_breakout",
            best_timeframe="5m",
            best_profit_factor=0.77,
            best_expectancy=-0.05,
            average_best_profit_factor=0.64,
            average_best_expectancy=-0.06,
        ),
    )

    lines = render_memory_summary()

    assert "Total experiments: 2" in lines
    assert "Completed experiments: 2" in lines
    assert "Strategies tested: donchian_breakout, mean_reversion" in lines
    assert "Timeframes tested: 1m, 5m" in lines
    assert "Best experiment ID: EXP-000002" in lines
    assert "Best strategy: donchian_breakout" in lines
    assert "Best timeframe: 5m" in lines


def test_optimizer_failure_does_not_create_experiment_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    index_path = _patch_memory_paths(monkeypatch, tmp_path)
    config = _sample_config(tmp_path)

    import app.research.optimizer.grid_search as grid_search_module

    monkeypatch.setattr(grid_search_module, "parse_args", lambda: argparse.Namespace())
    monkeypatch.setattr(grid_search_module, "build_config_from_args", lambda args: config)
    monkeypatch.setattr(
        grid_search_module,
        "load_featured_data",
        lambda data, timeframe, **kwargs: object(),
    )

    def _boom(**kwargs):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(grid_search_module, "run_grid_search", _boom)

    with pytest.raises(RuntimeError, match="optimizer failed"):
        optimizer_main()

    index_df = load_memory_index(index_path)
    assert index_df.empty
    assert not (tmp_path / "research" / "journal" / "EXP-000001.md").exists()


def test_successful_historical_import_creates_imported_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    index_path = _patch_memory_paths(monkeypatch, tmp_path)
    leaderboard_path = tmp_path / "research" / "leaderboards" / "donchian_breakout_5m_smoke_v1.csv"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(
        "\n".join(
            [
                "strategy,timeframe,rank,total_trades,win_rate,profit_factor,expectancy,max_drawdown,gross_pnl,net_pnl,lookback,volume_ratio,take_profit_pct,stop_loss_pct,max_holding_candles",
                "donchian_breakout,5m,1,100,0.44,0.80,-0.05,60.0,10.0,-5.0,10,0.8,0.006,0.005,36",
                "donchian_breakout,5m,2,150,0.45,0.80,-0.05,55.0,12.0,-4.0,20,1.0,0.006,0.005,24",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "research" / "optimization" / "grid_search" / "donchian_breakout_5m_smoke.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "strategy: donchian_breakout",
                "data: data/BTCUSDT_1m.csv",
                "timeframe: 5m",
                f"output: {leaderboard_path.as_posix()}",
                "min_trades: 50",
                "parameters:",
                "  lookback:",
                "    - 10",
                "    - 20",
                "  volume_ratio:",
                "    - 0.8",
                "    - 1.0",
                "  take_profit_pct:",
                "    - 0.006",
                "  stop_loss_pct:",
                "    - 0.005",
                "  max_holding_candles:",
                "    - 24",
                "    - 36",
            ]
        ),
        encoding="utf-8",
    )

    imported, message = import_historical_leaderboard(leaderboard_path, config_path)

    index_df = load_memory_index(index_path)
    assert imported is True
    assert "Imported experiment ID: EXP-000001" in message
    assert "Leaderboard:" in message
    assert "Journal:" in message
    assert "Memory index:" in message
    assert len(index_df) == 1
    row = index_df.iloc[0].to_dict()
    assert row["experiment_id"] == "EXP-000001"
    assert row["status"] == "imported"
    assert row["strategy"] == "donchian_breakout"
    assert row["timeframe"] == "5m"
    assert int(row["total_configurations"]) == 8
    assert int(row["eligible_configurations"]) == 2
    assert float(row["best_max_drawdown"]) == pytest.approx(55.0)
    assert int(row["best_total_trades"]) == 150
    assert json.loads(row["best_configuration"])["lookback"] == 20
    journal_text = (tmp_path / "research" / "journal" / "EXP-000001.md").read_text(encoding="utf-8")
    assert "Source: historical leaderboard import" in journal_text
    assert "Optimizer was not rerun: yes" in journal_text


def test_historical_import_prevents_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    index_path = _patch_memory_paths(monkeypatch, tmp_path)
    existing_config = _sample_config(tmp_path)
    leaderboard_path = tmp_path / "research" / "leaderboards" / "mean_reversion_test.csv"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(
        "strategy,timeframe,rank,total_trades,win_rate,profit_factor,expectancy,max_drawdown,gross_pnl,net_pnl\n",
        encoding="utf-8",
    )
    existing_config = existing_config.model_copy(update={"output": leaderboard_path})
    existing_summary = _sample_summary()
    write_experiment_memory(existing_config, existing_summary)

    config_path = tmp_path / "research" / "optimization" / "grid_search" / "mean_reversion.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        (
            "strategy: mean_reversion\n"
            "data: data/BTCUSDT_1m.csv\n"
            "timeframe: 1m\n"
            f"output: {leaderboard_path.as_posix()}\n"
            "parameters:\n"
            "  rsi_threshold:\n"
            "    - 25\n"
        ),
        encoding="utf-8",
    )

    imported, message = import_historical_leaderboard(leaderboard_path, config_path)

    index_df = load_memory_index(index_path)
    assert imported is False
    assert "already imported" in message
    assert len(index_df) == 1
    assert index_df.iloc[0]["experiment_id"] == "EXP-000001"


def test_historical_import_missing_leaderboard(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("strategy: mean_reversion\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Leaderboard file not found"):
        import_historical_leaderboard(tmp_path / "missing.csv", config_path)


def test_historical_import_missing_config(tmp_path: Path):
    leaderboard_path = tmp_path / "leaderboard.csv"
    leaderboard_path.write_text("strategy,timeframe,rank,total_trades,win_rate,profit_factor,expectancy,max_drawdown,gross_pnl,net_pnl\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Config file not found"):
        import_historical_leaderboard(leaderboard_path, tmp_path / "missing.yaml")


def test_historical_import_best_row_tie_breaking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_memory_paths(monkeypatch, tmp_path)
    leaderboard_path = tmp_path / "research" / "leaderboards" / "tie.csv"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(
        "\n".join(
            [
                "strategy,timeframe,rank,total_trades,win_rate,profit_factor,expectancy,max_drawdown,gross_pnl,net_pnl,lookback,volume_ratio,take_profit_pct,stop_loss_pct,max_holding_candles",
                "donchian_breakout,5m,1,100,0.44,0.90,-0.04,70.0,10.0,-5.0,10,0.8,0.006,0.005,36",
                "donchian_breakout,5m,2,120,0.44,0.90,-0.04,65.0,10.0,-5.0,20,0.8,0.006,0.005,36",
                "donchian_breakout,5m,3,140,0.44,0.90,-0.04,65.0,10.0,-5.0,30,0.8,0.006,0.005,36",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "research" / "optimization" / "grid_search" / "tie.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "strategy: donchian_breakout",
                "data: data/BTCUSDT_1m.csv",
                "timeframe: 5m",
                f"output: {leaderboard_path.as_posix()}",
                "parameters:",
                "  lookback:",
                "    - 10",
                "    - 20",
                "    - 30",
                "  volume_ratio:",
                "    - 0.8",
                "  take_profit_pct:",
                "    - 0.006",
                "  stop_loss_pct:",
                "    - 0.005",
                "  max_holding_candles:",
                "    - 36",
            ]
        ),
        encoding="utf-8",
    )

    imported, _ = import_historical_leaderboard(leaderboard_path, config_path)
    index_df = load_memory_index(tmp_path / "research" / "memory" / "index.csv")

    assert imported is True
    assert json.loads(index_df.iloc[0]["best_configuration"])["lookback"] == 30


def test_historical_import_preserves_existing_experiments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    index_path = _patch_memory_paths(monkeypatch, tmp_path)
    write_experiment_memory(_sample_config(tmp_path), _sample_summary())
    imported_config = GridSearchConfig(
        strategy="donchian_breakout",
        data=Path("data/BTCUSDT_1m.csv"),
        timeframe="5m",
        output=Path("research/leaderboards/imported.csv"),
        config_file=tmp_path / "imported.yaml",
        parameters={"lookback": [10], "volume_ratio": [0.8]},
    )
    imported_summary = GridSearchSummary(
        strategy="donchian_breakout",
        timeframe="5m",
        evaluated_configurations=2,
        ranked_results=[
            GridSearchResult(
                strategy="donchian_breakout",
                parameters={"lookback": 10, "volume_ratio": 0.8},
                metrics=BacktestMetrics(
                    total_trades=50,
                    wins=0,
                    losses=0,
                    win_rate=0.4,
                    gross_pnl=1.0,
                    estimated_fees=0.0,
                    net_pnl=-1.0,
                    profit_factor=0.7,
                    expectancy=-0.02,
                    average_win=0.0,
                    average_loss=0.0,
                    max_drawdown=10.0,
                ),
                average_holding_candles=12.0,
            )
        ],
        leaderboard_rows=[],
    )

    write_imported_experiment_memory(imported_config, imported_summary)
    index_df = load_memory_index(index_path)

    assert len(index_df) == 2
    assert index_df.iloc[0]["experiment_id"] == "EXP-000001"
    assert index_df.iloc[1]["experiment_id"] == "EXP-000002"
