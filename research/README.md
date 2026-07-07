# Quant Research Lab

This top-level `research/` directory stores durable quant research artifacts and experiment outputs.

Important distinction:

- `app/research/` contains executable Python research code.
- `research/` contains durable artifacts such as datasets, reports, leaderboards, notebooks, feature stores, and experiment outputs.

## Folder structure

- `datasets/` - Dataset storage for research workflows.
- `datasets/raw/` - Immutable raw market or reference data captures.
- `datasets/processed/` - Cleaned, normalized, or transformed datasets ready for analysis.
- `features/` - Persisted feature outputs and feature analysis artifacts.
- `strategies/` - Durable strategy research artifacts, notes, and experiment outputs.
- `backtests/` - Saved backtest outputs, metrics, and run artifacts.
- `optimization/` - Optimization experiment artifacts and configuration snapshots.
- `optimization/grid_search/` - Grid-search optimization outputs and intermediate artifacts.
- `leaderboards/` - Ranked optimizer results and strategy comparison tables.
- `reports/` - Research summaries, charts, and final analysis reports.
- `walk_forward/` - Walk-forward validation outputs and analysis artifacts.
- `monte_carlo/` - Monte Carlo simulation outputs and robustness analysis artifacts.
- `ml/` - Machine learning research artifacts, model outputs, and evaluation results.
- `feature_store/` - Versioned feature datasets intended for reuse across experiments.
- `candidate_signals/` - Candidate signal outputs awaiting validation or promotion.
- `notebooks/` - Research notebooks and exploratory analysis.

## Optimizer output convention

Future optimizer leaderboard outputs should be written to `research/leaderboards/`.

Example:

```text
research/leaderboards/mean_reversion_gridsearch_v1.csv
```
