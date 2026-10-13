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

## Consulta local del informe histórico

La interfaz consulta exclusivamente `research/reports/failure_patterns_exp_000021_000024.md`, ya integrado en `main`. No regenera ese informe, abre sus enlaces ni ejecuta experimentos. Devuelve citas literales, números de línea y SHA-256 del archivo; no completa métricas ausentes ni transforma clasificaciones de triage en veredictos de gobernanza.

Desde la raíz del repositorio:

```bash
venv/bin/python -m app.research.query_report --list-sections
venv/bin/python -m app.research.query_report --experiment EXP-000024 --json
venv/bin/python -m app.research.query_report --section "Recurring mechanisms and interpretation"
venv/bin/python -m app.research.query_report --query FEE_DOMINATED
```

Los filtros se combinan con AND. `--query` busca una subcadena literal sin distinguir mayúsculas; no interpreta preguntas en lenguaje natural. `--section` selecciona un título exacto, también sin distinguir mayúsculas, hasta el siguiente título. Las filas de tablas incluyen su encabezado como contexto. Sin filtros devuelve todas las líneas no vacías. Sin coincidencias devuelve una lista vacía; un experimento o sección inexistente, o un informe ausente o vacío, produce un error con código de salida 2.

El modo local es el único conectado: no necesita credenciales ni acceso de red. La integración real de GPT-6 permanece sin conectar y deshabilitada por defecto; disponer de una API key no activa llamadas. Cualquier incorporación futura debe ser opcional y explícita, y conservar las citas y los límites de gobernanza.

Pruebas enfocadas: `venv/bin/python -m pytest -q -p no:cacheprovider app/research/test_query_report.py`.
