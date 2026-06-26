# crypto-scalping-core

Core modular para un bot de **crypto scalping** en Python, diseñado con arquitectura multi-agente y foco en **gestión de riesgo, control operativo y escalabilidad**.

## 🎯 Objetivo

- Separar estrategia, riesgo y ejecución.
- Permitir iteraciones rápidas en paper trading antes de operar en real.
- Definir apagado automático ante límites de pérdida o cumplimiento de objetivo.
- Mantener una base reusable para nuevas estrategias.

## 🧱 Estructura

```text
crypto-scalping-core/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── data/
│   │   ├── market_stream.py
│   │   └── features.py
│   ├── agents/
│   │   ├── strategy_agent.py
│   │   ├── risk_agent.py
│   │   └── supervisor_agent.py
│   ├── trading/
│   │   ├── executor.py
│   │   └── paper_wallet.py
│   ├── storage/
│   │   └── trades_repo.py
│   └── utils/
│       └── logger.py
├── diagrams/
│   └── architecture.puml
├── docker/
│   └── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙️ Requisitos

- Python 3.9+
- pip
- (Opcional) Docker

## 🚀 Instalación rápida

```bash
git clone https://github.com/tu-usuario/crypto-scalping-core.git
cd crypto-scalping-core
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## 🧪 Ejecución

Paper trading local:

```bash
python -m app.main --paper --steps 20
```


## 📚 V2 research: datos históricos y backtesting

Descargar velas históricas OHLCV de Binance USD-M Futures para investigación:

```bash
python -m app.research.download_data --symbol BTCUSDT --interval 1m --start 2025-01-01 --end 2026-01-01 --output data/BTCUSDT_1m.csv
```

El comando crea la carpeta `data/` si no existe y guarda un CSV compatible con el backtester V2. Los CSV descargados en `data/*.csv` quedan ignorados por Git para evitar commitear datasets pesados o generados localmente.

Ejecutar backtest sobre el CSV descargado:

```bash
python -m app.research.backtester --data data/BTCUSDT_1m.csv --strategy breakout
```

## 🤖 Flujo de agentes

1. `MarketStream` emite ticks simulados.
2. `FeatureBuilder` calcula momentum.
3. `StrategyAgent` produce señal (`buy`, `sell`, `hold`).
4. `RiskAgent` valida límites diarios y tamaño de posición.
5. `Executor` solo ejecuta si hay aprobación de riesgo.
6. `SupervisorAgent` puede detener toda la operativa si se alcanzan umbrales de PnL.
7. `TradesRepository` registra las operaciones.

## ⚠️ Disclaimer

Proyecto educativo y experimental.
No constituye asesoramiento financiero.
El trading conlleva riesgo real de pérdida de capital.
