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
