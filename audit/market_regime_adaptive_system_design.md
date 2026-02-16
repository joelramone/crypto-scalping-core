# Market Regime Adaptive System Redesign (Crypto Scalping)

## 1) Diagnóstico cuantitativo del estado actual

Con base en tus resultados reportados:

- **Expectancy negativa** en TRENDING, SIDEWAYS y HIGH_VOL.
- **BreakoutTrendStrategy** ~break-even en HIGH_VOL (PF≈0.99), lo cual sugiere que su edge está cerca de activarse pero con exits/risk sizing aún subóptimos.
- **RSI mean reversion** sin edge robusto (probable dependencia a micro-estructura del spread/slippage y ruido).
- **Sample size bajo (<10 trades/sim)**: estadísticamente no defendible para inferencia de edge.
- **Monte Carlo negativo**: evidencia de fragilidad del sistema actual.

Conclusión operativa: hoy no existe un framework adaptativo real; existe selección estática con baja potencia estadística.

---

## 2) Diseño conceptual objetivo (small quant fund-grade)

El sistema debe operar como **Regime-First Execution Framework**:

1. **Detectar régimen** en cada barra/ventana con variables cuantitativas robustas.
2. **Asignar probabilidad de régimen** (no etiqueta dura exclusivamente), con histéresis para evitar flips.
3. **Permitir trading solo si existe edge ex-ante** medido por un `EdgeScore` en ventana walk-forward reciente.
4. **Seleccionar estrategia por régimen** a partir de una tabla de elegibilidad basada en desempeño OOS.
5. **Aplicar risk/exits dinámicos** según volatilidad, tendencia y liquidez.
6. **Entrar en modo NO-TRADE** cuando:
   - incertidumbre de régimen es alta,
   - edge reciente < umbral,
   - degradación de ejecución (costes/slippage) rompe PF esperado.

Principio rector: *No trade is a position.*

---

## 3) Arquitectura modular propuesta (Clean Architecture + Strategy Pattern)

## 3.1 Capas

### Domain (núcleo puro, sin infraestructura)
- `entities/`
  - `MarketState` (features agregadas + probabilidades de régimen)
  - `Signal`, `OrderIntent`, `Position`, `Trade`
  - `RiskLimits`, `ExecutionContext`
- `value_objects/`
  - `RegimeType` (`TRENDING`, `SIDEWAYS`, `HIGH_VOL`, `UNDEFINED`)
  - `EdgeScore`, `ConfidenceScore`
- `services/`
  - `RegimeDetector` (interfaz)
  - `EdgeEvaluator` (interfaz)
  - `RiskManager` (interfaz)
  - `ExitEngine` (interfaz)
  - `Strategy` (interfaz base)
  - `StrategySelector` (interfaz)

### Application (casos de uso/orquestación)
- `use_cases/`
  - `ProcessMarketTick`
  - `EvaluateRegimeAndEdge`
  - `GenerateSignals`
  - `ApplyRiskAndExits`
- `orchestrators/`
  - `AdaptiveTradingOrchestrator`

### Infrastructure
- `data/` (ingesta, storage, feature store)
- `brokers/` (paper/live adapters)
- `models/` (implementaciones concretas de detector régimen)
- `repositories/` (trades, performance snapshots, cost model)

### Interface / Delivery
- CLI/API scheduler, jobs de backtest/walk-forward, observabilidad.

## 3.2 Módulos clave exigidos

### A) `regime_detector/` (separado)
Responsabilidades:
- calcular features de régimen;
- inferir probabilidades por régimen;
- aplicar smoothing + histéresis;
- emitir `RegimeDecision` con confianza.

Features mínimas:
- `ATR_norm = ATR(n) / Price`
- `ADX(n)`
- `RealizedVol` (EWMA/std intrabar)
- `VolClustering` (proxy: ratio vol_corta/vol_larga + autocorrelación de |retornos|)
- `TrendStrength` (slope + R² rolling de log-price)
- `ChopIndex` o rango/ATR para detectar lateralidad.

Enfoque recomendado (fase 1):
- clasificador híbrido por reglas calibradas + logistic score.

### B) `risk_manager/` (separado)
Responsabilidades:
- sizing por volatilidad y convexidad de edge;
- límites de pérdida diaria/semanal;
- drawdown guard;
- kill switch por degradación.

Controles:
- max risk per trade (bps de equity)
- max concurrent exposure
- max regime-specific loss streak
- reduce size cuando `RegimeConfidence` baja.

### C) `exit_engine/`
Responsabilidades:
- SL/TP dinámicos por régimen.
- trailing adaptativo por ATR.
- time-stop para scalping (decay del edge).

Plantillas recomendadas:
- **TRENDING:** SL `1.2–1.8*ATR`, TP abierto con trailing (capture convexidad).
- **SIDEWAYS:** TP corto `0.8–1.2*ATR`, SL simétrico/ligeramente menor, time-stop agresivo.
- **HIGH_VOL:** menor tamaño + SL más amplio en ATR pero riesgo monetario constante.

### D) `strategy_selector/`
Matriz de elegibilidad régimen→estrategia basada en OOS:
- habilita estrategias solo si cumplen umbrales de edge por régimen.
- ejemplo umbral mínimo por estrategia/regímen (rolling 90d OOS):
  - PF >= 1.10
  - Expectancy > 0
  - p-value bootstrap < 0.1 (o posterior probabilidad de edge > 60%)

Si no cumple: `NO_TRADE`.

---

## 4) Lógica de decisión adaptativa (pipeline en tiempo real)

1. Ingesta de mercado y actualización de features.
2. `RegimeDetector` emite `{regime_probs, top_regime, confidence}`.
3. `EdgeEvaluator` calcula `EdgeScore(strategy, regime)` en ventana walk-forward reciente.
4. `StrategySelector` filtra estrategias elegibles.
5. Si no hay elegibles o confianza baja -> `NO_TRADE`.
6. Estrategia activa genera `Signal`.
7. `RiskManager` transforma señal en `OrderIntent` (size/leverage límites).
8. `ExitEngine` inyecta plan de salida dinámico.
9. `ExecutionAdapter` envía órdenes y registra slippage/costes.
10. `PerformanceMonitor` actualiza métricas por régimen y gatilla kill-switch si hay degradación.

---

## 5) Pseudocódigo estructural

```python
class AdaptiveTradingOrchestrator:
    def __init__(self, regime_detector, strategy_selector, edge_evaluator,
                 risk_manager, exit_engine, execution, performance_monitor):
        ...

    def on_bar(self, bar, portfolio_state):
        market_state = self._build_market_state(bar)

        regime_decision = self.regime_detector.detect(market_state.features)
        # regime_decision: probs, top_regime, confidence

        candidate_strategies = self.strategy_selector.candidates(regime_decision.top_regime)

        scored = []
        for strat in candidate_strategies:
            edge = self.edge_evaluator.score(
                strategy_id=strat.id,
                regime=regime_decision.top_regime,
                asof=bar.timestamp
            )
            if edge.is_tradable and edge.confidence >= MIN_EDGE_CONF:
                scored.append((strat, edge))

        if regime_decision.confidence < MIN_REGIME_CONF or not scored:
            return Decision.no_trade(reason="low_confidence_or_no_edge")

        strat, edge = max(scored, key=lambda x: x[1].score)
        signal = strat.generate_signal(market_state, portfolio_state)

        if signal.is_flat:
            return Decision.no_trade(reason="strategy_flat")

        order_intent = self.risk_manager.allocate(
            signal=signal,
            edge=edge,
            regime=regime_decision,
            portfolio=portfolio_state
        )

        if not order_intent.allowed:
            return Decision.no_trade(reason=order_intent.block_reason)

        exit_plan = self.exit_engine.build_plan(
            entry=order_intent,
            regime=regime_decision.top_regime,
            volatility=market_state.features.atr_norm,
            liquidity=market_state.features.liquidity_score
        )

        execution_report = self.execution.submit(order_intent, exit_plan)
        self.performance_monitor.update(execution_report, regime_decision)

        if self.performance_monitor.trigger_kill_switch():
            self.execution.flatten_all()
            return Decision.no_trade(reason="kill_switch")

        return Decision.executed(execution_report)
```

---

## 6) Estrategias por régimen (propuesta concreta)

## 6.1 TRENDING
- Mantener `BreakoutTrendStrategy` como core candidate.
- Añadir filtro de calidad de ruptura:
  - expansión de rango + volumen relativo + ADX rising.
- Evitar entradas en extensión extrema (`zscore_move > threshold`).

## 6.2 SIDEWAYS
- RSI puro suele fallar en crypto microstructure. Reemplazar por:
  - mean reversion sobre **bandas de volatilidad adaptativa** + filtro de spread/coste.
  - entrada solo en extremos con reversión confirmada (micro pullback).

## 6.3 HIGH_VOL
- Estrategia híbrida:
  - breakout solo si dirección validada por flujo/tendencia intraday;
  - mean-reversion ultracorto en spikes con condiciones estrictas.
- Reducción automática de tamaño por varianza de retornos.

## 6.4 UNDEFINED / MIXED
- Estado explícito sin edge: `NO_TRADE`.

---

## 7) TP/SL dinámicos y gestión de salida

Regla base: el riesgo monetario por trade es estable; la distancia de stop varía con volatilidad.

- `stop_distance = k_sl(regime) * ATR`
- `tp_distance = k_tp(regime) * ATR`
- trailing activado después de `+x ATR` en favor.
- `time_stop` duro (p.ej. 20-60 barras según timeframe) para evitar capital atrapado.
- salida parcial escalonada para reducir varianza (ej. 50% en TP1, resto trailing).

Framework de calibración:
- calibrar `k_sl`, `k_tp`, `time_stop` en walk-forward nested.
- elegir parámetros por robustez (percentil 25 de performance), no por máximo in-sample.

---

## 8) Aumento de sample size sin curve fitting

1. **Expandir universo**: múltiples pares líquidos (BTC, ETH, SOL, etc.) con normalización de costes.
2. **Multi-timeframe coherente**: señal en 1-5m, filtro de régimen en 15-60m.
3. **Walk-forward rolling** con ventanas cortas/medias (ej. train 60d, test 15d).
4. **Combinación cross-sectional**: validar edge por instrumento y pooled.
5. **Bootstrap/Monte Carlo por bloques** para preservar autocorrelación.
6. **Purged/embargo split** para evitar leakage temporal.
7. **Mínimos estadísticos para promoción a producción**:
   - trades >= 200 por régimen-estrategia en OOS acumulado,
   - PF robusto en percentiles bajos,
   - estabilidad de expectancy por subperiodo.

---

## 9) Plan de implementación por fases

## Fase 0 — Hardening de medición (1 semana)
- estandarizar coste realista (fees, slippage, spread).
- dashboard de métricas por régimen-estrategia.
- tagging obligatorio de cada trade con régimen detectado y confianza.

**Deliverable:** baseline confiable + auditoría de datos.

## Fase 1 — Núcleo adaptativo (2 semanas)
- crear módulos `regime_detector`, `strategy_selector`, `risk_manager`, `exit_engine`.
- integrar orquestador único y modo `NO_TRADE`.
- thresholds iniciales conservadores.

**Deliverable:** motor end-to-end adaptativo en paper/backtest.

## Fase 2 — Edge gating robusto (2 semanas)
- implementar `EdgeEvaluator` OOS rolling.
- política de elegibilidad y desactivación automática.
- kill-switch por degradación en vivo.

**Deliverable:** sistema que solo opera cuando hay edge medible.

## Fase 3 — Exits y sizing avanzados (2 semanas)
- TP/SL adaptativos por régimen.
- trailing + time-stop + partial exits.
- position sizing volatility-targeted.

**Deliverable:** mejora de PF y reducción de tails negativos.

## Fase 4 — Validación estadística institucional (2 semanas)
- walk-forward nested + bootstrap bloques + Monte Carlo stress.
- análisis de sensibilidad y fragilidad paramétrica.
- criterios formales de go-live.

**Deliverable:** decision pack cuantitativo para despliegue controlado.

---

## 10) Métricas objetivo (12 semanas)

Objetivos realistas para scalping cripto (neto de costes):

- **Profit Factor (global):** >= 1.15 (objetivo stretch 1.25)
- **Sharpe neto:** >= 1.2 (objetivo stretch 1.8)
- **Expectancy por trade:** > 0, ideal > 0.05R
- **Max Drawdown:** < 12% (paper/live inicial)
- **Hit ratio:** secundario; priorizar expectancy y PF
- **Regime coverage:** >= 70% de tiempo en estados con decisión explícita (trade o no-trade)
- **No-trade discipline:** porcentaje de periodos bloqueados cuando no hay edge (debe ser significativo, no “siempre dentro”).

---

## 11) Riesgos técnicos y estadísticos críticos

## 11.1 Técnicos
- latencia y slippage no modelados destruyen edge de scalping;
- errores de sincronización multi-timeframe;
- dependencia excesiva en features frágiles (ADX/ATR mal calibrados por exchange).

Mitigación:
- simulador de ejecución realista;
- tests de consistencia temporal;
- feature monitoring + alertas de drift.

## 11.2 Estadísticos
- **overfitting por régimen** (demasiadas reglas para poca muestra);
- **selection bias** al escoger estrategias por performance reciente;
- **non-stationarity** estructural cripto.

Mitigación:
- nested walk-forward + purged CV;
- promoción por robustez (percentiles), no por media;
- recalibración programada con límites de cambio paramétrico.

---

## 12) Blueprint de paquetes Python 3.12+

```text
app/
  domain/
    entities/
    value_objects/
    services/
  application/
    use_cases/
    orchestrators/
  infrastructure/
    data/
    brokers/
    repositories/
    models/
  strategies/
    breakout_trend/
    mean_reversion_volband/
    high_vol_hybrid/
  regime_detector/
    features.py
    detector.py
    smoothing.py
  risk_manager/
    sizing.py
    limits.py
    kill_switch.py
  exit_engine/
    dynamic_exits.py
  monitoring/
    metrics.py
    drift.py
    alerts.py
```

Notas de implementación:
- interfaces vía `Protocol`/ABC para inversión de dependencias;
- dataclasses inmutables en entidades clave;
- configuración tipada (pydantic-settings o similar);
- logging estructurado + trazabilidad por `trade_id` y `regime_snapshot_id`.

---

## 13) Criterio de éxito práctico

El rediseño será exitoso si, durante validación OOS extendida:
1. el sistema **reduce drásticamente operaciones sin edge**;
2. el performance mejora por régimen (no solo agregado);
3. la distribución de Monte Carlo muestra percentiles aceptables (P10/P25 positivos o levemente positivos, no colapso sistemático);
4. la degradación en vivo activa controles sin intervención manual.

En términos de arquitectura: éxito = módulos desacoplados, testeables y reemplazables sin romper el motor.
