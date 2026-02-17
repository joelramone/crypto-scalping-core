# Market Regime Adaptive System v2 (Edge-Focused)

## 1) Diagnóstico cuantitativo (base line)

### Hallazgos de desempeño
- **HIGH_VOL**: único régimen con señal explotable (BreakoutTrend PF ~1.0, expectancy ~0, 46% runs rentables).
- **SIDEWAYS**: destrucción sistemática de capital (PF ~0.06, 0% runs rentables, expectancy fuertemente negativa).
- **TRENDING**: edge insuficiente con configuración actual (PF ~0.35, expectancy negativa).
- **Baja frecuencia**: ~6 trades/simulación, insuficiente para converger estadísticamente y para capturar micro-regímenes intradía.

### Conclusión operativa
El sistema v2 debe ser un **single-regime engine**: operar solo en **HIGH_VOL** y bloquear sideway/trending hasta nueva evidencia estadística.

---

## 2) Diseño de Market Regime Detector (solo HIGH_VOL operable)

## 2.1 Variables base por timeframe (ej. 1m con contexto 5m)
1. **ATR Percentile (rolling)**
   - `atr_t = ATR(n_atr)`
   - `atr_pct_t = percentile_rank(atr_t, window=W_atr)`
2. **Realized Volatility (RV)**
   - `rv_t = sqrt(sum(log_ret^2, k)) * annualization_factor`
   - `rv_pct_t = percentile_rank(rv_t, window=W_rv)`
3. **Range Expansion Ratio (RER)**
   - `rer_t = (high_t - low_t) / EMA(high-low, n_range)`
4. **Volume Impulse (VI)** (si volumen confiable)
   - `vi_t = volume_t / EMA(volume, n_vol)`
5. **Directional impulse gate (DIG)**
   - Magnitud neta de desplazamiento: `|close_t - open_t| / ATR`

## 2.2 Score de régimen
Usar score continuo y umbral con histéresis para evitar chattering:

```text
regime_score =
  w1 * z(atr_pct_t) +
  w2 * z(rv_pct_t) +
  w3 * z(rer_t) +
  w4 * z(vi_t)  +
  w5 * z(DIG)
```

**Entrada HIGH_VOL**:
- `regime_score >= enter_th` durante `m` barras.
- `atr_pct_t >= p_atr_min` y `rv_pct_t >= p_rv_min` (doble confirmación).

**Salida HIGH_VOL**:
- `regime_score <= exit_th` durante `q` barras (`exit_th < enter_th`, histéresis).

### Parámetros iniciales robustos (no optimizar fino al inicio)
- `W_atr, W_rv`: 250–500 barras.
- `enter_th`: percentil 70–80 histórico del `regime_score`.
- `exit_th`: 55–65.
- `m=2`, `q=3`.

---

## 3) Strategy Layer: BreakoutTrendStrategy v2

## 3.1 Señal de entrada
Operar únicamente si `RegimeDetector.state == HIGH_VOL`.

Long setup (simétrico para short):
1. **Breakout estructural**: `close_t > rolling_high(N_break)`.
2. **Confirmación de impulso**: vela de ruptura con cuerpo mínimo (`body/ATR >= min_body_atr`).
3. **No-entry zones**: evitar entradas si distancia a nivel invalidante implica RR<`rr_min`.
4. **Cooldown**: evitar reentrada inmediata en misma dirección durante `cooldown_bars` tras stop.

## 3.2 Aumento de frecuencia sin curve fitting
- Multi-trigger dentro del mismo régimen HIGH_VOL:
  1) breakout de rango corto (`N_break_fast`)
  2) breakout de rango medio (`N_break_slow`)
  3) pullback-breakout (retesteo + continuación)
- Unificar todos bajo el mismo motor de riesgo y con **feature set mínimo**.
- Limitar grados de libertad: máximo 1–2 variantes de entrada por lado.

---

## 4) RiskManager y rediseño de exits

## 4.1 Stop loss dinámico ATR
Al abrir:
- `initial_stop = entry - k_sl * ATR` (long).
- `k_sl` dependiente de régimen interno:
  - HIGH_VOL bajo: `k_sl ~ 1.2`
  - HIGH_VOL extremo: `k_sl ~ 1.6`

## 4.2 RR mínimo hard constraint
- `take_profit_candidate = entry + rr_min * (entry - stop)`
- Si estructura de mercado no permite `rr_min in [1.5, 2.0]`, **no trade**.

## 4.3 Trailing adaptativo (dos fases)
1. **Phase A (protección)**:
   - al alcanzar `+1R`, mover stop a `breakeven + fees_buffer`.
2. **Phase B (captura de cola)**:
   - trailing por `ATR_trail = k_trail * ATR` sobre swing low/high o `chandelier`.
   - `k_trail` decrece suavemente cuando sube `regime_score` (dejar respirar menos en vol extrema reversiva).

## 4.4 Position sizing
- Riesgo fijo por trade en bps de equity (`risk_bps`), p.ej. 10–25 bps.
- Cap por exposición simultánea (si hay multi-trigger).
- Daily max loss y kill-switch siguen vigentes.

---

## 5) Arquitectura modular en Python

```text
app/
  regime/
    detector.py         # RegimeDetector
    features.py         # ATR, RV, RER, percentiles
    state_machine.py    # hysteresis enter/exit
  strategies/
    breakout_trend_v2.py
    signals.py
  risk/
    risk_manager_v2.py  # sizing + exits policy
    exit_engine.py      # stop/trailing/tp transitions
  execution/
    execution_engine.py # order routing + slippage model
  research/
    backtest_runner.py
    walk_forward.py
    monte_carlo.py
  analytics/
    metrics.py
    diagnostics.py
```

### Interfaces propuestas

```python
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class MarketFeatures:
    atr: float
    atr_pct: float
    rv: float
    rv_pct: float
    rer: float
    vi: float
    dig: float

class RegimeDetector:
    def update(self, feat: MarketFeatures) -> str:
        """Return: HIGH_VOL | NON_TRADABLE"""

@dataclass
class TradeSignal:
    side: str              # long/short
    entry: float
    stop: float
    rr_min: float
    tag: str               # trigger type

class Strategy:
    def generate_signal(self, bars, regime_state: str) -> Optional[TradeSignal]:
        ...

class RiskManager:
    def approve_and_size(self, signal: TradeSignal, equity: float) -> Optional[Dict]:
        ...

class ExecutionEngine:
    def execute(self, order: Dict) -> Dict:
        ...
```

---

## 6) Pseudocódigo estructural (end-to-end)

```text
for each new bar:
    feat = build_features(bar_history)
    regime_state = regime_detector.update(feat)

    if regime_state != HIGH_VOL:
        manage_open_positions_only()
        continue

    signal = breakout_strategy.generate_signal(bar_history, regime_state)
    if signal is None:
        manage_open_positions_only()
        continue

    risk_order = risk_manager.approve_and_size(signal, equity)
    if risk_order is None:
        continue

    fill = execution_engine.execute(risk_order)
    portfolio.register(fill)

    for position in portfolio.open_positions:
        exit_event = exit_engine.update(position, feat, bar)
        if exit_event:
            execution_engine.execute(exit_event)

    analytics.log_state(feat, regime_state, signal, risk_order, portfolio)
```

---

## 7) Plan de validación de edge (institucional-lite)

## 7.1 Meta de muestra
- Objetivo mínimo: **10,000 trades netos** del sistema HIGH_VOL-only.
- Si histórico no alcanza: ampliar universo (más símbolos líquidos) y/o ampliar horizonte temporal.

## 7.2 Pipeline de evaluación
1. **Backtest con costos realistas**:
   - maker/taker fees, slippage dependiente de vol y spread modelado.
2. **Walk-forward validation**:
   - bloques rolling (train/validate/test) temporales, sin leakage.
   - recalibración solo en ventanas permitidas.
3. **Monte Carlo robusto**:
   - bootstrap por bloques (para preservar autocorrelación).
   - permutación de secuencia de trades.
   - shock tests de costos (`+25%`, `+50%` slippage).

## 7.3 Métricas objetivo (go/no-go)
- `PF > 1.20` (neto de costos).
- `Sharpe > 0.50`.
- `Expectancy > 0` por trade y por día.
- `% de ventanas walk-forward rentables > 60%`.
- `MaxDD` compatible con mandato de riesgo interno.

## 7.4 Criterios de robustez adicionales
- Edge persistente en submuestras por símbolo/hora del día.
- No dependencia crítica de 1 parámetro (sensibilidad local suave).
- Degradación controlada bajo costos más altos.

---

## 8) Riesgos estadísticos y mitigación

1. **Overfitting por hiperparametría**
   - Mitigar con pocos parámetros libres y rangos amplios.
   - Penalizar complejidad en selección final.
2. **Selection bias de régimen**
   - Definir detector ex-ante y congelar reglas antes de test final.
3. **Data snooping / leakage temporal**
   - Features con ventanas solo backward.
   - Split temporal estricto en walk-forward.
4. **Non-stationarity en microestructura crypto**
   - Recalibración periódica limitada (mensual/trimestral).
   - Alertas de drift en distribución de features y PnL.
5. **Riesgo de ejecución real vs backtest**
   - Modelar latencia/slippage variable por régimen.
   - Shadow-live antes de capital real.

---

## 9) Plan de implementación por fases

### Fase 0 — Instrumentación (1 semana)
- Logging granular de features, régimen, señales, fills y motivos de rechazo.
- Dataset limpio para investigación reproducible.

### Fase 1 — RegimeDetector + gating (1 semana)
- Implementar detector HIGH_VOL con histéresis.
- Desactivar trading fuera de HIGH_VOL.

### Fase 2 — Breakout v2 + exits ATR/trailing (1–2 semanas)
- Entradas multi-trigger controladas.
- Stop ATR, BE en 1R, trailing adaptativo, filtro RR mínimo.

### Fase 3 — Validation stack (2 semanas)
- Backtest costos realistas + walk-forward + Monte Carlo.
- Meta intermedia: 3,000–5,000 trades.

### Fase 4 — Escalado estadístico (continuo)
- Extender universo/periodo hasta 10,000 trades.
- Congelar versión candidata (release candidate).

### Fase 5 — Paper trading y promoción a live (2–4 semanas)
- Shadow mode, luego paper con reglas idénticas.
- Go-live solo si métricas permanecen sobre umbrales objetivo.

---

## 10) Reglas de gobierno cuantitativo (recomendadas)
- **Research lock**: no cambiar reglas durante ventana de evaluación.
- **Change log estricto** de parámetros y versiones.
- **Promotion committee** (aunque sea de 1–2 personas): separar investigación de aprobación.
- **Kill-switch estadístico**: apagar estrategia si rolling PF/expectancy cae bajo umbral crítico.

Este diseño prioriza supervivencia estadística y escalabilidad operacional: primero filtrar régimen con probabilidad de edge, luego extraer convexidad de rupturas mediante exits adaptativos y validación robusta.
