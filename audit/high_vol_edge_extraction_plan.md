# HIGH_VOL BreakoutTrend — Edge Extraction Blueprint

## 1) Diagnóstico matemático del problema actual

### 1.1 Métricas observadas
- Profit Factor (PF): ~0.94
- Expectancy por trade: ~-0.038R
- Win rate: ~42%
- Runs rentables: ~46%
- Sharpe: ~-0.06
- Frecuencia: ~4 trades por simulación

### 1.2 Descomposición de expectancy
Modelando en unidades de riesgo (`R = distancia SL`):

\[
E[R] = p_w \cdot \bar{W} - (1-p_w) \cdot \bar{L}
\]

Con `p_w = 0.42`, `E[R] = -0.038` y asumiendo `\bar{L} \approx 1.0R`:

\[
0.42\bar{W} - 0.58(1.0) = -0.038 \Rightarrow \bar{W} \approx 1.29R
\]

Break-even teórico para `p_w = 0.42`:

\[
\bar{W}_{BE} = \frac{1-p_w}{p_w}\bar{L} = \frac{0.58}{0.42} \approx 1.38R
\]

**Conclusión:** el sistema pierde principalmente por **asimetría insuficiente en la salida** (ganancia media < umbral de break-even), no necesariamente por timing de entrada.

### 1.3 Problema estadístico de baja frecuencia
Con ~4 trades/simulación, para 900 simulaciones solo hay ~3,600 trades efectivos. Con esta muestra:
- alta varianza de estimadores (PF/Sharpe inestables),
- sensibilidad extrema al path dependency,
- intervalos de confianza anchos para expectancy.

Por eso el objetivo mínimo de validación debe superar **10,000 trades**, y el target operativo de diseño debe elevar la frecuencia a **>=20 trades/simulación** en HIGH_VOL.

---

## 2) Diseño de exit asimétrico (sin añadir nuevas estrategias)

## 2.1 Principios
1. Mantener la lógica de entrada de `BreakoutTrendStrategy`.
2. Redefinir completamente la lógica de salida en términos de `R`.
3. Forzar distribución de payoff con cola positiva:
   - pérdidas acotadas y dinámicas,
   - ganadores con espacio a 1.5R–2R+
   - trailing que preserve convexidad en tendencias.

## 2.2 Stop Loss dinámico (ATR)
Definir en la entrada:
- `ATR_n` (ej. 14 periodos)
- `SL_dist = k_sl * ATR_n`
- `R = SL_dist`

Parámetros iniciales sugeridos (HIGH_VOL):
- `k_sl in [1.2, 2.0]` (barrido WF)
- hard floor anti-microstop: `SL_dist >= min_tick * m`

Racional:
- En alta volatilidad, stop fijo distorsiona riesgo efectivo.
- ATR permite normalizar riesgo entre episodios volátiles heterogéneos.

## 2.3 Take Profit base asimétrico
En vez de TP discrecional, establecer:
- `TP1 = 1.5R`
- `TP2 = 2.0R` (parcial o salida completa según configuración)

Dos modos permitidos:
1. **Full exit en 1.8R–2.0R** (simple, robusto).
2. **Escalonado:** 50% en 1.5R + 50% con trailing.

Objetivo: mover `\bar{W}` por encima de 1.40R con `p_w` estable (~40–45%).

## 2.4 Trailing stop adaptativo
Activación solo después de confirmar desplazamiento favorable:
- trigger: `unrealized_pnl >= 1.0R`
- una vez activado:
  - `trail_dist = k_trail * ATR_current`
  - `stop = max(stop_prev, highest_high_since_entry - trail_dist)` (long)

Parámetros sugeridos:
- `k_trail in [1.0, 2.5]`
- opción de tightening progresivo:
  - si PnL > 1.5R, reducir `k_trail` 10–20%
  - si PnL > 2.5R, reducir adicionalmente 10–20%

Esto protege ganancias sin truncar demasiado pronto los runners.

## 2.5 Filtro de percentil de volatilidad (>70)
Entrada válida solo si:

\[
vol\_pct = PercentileRank(ATR_n, window=W) > 70
\]

Sugerencias:
- `W in [100, 300]`
- evaluar umbrales 65/70/75 en walk-forward.

Efecto esperado:
- reduce noise regimes con falsos breakouts,
- aumenta densidad de desplazamientos extendidos (mejor para payout asimétrico).

---

## 3) Arquitectura modular propuesta

```text
app/
  strategies/
    breakout_trend.py                 # entrada existente + hooks de salida
  risk/
    exit_models.py                    # ATR stop, TP ladder, trailing
    volatility_filter.py              # percentile filter HIGH_VOL
    position_risk.py                  # R normalization, sizing utilities
  optimization/
    wf_edge_extraction.py             # walk-forward tuning de exits
    mc_robustness.py                  # shuffle + bootstrap MC
  analysis/
    edge_extraction_report.py         # métricas, CI, estabilidad
```

### Interfaces clave
- `ExitModel.compute_initial_stop(entry_ctx) -> stop_price`
- `ExitModel.compute_targets(entry_ctx) -> list[target_levels]`
- `ExitModel.update_stop(position_ctx, market_ctx) -> new_stop`
- `VolatilityGate.allow_trade(market_ctx) -> bool`

Separar entrada/exit evita contaminar la hipótesis original y acelera iteraciones controladas.

---

## 4) Pseudocódigo técnico

```python
class HighVolEdgeExtractor:
    def __init__(self, atr_period=14, sl_k=1.5, tp1_r=1.5, tp2_r=2.0,
                 trail_k=1.8, vol_window=200, vol_pct_threshold=70):
        ...

    def allow_entry(self, bar, features):
        vol_pct = percentile_rank(features.atr, history_atr[-self.vol_window:])
        return vol_pct > self.vol_pct_threshold

    def on_entry(self, entry_price, side, atr):
        R = self.sl_k * atr
        if side == "long":
            stop = entry_price - R
            tp1 = entry_price + self.tp1_r * R
            tp2 = entry_price + self.tp2_r * R
        else:
            stop = entry_price + R
            tp1 = entry_price - self.tp1_r * R
            tp2 = entry_price - self.tp2_r * R

        return PositionState(R=R, stop=stop, tp1=tp1, tp2=tp2,
                             trail_active=False, max_favorable_R=0.0)

    def on_bar(self, pos, bar, atr):
        pnl_r = compute_unrealized_R(pos, bar.close)
        pos.max_favorable_R = max(pos.max_favorable_R, pnl_r)

        # activar trailing después de 1R
        if pnl_r >= 1.0 and not pos.trail_active:
            pos.trail_active = True

        # salida parcial en TP1 (opcional)
        if not pos.tp1_hit and hit_level(bar, pos.tp1, pos.side):
            execute_partial_exit(pos, fraction=0.5)
            pos.tp1_hit = True
            pos.stop = max(pos.stop, pos.entry_price)  # mover a BE opcional

        # trailing adaptativo
        if pos.trail_active:
            adaptive_trail_k = dynamic_trail_k(base=self.trail_k,
                                               max_favorable_R=pos.max_favorable_R)
            trail_dist = adaptive_trail_k * atr
            pos.stop = update_trailing_stop(pos, bar, trail_dist)

        # TP2 o SL
        if hit_level(bar, pos.tp2, pos.side):
            close_position(pos, reason="tp2")
        elif hit_level(bar, pos.stop, pos.side):
            close_position(pos, reason="stop")
```

---

## 5) Plan de implementación por fases

## Fase 0 — Baseline y dataset de validación
- Congelar baseline actual de HIGH_VOL + BreakoutTrend.
- Construir set de evaluación con **>=10,000 trades** agregados.
- Reportar métricas con IC por bootstrap.

## Fase 1 — Exit mínimo viable asimétrico
- Introducir ATR stop + TP fijo 1.8R (sin trailing).
- Objetivo: verificar mejora directa de expectancy.
- Mantener entradas idénticas.

## Fase 2 — Trailing adaptativo
- Activación >1R.
- Barrido de `k_trail` y lógica de tightening.
- Analizar trade-off entre `\bar{W}` y win rate.

## Fase 3 — Volatility percentile gate
- Activar filtro >70p y comparar con 65/75.
- Medir impacto en frecuencia (target >=20 trades/sim).

## Fase 4 — Robustez estadística
- Monte Carlo con:
  - **shuffle** de secuencias de trades,
  - **bootstrap** con reemplazo.
- Mínimo 5,000–10,000 resamples por configuración.
- Reportar distribución de PF, expectancy, maxDD, Sharpe.

## Fase 5 — Walk-forward
- Esquema rolling/anchored:
  - Train: optimiza solo parámetros de salida,
  - Test: validación out-of-sample estricta.
- Consolidar parámetros por estabilidad, no por máximo retorno puntual.

---

## 6) Métricas objetivo (criterio de aceptación)

Mínimos para considerar edge extraído en HIGH_VOL:
- `PF >= 1.15`
- `Expectancy >= +0.10R`
- `Win rate 38%–48%` (no es necesario elevarlo si sube payoff)
- `% runs rentables >= 60%`
- `Sharpe > 0.30`
- `Avg trades/simulation >= 20`
- `MaxDD` controlado vs baseline (no deterioro >15% relativo)

Robustez:
- En MC shuffle/bootstrap, percentil 25 de expectancy > 0.
- En walk-forward, >=70% de ventanas OOS con expectancy positiva.

---

## 7) Riesgos de overfitting y mitigaciones

Riesgos principales:
1. Sobreajuste de `k_sl`, `k_trail`, `tp_r` a un único tramo de volatilidad.
2. Data snooping al probar demasiadas combinaciones sin corrección.
3. Optimizar win rate en lugar de distribución de payoff.
4. Inestabilidad por baja N efectiva de trades.

Mitigaciones:
- Limitar dimensión del grid (parsimonia).
- Selección por **estabilidad inter-ventana** (WF), no por mejor punto.
- Usar intervalos de confianza y percentiles, no solo medias.
- Fijar criterios de aceptación ex-ante.
- Congelar configuración final y correr validación ciega final.

---

## 8) Decisión operativa recomendada

No añadir nuevas estrategias. Enfocar el roadmap en **Exit Engineering** de BreakoutTrend en HIGH_VOL:
1. ATR stop normalizado por régimen.
2. TP asimétrico mínimo 1.5R–2.0R.
3. Trailing adaptativo activado por avance >1R.
4. Filtro de percentil de volatilidad >70.
5. Validación robusta con 10,000+ trades, MC y walk-forward.

Este enfoque ataca directamente la causa raíz (asimetría de salida), preservando la hipótesis de entrada y maximizando probabilidad de convertir un PF sub-1 en edge explotable.
