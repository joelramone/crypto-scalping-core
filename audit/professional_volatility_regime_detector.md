# Professional Volatility Regime Detector (Institutional Design)

## 1) Métricas cuantitativas (núcleo)

El detector usa 6 familias de métricas, cada una diseñada para capturar **expansión estructural** y filtrar ruido:

1. **ATR percentile rolling**
   - ATR(14) sobre true range.
   - Percentil rolling sobre ventana larga (ej. 120 barras).
   - Señal robusta cuando `atr_percentile >= 0.70-0.80`.

2. **Realized volatility**
   - RV como `sqrt(sum(r_t^2)/N)` con retornos de close-to-close.
   - Percentil rolling igual que ATR para comparación cross-regime.

3. **Range expansion ratio**
   - `mean(range_t, window corto) / mean(range_t, ventana previa equivalente)`.
   - Captura transición de compresión a expansión, no solo spike puntual.

4. **Ratio current range / rolling mean range**
   - `range_actual / mean(range, window corto)`.
   - Aporta sensibilidad a ruptura del micro-rango vigente.

5. **Volatility clustering**
   - Variación de ATR (`|ATR_t - ATR_{t-1}|`) suavizada.
   - Ratio vs su media histórica para detectar persistencia (cluster), no evento único.

6. **Volume expansion confirmation**
   - `mean(volume, window corto) / mean(volume, ventana previa)`.
   - Condición crítica para descartar fake breakout con baja participación.

## 2) Lógica combinada (score compuesto)

En vez de un trigger binario único:

- Cada métrica se normaliza a `[0,1]`.
- Se construye score ponderado:

`score = 0.25*ATR + 0.20*RV + 0.20*RANGE + 0.15*CLUSTER + 0.20*VOLUME`

- Se exige **breadth**: mínimo 3 factores activos (`>= 0.60`) para evitar dependencia de una sola variable.
- Se exige **persistencia temporal**: 2 barras consecutivas sobre threshold.

Regla recomendada:

- `HIGH_VOL` si `score >= 0.62` y `active_features >= 3` por `>=2` barras.
- En otro caso: `NORMAL_VOLATILITY`.

## 3) Arquitectura modular en Python

- `VolatilityMetrics`: cálculo de features robustas por barra.
- `RegimeScoreCalculator`: normalización y score compuesto.
- `RegimeClassifier`: umbral + breadth + confirmación temporal.

Implementación de referencia en:
`app/agents/volatility_regime_detector.py`.

## 4) Pseudocódigo técnico detallado

```text
INPUTS:
  close[], high[], low[], volume[]
  params: atr_window, lookback, rv_window, range_window,
          clustering_window, volume_window,
          score_threshold, min_active_features, confirmation_bars

STATE:
  positive_streak = 0

FOR each new bar t:
  if not enough history:
      return None

  # Step A: Metrics
  tr_t = max(high_t-low_t, abs(high_t-close_{t-1}), abs(low_t-close_{t-1}))
  ATR_t = rolling_mean(TR, atr_window)
  atr_pct_t = percentile_rank(ATR_{t-lookback:t}, ATR_t)

  rv_t = sqrt(mean(returns_{t-rv_window:t}^2))
  rv_pct_t = percentile_rank(RV_{t-lookback:t}, rv_t)

  range_t = high_t - low_t
  range_ratio_t = range_t / mean(range_{t-range_window:t})
  range_expansion_t = mean(range_{t-range_window:t}) /
                      mean(range_{t-2*range_window:t-range_window})

  cluster_t = rolling_mean(abs(ATR_i-ATR_{i-1}), clustering_window)
  cluster_ratio_t = cluster_t / mean(cluster_{t-lookback:t})

  vol_exp_t = mean(volume_{t-volume_window:t}) /
              mean(volume_{t-2*volume_window:t-volume_window})

  # Step B: Normalization
  f_atr = clip(atr_pct_t, 0, 1)
  f_rv = clip(rv_pct_t, 0, 1)
  f_range = clip(0.5*range_expansion_t + 0.5*range_ratio_t - 1.0, 0, 1)
  f_cluster = clip(cluster_ratio_t - 1.0, 0, 1)
  f_volume = clip(vol_exp_t - 1.0, 0, 1)

  # Step C: Composite score
  score_t = 0.25*f_atr + 0.20*f_rv + 0.20*f_range +
            0.15*f_cluster + 0.20*f_volume

  active_features = count(features >= 0.60)

  # Step D: Classification with persistence
  if score_t >= score_threshold and active_features >= min_active_features:
      positive_streak += 1
  else:
      positive_streak = 0

  if positive_streak >= confirmation_bars:
      regime_t = HIGH_VOLATILITY
  else:
      regime_t = NORMAL_VOLATILITY

  return regime_t, score_t, metrics_snapshot
```

## 5) Validación cuantitativa seria

### A. Backtest del detector aislado (sin estrategia)

1. Construir dataset de barras con etiquetas derivadas de outcome forward:
   - Define `breakout_success` si en las próximas `k` barras el retorno ajustado por costo supera umbral.
2. Ejecutar detector barra a barra (sin look-ahead).
3. Guardar predicción `HIGH_VOL` / `NORMAL` + score continuo.
4. Métricas de clasificación:
   - Precision, Recall, F1 para clase `HIGH_VOL` contra `breakout_success`.
   - PR-AUC (preferible a ROC en clases desbalanceadas).
   - Calibration curve por buckets de score.

### B. Comparativa contra detector anterior

Comparar **A/B** old vs new:

- Precision@HIGH_VOL
- Recall@HIGH_VOL
- Lift de breakout exitoso condicionado a HIGH_VOL:
  - `P(success | HIGH_VOL_new)` vs `P(success | HIGH_VOL_old)`
- Cambio de distribución de duración de regímenes (evitar flicker).
- PSI/KL divergence de features out-of-sample para robustez de régimen.

### C. Validación de impacto en trading

Correr estrategia breakout idéntica, cambiando solo el detector:

- Delta Winrate, Profit Factor, Expectancy en subset HIGH_VOL.
- Sharpe/Sortino y max DD en portfolio completo.
- Test de significancia por bootstrap de trades y SPA/Reality Check si aplica.

## 6) Riesgos y mitigaciones

1. **Volatility mean reversion trap**
   - Riesgo: entrar al final de un spike revertido.
   - Mitigación: confirmar clustering + volumen, no solo ATR alto.

2. **Late regime detection**
   - Riesgo: perder primera parte del breakout por confirmaciones.
   - Mitigación: usar score continuo para sizing gradual (pilot size antes de confirmación completa).

3. **Structural regime shift**
   - Riesgo: pesos/threshold dejan de representar mercado actual.
   - Mitigación: recalibración walk-forward con ventanas estables y regularización de parámetros.

4. **Sobreoptimización**
   - Riesgo: overfit de umbrales por activo/periodo.
   - Mitigación: parámetros coarse, validación por múltiples mercados y nested walk-forward.

## Recomendación operativa

- Empezar con pesos conservadores definidos arriba.
- Congelar parámetros en un periodo OOS amplio.
- Monitorear trimestralmente precision/recall y drift de features.
- Si `P(success | HIGH_VOL)` cae de forma sostenida, gatillar recalibración controlada.
