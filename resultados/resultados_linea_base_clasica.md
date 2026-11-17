# Línea base clásica de detección en simulación frente a la red s1

¿Cuánto aporta la red frente a un detector clásico? El detector es el de manchas de `tracking/reflejos.py`:
1. diferencia de gaussianas (σ 1.5 y 4 px);
2. máximos locales de radio r por encima de k veces el ruido (MAD) de la DoG;
3. ajuste gaussiano 2D en ±5 px, con σ entre 0.6 y 4 px.

Se aplica a las partículas reales y se compara con la red `hypatia_celula_v2_b45_cerca_s1` (`postproc.umbral` 0.4), sin reentrenar. Los datos son los de `resultados_celula_v2.md` y `resultados_tracking_3d.md`:
- **test v2:** `localizacion/datos/test_celula_v2_b45_cerca`, 2000 imágenes de 96 px con célula, espejo y reflejo. Para la red se usa `predicciones.csv` de su evaluación local.
- **videos de tracking:** los 200 de `tracking_base.py` (50 cuadros, 4 D × 50 muestras base, 1–12 emisores). Para la red, `detecciones_u02.csv` con conf ≥ 0.4, que es su línea base.

```bash
PY=~/miniforge3/envs/tesis/bin/python
$PY figuras/linea_base_clasica.py detectar     # detector con k = 3 y r = 1, 2, 3 en los dos conjuntos (~6 min, 14 procesos)
$PY figuras/linea_base_clasica.py analisis     # ajuste, comparación emisor a emisor y tracking (~5 min)
```

**Mitades.**
- El detector clásico tiene dos parámetros, el umbral k y el radio r. Se eligen en la **mitad de selección**: imágenes 0–999 del test y muestras base 0–24 de los videos. Las curvas para k > 3 salen filtrando por SNR la corrida con k = 3, porque el resto del detector no depende de k.
- Todo lo demás se mide en la **mitad de prueba**: imágenes 1000–1999 y muestras base 25–49, es decir 100 videos con los 4 D.
- El σ lateral que usa el costo del enlace también sale de la mitad de selección.

**Criterio de emparejamiento.** El detector clásico no da z, así que los dos métodos se emparejan por cuadro con el mismo criterio lateral:
- asignación húngara con tolerancia de 150 nm;
- GT evaluable = en el FOV y dentro del rango z de la red (`evaluable`); las detecciones emparejadas con un GT no evaluable no cuentan;
- un **FP sobre un reflejo** es una detección no emparejada a ≤ 150 nm del reflejo GT de alguna partícula;
- también se da la red con su criterio 3D (cilindro de 150 / 300 nm).

**IC 95 %:** bootstrap de imágenes (test) o de videos (tracking).

**Salidas:**
- `figuras/linea_base_clasica/`: CSV agregados y 4 PNG a 300 dpi;
- manchas crudas y tablas por video en `localizacion/datos/linea_base_clasica/`, sin versionar.

## Resumen

| | resultado |
|---|---|
| ajuste del detector | **r = 1, k = 5.5** (Jaccard 0.328 en la mitad de selección). La curva es plana: k entre 4.5 y 6.5 y r = 2 quedan a ≤ 0.003. Aun con k = 3, el recall no pasa de 0.53 en el test y de 0.61 en los videos |
| detección, test v2 (mitad de prueba) | **Red: recall 0.721 [0.707, 0.734], precisión 0.957, RMSE lateral 44.0 nm. Clásico: recall 0.496 [0.481, 0.510], precisión 0.448, RMSE lateral 56.6 nm** |
| detección, videos | **Red: 0.809 / 0.959 / 38.1 nm. Clásico: 0.572 / 0.435 / 51.2 nm** |
| reflejos como partículas | **El clásico detecta 2.95 reflejos por cuadro en el test y 3.17 en los videos**, que son el 75 % y el 68 % de sus FP. Sin contarlos, su precisión sería 0.76 y 0.71. La red: 0.02 reflejos por cuadro, el 10 % de sus 0.21 FP por cuadro |
| a igual precisión | Barriendo k y la conf de la red, la curva del clásico queda siempre por debajo. Aun sin contar los reflejos, el clásico no llega a precisión 0.90 (máximo 0.87–0.90, con recall 0.35–0.40). La red tiene recall 0.84 con precisión 0.90 |
| por rangos | **La red gana en todos los bins** de fotones, fondo, distancia al vecino, autofluorescencia y z, en recall y en RMSE lateral, en los dos conjuntos |
| tracking (LAP de `tracking_3d`, métricas en x, y) | JSC: **red con enlace 3D 0.550 [0.490, 0.611], red 2D 0.491, clásico 0.211 [0.187, 0.238]**; clásico sin reflejos (oráculo) 0.292. Error relativo mediano del D lateral: **0.128 (red 3D), 0.148 (red 2D) y 0.291 (clásico)**. Los cambios de identidad por pista GT son parecidos: 2.4, 4.1 y 4.3 |
| dónde gana cada uno | **El clásico no gana en ningún régimen.** La red aporta más con señal media y vecinos cercanos: +0.40 a +0.43 de recall con 5000–20 000 fotones y vecino a < 500 nm. **Donde la red no aporta** es con mucha señal y partículas aisladas: con ≥ 30 000 fotones y vecino a > 2 µm, +0.01 [0.00, 0.03] |

## 1. Ajuste del detector clásico

`linea_base_clasica_ajuste.csv` y `fig_linea_base_clasica_ajuste.png`.

**Objetivo:** el Jaccard lateral TP / (TP + FP + FN), contando como FP los reflejos detectados, sumado sobre las dos mitades de selección.

| | r | k | Jaccard | recall | precisión | precisión sin reflejos |
|---|---|---|---|---|---|---|
| **elegido (los dos conjuntos)** | **1** | **5.5** | **0.328** | 0.534 | 0.459 | 0.768 |
| mejor solo en el test | 1 | 5.0 | 0.301 | 0.493 | 0.437 | — |
| mejor solo en los videos | 1 | 5.5 | 0.333 | 0.545 | 0.462 | — |

- **El óptimo es plano.** Entre k = 4.5 y 6 el Jaccard varía < 0.003, y r = 2 da casi lo mismo. El r = 3 pierde ~0.01 porque fusiona vecinos.
- **El recall tiene un techo bajo.** Ni con k = 3 pasa de 0.53 (test) o 0.61 (videos). Los emisores defocalizados, que tienen anillos y no son gaussianos, y los vecinos a < 500 nm no dan un máximo que el ajuste acepte (ver z y vecino en la sección 2).
- **No se ajustó nada más** que k y r. Los σ de la DoG, la ventana y los límites de σ son los de `tracking/reflejos.py`. Un detector clásico pensado para la PSF esférica podría correr ese techo; aquí no se exploró.

## 2. Detección emisor a emisor (mitad de prueba)

`linea_base_clasica_global.csv`, `_rangos.csv`, `_precision_recall.csv` y `fig_linea_base_clasica_deteccion.png`.

| conjunto | método | recall | precisión | precisión sin reflejos | Jaccard | FP / cuadro | reflejos / cuadro | RMSE lateral |
|---|---|---|---|---|---|---|---|---|
| test v2 (1000 imágenes, 6479 emisores) | clásico | 0.496 [0.481, 0.510] | 0.448 [0.440, 0.457] | 0.762 | 0.308 | 3.96 | **2.95** | 56.6 [55.1, 58.0] nm |
| | red (lateral) | **0.721** [0.707, 0.734] | **0.957** [0.951, 0.963] | 0.961 | **0.698** | 0.21 | 0.02 | **44.0** [42.6, 45.3] nm |
| | red (cilindro 3D) | 0.672 | 0.893 | 0.897 | 0.622 | 0.52 | 0.02 | 42.3 nm |
| videos (100, 31 391 emisores-cuadro) | clásico | 0.572 [0.548, 0.599] | 0.435 [0.416, 0.452] | 0.705 | 0.328 | 4.68 | **3.17** | 51.2 [47.9, 54.3] nm |
| | red (lateral) | **0.809** [0.784, 0.836] | **0.959** [0.950, 0.967] | 0.963 | **0.782** | 0.22 | 0.02 | **38.1** [34.7, 41.0] nm |
| | red (cilindro 3D) | 0.763 | 0.904 | 0.908 | 0.706 | 0.51 | 0.02 | 36.5 nm |

- **Los reflejos explican tres cuartos de los FP del clásico.** El detector no tiene forma de saber cuál de las dos imágenes es la partícula. La red, entrenada para ignorarlos, pone 0.02 por cuadro.
- **Aun sin contar los reflejos, el clásico tiene un FP por cada 2.4–3.2 TP.** Los restantes son anillos de PSF defocalizadas, manchas de la autofluorescencia y dobles detecciones.
- **A igual precisión** (`fig_linea_base_clasica_ajuste`, derecha; barrido de k y de la conf de la red):
  - la red tiene recall 0.84 con precisión 0.90 (conf 0.2, videos);
  - el clásico, sin contar reflejos, solo llega a precisión 0.86–0.90 con recall 0.35–0.40;
  - contando los reflejos, su precisión no pasa de 0.48.
- **El criterio 3D** le cuesta a la red ~0.05 de recall (z fuera de ±300 nm). Aun así, con la z exigida, la red supera al clásico sin z.

**Por rangos** (recall, red / clásico; RMSE lateral entre paréntesis):

| variable | rango | test v2 | videos |
|---|---|---|---|
| fotones | 1000–2000 | 0.26 / 0.15 (58 / 74 nm) | 0.36 / 0.24 |
| | 2000–5000 | 0.58 / 0.32 (49 / 62) | 0.63 / 0.36 |
| | 5000–10 000 | 0.83 / 0.52 (45 / 60) | 0.89 / 0.58 |
| | 10 000–20 000 | 0.91 / 0.66 (41 / 55) | 0.95 / 0.65 |
| | 30 000–50 000 | 0.99 / 0.83 (37 / 48) | 1.00 / 0.85 (24 / 44) |
| fondo [e⁻/px] | 0–3 | 0.72 / 0.50 | 0.86 / 0.76 |
| | 12–25 | 0.74 / 0.51 | 0.82 / 0.59 |
| | 50–200 | 0.64 / 0.40 | 0.92 / 0.38 |
| vecino más cercano | < 300 nm | 0.54 / 0.32 (50 / 69 nm) | 0.66 / 0.39 |
| | 300–500 nm | 0.67 / 0.35 | 0.73 / 0.40 |
| | 0.5–1 µm | 0.74 / 0.54 | 0.83 / 0.62 |
| | > 2 µm | 0.88 / 0.72 (39 / 46) | 0.92 / 0.71 |
| autofluorescencia (contraste) | 0–0.25 | 0.73 / 0.51 | 0.74 / 0.54 |
| | 1–2 | 0.71 / 0.48 | 0.80 / 0.55 |
| | 2–4 | 0.62 / 0.43 | — |
| z | −3 a −2 µm | 0.71 / 0.66 (69 / 75 nm) | 0.77 / 0.66 |
| | 0 a 1 µm | 0.78 / 0.53 (28 / 42) | 0.78 / 0.60 |
| | 2 a 3 µm | 0.60 / 0.29 (38 / 62) | 0.66 / 0.37 |

- **Fotones.** La diferencia es máxima a señal media: +0.26 a +0.31 entre 2000 y 20 000 fotones. Con < 2000 fotones los dos fallan (red 0.26–0.36, clásico 0.15–0.24), pero la red sigue sacando +0.11 a +0.12.
- **Vecinos.** El clásico cae a 0.32–0.40 con vecinos a < 500 nm, donde la DoG funde las dos PSF. La red mantiene 0.54–0.73.
- **Fondo y autofluorescencia.** Con fondo alto (> 50 e⁻/px) el clásico baja a 0.38–0.40, mientras la red se mantiene en 0.64–0.92. La autofluorescencia afecta poco a los dos. La precisión del clásico es 0.32–0.52 en todos los bins (reflejos incluidos); la de la red, 0.87–0.99.
- **z.** El clásico rinde mejor del lado compacto de la PSF (z < −2 µm: 0.66, cerca de la red, 0.71–0.77). Del lado de los anillos (z > 2 µm) cae a 0.29–0.37, contra 0.60–0.66 de la red. Es la única región donde la diferencia de recall es chica sin que la señal sea alta: +0.05 en el test y +0.11 en los videos. En esa misma región el RMSE lateral de los dos es el peor (61–75 nm).
- **RMSE lateral.** La red es más precisa en todos los bins, por 2–22 nm (24 frente a 44 nm con ≥ 30 000 fotones en los videos). La diferencia mínima, 2 nm, es con contraste de autofluorescencia 2–4 en el test (126 emisores).

**Régimen fotones × vecino** (`linea_base_clasica_regimen.csv`, `fig_linea_base_clasica_regimen.png`; los dos conjuntos juntos, diferencia pareada por emisor):

| fotones \ vecino | < 500 nm | 0.5–1 µm | 1–2 µm | > 2 µm |
|---|---|---|---|---|
| 1000–2000 | +0.12 | +0.10 | +0.15 | +0.19 |
| 2000–5000 | +0.20 | +0.22 | +0.40 | +0.43 |
| 5000–10 000 | +0.40 | +0.27 | +0.28 | +0.25 |
| 10 000–20 000 | **+0.43** | +0.28 | +0.18 | +0.07 |
| 20 000–30 000 | +0.34 | +0.11 | +0.10 | +0.04 [0.01, 0.09] |
| 30 000–50 000 | +0.20 | +0.17 | +0.07 | **+0.01 [0.00, 0.03]** |

Todas las diferencias son positivas, y todas las celdas menos la última tienen IC que excluye 0. En las 24 celdas (37 381 emisores) hay 9 509 que solo detecta la red y 681 que solo detecta el clásico.

## 3. Tracking con el LAP mejorado

`linea_base_clasica_tracking.csv` y `fig_linea_base_clasica_tracking.png`.

**Enlace.** El elegido en B1 de `tracking_3d`: costo normalizado 2σ² + 2DΔt, huecos ≤ 10, pistas de ≥ 8 detecciones y D conocido por condición.
- σ por eje medido en la mitad de selección: clásico 27.2 nm, red 21.1 nm (lateral). Para el enlace 3D de la red, los σ de `tracking_3d` (26.9, 27.6 y 97.7 nm).
- **Entradas:**
  - red con enlace 3D (la de `tracking_3d`);
  - red con enlace 2D, el mismo que recibe el clásico;
  - clásico 2D;
  - clásico sin las detecciones sobre reflejos GT, un **oráculo** que acota lo que se ganaría separando los reflejos.

**Métricas.** ISBI e identidades en (x, y) con ε = 150 nm, y D lateral por MSD (lags 1–4, con ordenada) contra el mismo estimador sobre la pista GT. Mitad de prueba, 100 videos.

| | red, 3D | red, 2D | clásico, 2D | clásico sin reflejos (oráculo) |
|---|---|---|---|---|
| JSC, todos | **0.550** [0.490, 0.611] | 0.491 [0.431, 0.554] | 0.211 [0.187, 0.238] | 0.292 [0.257, 0.331] |
| α / β | 0.517 / 0.496 | 0.479 / 0.459 | 0.284 / 0.182 | 0.287 / 0.246 |
| pistas estimadas por pista GT | 1.05 | 1.05 | 1.96 | 1.39 |
| cambios de identidad por pista GT | 2.4 [1.6, 3.2] | 4.1 [2.8, 5.4] | 4.3 [3.1, 5.5] | 4.1 |
| fragmentaciones por pista GT | 1.9 | 1.9 | 3.9 | 3.8 |
| error relativo mediano de D lateral | **0.128** [0.105, 0.147] | 0.148 | 0.291 [0.264, 0.320] | 0.293 |
| D / D_GT (mediana) | 1.00 | 0.98 | 1.02 | 1.01 |

**JSC por D y por densidad:**

| | D = 0.001 | 0.02 | 0.2 | 1.5 | 1–4 emisores | 5–8 | 9–12 |
|---|---|---|---|---|---|---|---|
| red, 3D | 0.75 | 0.74 | 0.57 | 0.25 | 0.61 | 0.57 | 0.52 |
| red, 2D | 0.77 | 0.71 | 0.44 | 0.20 | 0.59 | 0.52 | 0.45 |
| clásico, 2D | 0.34 | 0.30 | 0.17 | 0.08 | 0.22 | 0.19 | 0.22 |
| clásico sin reflejos | 0.48 | 0.42 | 0.24 | 0.11 | 0.30 | 0.27 | 0.31 |

**Cambios de identidad por pista GT** (red 3D / red 2D / clásico): 0.32 / 0.22 / **0.04** con D = 0.001, 0.41 / 0.59 / 0.47 con 0.02, 1.6 / 3.3 / 3.9 con 0.2 y 7.2 / 12.2 / 12.7 con 1.5. **Error del D lateral** (ídem): 0.13 / 0.13 / 0.19, 0.04 / 0.04 / 0.17, 0.08 / 0.18 / 0.35 y 0.27 / 0.30 / 0.34.

- **El clásico rinde menos de la mitad en JSC** en todos los D y densidades.
  - **Reflejos (β 0.18 frente a 0.50):** generan pistas espurias, 1.96 pistas por pista GT. Con el oráculo el JSC sube a 0.29, que sigue siendo poco más de la mitad del de la red.
  - **Recall (α 0.28):** cada partícula queda cubierta en menos cuadros y más fragmentada (3.9 fragmentaciones por pista frente a 1.9).
- **Cambios de identidad.** Con D = 0.001 el clásico tiene menos (0.04 [0.00, 0.09] frente a 0.32 [0.09, 0.62]). Es la única métrica en que gana, y es un efecto de su recall bajo: con menos cuadros emparejados hay menos ocasiones de cambiar. Para D ≥ 0.02 los IC se solapan con los de la red 2D, y la red 3D tiene menos que ambos con D = 0.2 y 1.5, porque la z le ayuda a no cruzar pistas.
- **D lateral.** Los dos métodos lo estiman sin sesgo (mediana de D / D_GT de 0.98 a 1.02), pero el error relativo del clásico duplica al de la red: 0.29 frente a 0.13–0.15. Las pistas del clásico son más cortas y su error lateral es mayor. La ventaja es más clara con D = 0.02 (0.17 frente a 0.04).
- **Densidad.** El JSC del clásico no baja con la densidad (0.22 / 0.19 / 0.22), porque lo limitan los reflejos y el recall, no las confusiones. El de la red sí baja (0.61 → 0.52), pero sigue siendo 2.4 veces mayor con 9–12 emisores.

## 4. ¿En qué régimen gana cada método?

- **El detector clásico no gana en ningún régimen** de detección, ni en recall ni en RMSE lateral. Tampoco en tracking, salvo por los cambios de identidad con D = 0.001, que son un artefacto de su recall bajo.
- **Baja señal (< 5000 fotones):** los dos fallan, pero la red recupera +0.10 a +0.43 de recall y tiene 13–16 nm menos de RMSE lateral. Con < 2000 fotones la ventaja es la menor (+0.10 a +0.19): ahí el límite es la señal.
- **Alta densidad (vecino a < 500 nm):** es donde más aporta la red. Con 5000–20 000 fotones, el clásico detecta 0.30–0.41 de esos emisores y la red 0.70–0.85 (+0.40 a +0.43). En tracking, con 9–12 emisores por célula, el JSC es 0.52 frente a 0.22.
- **Donde la red no aporta:** con mucha señal (≥ 20 000 fotones) y partículas aisladas (vecino a > 2 µm), el recall del clásico es 0.96–0.99 y la diferencia +0.01 a +0.04. Aun ahí la red es más precisa en x, y (24–37 nm frente a 44–48 nm con ≥ 30 000 fotones), y ese régimen es una fracción chica de los datos (292 de los 37 381 emisores de la tabla de régimen están en esas dos celdas).
- **Aun en el régimen favorable, el clásico necesita resolver los reflejos.** Con el espejo, detecta una mancha por cada partícula y otra por su reflejo, con contraste parecido. Separarlas exige emparejarlas (`tracking/espejo.py`), y aun con un oráculo perfecto su JSC de tracking queda en 0.29 frente a 0.55. **La red aporta además la z,** que el clásico no da sin la triangulación con el reflejo.

## Figuras (PNG 300 dpi, `figuras/linea_base_clasica/`)

- `fig_linea_base_clasica_ajuste.png`: Jaccard, recall y precisión del clásico frente a k para r = 1, 2, 3 en la mitad de selección (los dos conjuntos juntos y por separado); curvas de precisión–recall del clásico (k) y de la red (conf) en la mitad de prueba, con y sin contar reflejos.
- `fig_linea_base_clasica_deteccion.png`: recall (con la precisión a trazos donde corresponde) por fotones, fondo, vecino, autofluorescencia y z en los dos conjuntos, y RMSE lateral por los mismos rangos en el test v2.
- `fig_linea_base_clasica_regimen.png`: recall del clásico, de la red y su diferencia por fotones × distancia al vecino.
- `fig_linea_base_clasica_tracking.png`: JSC, cambios de identidad por pista y error relativo del D lateral por D y por densidad, con las cuatro entradas.

## Límites

- **El clásico es el detector de reflejos de `tracking/` con dos parámetros ajustados.** Un detector clásico dedicado, por ejemplo con un ajuste de la PSF esférica o con una DoG a la escala del halo, podría subir su techo de recall (0.53–0.61).
- **La comparación es lateral.** El clásico no da z. La red se reporta también con su criterio 3D, que la penaliza en ~0.05 de recall.
- **Simulación.** En datos reales (`resultados_tracking_yeast6_1.md`), la red pone detecciones sobre los anillos de la PSF real y el clásico localiza la partícula con un ruido igual o menor. Esta comparación no dice nada de ese régimen.
