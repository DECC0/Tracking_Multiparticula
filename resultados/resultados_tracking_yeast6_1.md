# Tracking 3D en Yeast6_1 (canal derecho)

Se aplica a datos reales el pipeline de `resultados_tracking_3d.md`: la red s1, el enlace LAP de B1, el detector clásico de reflejos, el emparejamiento real ↔ reflejo y la z triangulada. Los datos son los de Yeast6_1 (`resultados_datos_reales.md`): una partícula, 100 cuadros a 10 Hz con exposición de 100 ms, canal derecho. Es solo análisis, sin reentrenar y sin Hypatia.

```bash
PY=~/miniforge3/envs/tesis/bin/python
LD_PRELOAD=$HOME/miniforge3/envs/tesis/lib/libstdc++.so.6 $PY figuras/tracking_yeast6_1.py localizar [--tif RUTA]   # ~20 s
$PY figuras/tracking_yeast6_1.py analisis                                                                             # ~8 min (bootstrap)
```

**Montaje** (el de `celula_v2.py reales`):
- **Par clásico real/reflejo** de `analisis_datos_reales.py`.
- **Geometría:** φ = −169.9°, la dirección media reflejo → real. La línea espejo/plano focal se toma en el borde de la franja, a −32.0 px del punto medio. Es un supuesto: `resultados_calibracion_pupila.md`, sección 7, la corre 1.52 µm.
- **Red:** ventana de 96 px centrada en el punto medio, con umbral 0.2, que es un superconjunto del 0.4.
- **Manchas:** las de `tracking/reflejos.py` (DoG > 5σ y ajuste gaussiano), en la misma ventana.

**Criterios de evaluación:**
- La "partícula" del pipeline es el componente conexo de pistas reales y de reflejo unidas por pares con más cuadros triangulados. No usa el par clásico.
- El par clásico solo sirve de referencia. Un real "correcto" está a ≤ 3 px del real clásico; un reflejo "correcto", a ≤ 2 px del reflejo clásico.
- **MSD:** pendiente de MSD(τ) con τ = 1–4 cuadros, con ordenada.
- **IC 95 %:** bootstrap de bloques móviles de 10 cuadros (10 bloques por remuestreo, 2000 remuestreos, solo pares dentro de un bloque).

**Salidas:**
- `figuras/tracking_yeast6_1/`: CSV y 5 PNG a 300 dpi;
- caché sin versionar en `localizacion/datos/yeast6_1/`.

## Resumen

| | resultado |
|---|---|
| 1. desfase dt = t_real − t_reflejo, lineal en L | **El modelo no explica el desfase.** Ajustado a todos los cuadros, r² = 0.010 en el par clásico y 0.005 en el pipeline. El giro implícito de m queda indeterminado: **−4.0° [−15.6, +17.7]** con el par clásico y **+1.7° [−46.7, +7.7]** con el pipeline. El borde de la franja pediría −6.5°, dentro de los dos IC, pero también lo está 0°. La causa: L varía solo ±5 % (4630 ± 232 nm), así que un giro de m es casi una constante. La varianza de dt está en z (r = −0.89 con la z triangulada). σ_dt sale del residuo: **160 nm** (172 sin modelo) |
| efecto en el emparejamiento | El pipeline final, con el modelo L, triangula **95 cuadros**: reflejo correcto en 93 y real en 79. Sin modelo eran 50. **Toda la ganancia viene de la constante del modelo** (centrar dt; el sesgo de la red en t es de ~70 nm), no de la pendiente: restar solo la mediana de dt da exactamente los mismos 95 / 93 / 79 cuadros |
| 2. ruido de localización (x, y) | **Clásico:** σ = 54–60 nm en x y ≈ 0 ± 15 nm en y (IC desde < 0 hasta 55–93 nm). **Red, umbral 0.4:** 94–100 nm en x y 64–75 nm en y (IC 52–131 en x). Las estimaciones favorecen al clásico, pero los IC se solapan. La pista del pipeline final da 115–156 nm, por los saltos a anillos |
| sesgo red − clásico | Media (−55, −101) nm. **No es constante**: las medias por bloque de 10 cuadros van de −129 a −13 nm en x y de −170 a −9 en y. Su dispersión es 1.7–1.8 veces la esperable por el ruido de cada bloque (χ² 23 y 63 con 9 gl). No hay tendencia (pendiente −2 [−117, +60] y +10 [−145, +69] nm por 100 cuadros), ni correlación con z (−0.17, −0.12) ni con el brillo (−0.11, −0.20), todo con IC que incluyen 0. En y el sesgo deriva lentamente (autocorrelación de paso 1: 0.37) |
| 3. atípicos de z (k = 3) | Con β = 45°, **3 de 95** (pipeline) y **18 de 100** (par clásico, cuya z es más suave: σ del residuo 30 nm frente a 54). Sensibilidad en β 40–50°: con k = 2.5, 5–10 y 13–22; con k = 3, 2–5 y 7–18; con k = 3.5, 1–2 y 5–14 |
| 4. D, µm²/s (β = 45°, k = 3) | **Par clásico:** lateral 0.078 [0.039, 0.113], z 0.040 [0.011, 0.080], 3D 0.075 [0.036, 0.113]. **Pipeline final:** lateral 0.114 [0.065, 0.194] (inflado por los anillos), z 0.056 [0.017, 0.108], 3D 0.084 [0.049, 0.125]. D_z baja de 0.062–0.073 a 40° a 0.030–0.043 a 50°. Con k = 2.5, 3.5 o sin rechazo, D_z cambia ≤ 0.007 |

**Conclusión:**
- El desfase real–reflejo de estos datos **no es un giro de m**. Depende de z, probablemente por un sesgo de los centroides con el desenfoque: anillos del real asimétricos o recorte de pupila del reflejo. En el emparejamiento solo hacía falta centrarlo, y eso lo hace la constante del modelo, que también la habría dado la mediana sola.
- **Con el pipeline final**, la z con β = 45° reproduce la del par clásico a 81 nm RMS (media −63 nm), y D_z coincide con la del par clásico dentro de los IC.
- **Lo que limita D_z es β y la estadística, no el tracking:** con una partícula y 100 cuadros, los IC son de un factor 5–10. La lateral del pipeline está inflada por los cuadros en que el real es un anillo.

## 1. Modelo del desfase real–reflejo

`tracking_yeast6_1_desfase.csv` y `fig_tracking_yeast6_1_deteccion.png` (paneles 3 y 4).

**Modelo:**
- dt = a + b·L, con L = s_real − s_reflejo. Es lo que se ve si el segmento real → reflejo es paralelo a m + b·t en lugar de a m: la dirección de m está girada δ = atan b.
- Se ajusta con recorte iterativo de residuos > 3σ (σ = 1.4826 MAD). El IC del giro sale de un bootstrap de bloques.

**Aplicación en el pipeline:**
- Se ajusta en los cuadros comunes de los pares de la partícula del primer pase (σ_dt = 200 nm y D = 0.2 µm²/s). No se usa el par clásico.
- Se aplica a todos los cuadros: se gira el sistema (s, t) en δ (φ = −168.3° en lugar de −169.9°) y se resta la mediana del dt girado (−75 nm).
- El segundo pase usa σ_dt = 1.4826 MAD del residuo y el D lateral medido en el primero.
- El giro mueve la z del pipeline 9 nm RMS (β = 45°).

| ajuste | n | δ [IC 95 %] | r² | σ de dt sin modelo / residuo |
|---|---|---|---|---|
| par clásico, dt ~ L | 100 | **−4.0° [−15.6, +17.7]** | 0.010 | 212 / 202 nm |
| pipeline (primer pase), dt ~ L | 99 (90 dentro) | **+1.7° [−46.7, +7.7]** | 0.005 | 172 / **160 nm** |
| *diagnóstico:* par clásico, dt ~ s_reflejo | 100 | pendiente 0.55 [0.46, 0.70] | 0.79 | 212 / 79 nm |
| *diagnóstico:* pipeline, dt ~ s_reflejo | 99 (68 dentro) | pendiente 0.32 | 0.84 | 172 / 34 nm |

- **Comparación con la franja.** El borde de la franja forma +6.5° con t, el mismo valor que `resultados_datos_reales.md`, sección 3, da para el canal derecho. Si el borde fuera paralelo a la línea del espejo, m debería girarse −6.5°. El par clásico da −4.0°, con el mismo signo y un orden de magnitud parecido, y el pipeline da +1.7°. Pero los dos IC contienen a la vez −6.5° y 0°. **Estos datos no pueden confirmar ni descartar ese giro.**
  - Con L casi constante (±232 nm sobre 4630), un giro de 4° mueve dt en ~320 nm de forma casi uniforme y solo ±16 nm de un cuadro a otro.
  - Por eso la pendiente y la constante son casi la misma cosa: el ajuste clásico compensa δ = −4.0° con a = +321 nm.
  - Para medir el giro hacen falta reflejos a distancias L muy distintas: varias partículas o beads (`resultados_calibracion_espejo.md`).
- **Dónde está la varianza de dt.** Está en s_reflejo, que con β = 45° es exactamente −z. En el par clásico, dt ~ s_reflejo deja un residuo de 79 nm frente a 202 (r² 0.79).
  - **No es una geometría del espejo.** Si la normal del espejo tuviera una componente a lo largo de t, el desfase dependería de la distancia al espejo, d = s cos β + z sin β, y no de la posición del reflejo sola.
  - Lo más probable es un sesgo de los centroides a lo largo de t que cambia con el desenfoque.
  - Se reporta como diagnóstico. Usarlo en el pipeline (variante `desfase_en_s_reflejo`) da σ_dt = 34 nm, demasiado estrecho: 76 cuadros triangulados en lugar de 95.

## 2. x, y del real: red frente a la detección clásica

`tracking_yeast6_1_xy_red_vs_clasico.csv`, `_ruido_localizacion.csv`, `_sesgo_tiempo.csv` y `fig_tracking_yeast6_1_ruido.png`.

**Detecciones.** En cada cuadro, la detección de la red más cercana al real clásico, si está a ≤ 3 px.
- Con umbral 0.4, la red detecta el real en 88 cuadros. Faltan 4, 6, 7, 9, 10, 72, 76, 78, 82, 90, 97 y 98.
- Con 0.2, lo detecta en 99 (falta el 76).
- Además pone 1.55 detecciones por cuadro (4.42 con umbral 0.2) sobre los anillos de la PSF del real, a 3–20 px y con la misma conf.
- Diferencia red − clásico, con umbral 0.4: x −55 ± 70 nm, y −101 ± 89 nm, RMS radial 161 nm. Con 0.2: −44 ± 74 y −70 ± 94 nm, RMS 145 nm.

**Ruido de localización.** Se estima de cada serie de posiciones por separado, con el D de la misma pendiente:
- **por el MSD:** la ordenada b del ajuste MSD(τ) = 2Dτ + b, con τ = 1–4;
- **por los pasos:** var(paso entre cuadros consecutivos) − 2DΔt.

Con exposición t_E = Δt, el desenfoque de movimiento resta 2D·t_E/3 a las dos cosas, así que σ² = (b + 2D·t_E/3)/2 = (var − 2DΔt + 2D·t_E/3)/2. Una varianza negativa se informa como σ < 0.

| serie | eje | cuadros | σ por la ordenada del MSD [IC] | σ por los pasos [IC] | sin el término de desenfoque (MSD / pasos) |
|---|---|---|---|---|---|
| par clásico | x | 100 | 60 [−25, 93] nm | 54 [−18, 82] | 38 / 27 |
| par clásico | y | 100 | −15 [−66, 56] | 15 [−53, 53] | −57 / −52 |
| red, umbral 0.4 | x | 88 | 100 [58, 131] | 94 [52, 120] | 91 / 84 |
| red, umbral 0.4 | y | 88 | 75 [−31, 106] | 64 [−10, 90] | 52 / 35 |
| red, umbral 0.2 | x | 99 | 97 [55, 128] | 93 [54, 123] | 88 / 83 |
| red, umbral 0.2 | y | 99 | 70 [−28, 92] | 67 [19, 86] | 38 / 32 |
| pipeline final (pista) | x | 97 | 156 [88, 200] | 150 [97, 187] | 142 / 136 |
| pipeline final (pista) | y | 97 | 121 [76, 148] | 115 [75, 142] | 106 / 99 |

- **Los dos estimadores coinciden** a menos de 11 nm en cada serie, como corresponde: comparten D, y el de los pasos es el lag 1 del mismo MSD. La excepción es la y del clásico (−15 frente a +15 nm), donde los dos dan una varianza cercana a 0.
- **El clásico parece más preciso:** 54–60 nm en x y casi 0 en y, frente a 64–100 nm de la red.
  - En x, el IC de la red no baja de 52 nm y el del clásico llega a 0, pero se solapan entre 52 y 93 nm.
  - Con un D de ~0.08 µm²/s, el paso browniano por eje es de ~125 nm (√(2DΔt)). Un ruido de 50–100 nm queda en el borde de lo que 100 cuadros pueden separar de la difusión.
  - Que el clásico dé σ² ≤ 0 en y indica que su ruido allí está por debajo de lo resoluble.
- **Consistencia cruzada.** var(red − clásico) = σ_red² + σ_clásico² − 2·cov.
  - En x, la std de la diferencia (70 nm) es menor que la suma en cuadratura de los σ (~115 nm): hay covarianza positiva, porque los dos métodos localizan sobre los mismos fotones.
  - En y es mayor (89 frente a ~76 nm), lo que encaja con el sesgo que deriva en y (tabla siguiente).
- **Pista del pipeline.** Su ruido de 115–156 nm no es el de la red sino el de los cuadros en que el representante es un anillo (sección 3). Por eso su D lateral sale inflado (sección 5).

**¿El sesgo red − clásico es constante en el tiempo?** No del todo.

| eje | media | medias por bloque de 10 cuadros | dispersión de las medias / error esperado | χ² (9 gl) | pendiente por 100 cuadros [IC] | corr. con z [IC] | corr. con el brillo [IC] | autocorr. de paso 1 |
|---|---|---|---|---|---|---|---|---|
| x | −55 nm | −112 −30 −76 −13 −35 −48 −62 −129 −37 −48 | 37 / 21 nm | 23 (p = 0.006) | −2 [−117, +60] nm | −0.17 [−0.41, +0.05] | −0.11 [−0.33, +0.09] | 0.02 |
| y | −101 nm | −170 −91 −147 −9 −77 −90 −146 −104 −123 −70 | 47 / 26 nm | 63 | +10 [−145, +69] nm | −0.12 [−0.44, +0.12] | −0.20 [−0.42, +0.09] | 0.37 |

- **No hay tendencia** a lo largo del video (100 cuadros, con un fotoblanqueo de ~3×), **ni dependencia de z ni del brillo** distinguible de 0.
- **Pero las medias por bloque varían más que su error** (factor 1.7–1.8):
  - en x el sesgo cuadro a cuadro no está autocorrelacionado y el exceso entre bloques es moderado (p = 0.006);
  - en y está autocorrelacionado (0.37), así que el χ² exagera la significancia, pero esa misma autocorrelación muestra que el sesgo en y deriva en escalas de unos pocos cuadros.
- **En resumen:** un desfase medio de ~115 nm, con fluctuaciones de ±40–50 nm de un tramo a otro que no siguen ni a z ni al brillo.

## 3. Pipeline: variantes

`tracking_yeast6_1_variantes.csv` y `fig_tracking_yeast6_1_deteccion.png`.

**Enlace.** El de B1 (costo normalizado 2σ² + 2DΔt, huecos ≤ 10, pistas de ≥ 8 detecciones), en 2D. Dos pases:
1. con D = 0.2 µm²/s y σ_dt = 200 nm;
2. con el D lateral de la partícula del primer pase (sin atípicos laterales) y el σ_dt y el giro del modelo de desfase (sección 1).

**Variantes:**
- **Tal cual:** la configuración de `tracking_3d` (umbral 0.4; `filtrar_manchas`, sin supresión de anillos) y sin modelo de desfase.
- **Final:**
  - umbral 0.2;
  - supresión de anillos en 16 px: de cada grupo de detecciones, la más cercana a su centro ponderado por conf;
  - descarte de las detecciones que están del lado del espejo de otra, a más de 16 px a lo largo de m y a < 600 nm en t, porque son el reflejo detectado como real (0.39 por cuadro);
  - exclusión de las manchas a ≤ 16 px de una detección;
  - modelo de desfase en L.

| variante | σ_dt | det./cuadro | pistas reales / de reflejo | cuadros triangulados | reflejo correcto | real correcto | z − z clásica, β 45° (media / RMS) |
|---|---|---|---|---|---|---|---|
| tal cual | 157 nm | 2.87 | 8 / 9 | 77 | 75 | 44 | −55 / 89 nm |
| **final** | **160 nm** | 1.51 | 4 / 2 | **95** (faltan 10, 42, 86, 87, 88) | **93** | **79** | −63 / 81 nm |

- **Tal cual.**
  - La red da 8 pistas reales, la mayoría sobre anillos, y el real de los pares es un anillo en 33 de los 77 cuadros.
  - Con β = 45° eso no cambia z (z = −s_reflejo), pero con 40° o 50° entra con un factor ∓0.18, y la pista lateral queda inservible (σ ~ 250 nm).
- **Final.** El reflejo emparejado es el correcto en 93 de los 95 cuadros triangulados.
- **La diferencia de z con el par clásico** es la del reflejo: ajuste gaussiano en ±5 px frente a centroide iterativo de r = 10 px sobre un reflejo con la pupila recortada (`resultados_recorte_pupila.md`).

## 4. z triangulada y atípicos

`tracking_yeast6_1_z_beta.csv` y `fig_tracking_yeast6_1_z.png`.

**Fórmula.** z = −s_M/tan β + (L/2)·tan β por cuadro, con β entre 40° y 50° cada 0.5°.

**Atípicos:** |z − mediana móvil de 9 cuadros| > k·σ, con σ = 1.4826 MAD de los residuos del video. **k = 3, fijo.** Es un criterio de consistencia temporal: no usa la z de la red.

| fuente | cuadros | z media (β 40 / 45 / 50) | σ del residuo (β 45) | atípicos k = 3, β 45 | k = 2.5 / 3 / 3.5 en β 40–50° |
|---|---|---|---|---|---|
| par clásico | 100 | −1813 / −836 / +115 nm | 30 nm | **18** | 13–22 / 7–18 / 5–14 |
| pipeline final | 95 | −1888 / −903 / +54 | 54 | **3** | 5–10 / 2–5 / 1–2 |
| tal cual | 77 | −1802 / −839 / +99 | 79 | **2** | 2–7 / 1–5 / 0–3 |

- **La z clásica es la más suave**, así que su umbral relativo es el más estricto: rechaza excursiones de pocos cuadros que la propia serie sigue. El criterio dice qué cuadros se apartan de su entorno, no cuáles están mal. Con D_z ≈ 0.04 µm²/s, el paso browniano en z es de ~90 nm, del orden de esos residuos.
- **La z absoluta se corre ~190 nm por grado de β.** La línea del espejo la desplaza en δ/tan β; con los 1.52 µm de `resultados_calibracion_pupila.md` serían 1.5 µm. Ninguna de las dos cosas afecta a D, salvo por el cambio de escala con β (sección 5).

## 5. D por MSD

`tracking_yeast6_1_D.csv`, `_msd.csv`, `fig_tracking_yeast6_1_msd.png` y `fig_tracking_yeast6_1_trayectoria.png`.

**Qué entra:**
- lateral: todos los cuadros de las pistas reales de la partícula (no depende de β);
- z y 3D: los cuadros con z válida y sin atípicos (k = 3), con las (x, y) del real triangulado.

**D en µm²/s [IC 95 %]:**

| β | par clásico, 3D | par clásico, z | pipeline final, 3D | pipeline final, z |
|---|---|---|---|---|
| 40° | 0.075 [0.036, 0.114] | 0.062 [0.018, 0.122] | 0.090 [0.055, 0.132] | 0.073 [0.019, 0.142] |
| 42.5° | 0.075 [0.037, 0.113] | 0.053 [0.014, 0.112] | 0.090 [0.054, 0.134] | 0.065 [0.020, 0.127] |
| **45°** | **0.075 [0.036, 0.113]** | **0.040 [0.011, 0.080]** | **0.084 [0.049, 0.125]** | **0.056 [0.017, 0.108]** |
| 47.5° | 0.072 [0.039, 0.104] | 0.037 [0.009, 0.078] | 0.080 [0.048, 0.121] | 0.049 [0.017, 0.095] |
| 50° | 0.069 [0.037, 0.101] | 0.030 [0.007, 0.066] | 0.072 [0.042, 0.104] | 0.043 [0.012, 0.083] |
| **lateral** (todo β) | **0.078 [0.039, 0.113]**; x 0.066, y 0.089 | | **0.114 [0.065, 0.194]**; x 0.127, y 0.102 | |

**Sensibilidad al umbral de atípicos** (β = 45°, z / 3D):

| | k = 2.5 | k = 3 | k = 3.5 | sin rechazo |
|---|---|---|---|---|
| par clásico | 0.041 / 0.073 | 0.040 / 0.075 | 0.045 / 0.074 | 0.047 / 0.068 |
| pipeline final | 0.051 / 0.081 | 0.056 / 0.084 | 0.056 / 0.082 | 0.057 / 0.083 |

- **D_z del pipeline y del par clásico coinciden dentro de los IC** en todo β. Con 45° comparten casi todo el reflejo, y el pipeline queda ~0.015 por encima por su reflejo más ruidoso (σ del residuo de z 54 frente a 30 nm).
- **D_z se reduce a la mitad entre 40° y 50°**, porque los pesos de s_real y s_reflejo en z cambian con β. En todo el rango queda por debajo del D lateral clásico (0.078), con IC que lo alcanzan.
- **Isotropía.** D_z = D_lateral pediría β ≤ 40°, en la línea del ~38° que daba el acople en m (`resultados_datos_reales.md`, sección 4). Pero ni los IC ni el supuesto de isotropía (una partícula en una levadura que toca el espejo) lo sostienen. **No sirve para fijar β.**
- **Ordenadas.** En el par clásico son negativas (−3.5·10³ nm² en la lateral y en z, β 45°), como corresponde al desenfoque (−2D·t_E/3 por eje). En el pipeline, la ordenada lateral de 6.3·10⁴ nm² son los anillos, y por eso su D lateral (0.114) supera al clásico. La lateral fiable es la clásica.

## Figuras (PNG 300 dpi, `figuras/tracking_yeast6_1/`)

- `fig_tracking_yeast6_1_deteccion.png`:
  - la ventana con las detecciones de la red, las candidatas del pipeline final, las manchas, el par clásico y los cuadros triangulados;
  - s del real de la pista menos s clásico (tal cual y final);
  - dt frente a L, con los ajustes del par clásico y del pipeline y el giro que pediría el borde de la franja;
  - dt frente a la z triangulada.
- `fig_tracking_yeast6_1_ruido.png`: sesgo red − clásico por cuadro y por bloque de 10 cuadros en x e y; σ de localización por los dos estimadores, con IC.
- `fig_tracking_yeast6_1_z.png`: z(t) con β = 45° (pipeline final y par clásico; banda del clásico para β de 40–50°), mediana móvil y atípicos con k = 3; % rechazado frente a β para k = 2.5, 3 y 3.5.
- `fig_tracking_yeast6_1_trayectoria.png`: trayectoria 3D del pipeline final y del par clásico, proyección lateral y z(t) sin atípicos.
- `fig_tracking_yeast6_1_msd.png`: MSD por eje (lateral, z y 3D, normalizado por el número de ejes) del pipeline final y del par clásico, y D_z(β) frente a D lateral.

## Elecciones hechas después de ver los datos

Cada fila cambia **una** elección del pipeline final y deja las demás igual (`tracking_yeast6_1_variantes.csv`; D con β = 45° y k = 3, 500 remuestreos).

| elección (lo que motivó verla) | alternativa | cuadros triangulados | reflejo / real correcto | D lateral | D_z | D 3D |
|---|---|---|---|---|---|---|
| **pipeline final** | — | **95** | 93 / 79 | 0.114 | 0.056 | 0.084 |
| **supresión de anillos en 16 px**: la red pone 1.5–4.4 detecciones por cuadro sobre los anillos de la PSF real | sin suprimir | 26 | 26 / 9 | 0.016 | 0.006 | 0.013 |
| **descarte de las detecciones que parecen el reflejo**: sin él, la mancha del reflejo se empareja con la detección del reflejo (L ≈ 0) | sin descartar | 40 | 40 / 33 | 0.095 | 0.041 | 0.090 |
| **exclusión de manchas a ≤ 16 px** de una detección (las manchas de los anillos) | sin excluir | 95 | 93 / 79 | 0.114 | 0.056 | 0.084 |
| **umbral 0.2** (la red detecta el real en 99 cuadros frente a 88) | umbral 0.4 | 45 | 44 / 44 | 0.074 | 0.035 | 0.057 |
| **enlace 2D**: la z de la red salta entre modos en datos reales (`resultados_celula_v2.md`, sección 4) | 3D con la z de la red (σ_z = 99.6 nm, el de la simulación) | 67 | 66 / 61 | 0.083 | 0.033 | 0.053 |
| **autocalibración de D**: el D lateral del primer pase (0.060) entra en el segundo | D fijo 0.02 | 97 | 95 / 82 | 0.084 | 0.053 | 0.074 |
| | D fijo 0.2 | 87 | 87 / 76 | 0.180 | 0.053 | 0.067 |
| | D fijo 1.5 | 0 (su primer pase da δ = −18.5° y σ_dt = 108 nm, y no queda ningún par) | — | — | — | — |
| **modelo del desfase** (pedido; sus parámetros se ajustan con los datos) | sin modelo (σ_dt = MAD de dt) | 50 (faltan 0–47) | 50 / 40 | 0.156 | 0.053 | 0.100 |
| | solo la constante (mediana de dt) | 95 | 93 / 79 | 0.114 | 0.056 | 0.084 |
| | dt lineal en s_reflejo (diagnóstico) | 76 | 75 / 69 | 0.086 | 0.027 | 0.059 |

Lectura:
- **Imprescindibles:** la supresión de anillos y el descarte de las detecciones del reflejo. Sin la supresión, el pipeline sigue anillos y el D se desploma (0.006–0.016). Sin el descarte se pierde la mitad de los cuadros.
- **Sin efecto, una vez que están las anteriores:** la exclusión de manchas en 16 px. Se puede quitar.
- **Umbral 0.2:** duplica los cuadros triangulados frente al 0.4.
- **Enlace 2D:** con el enlace 3D, la z de la red rompe las pistas y se pierden 28 cuadros. D_z baja a 0.033, dentro de los IC.
- **Autocalibración de D:**
  - Lo que mueve el resultado es que el primer pase y el ajuste del desfase dependen de D. Con D = 1.5, el ajuste de su primer pase da un giro absurdo (−18.5°) y el segundo pase no encuentra pares.
  - Con D = 0.02 se triangulan 2 cuadros más que con el D autocalibrado.
  - En los tres casos con pares, D_z queda entre 0.053 y 0.056. D_z no depende de esta elección; la cobertura sí.
- **Modelo del desfase:** su efecto útil es la constante. Con solo la mediana de dt el resultado es idéntico.
- **Elecciones heredadas**, no revisadas aquí:
  - la ventana de 9 cuadros de la mediana móvil (con 5, 9 y 15 cuadros el par clásico rechaza 22, 18 y 1 con k = 3);
  - el radio de 16 px (de la distribución de distancias de las detecciones de anillo, 4–20 px);
  - la línea en el borde de la franja y la ventana de 96 px (de `celula_v2.py`).

## Límites

- **Partículas aisladas.** La supresión de anillos y el descarte de reflejos lo suponen: dos partículas a < 1.6 µm se fundirían.
- **Una partícula y 100 cuadros.** Los IC de D son de un factor 3–10, los IC de bloques son aproximados (10 bloques) y no hay dispersión entre partículas. Por lo mismo, el giro de m no se puede determinar: L casi no varía.
- **β y la línea no están calibrados** (`resultados_calibracion_espejo.md`). La z absoluta es indicativa y D_z cambia un factor 2 en [40, 50]°.
- **El desfase dependiente de z** del par clásico (sección 1) queda sin explicar. Si es un sesgo de centroides con el desenfoque, también afecta la s de cada imagen y, con ella, la z triangulada.
