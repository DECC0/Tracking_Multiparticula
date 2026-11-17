# Tracking 3D en Yeast6_1 (canal derecho)

Se aplica a datos reales el pipeline de `resultados_tracking_3d.md`: la red s1, el enlace LAP elegido en B1, el detector clásico de reflejos, el emparejamiento real ↔ reflejo y la z triangulada. Los datos son Yeast6_1 (`resultados_datos_reales.md`): una sola partícula, 100 cuadros a 10 Hz, exposición de 100 ms y canal derecho. Es solo análisis: no se reentrena nada y no se usó Hypatia.

```bash
PY=~/miniforge3/envs/tesis/bin/python
LD_PRELOAD=$HOME/miniforge3/envs/tesis/lib/libstdc++.so.6 $PY figuras/tracking_yeast6_1.py localizar [--tif RUTA]   # ~20 s
$PY figuras/tracking_yeast6_1.py analisis                                                                             # ~5 min (bootstrap)
```

**Montaje.** Es el mismo de `celula_v2.py reales`:
- Par clásico real/reflejo de `analisis_datos_reales.py`.
- φ = −169.9°, la dirección media reflejo → real.
- Línea espejo/plano focal en el borde de la franja, a −32.0 px del punto medio del par. Es un supuesto: `resultados_calibracion_pupila.md`, sección 7, la corre +1.52 µm.
- La red ve una ventana de 96 px centrada en el punto medio, columnas 303–398 y filas 399–494. Su canal de distancia usa esa línea.
- La red se corre una vez con umbral 0.2, que es un superconjunto del 0.4.
- Las manchas (`tracking/reflejos.py`, DoG > 5σ y ajuste gaussiano) se buscan en la misma ventana.

**Cómo se elige la partícula.** Es el componente conexo de pistas reales y de reflejo unidas por pares que tiene más cuadros triangulados. El criterio es del propio pipeline y no usa el par clásico. El par clásico sirve solo como referencia: un real o un reflejo "correcto" es el que cae a ≤ 3 px o ≤ 2 px de la posición clásica.

**MSD.** Pendiente de MSD(τ) para τ = 1–4, con ordenada. El IC 95 % sale de un bootstrap de bloques móviles de 10 cuadros: 10 bloques por remuestreo, 2000 remuestreos, contando solo los pares dentro de un bloque.

**Salidas.**
- `figuras/tracking_yeast6_1/`: CSV agregados y 4 PNG a 300 dpi.
- Caché en `localizacion/datos/yeast6_1/`, no versionada: par clásico, geometría, detecciones de la red, manchas y la media de la ventana.

## Resumen

| paso | resultado |
|---|---|
| 1. x, y del real: red frente a la detección clásica | **Con umbral 0.4, la red detecta el real en 88 de 100 cuadros**; faltan 4, 6, 7, 9, 10, 72, 76, 78, 82, 90, 97 y 98. Con 0.2, en 99 (falta el 76). Diferencia red − clásico: media (−55, −101) nm en (x, y), desviación (70, 89) nm, **RMS radial 161 nm** (145 con 0.2). Además pone 1.55 detecciones por cuadro **sobre los anillos de la PSF del real** (3–20 px; 4.4 con umbral 0.2), con la misma conf que el real |
| 2. reflejo y emparejamiento | El detector de manchas encuentra el reflejo en 98 de 100 cuadros, a 0.84 px de mediana del clásico. **Tal cual, el pipeline falla:** la red da 8 pistas reales, la mayoría sobre anillos; la partícula queda repartida en 3, y el real de los pares cae sobre un anillo en 32 de 77 cuadros. Con la **variante adaptada** (anillos suprimidos y sin las detecciones que la red pone sobre el reflejo), los 50 cuadros triangulados tienen el reflejo correcto y el real correcto en 40[^a], pero no se triangulan los cuadros 0–47. Con σ_dt × 2 se triangulan **99 cuadros**: reflejo correcto en 97, real en 81 |
| problema de los datos | **dt = t_real − t_reflejo no es constante**, contra el modelo (t' = t). En el par clásico se correlaciona con la z triangulada (r = −0.89, pendiente −0.55 nm/nm) y deriva ±200 nm. Por eso el σ_dt que se mide es 150–175 nm, frente a ~67 nm en simulación |
| 3. z triangulada | Con β = 45°, z = −s_reflejo, así que la z del pipeline solo difiere de la clásica por la posición del reflejo: **media −47 a −65 nm, RMS 68–89 nm**. Media de z con β = 45°: −836 nm (clásico) y −904 nm (pipeline, 99 cuadros). La media se corre ~190 nm por grado de β: **de −1813 nm a 40° a +115 nm a 50°** |
| atípicos (mediana móvil de 9 cuadros, k = 3) | **Pipeline: 2 de 50, 5 de 99 y 2 de 77 cuadros con β = 45°**. En el barrido de 40–50° hay entre 1 y 8 por variante. El par clásico rechaza **7–18 de 100**, porque su z es más suave (σ del residuo 30–42 nm, frente a 47–69 del pipeline) y el umbral relativo corta excursiones reales. Hay sensibilidad fuerte a la ventana: el clásico rechaza 22, 18 y 1 con ventanas de 5, 9 y 15 cuadros |
| 4. D (β = 45°, sin atípicos), µm²/s | Par clásico: **lateral 0.078 [0.039, 0.111], z 0.040 [0.011, 0.079], 3D 0.075 [0.039, 0.112]**. Pipeline, 99 cuadros: lateral 0.079 [0.031, 0.169], z 0.055 [0.016, 0.108], 3D 0.052 [0.023, 0.102][^b]. Con una partícula y 100 cuadros, los IC abarcan un factor 3–10 |
| D_z(β) | **Baja al subir β:** 0.062 → 0.040 → 0.030 (clásico, 40 → 45 → 50°) y 0.069 → 0.055 → 0.043 (pipeline). D_z < D_lateral en todo [40, 50]°, pero los IC se solapan en todo el rango. **β no se puede fijar por isotropía con estos datos** |

[^a]: Real correcto: a ≤ 3 px del real clásico.
[^b]: El 3D del pipeline queda por debajo de su lateral y de su z porque usa las (x, y) de los cuadros triangulados. Los saltos a los anillos le dan una ordenada de 1.9·10⁵ nm² y aplanan la pendiente.

**Conclusión.**
- **En datos reales, el pipeline de `tracking_3d.py` no funciona sin adaptar.** Hay dos fallas que la simulación no tiene:
  - la red detecta los anillos de la PSF real (y, a veces, el reflejo) como emisores;
  - el desfase real–reflejo a lo largo de la línea del espejo depende de z.
- **Con dos adaptaciones**, la supresión de anillos en 16 px y σ_dt holgado, el reflejo se empareja bien en 97 de 99 cuadros y la z con β = 45° reproduce la del par clásico a ~85 nm RMS.
- **Lo que limita el D en z no es el tracking**, sino una partícula, 100 cuadros y β: D_z cambia un factor 2 entre 40 y 50°.

## 1. x, y del real: red s1 frente a la detección clásica

`tracking_yeast6_1_xy_red_vs_clasico.csv` y `_resumen.csv`. En cada cuadro, la detección de la red más cercana al real clásico, si está a ≤ 3 px.

| entrada | cuadros con el real | det/cuadro | sobre anillos | sobre el reflejo | x: media ± std | y: media ± std | s (a lo largo de m) | t (a lo largo de la línea) | RMS radial |
|---|---|---|---|---|---|---|---|---|---|
| umbral 0.4 | 88 | 2.87 | 1.55 | 0.11 | −55 ± 70 | −101 ± 89 | +72 ± 73 | +89 ± 86 | 161 nm |
| umbral 0.2 | 99 | 6.97 | 4.42 | 0.28 | −44 ± 74 | −70 ± 94 | +56 ± 78 | +61 ± 90 | 145 nm |
| umbral 0.2, anillos suprimidos | 84 | 1.90 | 0.48 | 0.27 | −46 ± 92 | −78 ± 103 | +59 ± 96 | +69 ± 99 | 165 nm |

- **Sesgo y dispersión.** Hay un sesgo constante de ~110 nm en (x, y), casi todo en la dirección (−x, −y). La dispersión es de 70–90 nm por eje. El centroide clásico (r = 10 px sobre una PSF con anillos) no es una referencia sin error, así que esto mide la discrepancia, no el error de la red.
- **Ninguno de los dos se puede declarar mejor con estos datos.** La referencia independiente sería el reflejo, porque t' = t, pero dt deriva con z (sección 2). La std de dt contra el reflejo clásico es 156 nm con la red y 160 nm con el centroide clásico. El paso cuadro a cuadro de dt del par clásico (std del paso / √2) da un ruido combinado de 46 nm por cuadro.
- **Los anillos son el problema principal.** Las detecciones extra caen a 4–20 px del real, en los anillos de la PSF esférica (`resultados_psf_real.md`), y su z mediana es de +1.5 µm, frente a +0.1 µm en el núcleo. La conf máxima del cuadro corresponde al real en solo 74 de 98 cuadros, así que una supresión de no-máximos por conf no basta.
- **La supresión adaptada** agrupa las detecciones a ≤ 16 px de la de más conf y se queda con la más cercana al centro ponderado por conf, porque los anillos rodean al real. Así recupera el real en 84 cuadros y deja 0.48 detecciones de anillo por cuadro. En los 16 cuadros restantes no queda ninguna detección a ≤ 3 px del real: el representante es un anillo o no hay detección (saltos de 300–700 nm en `fig_tracking_yeast6_1_deteccion`, centro).

## 2. Reflejo, pistas y emparejamiento

`tracking_yeast6_1_pistas.csv`, `_pares.csv`, `_reflejo_vs_clasico.csv`, `_autocalibracion.csv`, `_sensibilidad_enlace.csv`.

**Enlace.** Se usa el de B1: costo normalizado con 2σ² + 2DΔt, huecos ≤ 10 y pistas de ≥ 8 detecciones, pero **en 2D**, porque la z de la red no es confiable en datos reales (`resultados_celula_v2.md`, sección 4). Los reflejos se enlazan con el mismo costo. Sin GT, D y σ_dt se auto-calibran en dos pases:
1. primer pase con D = 0.2 µm²/s y σ_dt = 200 nm;
2. segundo pase con el D lateral de la partícula (sin atípicos laterales) y σ_dt = 1.4826 MAD de dt en todos los cuadros comunes de sus pares.

Seguir iterando no converge. Cada emparejamiento se queda con los cuadros de dt chico, así que σ_dt se estrecha (adaptada: 173 → 72 → 23 nm, hasta 0 pares) u oscila (tal cual: 149 → 108 → 314 → 229 → 165 → 88 nm).

**Variantes.**
- **Tal cual:** la configuración de `tracking_3d` (umbral 0.4 y `filtrar_manchas`, que quita las manchas a ≤ 2 px de una detección y las que no están del lado del espejo de ninguna).
- **Adaptada:**
  - umbral 0.2, que detecta el real en 99 cuadros frente a 88;
  - supresión de anillos en 16 px;
  - se descartan como candidatas a real las detecciones del lado del espejo de otra, a más de 16 px a lo largo de m y a < 600 nm en t (0.39 por cuadro). Son el reflejo detectado como real. Sin este paso, la mancha del reflejo se empareja con la propia detección del reflejo (L ≈ 0);
  - se excluyen las manchas a ≤ 16 px de una detección (los anillos).
- **Adaptada con σ_dt × 2** (346 nm): sensibilidad, no auto-calibrada.

| variante | σ_dt | candidatas/cuadro (con el reflejo clásico) | pistas reales / de reflejo | pares | cuadros triangulados | reflejo correcto | real correcto |
|---|---|---|---|---|---|---|---|
| tal cual | 149 nm | 2.35 (85 cuadros) | 8 / 8 | 4 | 77 | 75 | 45 |
| adaptada | 173 nm | 1.27 (97) | 4 / 2 | 2 | 50 (cuadros 45–99) | 50 | 40 |
| adaptada, σ_dt × 2 | 346 nm | 1.27 (97) | 6 / 2 | 5 | 99 (falta el 86) | 97 | 81 |

- **Tal cual.**
  - Hay 2.35 manchas candidatas por cuadro. Las ~1.5 que sobran están sobre los anillos del real (a ~10 px), del lado del espejo y moviéndose rígidamente con él.
  - La pista del reflejo verdadero (87 cuadros, a 0.83 px del clásico) sí se forma. Pero la partícula se reparte en 3 pistas reales, y la mayoría de las demás pistas de la red siguen anillos.
  - En 32 de los 77 cuadros triangulados, el "real" es un anillo. Con β = 45° no importa, porque z = −s_reflejo, pero con β ≠ 45° entra con un factor (tan β − cot β)/2 = ∓0.18 a 40° y 50°.
- **Adaptada.**
  - Una sola pista de reflejo de 99 cuadros, a 0.84 px del clásico.
  - En los cuadros 0–47 dt ≈ +200 nm: la deriva de dt con z (+60 a +220 nm de media por bloque de 10 cuadros) más el sesgo de +70 nm de la red en t. Eso es 1.3σ_dt, y la contribución por cuadro al puntaje, C_LR − (dt/σ)²/2 con C_LR = 0.87, sale negativa. Por eso la pista de la partícula de esos cuadros (48 cuadros a 1.6 px del real) no se empareja.
  - Con σ_dt × 2 se emparejan todas: el primer pase, con σ_dt = 200 nm, daba también 99 cuadros. Sobre la rejilla de D (0.02, 0.2, 1.5 µm²/s) × σ_dt (×0.5, ×1, ×2), los cuadros triangulados van de 27 a 99. Dependen de σ_dt y también de D: con σ_dt = 173 nm son 50 y 52 con D = 0.02 y 0.2, y 98 con D = 1.5, porque un D grande deja enlazar la partícula en menos pistas.
- **Mancha emparejada frente al reflejo clásico** (adaptada × 2, 99 cuadros):
  - desfase (−68, +13) nm en (x, y) y +65 nm en s;
  - std 47 / 40 nm, RMS radial 93 nm.

  Es la diferencia entre el ajuste gaussiano en ±5 px y el centroide iterativo con r = 10 px de un reflejo que no es gaussiano (pupila recortada, `resultados_recorte_pupila.md`).

**El modelo t' = t no se cumple en estos datos.** Con el par clásico (`fig_tracking_yeast6_1_deteccion`, derecha):
- dt se correlaciona con la z triangulada (r = −0.89, pendiente −0.55 nm por nm de z), con s del real (+0.81) y con el tiempo (−0.73);
- al quitarle la parte lineal en z, su std baja de 160 a 73 nm.

Puede venir de un sesgo de los centroides a lo largo de t que depende del desenfoque (anillos asimétricos del real, recorte de pupila del reflejo) o de que la normal del espejo tenga una componente a lo largo de t. Es lo mismo que el ángulo de ~5° entre el borde de la franja y t que ya registraba `resultados_datos_reales.md`, sección 3. Para el emparejamiento significa que σ_dt tiene que absorber una deriva de ±200 nm en lugar del ruido de ~67 nm de la simulación. Un modelo dt = a + b·z por par lo evitaría.

## 3. z triangulada, barrido de β y atípicos

`tracking_yeast6_1_z_beta.csv`, `_atipicos_ventana.csv` y `fig_tracking_yeast6_1_z.png`.

**Fórmula.** z = −s_M/tan β + (L/2)·tan β, por cuadro, con β entre 40° y 50° cada 0.5°.

**Atípicos.** Se marca un cuadro cuando |z − mediana móvil de 9 cuadros| > k·σ, con σ = 1.4826 MAD de los residuos de todo el video. Es un criterio de consistencia temporal: **no usa la z de la red**.

| fuente | cuadros | z media, β 40 / 45 / 50 | std de z (β 45) | σ del residuo | atípicos k = 2.5 / 3 / 4 (β 45) | atípicos k = 3 en 40–50° |
|---|---|---|---|---|---|---|
| par clásico | 100 | −1813 / −836 / +115 nm | 260 nm | 30 nm | 22 / 18 / 7 | 7–18 |
| tal cual | 77 | −1795 / −839 / +92 | 271 | 79 | 4 / 2 / 0 | 1–4 |
| adaptada | 50 | −1707 / −752 / +179 | 254 | 57 | 2 / 2 / 1 | 2–5 |
| adaptada, σ_dt × 2 | 99 | −1894 / −904 / +59 | 284 | 49 | 8 / 5 / 2 | 2–8 |

- **Pipeline frente a par clásico, en los mismos cuadros:** z_pipeline − z_clásico tiene media −41 a −78 nm y RMS 68–113 nm según la variante y β (β = 45°: 68, 83 y 89 nm). Con 45° solo entra el reflejo, así que es exactamente el −Δs de la mancha (sección 2).
- **El criterio relativo es frágil.**
  - La z clásica es la más suave (σ del residuo 30 nm), así que su k·σ es el más estricto: rechaza el 18 % de los cuadros con k = 3, en su mayoría excursiones de pocos cuadros que la propia z clásica sigue (cruces en `fig_tracking_yeast6_1_z`). Con D_z ≈ 0.04 µm²/s el paso browniano en z es de ~90 nm por cuadro, del orden de esos residuos.
  - Depende mucho de la ventana: el clásico rechaza 22, 18 y 1 cuadros con 5, 9 y 15.
  - En la variante adaptada (50 cuadros con huecos), la ventana de 5 degenera (σ = 5 nm, 21 atípicos), porque con 3 valores la mediana suele ser el propio punto.

  Los conteos dicen qué cuadros se apartan de su entorno, no cuáles están mal. El MSD se reporta con y sin rechazo (sección 4).
- **β mueve la z absoluta ~190 nm por grado.** La línea del espejo la corre en δ/tan β; con el +1.52 µm de `resultados_calibracion_pupila.md`, sección 7, serían +1.5 µm. Ninguna de las dos cosas cambia el D: la línea es un desfase constante, y β cambia la escala (sección 4).

## 4. Trayectoria 3D y D por MSD

`tracking_yeast6_1_D.csv`, `_msd.csv`, `fig_tracking_yeast6_1_trayectoria.png` y `fig_tracking_yeast6_1_msd.png`.

**Qué entra en cada MSD:**
- lateral: todos los cuadros de las pistas reales de la partícula;
- z y 3D: los cuadros con z válida (sin los atípicos de z cuando se rechazan), con las (x, y) del real usado en la triangulación.

D en µm²/s, con β = 45° e IC 95 %:

| fuente | lateral | x / y | z sin atípicos | z con todos | 3D sin atípicos | ordenada de la lateral / z |
|---|---|---|---|---|---|---|
| par clásico | **0.078** [0.039, 0.111] | 0.066 / 0.089 | **0.040** [0.011, 0.079] | 0.047 [0.016, 0.092] | **0.075** [0.039, 0.112] | −3.5·10³ / −3.6·10³ nm² |
| tal cual | 0.033 [−0.078, 0.110] | 0.041 / 0.025 | 0.040 [0.010, 0.087] | 0.039 | 0.094 [0.038, 0.166] | 2.2·10⁵ / 1.5·10³ |
| adaptada (50 cuadros) | 0.156 [0.064, 0.326] | 0.205 / 0.108 | 0.053 [0.004, 0.127] | 0.056 | 0.100 [0.038, 0.189] | 4.4·10⁴ / −4.6·10³ |
| adaptada, σ_dt × 2 (99) | **0.079** [0.031, 0.169] | 0.060 / 0.099 | **0.055** [0.016, 0.108] | 0.058 [0.021, 0.107] | 0.052 [0.023, 0.102] | 2.4·10⁵ / −3.6·10³ |

- **Ordenadas.**
  - **Par clásico:** las ordenadas negativas son las que se esperan del desenfoque de movimiento con exposición = período (100 ms): −2D·t_E/3 por eje, unos −2.7·10³ nm² en z con D = 0.04. El ajuste con ordenada lo absorbe sin sesgar la pendiente. Lo que queda es compatible con un error de localización de ~40 nm lateral y ≲ 30 nm en z.
  - **Pipeline en lateral:** la ordenada de 0.4–2.4·10⁵ nm² (~150–250 nm por eje) son los saltos a los anillos. Hacen la pendiente lateral inestable: 0.033 tal cual, 0.156 adaptada. Con 99 cuadros (σ_dt × 2), la lateral vuelve a coincidir con la clásica (0.079 frente a 0.078), aunque con un IC el doble de ancho.
- **En z, el pipeline y el par clásico coinciden dentro de los IC** (0.040–0.055). Es lo que se espera, porque a 45° comparten casi todo el reflejo.
- **D_z en función de β** (`fig_tracking_yeast6_1_msd`, derecha; sin atípicos):

  | β | 40° | 42° | 44° | 45° | 46° | 48° | 50° |
  |---|---|---|---|---|---|---|---|
  | par clásico | 0.062 [0.019, 0.122] | 0.055 | 0.048 | 0.040 [0.011, 0.079] | 0.043 | 0.036 | 0.030 [0.007, 0.065] |
  | adaptada, σ_dt × 2 | 0.069 [0.025, 0.127] | 0.064 | 0.059 | 0.055 [0.016, 0.108] | 0.053 | 0.047 | 0.043 [0.010, 0.087] |

  D_z baja ~2× entre 40° y 50° porque los coeficientes de s_real y s_reflejo en z cambian con β. Con difusión isótropa, D_z = D_lateral (~0.078) pediría β ≤ 40°, en la dirección del β ≈ 38° que el acople en m sugería para este canal (`resultados_datos_reales.md`, sección 4). Pero los IC de D_z alcanzan 0.078 en todo el rango, y una partícula confinada en una levadura que toca el espejo no tiene por qué difundir igual en z que en el plano. **No se puede usar para fijar β.**
- **El 3D** del par clásico (0.075) es la media ponderada esperable de lateral y z. El del pipeline (0.052) usa las (x, y) de los cuadros triangulados, con los saltos a anillos (ordenada 1.9·10⁵ nm²), y no es comparable.

## Figuras (PNG 300 dpi, en `figuras/tracking_yeast6_1/`)

- `fig_tracking_yeast6_1_deteccion.png`:
  - izquierda: la ventana con las detecciones de la red (todas las de umbral 0.2 y las que quedan tras suprimir anillos), las manchas candidatas, el par clásico y los cuadros triangulados;
  - centro: s del real de la pista de la partícula menos s clásico (los saltos son anillos);
  - derecha: dt frente a la z triangulada, la violación de t' = t.
- `fig_tracking_yeast6_1_z.png`:
  - izquierda: z(t) triangulada con β = 45° (pipeline adaptado, adaptado × 2 y par clásico; banda del clásico para β de 40–50°), la mediana móvil y los atípicos marcados;
  - derecha: % de cuadros rechazados en función de β.
- `fig_tracking_yeast6_1_trayectoria.png`: trayectoria 3D (pipeline adaptado × 2 y par clásico), proyección lateral y z(t) sin atípicos, coloreadas por cuadro.
- `fig_tracking_yeast6_1_msd.png`: MSD por eje (lateral, z y 3D, normalizado por el número de ejes) con los ajustes de las variantes adaptada y adaptada × 2 y del par clásico, y D_z(β) con sus IC frente a D lateral.

## Decisiones y límites

- **Adaptaciones que no estaban en el pipeline simulado:**
  - supresión de anillos (16 px);
  - descarte de las detecciones que parecen el reflejo;
  - exclusión de manchas en 16 px;
  - enlace en 2D.

  Las tres primeras suponen partículas aisladas: con dos partículas a menos de ~1.6 µm, la supresión las fundiría. En la simulación no hicieron falta (`resultados_tracking_3d.md`).
- **σ_dt × 2 es una elección a posteriori**, hecha para mostrar cuánto se gana si σ_dt absorbe la deriva de dt. La versión que se puede defender es modelar dt = a + b·z en el emparejamiento.
- **Una partícula y 100 cuadros.** Los IC de D son de un factor 3–10 y no hay dispersión entre partículas. Los IC de bloques de 10 cuadros, con solo 10 bloques, son aproximados.
- **β y la línea no están calibrados** (`resultados_calibracion_espejo.md`). La z absoluta es indicativa y D_z depende de β en un factor 2 en [40, 50]°.
