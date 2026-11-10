# Información del reflejo sobre z: CRLB, triangulación y redes

> **Nota (ángulo del espejo):** estas tablas se calcularon con α = 34.4°, el valor por defecto anterior. El montaje está a 56° del eje óptico (valor medido, redondeado al grado), así que ahora α = 90 − 56 = 34.0°. `crlb_espejo.py` toma el valor por defecto y ya se volvió a correr con 34.0°: la CRLB conjunta cambia como mucho 1.4% (≤ 0.8 nm), y la CRLB del emisor solo y la triangulación (que usa la geometría de cada muestra del test) no cambian. Las figuras y CSV de `figuras/` están con 34.0°. Las redes con espejo vieron α ∈ [29.4°, 39.4°].

Generado con `figuras/crlb_espejo.py` (`~/miniforge3/envs/tesis/bin/python figuras/crlb_espejo.py`, ~35 s).

Figuras:
- `figuras/fig_crlb_espejo_sz+1.png` (y `.pdf`)
- `figuras/fig_crlb_espejo_sz-1.png` (y `.pdf`)

Datos:
- `figuras/crlb_espejo_rangos_sz±1.csv`: por rango (d, fotones), con CRLB, triangulación, RMSE de las redes, IC y diferencias.
- `figuras/crlb_espejo_sz±1.csv`: rejilla completa de la CRLB, 44 d × 9 z × 13 N × 5 fondos.

Redes comparadas, cada una sobre el test con espejo de su signo (`localizacion/datos/test_espejo_sz±1`, el mismo GT emisor por emisor):
- `hypatia_espejo_sz+1_20261109-234610` y `hypatia_espejo_sz-1_20261109-234650`: entrenadas con espejo (precisión 0.980, RMSE z global 65 y 63 nm).
- `hypatia_mezcla_renovar_largo_20261027-171202`: entrenada sin espejo, evaluada sobre los dos tests.

## Conclusión

Desde 2 µm las dos redes localizan z igual. Por debajo de 1 µm (y entre 1 y 2 µm con sz−1) la red con espejo es significativamente mejor, pero solo vuelve a su propio nivel lejos del espejo: la red sin espejo empeora junto al espejo por el reflejo superpuesto, y la red con espejo aprendió a no confundirse con él (la precisión sube de 0.73 a 0.98). No hay señal de que lo use para z.

Si lo usara, el RMSE bajaría justo donde la CRLB conjunta es 2–3× menor que la del emisor solo (0–2 µm: 4.5–6.6 nm frente a 15 nm a 13k–30k fotones). En cambio, la red queda ~2× por encima de la CRLB del emisor solo en todos los rangos.

## 1. Red con espejo frente a red sin espejo, rango por rango

Cada celda da el RMSE z [nm] de las dos redes (con espejo / sin espejo). La diferencia pareada es el RMSE z con espejo menos el sin espejo, calculado sobre los emisores que detectan las dos redes. Los IC 95% salen de 2000 remuestreos de imágenes (`video_id`), porque los emisores de una imagen no son independientes. La diferencia sobre los detectados de cada red (`dif_propia` en el CSV) da lo mismo, así que no es un efecto de supervivencia. En negrita, los IC que excluyen el 0 a favor de la red con espejo. La estimación puntual reproduce exactamente cada `por_dist_espejo_fotones.csv`.

| d (µm) | fotones | con/sin, sz+1 | dif. pareada sz+1 | con/sin, sz−1 | dif. pareada sz−1 |
|---|---|---|---|---|---|
| 0–1 | 1.0–2.3k | 114 / 117 | −1.9 [−9.0, 5.2] | 120 / 123 | −9.3 [−18.6, 0.0] |
| 0–1 | 2.3–5.5k | 78 / 88 | **−10.1 [−14.8, −5.6]** | 77 / 95 | **−17.0 [−22.6, −11.6]** |
| 0–1 | 5.5–12.8k | 52 / 57 | **−5.3 [−8.7, −1.8]** | 53 / 73 | **−16.5 [−21.5, −11.7]** |
| 0–1 | 12.8–30k | 32 / 41 | **−8.9 [−11.7, −6.4]** | 35 / 52 | **−15.5 [−19.2, −11.7]** |
| 1–2 | 1.0–2.3k | 112 / 112 | −0.1 [−5.5, 5.3] | 104 / 109 | −5.7 [−11.8, 0.5] |
| 1–2 | 2.3–5.5k | 82 / 82 | 0.5 [−2.7, 3.6] | 78 / 81 | **−3.5 [−6.8, −0.1]** |
| 1–2 | 5.5–12.8k | 54 / 53 | 0.5 [−1.9, 3.0] | 50 / 55 | **−7.8 [−11.0, −4.9]** |
| 1–2 | 12.8–30k | 27 / 30 | −1.5 [−3.0, −0.0] | 28 / 33 | **−5.3 [−7.4, −3.4]** |
| 2–3 | 1.0–2.3k | 113 / 114 | 2.6 [−2.9, 8.3] | 105 / 106 | 3.7 [−1.8, 9.3] |
| 2–3 | 2.3–5.5k | 87 / 84 | 4.0 [0.5, 7.6] | 81 / 78 | 3.3 [−0.3, 6.7] |
| 2–3 | 5.5–12.8k | 44 / 43 | 1.2 [−0.7, 3.3] | 42 / 42 | 0.2 [−1.6, 2.0] |
| 2–3 | 12.8–30k | 26 / 28 | −1.6 [−3.3, −0.1] | 25 / 25 | −0.0 [−1.3, 1.1] |
| 3–4 | 1.0–2.3k | 98 / 102 | 4.8 [−1.1, 10.8] | 118 / 108 | 2.6 [−7.9, 11.5] |
| 3–4 | 2.3–5.5k | 87 / 85 | 3.7 [−0.9, 8.3] | 73 / 73 | 0.1 [−4.3, 4.2] |
| 3–4 | 5.5–12.8k | 49 / 47 | 1.8 [−2.9, 6.7] | 45 / 44 | 2.2 [−2.1, 7.0] |
| 3–4 | 12.8–30k | 30 / 28 | 1.4 [−0.9, 3.6] | 34 / 32 | 2.2 [0.4, 4.3] |

- **Desde 2 µm las dos redes son iguales:** las diferencias quedan entre −2 y +4 nm, y casi todos los IC incluyen el 0.
- **Por debajo de 1 µm la red con espejo es claramente mejor:** 5–10 nm con sz+1 y 15–17 nm con sz−1, desde 2300 fotones. Con sz−1 también lo es entre 1 y 2 µm.
- **Esa mejora solo devuelve a la red con espejo a su propio nivel lejos del espejo.** Con sz−1 a 13k–30k fotones, su RMSE es 35 / 28 / 25 / 34 nm de 0–1 a 3–4 µm. La red sin espejo, en cambio, empeora junto al espejo: 52 nm a 0–1 µm frente a 25 nm a 2–3 µm.

## 2. CRLB con y sin reflejo, y triangulación, por rango

√CRLB_z [nm], como RMS sobre z ∈ [−800, 800] nm, cinco fondos (quintiles del fondo medio del test: 16, 27, 35, 43 y 54 fotones/px) y las d y N de cada rango. La cota del emisor solo no depende del signo. La triangulación es un RMSE: z por mínimos cuadrados ponderados a partir de las posiciones laterales del GT, con real + N(0, 14 nm) y reflejo + N(0, su CRLB lateral), usando la geometría exacta de cada muestra. Incluye solo los reales con el reflejo en el FOV (6471 de 7997 con sz+1 y 6464 de 8030 con sz−1).

| d (µm) | fotones | solo | con reflejo +1 | triang. +1 | con reflejo −1 | triang. −1 |
|---|---|---|---|---|---|---|
| 0–1 | 1.0–2.3k | 119 | 44 | 34 | 46 | 36 |
| 0–1 | 2.3–5.5k | 59 | 23 | 17 | 23 | 17 |
| 0–1 | 5.5–12.8k | 31 | 13 | 9.6 | 13 | 9.6 |
| 0–1 | 12.8–30k | 15 | 6.4 | 7.2 | 6.6 | 7.3 |
| 1–2 | 1.0–2.3k | 119 | 51 | 72 | 31 | 36 |
| 1–2 | 2.3–5.5k | 59 | 25 | 34 | 16 | 18 |
| 1–2 | 5.5–12.8k | 31 | 13 | 16 | 8.8 | 10 |
| 1–2 | 12.8–30k | 15 | 6.3 | 8.9 | 4.5 | 7.3 |
| 2–3 | 1.0–2.3k | 119 | 77 | 126 | 59 | 80 |
| 2–3 | 2.3–5.5k | 59 | 38 | 57 | 29 | 40 |
| 2–3 | 5.5–12.8k | 31 | 20 | 24 | 16 | 17 |
| 2–3 | 12.8–30k | 15 | 9.0 | 12 | 7.8 | 10 |
| 3–4 | 1.0–2.3k | 119 | 97 | 217 | 102 | 209 |
| 3–4 | 2.3–5.5k | 59 | 47 | 107 | 50 | 101 |
| 3–4 | 5.5–12.8k | 31 | 25 | 42 | 27 | 43 |
| 3–4 | 12.8–30k | 15 | 12 | 21 | 13 | 21 |

- La CRLB conjunta mejora la del emisor solo 2.5–4× entre 0.1 y 1.5 µm. La ganancia baja a ~1.1× a 4 µm, donde el reflejo está desenfocado 5.8–7.4 µm.
- Con sz−1 la ganancia dura más (hasta ~2 µm), porque el reflejo cae en −z, donde la PSF esférica es compacta.
- La triangulación queda en 7–10 nm a muchos fotones hasta 2 µm. Puede quedar bajo la CRLB conjunta porque el real usa 14 nm fijos (mejor que su CRLB lateral a pocos fotones) y no hay superposición de PSF. Desde 3 µm es peor que la forma de la PSF sola, así que hay que combinarla con z_forma, no reemplazarla.

## 3. CRLB por distancia, en una rejilla más fina

√CRLB_z solo / con reflejo [nm], como RMS sobre z ∈ [−800, 800] y los cinco fondos.

### signo_z = +1

| d (nm) | N=1000 | N=2340 | N=5477 | N=12819 | N=30000 | z' (µm) |
|---|---|---|---|---|---|---|
| 0 | 197 / 722 | 86 / 321 | 39 / 149 | 18 / 73 | 9 / 38 | −0.8..+0.8 |
| 50 | 197 / 93 | 86 / 42 | 39 / 20 | 18 / 9.7 | 9 / 5.2 | −0.7..+0.9 |
| 100 | 197 / 55 | 86 / 25 | 39 / 12 | 18 / 5.9 | 9 / 3.2 | −0.6..+1.0 |
| 200 | 197 / 50 | 86 / 23 | 39 / 11 | 18 / 5.5 | 9 / 3.0 | −0.5..+1.1 |
| 500 | 197 / 49 | 86 / 22 | 39 / 11 | 18 / 5.4 | 9 / 2.9 | +0.0..+1.6 |
| 1000 | 197 / 54 | 86 / 24 | 39 / 11 | 18 / 5.7 | 9 / 3.0 | +0.9..+2.5 |
| 1500 | 197 / 78 | 86 / 34 | 39 / 15 | 18 / 7.3 | 9 / 3.7 | +1.7..+3.3 |
| 2000 | 197 / 109 | 86 / 47 | 39 / 21 | 18 / 9.6 | 9 / 4.6 | +2.5..+4.1 |
| 2500 | 197 / 130 | 86 / 56 | 39 / 25 | 18 / 11 | 9 / 5.3 | +3.3..+4.9 |
| 3000 | 197 / 144 | 86 / 63 | 39 / 28 | 18 / 13 | 9 / 5.9 | +4.2..+5.8 |
| 3500 | 197 / 162 | 86 / 70 | 39 / 31 | 18 / 14 | 9 / 6.9 | +5.0..+6.6 |
| 4000 | 197 / 180 | 86 / 79 | 39 / 35 | 18 / 16 | 9 / 7.9 | +5.8..+7.4 |

### signo_z = −1

| d (nm) | N=1000 | N=2340 | N=5477 | N=12819 | N=30000 | z' (µm) |
|---|---|---|---|---|---|---|
| 0 | 197 / 722 | 86 / 321 | 39 / 149 | 18 / 73 | 9 / 38 | −0.8..+0.8 |
| 50 | 197 / 94 | 86 / 42 | 39 / 20 | 18 / 9.8 | 9 / 5.2 | −0.9..+0.7 |
| 100 | 197 / 56 | 86 / 25 | 39 / 12 | 18 / 6.0 | 9 / 3.3 | −1.0..+0.6 |
| 200 | 197 / 52 | 86 / 24 | 39 / 11 | 18 / 5.7 | 9 / 3.1 | −1.1..+0.5 |
| 500 | 197 / 52 | 86 / 23 | 39 / 11 | 18 / 5.7 | 9 / 3.1 | −1.6..−0.0 |
| 1000 | 197 / 48 | 86 / 22 | 39 / 10 | 18 / 5.3 | 9 / 2.9 | −2.5..−0.9 |
| 1500 | 197 / 47 | 86 / 21 | 39 / 10 | 18 / 5.2 | 9 / 2.9 | −3.3..−1.7 |
| 2000 | 197 / 63 | 86 / 28 | 39 / 13 | 18 / 6.5 | 9 / 3.5 | −4.1..−2.5 |
| 2500 | 197 / 97 | 86 / 43 | 39 / 20 | 18 / 9.4 | 9 / 4.9 | −4.9..−3.3 |
| 3000 | 197 / 141 | 86 / 62 | 39 / 28 | 18 / 13 | 9 / 6.5 | −5.8..−4.2 |
| 3500 | 197 / 173 | 86 / 76 | 39 / 34 | 18 / 16 | 9 / 7.9 | −6.6..−5.0 |
| 4000 | 197 / 189 | 86 / 82 | 39 / 37 | 18 / 17 | 9 / 8.6 | −7.4..−5.8 |

**Fila d = 0.** Ahí real y reflejo coinciden, y un paso a lo largo de la normal n los mueve en sentidos opuestos. A primer orden la información sobre d se cancela, salvo por la diferencia de amplitud (1 − 0.8), y por eso la cota se dispara. La CRLB no incluye el prior d ≥ 0. El efecto se limita a los primeros ~50 nm.

## Método

- **CRLB del emisor solo:** el mismo cálculo que `crlb_3d` de `simulador_nuevo/simulador.py`, con parámetros (x, y, z, N, b) y ruido Poisson + lectura (Orca, 1.5 e-). Coincide con `crlb_3d` con tolerancia 1e-9, comprobado por una aserción en el script.
- **CRLB conjunta:** μ = b + N·g(p) + r·N·g(p0 + R(p − p0)), con r = `ratio_reflejo` = 0.8, α = 34.4° y φ = 0. La PSF del reflejo se mueve con p a través de R, y las dos comparten fondo y ruido. El jacobiano se comprobó contra diferencias finitas del modelo conjunto. La ROI es la caja que contiene al real y al reflejo, más 40 px por lado; no hay recorte por el FOV ni otros emisores.
- **Banco de PSF:** esférica, z en [−8000, 8000] nm cada 20 nm, K = 256.
- **Triangulación:** se comprobó que la geometría de `muestras.csv` reproduce los reflejos del GT (error < 1e-6 nm). La CRLB lateral del reflejo se interpola en una tabla de (fotones, z', fondo). Se hicieron 50 sorteos de ruido por emisor.
- **Redes:** el RMSE z se calcula sobre los detectados, lo que introduce sesgo de supervivencia a pocos fotones (recall ~0.35–0.44 en 1.0–2.3k). La red además aprende el prior z ∈ [−800, 800], que la CRLB no incluye; por eso puede igualar o mejorar la cota sin reflejo a pocos fotones.

## Limitaciones

- **El reflejo se modela como una fuente puntual vista con la pupila completa.** El espejo está a 56° del eje óptico (medido, redondeado; prácticamente arcsin(NA/n) = 55.6°), así que parte del cono de colección del reflejo no pasa por el espejo. Ese recorte de pupila no está modelado: le quitaría fotones al reflejo, volvería asimétrica su PSF y desplazaría su centroide con z'. Eso sesgaría la triangulación si no se modela.
- **Espejo plano ideal e infinito, con `ratio_reflejo` constante.**
- **La CRLB usa α fijo (34.4° en estas tablas; 34.0° en los CSV actuales)**, mientras que el test sortea α entre 29.4° y 39.4°. La triangulación sí usa la geometría de cada muestra.

## Próximo paso propuesto

Hacer un control con `ratio_reflejo = 0` sobre las mismas posiciones (mismos seed/idx y geometría, con el reflejo sin dibujar) y evaluar la red con espejo con ratio 0.8 y con ratio 0, rango por rango. Si el RMSE z no cambia, queda cerrado que la red no usa el reflejo. Solo hace falta inferencia, más dos cambios chicos:
- aceptar 0 en `Config.validar`, que hoy exige ratio en (0, 1];
- agregar `--ratio-reflejo` a `localizacion/generar_espejo.py`.

El `--sin-espejo` actual no sirve como control porque cambia las posiciones.
