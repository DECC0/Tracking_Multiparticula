# Combinar frames con y sin el reflejo: suavizador de Kalman/RTS (oráculo)

Generado con `figuras/temporal_espejo.py` (`~/miniforge3/envs/tesis/bin/python figuras/temporal_espejo.py`, ~75 s). No entrena ni corre la red.

- Figura: `figuras/fig_temporal_espejo.png` (y `.pdf`).
- Datos:
  - `figuras/temporal_espejo_rmse.csv`: RMSE z y lateral con IC 95% por rango, estimador, escenario y signo.
  - `figuras/temporal_espejo_D.csv`: D verdadera y estimada por trayectoria.

## Montaje del estudio

- **Trayectorias:** las del GT de `simulador_nuevo` con espejo.
  - 500 videos de 50 frames a 10 Hz, 1–12 emisores por video: 3381 trayectorias por signo.
  - D uniforme en [0.01, 0.3] µm²/s; fotones log-uniformes en [1000, 30 000].
  - α = 34.0° y φ = 0. La línea del espejo está en el borde izquierdo de un FOV de 64 × 96 px, para que haya reales en el FOV a todas las distancias de 0 a 4 µm; las posiciones iniciales están a menos de 4 µm del plano del espejo.
  - Se reproducen exactamente los sorteos de `generar_video` (comprobado contra su GT), sin renderizar imágenes.
  - El enlazado es perfecto (por `particle_id`).
- **Modelo del filtro:**
  - Estado: la posición 3D del real.
  - Dinámica: p_{k+1} = p_k + w, con w ~ N(0, 2·D·Δt·I) y D por partícula; el modelo no incluye paredes.
  - Medidas por frame:
    - el real, con covarianza diag(σ_lat², σ_lat², σ_z²);
    - si está en el FOV, el reflejo mapeado con la geometría exacta. Como R es una involución, la localización del reflejo equivale a una medida de p, R(y' − c), con covarianza R·Σ'·R. Así se suma su información a la del real; no es solo una corrección lateral.
  - Filtro de información más suavizado RTS.
- **Dos escenarios de ruido** (en los dos el ruido es gaussiano y su covarianza es conocida):
  - **`crlb`:** real y reflejo con su CRLB de emisor solo (fotones, z y fondo medio del video), siempre detectados si están en el FOV.
  - **`red` (el realista):**
    - Real: el RMSE lateral y z y el recall de la red con espejo de ese signo, medidos sobre su test por rango de fotones y de distancia al espejo.
    - Reflejo: un localizador hipotético. Su σ es su CRLB multiplicada por κ = (RMSE de la red / CRLB del real) a iguales fotones (κ_lat ≈ 1.2–2.5, κ_z ≈ 0.9–2.4), y se detecta con el recall de la red a 0.8·N fotones.
- **Estimadores:**
  - frame a frame: solo el real, o real + reflejo del mismo frame;
  - suavizado con D verdadera: solo el real, o real + reflejo;
  - suavizado con D estimada de la propia trayectoria (máxima verosimilitud o MSD).
- **Evaluación:** sobre los frames con el real en el FOV y detectado. IC 95% por bootstrap de trayectorias.

## Resultados

### RMSE z [nm], escenario `red`

En cada celda: frame real / frame real + reflejo / suavizado real / suavizado real + reflejo.

| variable | rango | signo_z = +1 | signo_z = −1 |
|---|---|---|---|
| D [µm²/s] | 0.01–0.05 | 65 / 50 / 45 / 35 | 64 / 49 / 45 / 34 |
| D | 0.05–0.1 | 67 / 53 / 53 / 42 | 66 / 50 / 53 / 40 |
| D | 0.1–0.2 | 67 / 52 / 58 / 45 | 65 / 51 / 57 / 44 |
| D | 0.2–0.3 | 65 / 52 / 59 / 47 | 64 / 50 / 59 / 45 |
| fotones | 1.0k–2.3k | 109 / 100 / 92 / 83 | 112 / 103 / 93 / 83 |
| fotones | 2.3k–5.5k | 85 / 65 / 69 / 54 | 77 / 55 / 64 / 46 |
| fotones | 5.5k–12.8k | 49 / 31 / 45 / 29 | 47 / 30 / 43 / 28 |
| fotones | 12.8k–30k | 29 / 18 / 28 / 18 | 31 / 19 / 30 / 18 |
| d al espejo [µm] | 0–1 | 65 / 47 / 56 / 40 | 67 / 49 / 57 / 39 |
| d | 1–2 | 66 / 47 / 56 / 40 | 64 / 43 / 55 / 36 |
| d | 2–3 | 66 / 53 / 56 / 44 | 63 / 47 / 54 / 40 |
| d | 3–4 | 66 / 57 / 56 / 49 | 64 / 58 / 55 / 50 |
| d | > 4 | 65 / 63 / 57 / 56 | 65 / 64 / 57 / 55 |

Los IC 95% son de ±1–3 nm (mediana ±1.6 nm; el máximo, ±3.5 nm, en el rango > 4 µm). El detalle está en el CSV.

### RMSE lateral [nm], escenario `red`

Mismo orden de columnas (frame real / frame + reflejo / suavizado real / suavizado + reflejo).

| rango | signo_z = +1 | signo_z = −1 |
|---|---|---|
| todos | 20.0 / 19.4 / 19.7 / 19.1 | 20.0 / 19.2 / 19.8 / 18.9 |
| 1.0k–2.3k fotones | 38 / 37 / 37 / 36 | 37 / 36 / 36 / 35 |
| 12.8k–30k fotones | 10 / 9 / 10 / 9 | 9 / 9 / 9 / 9 |

### RMSE z [nm], escenario `crlb` (cota con estimadores eficientes)

| variable | rango | signo_z = +1 | signo_z = −1 |
|---|---|---|---|
| D | 0.01–0.05 | 80 / 47 / 44 / 31 | 79 / 47 / 44 / 29 |
| D | 0.2–0.3 | 81 / 55 / 64 / 46 | 81 / 49 / 64 / 41 |
| fotones | 1.0k–2.3k | 140 / 94 / 96 / 71 | 141 / 84 / 96 / 63 |
| fotones | 12.8k–30k | 14 / 9 / 14 / 9 | 14 / 8 / 14 / 8 |
| d | 0–1 | 75 / 42 / 55 / 32 | 79 / 38 / 57 / 29 |
| d | 1–2 | 79 / 41 / 58 / 33 | 80 / 25 / 58 / 23 |
| d | 3–4 | 81 / 66 / 59 / 51 | 80 / 69 / 58 / 51 |

A pocos fotones, el frame a frame del escenario `crlb` (140 nm) es peor que el de `red` (109 nm). Hay dos razones: aquí se incluyen todos los reales, también los que la red no detecta, y la red aprende el prior z ∈ [−800, 800] nm, mientras que la CRLB no.

### Conclusiones

1. **En z, el reflejo aporta más que combinar frames, salvo a D muy baja.**
   - Con el reflejo, un solo frame baja el RMSE z de ~65 a ~50 nm (escenario `red`).
   - Suavizar 50 frames sin el reflejo lo baja a 45 nm con D < 0.05 µm²/s, pero solo a 53–59 nm con D ≥ 0.05.
   - Las dos cosas se suman: suavizado + reflejo da 34–35 nm con D < 0.05 y 45–47 nm con D 0.2–0.3.
2. **La ganancia del reflejo depende de la distancia al espejo y de los fotones.**
   - Es máxima entre 0 y 2 µm (frame a frame, 65 → 43–49 nm) y desaparece por encima de 4 µm (65 → 63), donde el reflejo está desenfocado 5–7 µm.
   - Con `signo_z = −1` es algo mayor entre 1 y 3 µm (el reflejo cae en −z', donde la PSF es más compacta). En el escenario `crlb` la diferencia es clara: 25 frente a 41 nm entre 1 y 2 µm.
   - A muchos fotones, el reflejo baja z de 29 a 18 nm y el suavizado no agrega nada (D ≥ 0.01 µm²/s da un paso por frame de ≥ 45 nm por eje, mayor que el error).
3. **El suavizado ayuda más cuando el paso browniano por frame es comparable o menor que el error de localización.**
   - Eso pasa en z con D baja y pocos fotones: 109 → 92 nm sin reflejo y 100 → 83 nm con reflejo, a 1000–2300 fotones.
   - **En lateral casi no hay ganancia** (20.0 → 19.1 nm en total): el error lateral (9–38 nm) es chico frente al paso por frame, √(2·D·Δt) = 45–245 nm por eje.
4. **El suavizado rellena los frames en que el real no se detecta** (~21% de los frames en el FOV en el escenario `red`), pero con errores grandes:
   - RMSE z de 171 nm con reflejo y 217 nm sin él; RMSE lateral de 310 y 430 nm.
   - Esos frames son sobre todo de pocos fotones, en huecos largos. Con 5.5k–12.8k fotones, el z de esos frames baja a 58 nm con reflejo (137 nm sin él).
5. **No conocer D casi no cuesta.**
   - D por máxima verosimilitud (verosimilitud de las innovaciones del filtro): el RMSE z empeora ≤ 0.3%.
   - D por MSD (lags 1–3, ruido de localización conocido restado, tolera huecos): empeora ≤ 2%.
   - Las D estimadas salen 5–7% bajas (mediana de D_est/D: 0.93 por ML y 0.95 por MSD; IQR 0.82–1.04 por ML y 0.82–1.10 por MSD). La causa principal son las paredes reflectantes (z = ±800 nm y el espejo), que el filtro no modela. En las trayectorias verdaderas, el MSD en z es 0.88 × 2·D·Δt (0.71 cerca de |z| = 800 nm), y el de x cerca del espejo es 0.93×. El MSD no da estimación en el 4–13% de las trayectorias (menos de 5 pares de frames detectados); a esas se les asigna la mediana.

### Limitaciones del oráculo

- **Enlazado perfecto** y **errores gaussianos con covarianza conocida y sin sesgo**. Los errores z de la red tienen colas y sesgos, y las detecciones perdidas no son independientes entre frames como aquí (Bernoulli con el recall por rango).
- **El localizador de reflejos es hipotético:** la red no localiza reflejos. Además, el estudio usa la geometría exacta y no modela el recorte de pupila del reflejo, que le quitaría fotones y podría desplazar su centroide con z'.
- **El filtro no modela las paredes** (z = ±800 nm y el espejo) **ni el desenfoque de movimiento**. El simulador usa `n_subpasos = 1`, así que el GT es la posición instantánea.
- La σ de la red depende aquí solo de los fotones y de la distancia al espejo, no de z. Viene de redes que vieron α ∈ [29.4°, 39.4°] y geometría aleatoria.
- La geometría (FOV de 96 px y línea en el borde) cambia la distribución de d respecto del test. Los números comparables son los de cada rango, no los globales.

## Propuesta: llevarlo a las detecciones de la red

Todo esto se puede hacer sin reentrenar la red.

1. **Enlazado (tracking por detección).**
   - **Asignación frame a frame:** usar una LAP (Jaqaman et al. 2008, el esquema de u-track). El costo de unir una detección con una pista es la distancia de Mahalanobis entre la detección y la predicción del mismo filtro de Kalman, con covarianza P_pred + Σ_det. Se agregan costos de nacimiento y muerte, y una segunda pasada que cierra huecos de hasta G frames (por ejemplo 3–5) con la covarianza propagada.
   - **D al enlazar:** al principio, una D poblacional alta (0.3 µm²/s) para no cortar pistas; después, la D estimada de cada pista por máxima verosimilitud.
   - **Densidad:** con 1–12 emisores en 64 × 64 px y un paso de hasta ~2.5 px por eje, los cruces son raros. Un umbral χ² sobre la innovación debería bastar para evitar uniones erróneas.
2. **Covarianza de cada detección.**
   - Calibrarla en simulación con la misma red: tablas del RMSE por fotones estimados, distancia a la línea del espejo y z, como las de este estudio.
   - Los fotones de cada detección se pueden estimar con su `conf` (masa) o con un ajuste local del PSF del banco.
   - Para las colas: actualizaciones robustas (innovaciones fuera de k·σ descartadas o con peso de Student-t).
3. **Medir el reflejo sin reentrenar.**
   - Para cada pista, predecir dónde debe estar su reflejo: p̂' = c + R·p̂, con su z' y su covarianza R·P·R.
   - Ajustar por máxima verosimilitud el PSF del banco en una ROI alrededor de esa predicción (x', y', z' y fotones, con el fondo local), y aceptar el resultado solo si pasa la compuerta χ².
   - La covarianza de esa medida sale de la Fisher del ajuste (como `crlb_3d`).
   - Esto reemplaza al "localizador hipotético" de este estudio. Una alternativa, si se reentrena, es una segunda salida 2D de reflejos en la red.
4. **Calibrar la geometría antes de usar el reflejo para z; es crítico.** Las sensibilidades de la triangulación, verificadas numéricamente:
   - un error δα en el ángulo sesga z en ≈ 2·m·δα, donde m es la distancia lateral del real a la línea del espejo en el plano focal. Con el valor medido redondeado (±0.5°), eso da 17 nm a 1 µm de la línea y 35 nm a 2 µm;
   - un error δ en la posición de la línea sesga z en δ·tan α ≈ 0.67·δ, es decir 34 nm por cada 50 nm.

   Esos sesgos son del mismo orden que la ganancia esperada. Propongo autocalibrar α, φ y la línea por máxima verosimilitud conjunta sobre muchas pistas (la z de la forma de la PSF del real es independiente del reflejo, así que la geometría es identificable), o con beads a z conocida. Antes de eso hay que confirmar `signo_z` (z-stack de una bead), que fija si el reflejo cae en +z' o −z'.
5. **Suavizado y D.** Correr RTS por pista con la D de máxima verosimilitud. El costo de estimar D es despreciable. Opcionalmente, restringir z a [−800, 800] nm (actualización truncada) y modelar el desenfoque de movimiento si la exposición es comparable a Δt (Berglund 2010).
6. **Validación, en este orden, sin entrenar:**
   1. Videos de `simulador_nuevo` con espejo (50 frames), pasados por la red con espejo, el enlazado y el filtro, con solo el real. Comparar con la fila "suavizado, real" de este estudio y medir los errores de enlazado contra `particle_id`.
   2. El mismo pipeline con el ajuste del reflejo, con la geometría exacta y después con la autocalibrada.
   3. Datos reales, sin GT: medir la consistencia (la z del real y la triangulada en los frames donde hay ambas, la blancura de las innovaciones y la distribución de D).
