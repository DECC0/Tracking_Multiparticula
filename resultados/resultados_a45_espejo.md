# Referencia A45: ¿aprende la CNN a usar el reflejo con β = 45° fijo?

**Estado:** tests generados y verificados; las cuatro corridas (A45 s0, s1; A45_sin s0, s1) están **aprobadas** y verificadas para el lanzamiento (abajo). Todavía no hay resultados de redes.

**Pregunta.** La red de referencia A (`hypatia_celula_b45_cerca_20261110-234627`) se entrenó con β sorteado en [40, 50]° y no usa el reflejo para z (`resultados_celula.md`). Con β fijo, ¿la misma CNN, con la misma pérdida, la misma entrada y los mismos datos salvo β, aprende a aprovechar la información axial del reflejo? La fusión explícita de la Etapa 0 (`resultados_fusion_espejo.md`) muestra que esa información existe: −29 nm sobre los emisores con reflejo localizable.

**Hipótesis.**
- **H1:** la red con reflejo (A45) es más precisa en z que una idéntica entrenada sin luz de reflejo (A45_sin), sobre las mismas escenas. Además, esa ventaja depende de que el reflejo esté en la imagen.
- **H0:** no hay diferencia más allá de la variabilidad entre semillas. El cuello de botella sería el aprendizaje de la correspondencia no local, no la falta de información sobre β.

## Configuraciones

| config | diferencias |
|---|---|
| `localizacion/configs/celula_b45fijo.yaml` (A45) | Frente a `celula.yaml`: `espejo_angulo_eje_rango_deg: null` (β = 45 en todas las muestras); `simulador.z_banco_nm: [-7596.266658713868, 3000.0]`, el banco de A (con β fijo se recalcularía a −7243 nm); `entrenamiento.nombre` |
| `localizacion/configs/celula_b45fijo_sin_reflejo.yaml` (A45_sin) | Lo mismo, más `ratio_reflejo_rango: [0, 0]`: el reflejo se simula y queda en el GT, pero sin luz |

- El flujo aleatorio de la geometría sigue sorteando sus 4 uniformes, así que φ y los demás sorteos son los mismos que con β variable.
- `ratio_reflejo` tiene su propio flujo (`spawn_key (idx, 1003)`). Por eso A45 y A45_sin ven **las mismas escenas de entrenamiento y validación** (semillas 1 y 2) y difieren solo en la luz del reflejo.

## Tests

Dos tests, ambos con 2000 imágenes y **semilla 4**, independiente de train (1), val (2) y del test anterior (3):
- `localizacion/datos/test_celula45fijo_b45_cerca`: con reflejo;
- `localizacion/datos/test_celula45fijo_sinrefl_b45_cerca`: las mismas escenas, sin luz de reflejo.

```bash
PY=~/miniforge3/envs/tesis/bin/python
$PY localizacion/generar_espejo.py --config localizacion/configs/celula_b45fijo.yaml --anillos-hacia-objetivo --split test --n 2000 --seed 4 \
    --salida localizacion/datos/test_celula45fijo_b45_cerca                 # 108 s
$PY localizacion/generar_espejo.py --config localizacion/configs/celula_b45fijo_sin_reflejo.yaml --anillos-hacia-objetivo --split test --n 2000 --seed 4 \
    --salida localizacion/datos/test_celula45fijo_sinrefl_b45_cerca         # 67 s
$PY figuras/a45_espejo.py verificar_tests                                    # 21 comprobaciones, ~15 s
```

- **Contenido:** 13 108 emisores, 13 051 en el FOV y 12 968 evaluables. 13 066 reflejos en el FOV. Cero reintentos y cero reflejos sin dibujar.
- **Nombres:** son los que busca `hypatia/espejo.sbatch` con `PREFIJO=celula45fijo` y `PREFIJO=celula45fijo_sinrefl`. Si el job los encuentra, no los regenera; si no, los regeneraría con la semilla 3 por defecto. **Hay que copiarlos a Hypatia antes de lanzar.**

## Verificaciones antes de entrenar

Todas pasan (`figuras/a45_espejo/a45_verificacion.csv`, 21 de 21):

| | comprobación | resultado |
|---|---|---|
| C1 | `celula.yaml` → `celula_b45fijo.yaml`: solo cambian β, `z_banco_nm` y el nombre | pasa |
| C2 | `celula_b45fijo` → `..._sin_reflejo`: solo cambian `ratio_reflejo_rango` y el nombre | pasa |
| C3 | config de la corrida A → `celula_b45fijo`: además, solo `workers` (2 en el YAML; el job lo pone en 8 = `CPUS`, como en A) | pasa |
| B1 | campos de PSF y rejilla z de A y de A45, con y sin reflejo: 531 planos de −7600 a 3000 nm, paso 20 nm | pasa, idénticos |
| B2 | `K_psf` usado: A, con y sin = 256 | pasa |
| B3 | stack del banco de A frente al de A45 (531 × 256 × 256) | pasa, **bit a bit** (diferencia máxima 0) |
| T1 | β = 45.0 exacto en las 2000 muestras de cada test | pasa |
| T3 | GT con/sin idéntico en todas las columnas salvo `photons_reflejo` (posiciones, fotones del real, D, `en_fov`, `en_rango_z`, posición y z del reflejo, `snr`) | pasa (13 108 filas, igualdad exacta) |
| T4 | `photons_reflejo`: con = ratio × photons (ratio en [0.45, 1.0]); sin = 0 | pasa |
| T5 | `muestras.csv` idéntico salvo `ratio_reflejo` (fondo, fotones, geometría, célula, reintentos) | pasa |
| I1 | imágenes de los tests = regeneradas desde la config (40 por test) | pasa, bit a bit |
| I2 | fondo y subposiciones del real y del reflejo idénticos con/sin | pasa |
| I3 | imagen sin ruido: con − sin = ratio × (luz del reflejo con ratio 1) | pasa (error relativo ≤ 8.7·10⁻¹⁶): **la única diferencia es la luz del reflejo, proporcional a su ratio** |
| I4 | ruido: la realización del Poisson y de la cámara difiere (98 % de los píxeles con distinto ADU) | informativo |
| E1 | muestras de entrenamiento y validación (20 por split): escenas con/sin idénticas | pasa |

Además, un entrenamiento de humo (1 época, 16 muestras, en el scratchpad) corre con las dos configs. `evaluate.py` lee los tests nuevos, incluido el canal de distancia, y escribe sus tablas. Con una red sin entrenar, sin detecciones, falla al dibujar `metricas.png` con eje log: es un artefacto de la red vacía, no del test.

**Comparabilidad.** Las escenas son las mismas: misma semilla, mismos sorteos y el mismo banco de PSF. Lo único que cambia entre los dos tests es la luz del reflejo, más una realización distinta del ruido. Por eso las diferencias se miden pareadas por emisor, con IC por bootstrap pareado de imágenes.

## Diseño factorial (evaluación)

Cada red se evalúa en los dos tests: M(red, imagen), con red ∈ {A45, A45_sin} e imagen ∈ {con, sin}, para cada semilla.

| efecto | definición | qué mide |
|---|---|---|
| de la imagen, con la red fija | I(red) = M(red, con) − M(red, sin) | cuánto cambia una red cuando aparece la luz del reflejo. Para A45, quitarlo la saca de su distribución; para A45_sin, agregarlo también |
| del entrenamiento, con la imagen fija | E(img) = M(A45, img) − M(A45_sin, img) | cuánto cambia la red por haberse entrenado con reflejo |
| interacción | I(A45) − I(A45_sin) = E(con) − E(sin) | si la ventaja de A45 depende de que el reflejo esté en la imagen |
| diagonal (dentro de distribución) | M(A45, con) − M(A45_sin, sin) = E(con) + I(A45_sin) | el valor de la información del reflejo para una red entrenada con ella, frente a una que nunca la tuvo |

**Métricas.** Las de `evaluate.py`: cilindro de 150/300 nm; recall, precisión, Jaccard, RMSE z y lateral, sesgo; FP sobre reflejos. Se reportan siempre de dos formas, sin elegir los casos favorables:
- **globales:** cada celda con todas sus detecciones;
- **pareadas sobre los mismos emisores:** los detectados en las celdas que se comparan, y además el conjunto común a las cuatro celdas de cada semilla.

Desgloses:
- z, fotones del real y del reflejo, `ratio_reflejo`;
- distancia al espejo y vecinos;
- cobertura y ganancia de la fusión D de la Etapa 0 sobre cada red.

**Uso del reflejo** (criterio principal, todas las condiciones):
1. diagonal < 0 en RMSE z pareado, con IC que excluye 0 y |efecto| > 2× la diferencia entre semillas;
2. I(A45) < 0 (a A45 le sirve el reflejo de la imagen);
3. interacción < 0 (la ventaja es específica del reflejo, no una mejora general por el entrenamiento);
4. Jaccard no inferior (Δ ≥ −0.01).

**Firmas de apoyo** (no son condiciones necesarias):
- **dosis–respuesta:** I(A45) crece con `ratio_reflejo` y con los fotones del reflejo, y es máxima donde la CRLB predice más ganancia (0.5–2 µm del espejo);
- **ganancia residual de D:** la fusión explícita mejora a A45 bastante menos de lo que mejoró a A (−29 nm);
- **eficiencia:** RMSE/CRLB conjunta deja de crecer hacia el espejo.

**Diagnóstico auxiliar S2 (no es una prueba por sí solo).** Desplazar el canal de distancia en inferencia y medir dz/dδ, contrastando con y sin reflejo. Ese canal también informa el prior de la célula, así que un efecto podría no ser triangulación. **No se exige correlación positiva entre los errores de A45 y los de la triangulación:** una red que use el reflejo puede hacerlo con su propia localización.

**Rechazo de H1:**
- la diagonal y la interacción quedan dentro del margen (±3 nm o la diferencia entre semillas, el mayor);
- I(A45) ≈ 0;
- la ganancia de D sobre A45 ≈ la que tuvo sobre A.

**Dos fuentes de incertidumbre, que se reportan por separado.**
- **Imágenes del test:** IC 95 % por bootstrap pareado de imágenes, con cada red fija. Mide cuánto cambiaría la cifra con otras 2000 escenas, no si el entrenamiento es estable.
- **Entre entrenamientos:** cada efecto se calcula por semilla (s0 con s0, s1 con s1) y se reporta para las dos, junto con su diferencia. Con dos semillas por condición, esa diferencia es la única estimación de la variabilidad de entrenamiento (un grado de libertad). Un IC de bootstrap estrecho no se interpreta como estabilidad entre semillas.

**Intermedio:** si E(sin) ≈ E(con) < 0, A45 es mejor pero no gracias al reflejo (por ejemplo, regularización). Si hay firmas pero el efecto es mucho menor que el de D, el uso es parcial.

## Semillas y costo

| | corridas | GPU-h en Hypatia | qué queda sin estimar |
|---|---|---|---|
| **propuesta: 2 semillas por condición** | A45 s0, s1; A45_sin s0, s1 | ~36.5 (4 × 9.1 h; ~9.5 h de reloj si van en paralelo) | con 2 semillas, la varianza entre semillas es una estimación tosca (un grado de libertad por condición) |
| alternativa: 3 corridas | A45 s0, s1; A45_sin s0 | ~27.5 | la variabilidad de A45_sin. La diagonal, E y la interacción suponen la misma varianza entre semillas en las dos condiciones, y no se puede saber si la única A45_sin salió buena o mala |

- 9.1 h por corrida es lo medido en A: 326 s por época × 100 épocas, con `CPUS=8`.
- Por corrida: `TIEMPO=0-14:00:00`, `MEMORIA` 24 G, `CPUS=8`.
- Local, después: 8 evaluaciones (4 redes × 2 tests) más A en los dos tests, la Etapa 0 sobre A45, la CRLB del test nuevo y el análisis. Del orden de 1–1.5 h (RTX 3060).

**Lanzamiento** (aprobado), desde la raíz del repo en Hypatia:
1. Las dos configs deben estar en la rama clonada allí (`prueba`).
2. Copiar antes los dos tests a `localizacion/datos/`.
3. Lanzar:

```bash
CONFIG=localizacion/configs/celula_b45fijo.yaml PREFIJO=celula45fijo EPOCAS=100 TIEMPO=0-14:00:00 CPUS=8 SEMILLA=0 bash hypatia/lanzar_espejo.sh cerca
CONFIG=localizacion/configs/celula_b45fijo.yaml PREFIJO=celula45fijo EPOCAS=100 TIEMPO=0-14:00:00 CPUS=8 SEMILLA=1 bash hypatia/lanzar_espejo.sh cerca
CONFIG=localizacion/configs/celula_b45fijo_sin_reflejo.yaml PREFIJO=celula45fijo_sinrefl EPOCAS=100 TIEMPO=0-14:00:00 CPUS=8 SEMILLA=0 bash hypatia/lanzar_espejo.sh cerca
CONFIG=localizacion/configs/celula_b45fijo_sin_reflejo.yaml PREFIJO=celula45fijo_sinrefl EPOCAS=100 TIEMPO=0-14:00:00 CPUS=8 SEMILLA=1 bash hypatia/lanzar_espejo.sh cerca
```

Antes de los cuatro `sbatch`, en Hypatia: `sha256sum -c figuras/a45_espejo/a45_tests_sha256.txt` (los 10 archivos de los dos tests deben dar `OK`; si alguno falla, no se lanza, porque el job regeneraría el test con la semilla 3 si falta `imagenes.npy`).

Corridas que se crean: `hypatia_celula45fijo_b45_cerca_s{0,1}` y `hypatia_celula45fijo_sinrefl_b45_cerca_s{0,1}`. Con `SEMILLA=0`, la inicialización y el orden de los datos son los de A, cuya semilla era 0. Cada job evalúa su `mejor.pt` solo en su test. La evaluación cruzada (factorial) se hace en local, después de traer las corridas.

## Verificaciones antes de lanzar

`$PY figuras/a45_espejo.py verificar_lanzamiento` (`figuras/a45_espejo/a45_verificacion_lanzamiento.csv`, 10 de 10 pasan). Reproduce los `--set` que agrega `hypatia/espejo.sbatch` (anillos cerca, `epocas=100`, `workers=8`, nombre, semilla; `EXTRA_SET` vacío):

| | comprobación | resultado |
|---|---|---|
| L0 | `espejo.sbatch` arma el nombre, la ruta del test y los overrides como se supone | pasa |
| L1 | misma condición, semilla 0 → 1: solo cambian `entrenamiento.semilla` y el nombre | pasa (con y sin) |
| L2 | misma semilla, A45 → A45_sin: solo cambian `ratio_reflejo_rango` y el nombre | pasa (s0 y s1) |
| L3 | corrida A → A45 s0 efectiva: solo β, `z_banco_nm` y el nombre (ahora también `workers` y `epocas` coinciden) | pasa |
| L4 | (`semilla`, `seed_train`, `seed_val`) = (s, 1, 2) en las cuatro: la semilla cambia solo la inicialización y el orden de los lotes (`train.py`), no los datos | pasa |
| L5 | nombres únicos; el glob `${NOMBRE}_*` con que el job busca su corrida no cruza las otras; ninguna corrida previa con esos nombres. `utils.carpeta_nueva` agrega fecha-hora y un sufijo si la carpeta existe, así que no se sobrescribe nada | pasa |
| L6 | el job busca `localizacion/datos/test_celula45fijo_b45_cerca` y `..._sinrefl_b45_cerca`, que existen con `imagenes.npy`, semilla 4 y n = 2000 | pasa |

Manifiesto sha256 de los tests: `figuras/a45_espejo/a45_tests_sha256.txt`.

## Limitaciones previstas

- Solo la variante cerca, en simulación, con β exactamente 45° tanto en entrenamiento como en evaluación. No dice nada de la robustez a un β real distinto de 45°.
- La red A sigue siendo de β variable. Compararla con A45 confunde β con la semilla, así que es contexto, no un endpoint.
- Con 2 semillas por condición, el margen de equivalencia es aproximado.
