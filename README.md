# Tracking_Multiparticula

## Convencion de z

Todo el codigo (`simulador_psf`, `simulador_nuevo`, `geometria_espejo.py`,
`localizacion`, figuras) usa la convencion del asesor:

- objetivo ARRIBA y **+z HACIA el objetivo**, fijo (z = 0 en el plano focal);
- el angulo del espejo medido desde el **EJE OPTICO**:
  `angulo_espejo_eje_deg` (beta, en (-90, 90); 45 por defecto, el montaje);
- `anillos_hacia_objetivo` (bandera de la PSF, False por defecto): de que
  lado del foco gana anillos la PSF esferica. Es la unica incognita fisica y
  se decidira con un z-stack de beads (seccion Espejo).

Antes (hasta el commit `02d8be8`) era al reves: +z lejos del objetivo,
`angulo_espejo_deg` = alpha medido desde el plano focal, y `signo_z` /
`signo_inclinacion`. Equivalencias:

| convencion anterior                      | convencion actual                        |
|------------------------------------------|------------------------------------------|
| `angulo_espejo_deg` = alpha              | `angulo_espejo_eje_deg` = 90 - alpha     |
| `signo_z = +1`                           | `anillos_hacia_objetivo = False`, z = -z_anterior |
| `signo_z = -1`                           | `anillos_hacia_objetivo = True`, z = z_anterior   |
| `signo_inclinacion = -1`                 | `angulo_espejo_eje_deg` -> -beta         |

Con el espejo apagado y `anillos_hacia_objetivo = False`, las imagenes con la
misma semilla son identicas bit a bit a las de antes y en el GT solo cambia el
signo de `z_nm` (`tests/test_convencion_z.py` lo verifica contra `02d8be8`).

**Hay que reentrenar.** Las corridas (`localizacion/corridas/`) y los datasets
(`localizacion/datos/`, `config.json` de `simulador_nuevo`) anteriores usan la
convencion vieja y no se convierten: `cargar_config` y `cargar_checkpoint` los
rechazan (no tienen el campo `convencion_z`). Sus evaluaciones (CSV, figuras,
`resultados_*.md`) siguen siendo legibles, con z en la convencion vieja.
Para entrenar con el montaje nuevo: `hypatia/README.md`, seccion espejo.

`sim_imagenes.py`, `sim_videos.py` (y `DOCUMENTACION_VIDEOS.md`) no cambian:
conservan la z vieja (+z lejos del objetivo).

## Simulador (`simulador_psf/`)

Videos sinteticos de fluorescencia (un solo canal) con emisores en movimiento
browniano 3D y PSF con aberracion esferica (Zernike Z4^0), mas el ground truth
de cada emisor. Montaje: NA 1.1, agua (n = 1.333), 530 nm, pixel de camara
6.5 um, 66x -> 98.48 nm/px.

```bash
PY=~/miniforge3/envs/tesis/bin/python
$PY generar_dataset.py         # -> videos/video_XXX.tif + ground_truth.csv en el directorio de trabajo
```

### Flujo

```
config.Config  (parametros)
      |
posiciones     n emisores, (x, y, z) iniciales, fotones por emisor
      |
browniano      pasos sqrt(2 D dt) N(0,1) por eje; z con paredes reflectantes
      |
optica         pupila con aberracion esferica -> banco de PSF en z -> N * PSF(z)
      |        colocada con desplazamiento subpixel por Fourier
      + fondo  fondo de fluorescencia (fotones/px), fijo por video
      |
ruido          Poisson(senal + fondo) -> / ganancia -> + lectura -> + offset -> uint16
      |
io_gt          videos/video_XXX.tif + ground_truth.csv
```

`generacion.py` orquesta el flujo con un solo `np.random.Generator` que se pasa
explicitamente; su docstring fija el orden en que se consume (no cambiarlo:
`checks/check_e2e.py` y `figuras/figuras_documento.py` lo repiten).
`diagnosticos.py` (energia encerrada, SNR pico) no interviene en la generacion.

### Salidas

- `videos/video_XXX.tif`: (T, H, W) uint16 en ADU, ImageJ con ejes TYX.
- `ground_truth.csv`: `video_id, frame, particle_id, x_px, y_px, z_nm, photons, D_um2_s, en_fov`,
  una fila por (video, frame, particula). x, y en px continuos (el pixel de la
  fila i y la columna j esta centrado en x = j, y = i), z en nm. Las particulas
  fuera del FOV se conservan con `en_fov=False`.

### Cambios respecto de `sim_imagenesPSF.py`

El codigo de `sim_imagenesPSF.py` se reorganizo en el paquete sin cambiar su
comportamiento: con la misma semilla las salidas son identicas bit a bit.

- El punto de entrada es `generar_dataset.py` (antes `python sim_imagenesPSF.py`).
- `sim_imagenesPSF.py` queda como capa de compatibilidad que solo reexporta
  desde `simulador_psf`; ejecutarlo ya no genera nada.
- Se elimino la mascara corkscrew (`fase_corkscrew`, `angulo_lobulo`,
  `Config.anillos_corkscrew`): `BancoPSF` lanza `ValueError` con
  `mascara="corkscrew"`. Quedan `"esferica"` (por defecto) y `"ninguna"`.

### Espejo (vista reflejada, `simulador_psf/espejo.py`)

La geometria (normal, reflexion, plegado, rango de z', franja de `dist_max_espejo_um`)
esta en `geometria_espejo.py`, en la raiz, compartida con `simulador_nuevo`;
`simulador_psf/espejo.py` solo la lee de `Config` y agrega las unidades del GT.
Diagramas (objetivo arriba, +z hacia el) en el docstring de `simulador_psf/espejo.py`.

Con `Config(usar_espejo=True)` hay un espejo plano junto a la muestra y la
camara ve cada emisor real y su imagen virtual. Solo se simulan las
trayectorias reales; en cada frame el reflejo es
`p' = p - 2 [(p - p0) . n] n` (en nm), asi que `dp' = R dp` con
`R = I - 2 n n^T` y nunca tiene browniano propio. El espejo no consume el rng
y con `usar_espejo=False` sus parametros no cambian nada.

- Montaje (grupo de Biofisica): sin cubreobjetos, celulas pegadas al espejo y
  el espejo detras de la muestra vista desde el objetivo, a **45 grados del
  eje optico**. Segun un analisis del asesor, la informacion axial del
  reflejo, proporcional a S sin^2(2 beta) / (1/R + cos^2(2 beta)), es maxima
  cerca de 45 grados, a costa de perder ~6% de la luz del reflejo por el
  recorte de pupila (con el montaje anterior, a 56 grados medidos, ~ 55.6 =
  arcsin(NA/n), la coleccion era completa). El espejo se define respecto del
  plano focal, y `z_rango_nm` solo delimita el volumen simulado.
- Parametros: `angulo_espejo_eje_deg` (beta: angulo entre el espejo y el eje
  optico, en (-90, 90), sin los extremos donde el espejo queda paralelo al
  plano focal; beta > 0: espejo detras de la muestra, el montaje; 0: pared
  paralela al eje; 45 por defecto), `orientacion_espejo_deg` (azimut, de +x
  hacia +y, de la direccion desde la linea del espejo hacia la muestra),
  `espejo_punto_px` (un punto de esa linea en el plano de referencia),
  `z_ref_espejo_nm` (z del plano de referencia, +z hacia el objetivo; 0 por
  defecto: la linea es la interseccion del espejo con el plano focal),
  `ratio_reflejo` (fotones del reflejo / del real, en (0, 1], 0.8) y
  `dist_max_espejo_um` (si se da, las posiciones iniciales se sortean solo
  hasta esa distancia perpendicular al plano del espejo, dentro del FOV
  ampliado, reescalando los mismos sorteos; None -> todo el FOV).
- Geometria: `n = cos(beta) m + sin(beta) z` (normal hacia la muestra), asi
  que `p' - p = -2 d n`: separacion lateral `2 d cos(beta)` hacia la linea del
  espejo y diferencia axial `-2 d sin(beta)` (con beta > 0 el reflejo queda
  siempre mas lejos del objetivo); `tan(beta) = -dz / dlat`. beta -> -beta es
  la misma configuracion espejada en z (lo verifica `tests/test_espejo.py`).
- El espejo es una pared reflectante mas para el browniano: los emisores
  reales nunca lo cruzan (posiciones iniciales plegadas al lado de la muestra,
  pasos integrados uno a uno con reflexion especular).
- El reflejo se renderiza con la misma optica en su (x', y', z') con
  `ratio_reflejo * photons`; el ruido se aplica una vez sobre la suma. El banco
  de PSF se amplia solo (`z_banco_nm=None`) para cubrir las z' de la caja de
  posiciones iniciales (a 45 grados con el FOV de 64 px: [-4640, 800] nm, 273
  planos); un reflejo que aun asi queda fuera no se renderiza.
- Ground truth: con espejo se agregan `x_reflejo_px, y_reflejo_px,
  z_reflejo_nm, photons_reflejo, reflejo_en_fov, reflejo_en_banco` en la
  misma fila (`particle_id` vincula real y reflejo).

**PSF: de que lado del foco salen los anillos (`anillos_hacia_objetivo`,
pendiente).** La inclinacion esta confirmada: el espejo queda detras de la
muestra vista desde el objetivo, asi que el reflejo esta siempre mas lejos
del objetivo que la particula (beta > 0). Lo pendiente es de la PSF: la fase
`k q cos(theta)` del modelo de pupila no esta atada a una direccion fisica,
asi que el banco se indexa por `q = -z` (`anillos_hacia_objetivo = False`, por
defecto: la esferica gana anillos al alejarse el emisor del objetivo) o
`q = +z` (`True`: al acercarse). Equivale al `signo_z` anterior: `+1` <->
`False`, `-1` <-> `True`.

En los datos reales la particula es la mancha mas brillante y con anillos, y
el reflejo (mas lejos del objetivo) la mas tenue, compacta y con lobulos. Con
`False` el reflejo cae del lado de los anillos, al reves de lo observado; con
`True`, del lado compacto: **el frame real apunta a `True`, sin confirmar**,
porque el truncamiento de la pupila del reflejo (abajo) tambien
cambia su forma (los lobulos, que no salen de una PSF de simetria circular,
apuntan a eso). Se decidira con un z-stack de beads: si los anillos aparecen
al acercar la bead al objetivo (+z), `True`; si aparecen al alejarla, `False`.
Hasta entonces el valor por defecto del simulador (`False`) no se cambia.

**Decision para entrenar (2026-11-10): la corrida principal es "cerca"
(`anillos_hacia_objetivo = True`)**, con "lejos" solo como respaldo
(`bash hypatia/lanzar_espejo.sh cerca|lejos`; las configs dejan `false` y la
variante la fija el lanzador). Evidencia (`resultados_recorte_pupila.md`,
seccion 6, y `resultados_datos_reales.md`): en el frame real el reflejo, a
z' ~ z - 4.5 um, es compacto, y la fotometria da un cociente reflejo/real de
~0.43 > 0. Con "lejos" un reflejo a esa z' es un sistema de anillos de ~8 um
sin nucleo (`figuras/fig_recorte_pupila_psf.png`): la misma fotometria sobre
imagenes simuladas da un cociente negativo o casi nulo para casi cualquier z
del real, y el ajuste del cociente contra el radio deja residuos 3 a 4 veces
mayores que con "cerca". Con "cerca" el reflejo es compacto y el cociente
cae en el rango medido. **Falta confirmarlo con el z-stack de beads**
(`resultados_calibracion_espejo.md`): es una sola particula del montaje
anterior y su mascara de fase puede no ser la del simulador.

Supuestos (no modelados): espejo plano ideal sin aberraciones ni
polarizacion; una sola perdida constante `ratio_reflejo` (reflectividad y
coleccion); el reflejo es una fuente puntual vista por la misma optica, sin
el truncamiento de la pupila (a 45 grados parte del cono de coleccion del
reflejo no llega por el espejo: 4.7% de su luz con pupila uniforme; hace
asimetrica su PSF y desplaza su centroide con z'; con beta <= 0 el espejo no
manda luz al objetivo por el eje); espejo infinito; iluminacion uniforme (no
se modela la hoja de luz). El truncamiento si esta modelado, opcional, en
`simulador_nuevo` (`recorte_pupila_reflejo`, mascara w . n > 0 en la pupila
del reflejo, con espejo infinito; ver su README), y tambien el del real
(`recorte_pupila_real`: con la misma mascara pierde los rayos que chocan con
el espejo, que es un portaobjetos espejado de varios mm). Los dos estan
activos en `localizacion/configs/espejo.yaml` y `celula.yaml`.

### Verificacion

```bash
$PY checks/run_all_checks.py   # checks con figuras (sin aserciones: fallan solo si lanzan)
$PY tests/regresion.py         # regresion bit a bit contra tests/referencia/
$PY tests/regresion.py --modulo sim_imagenesPSF   # idem sobre la capa de compatibilidad
$PY tests/test_convencion_z.py # convencion de z contra el codigo anterior (02d8be8), ~10 s
```

`tests/referencia/` se escribio con el codigo anterior a la reorganizacion y
se regenero al cambiar la convencion de z (videos iguales, z negada; lo
verifica `tests/test_convencion_z.py`). `--generar` la reescribe; hacerlo solo
si se cambia el comportamiento a proposito.
