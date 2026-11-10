# Tracking_Multiparticula

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

Con `Config(usar_espejo=True)` hay un espejo plano junto a la muestra y la
camara ve cada emisor real y su imagen virtual. Solo se simulan las
trayectorias reales; en cada frame el reflejo es
`p' = p - 2 [(p - p0) . n] n` (en nm), asi que `dp' = R dp` con
`R = I - 2 n n^T` y nunca tiene browniano propio. El espejo no consume el rng
y con `usar_espejo=False` las salidas son identicas bit a bit a las de antes.

- Montaje (grupo de Biofisica): sin cubreobjetos, celulas pegadas al espejo y
  el espejo detras de la muestra vista desde el objetivo. El espejo se define
  respecto del plano focal, y `z_rango_nm` solo delimita el volumen simulado
  (el espejo lo corta a cualquier altura).
- Parametros: `angulo_espejo_deg` (angulo espejo-plano focal dentro de la
  cuna de la muestra, en (0, 180); medido desde el plano focal; si se mide
  desde el eje optico (beta = 56 grados en el montaje actual, valor medido
  redondeado al grado), `alpha = 90 - beta`, y 34.0 es el valor por defecto;
  56 grados coincide, dentro de ese redondeo, con el semiangulo del cono de
  coleccion arcsin(NA/n) = arcsin(1.1/1.333) = 55.6 grados), `orientacion_espejo_deg` (azimut, de +x hacia
  +y, de la direccion desde la linea del espejo hacia la muestra),
  `espejo_punto_px` (un punto de esa linea en el plano de referencia),
  `z_ref_espejo_nm` (z del plano de referencia respecto del foco; 0 por
  defecto: la linea es la interseccion del espejo con el plano focal),
  `ratio_reflejo` (fotones del reflejo / del real, en (0, 1], 0.8) y
  `dist_max_espejo_um` (si se da, las posiciones iniciales se sortean solo
  hasta esa distancia perpendicular al plano del espejo, dentro del FOV
  ampliado, reescalando los mismos sorteos; None -> todo el FOV).
- Convencion de signos: `Config.signo_z` (+1: +z lejos del objetivo) y `Config.signo_inclinacion`
  (+1: con alpha < 90 el espejo se inclina sobre la muestra), +1 por defecto; diagramas en el
  docstring de `espejo.py`. Solo entran como su producto: `signo_z = -1` equivale a
  `signo_inclinacion = -1` y a alpha -> 180 - alpha (lo verifica `tests/test_espejo.py`).
- El espejo es una pared reflectante mas para el browniano: los emisores
  reales nunca lo cruzan (posiciones iniciales plegadas al lado de la muestra,
  pasos integrados uno a uno con reflexion especular).
- El reflejo se renderiza con la misma optica en su (x', y', z') con
  `ratio_reflejo * photons`; el ruido se aplica una vez sobre la suma. El banco
  de PSF se amplia solo (`z_banco_nm=None`) para cubrir las z' de la caja de
  posiciones iniciales (a 34.0 grados con el FOV de 64 px: [-800, 4600] nm, 271
  planos); un reflejo que aun asi queda fuera no se renderiza.
- Ground truth: con espejo se agregan `x_reflejo_px, y_reflejo_px,
  z_reflejo_nm, photons_reflejo, reflejo_en_fov, reflejo_en_banco` en la
  misma fila (`particle_id` vincula real y reflejo).

**Signos: signo_inclinacion confirmado, signo_z pendiente.** En el montaje
las celulas estan pegadas al espejo y el espejo queda detras de la muestra
vista desde el objetivo, asi que el reflejo esta siempre mas lejos del
objetivo que la particula: `signo_inclinacion = +1` queda confirmado
(z' - z = 2 d cos(alpha) * signo_inclinacion * signo_z > 0 con `signo_z = +1`).
Sigue la discrepancia de los anillos: en los datos reales la particula es la
mancha mas brillante y con anillos, y el reflejo la mas tenue, compacta y con
lobulos; en la galeria de `checks/fig_espejo_comparacion.png` la PSF esferica
del codigo es compacta en -z y gana anillos hacia +z, asi que el reflejo (en
+z) sale con mas anillos que la particula, al reves de lo observado. Mover el
plano de referencia (`z_ref_espejo_nm`) no lo arregla. Con la inclinacion
fijada quedan dos causas:

- El lado del foco donde la PSF del codigo tiene anillos es el opuesto al
  fisico (la fase `k z cos(theta)` no esta atada a una direccion fisica):
  `signo_z = -1`.
- El truncamiento de la pupila (no modelado, abajo): solo parte del cono de
  coleccion pasa por el espejo, lo que cambiaria la forma de la PSF del
  reflejo; los lobulos, que no salen de una PSF de simetria circular, apuntan
  a esto.

Se decidira con un z-stack de una bead: de que lado del foco aparecen los
anillos al alejarla del objetivo. Si aparecen al alejarla, `signo_z = +1` es
correcto y la discrepancia viene del truncamiento; si aparecen al acercarla,
`signo_z = -1`. Hasta entonces los valores por defecto no se cambian.

Supuestos (no modelados): espejo plano ideal sin aberraciones ni
polarizacion; una sola perdida constante `ratio_reflejo` (reflectividad y
coleccion); el reflejo es una fuente puntual vista por la misma optica, sin
el truncamiento de la pupila (que podria hacer asimetrica su PSF; con
alpha >= 90 el espejo no manda luz al objetivo por el eje); espejo infinito;
iluminacion uniforme (no se modela la hoja de luz).

### Verificacion

```bash
$PY checks/run_all_checks.py   # checks con figuras (sin aserciones: fallan solo si lanzan)
$PY tests/regresion.py         # regresion bit a bit contra tests/referencia/
$PY tests/regresion.py --modulo sim_imagenesPSF   # idem sobre la capa de compatibilidad
```

`tests/referencia/` se escribio con el codigo anterior a la reorganizacion
(`--generar` la reescribe; hacerlo solo si se cambia el comportamiento a proposito).
