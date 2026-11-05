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

### Verificacion

```bash
$PY checks/run_all_checks.py   # checks con figuras (sin aserciones: fallan solo si lanzan)
$PY tests/regresion.py         # regresion bit a bit contra tests/referencia/
$PY tests/regresion.py --modulo sim_imagenesPSF   # idem sobre la capa de compatibilidad
```

`tests/referencia/` se escribio con el codigo anterior a la reorganizacion
(`--generar` la reescribe; hacerlo solo si se cambia el comportamiento a proposito).
