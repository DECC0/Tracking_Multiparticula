# simulador_nuevo

Simulador de videos de fluorescencia escrito a partir de sus tests. Usa la
misma optica, camara y rangos que `sim_imagenesPSF.py`, y es independiente de
el. La unica importacion de fuera de esta carpeta es `geometria_espejo.py` (en
la raiz del repo, compartida con `simulador_psf`): `simulador.py` agrega la
raiz al FINAL de `sys.path` (como `localizacion/data/simulacion.py` agrega esta
carpeta), asi que hay que conservar la estructura del repo.

```
simulador_nuevo/
  simulador.py          el simulador (Config, BancoPSF, generar_video, generar_dataset, diagnosticos)
  tests/                tests con asserts (pasa / falla)
    correr_tests.py     corre todos los tests, sin necesidad de pytest
  checks/               diagnosticos con figuras (fig_*.png), como checks/ de la raiz
    run_all_checks.py
  salida/               lo que escribe simulador.py (ignorado por git)
```

```bash
PY=~/miniforge3/envs/tesis/bin/python
$PY simulador_nuevo/simulador.py                     # dataset -> simulador_nuevo/salida/
$PY simulador_nuevo/tests/correr_tests.py            # 81 tests, ~9 s
$PY simulador_nuevo/tests/correr_tests.py psf        # solo los tests cuyo nombre contiene "psf"
$PY simulador_nuevo/checks/run_all_checks.py         # ~2 min (check_crlb hace ajustes MLE)
```

## Que se verifica

| Archivo | Que verifica |
|---|---|
| `test_config.py` | pixel en la muestra, Nyquist, rechazo de parametros invalidos, ida y vuelta a JSON |
| `test_psf.py` | normalizacion y no negatividad, recorte automatico, simetrias, PSF en foco = Airy, desenfoque en el eje vs integral teorica, interpolacion en z, integracion sobre el pixel vs sobremuestreo 5x5, DOF de la esferica, giro del corkscrew |
| `test_colocar.py` | ejes x/y, centro de pixel en enteros, conservacion de fotones, shift subpixel vs PSF exacta, traslaciones enteras, linealidad, bordes |
| `test_camara.py` | media y varianza en ADU (Poisson + lectura + cuantizacion), curva de transferencia de fotones, saturacion, conversion inversa |
| `test_fondo.py` | fondo uniforme / no uniforme acotado y suave |
| `test_difusion.py` | paredes reflectantes, MSD = 2Dt por eje, incrementos gaussianos independientes, z uniforme, desenfoque de movimiento (MSD con correccion t_e(n^2-1)/(3n^2)) |
| `test_video.py` | forma del video, columnas y filas del GT, en_fov, imagen limpia reconstruible desde el GT, residuo de ruido ~ N(0,1), semilla, componentes independientes, prefijo temporal, escritura a disco |
| `test_espejo.py` | validacion y JSON de los campos del espejo, misma geometria que `simulador_psf`, banco ampliado con los planos de `z_rango_nm` intactos, ningun real cruza el espejo (tambien en los subpasos), `dist_max_espejo_um`, espejo lejano = sin espejo, GT del reflejo, desenfoque ligado, imagen limpia reconstruible (con subpasos y desde el GT), el espejo no cambia los otros flujos, regeneracion por video, convencion de signos, espejo apagado |
| `test_regresion.py` | con el espejo apagado, salidas bit a bit iguales a las de antes del espejo (`referencia_regresion.json`; se salta si cambia la version de numpy; `--generar` la reescribe) |
| `test_diagnosticos.py` | SNR del pixel pico vs simulacion, escalamiento de la CRLB; CRLB 3D (`crlb_3d`, z desconocido) vs la PSF exacta de `psf_directa`, vs la 2D y escalamiento con N |

## Diferencias con sim_imagenesPSF.py

- **Integracion sobre el pixel** (`integrar_pixel=True`): la PSF se integra sobre el area del pixel en vez de muestrearse en su centro. Ensancha un poco la PSF; con `False` se recupera el muestreo puntual.
- **Recorte automatico de la PSF** (`K_psf=None`): se usa el menor K que pierde como mucho `energia_max_recortada` (2%). Queda K = 128 con la esferica (igual que antes), 64 sin mascara y 144 con el corkscrew (antes 48, que perdia ~7%).
- **Desenfoque de movimiento** opcional (`t_exp_s`, `n_subpasos`; por defecto apagado). El GT es la posicion media durante la exposicion.
- **Convencion de pixel**: el centro del pixel (fila i, col j) es (x=j, y=i) y el FOV es [-0.5, W-0.5) x [-0.5, H-0.5). Antes `en_fov` usaba [0, W).
- **Aleatoriedad por video y por componente**: cada video se regenera solo, y cambiar ruido, fondo o fotones no cambia las trayectorias. Por eso, con la misma semilla, los videos no son iguales a los de `sim_imagenesPSF.py`.
- La salida incluye `config.json` (Config + K usado), y hay diagnosticos nuevos: `crlb` (z conocido), `crlb_3d` (z desconocido), `modelo_roi`, `en_fov`, `psf_directa`.

## Espejo (`usar_espejo=True`)

Mismo modelo que `simulador_psf` (README de la raiz, seccion Espejo), con la
geometria de `geometria_espejo.py` y los mismos nombres de campos
(`angulo_espejo_deg`, `orientacion_espejo_deg`, `espejo_punto_px`,
`ratio_reflejo`, `z_ref_espejo_nm`, `dist_max_espejo_um`, `signo_z`,
`signo_inclinacion`, `z_banco_nm`):

- Solo se simulan las trayectorias reales. Con espejo se integran subpaso a
  subpaso en nm, plegando en las paredes en z y en el espejo, con los mismos
  sorteos del flujo `trayectorias`; con `dist_max_espejo_um` los sorteos de x
  e y se reescalan a la franja cercana al espejo.
- Cada subposicion del reflejo es el reflejo de la del real, asi que el
  desenfoque de movimiento queda ligado; se dibuja con
  `ratio_reflejo * fotones / n_subpasos`. El GT agrega `COLUMNAS_REFLEJO`
  (`x_reflejo_px, y_reflejo_px, z_reflejo_nm, photons_reflejo, reflejo_en_fov,
  reflejo_en_banco`); la posicion del reflejo es la media de sus subposiciones.
- El banco se amplia (`rango_banco_nm`) para cubrir las z' de la caja inicial,
  con los mismos planos que sin espejo dentro de `z_rango_nm`; `z_banco_nm`
  fija a mano el rango a cubrir. Un reflejo fuera del banco no se dibuja
  (`reflejo_en_banco = False`). `K_psf=None` elige el recorte sobre todo el
  banco: con la esferica, `signo_z = +1` (z' > 0) deja K = 128, pero
  `signo_z = -1` (z' hasta -4.6 um, donde la PSF es mas ancha) sube a K = 208,
  y cada muestra cuesta el doble (con K = 128 se perderia el 21% de la luz de
  los reflejos mas lejanos).
- No consume numeros aleatorios: n, D, fotones y fondo son los mismos con y
  sin espejo. Con el espejo apagado sus campos no se escriben en
  `config.json`, y todo es identico bit a bit a antes (`test_regresion.py`).
