# simulador_nuevo

Simulador de videos de fluorescencia escrito a partir de sus tests. Usa la
misma optica, camara y rangos que `sim_imagenesPSF.py`, y es independiente de
el: no importa ni modifica nada fuera de esta carpeta.

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
$PY simulador_nuevo/tests/correr_tests.py            # ~4 s
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
| `test_diagnosticos.py` | SNR del pixel pico vs simulacion, escalamiento de la CRLB; CRLB 3D (`crlb_3d`, z desconocido) vs la PSF exacta de `psf_directa`, vs la 2D y escalamiento con N |

## Diferencias con sim_imagenesPSF.py

- **Integracion sobre el pixel** (`integrar_pixel=True`): la PSF se integra sobre el area del pixel en vez de muestrearse en su centro. Ensancha un poco la PSF; con `False` se recupera el muestreo puntual.
- **Recorte automatico de la PSF** (`K_psf=None`): se usa el menor K que pierde como mucho `energia_max_recortada` (2%). Queda K = 128 con la esferica (igual que antes), 64 sin mascara y 144 con el corkscrew (antes 48, que perdia ~7%).
- **Desenfoque de movimiento** opcional (`t_exp_s`, `n_subpasos`; por defecto apagado). El GT es la posicion media durante la exposicion.
- **Convencion de pixel**: el centro del pixel (fila i, col j) es (x=j, y=i) y el FOV es [-0.5, W-0.5) x [-0.5, H-0.5). Antes `en_fov` usaba [0, W).
- **Aleatoriedad por video y por componente**: cada video se regenera solo, y cambiar ruido, fondo o fotones no cambia las trayectorias. Por eso, con la misma semilla, los videos no son iguales a los de `sim_imagenesPSF.py`.
- La salida incluye `config.json` (Config + K usado), y hay diagnosticos nuevos: `crlb` (z conocido), `crlb_3d` (z desconocido), `modelo_roi`, `en_fov`, `psf_directa`.
