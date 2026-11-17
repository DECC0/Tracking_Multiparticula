# figuras/

Los scripts y notebooks de análisis están en esta carpeta. Lo que generan (PNG, PDF, CSV, JSON) va a una subcarpeta por análisis, que es la salida por defecto de cada script (`--salida` la cambia). Los resultados escritos están en `resultados/` (raíz del repo).

| carpeta | qué hay | script | documento |
|---|---|---|---|
| `documento_simulador/` | banco de PSF (montaje, diapositiva, perfil axial) y cuadros con GT para la tesis | `figuras_documento.py` | `README.md` |
| `redes_sin_espejo/` | curvas de entrenamiento, recall comparado, recall frente al vecino y ejemplos de las redes sin espejo | `figuras_localizacion.py` | — |
| `datos_reales/` | canales y medidas de la adquisición real con espejo (Yeast6_1) | `analisis_datos_reales.py` | `resultados/resultados_datos_reales.md` |
| `psf_real/` | PSF medida en Yeast6_1: imágenes y medidas | `psf_real.py` | `resultados/resultados_psf_real.md` |
| `suma_canales/` | suma de los dos canales de polarización registrados | `suma_canales.py` | `resultados/resultados_suma_canales.md` |
| `autofluorescencia_celula/` | contraste de la autofluorescencia de la célula | `autofluorescencia_celula.py` | `resultados/resultados_simulador_v2.md` |
| `calibracion_pupila/` | ajuste de la pupila esférica (+ astigmatismo y coma), β y la línea del espejo | `calibracion_pupila.py` | `resultados/resultados_calibracion_pupila.md` |
| `recorte_pupila/` | recorte de pupila por el espejo: pérdida, forma, sesgo, CRLB y calibración fotométrica | `recorte_pupila.py` | `resultados/resultados_recorte_pupila.md` |
| `crlb_espejo/` | CRLB de z con espejo (rejilla y por emisor del test con célula) | `crlb_espejo.py` | `resultados/resultados_crlb_espejo.md`, `resultados/resultados_celula.md` |
| `temporal_espejo/` | información temporal: RMSE y D por trayectoria | `temporal_espejo.py` | `resultados/resultados_temporal_espejo.md` |
| `diagnostico_celula/` | corridas con célula: recall/RMSE por rango, clases de FP, control sin reflejo | `diagnostico_celula.py` | `resultados/resultados_celula.md` |
| `red_celula/` | redes con célula: Fisher en z, curva de transferencia, agujero de +400 nm, Yeast6_1 | `red_celula.py` | `resultados/resultados_celula.md` |
| `cuantizacion/` | ¿el piso de error es la rejilla? fases y estimadores subplano | `cuantizacion.py` | `resultados/resultados_cuantizacion.md` |
| `convergencia/` | convergencia de las corridas con célula v1 | `convergencia.py` | `resultados/resultados_convergencia.md` |
| `simulador_v2/` | simulador v2: distribuciones, galería de PSF y línea base | `simulador_v2.py` | `resultados/resultados_simulador_v2.md` |
| `celula_v2/` | corridas v2 (dos semillas): convergencia, tests, z y Yeast6_1 | `celula_v2.py` | `resultados/resultados_celula_v2.md` |
| `tracking_base/` | tracking base: red v2 s1 + enlace LAP en 200 videos simulados (ISBI, identidades, D por MSD) | `tracking_base.py` | `resultados/resultados_tracking_base.md` |

Otros:
- `analisis_corridas.ipynb` exporta a `notebook/`.
- `gif_video.py` escribe en `gif_video/` (los GIF se versionan, el tif y el GT no) y `muestra_espejo.py` en `muestra_espejo/` (no versionada); `--salida` cambia las dos.
- `comun_localizacion.py` y `analisis_datos_reales.py` también se importan desde otros scripts.
