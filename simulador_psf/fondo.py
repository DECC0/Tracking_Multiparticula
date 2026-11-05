"""
Fondo de fluorescencia de la muestra, en fotones por pixel por frame.

Es senal (luz desenfocada, autofluorescencia), no ruido: se suma a la imagen
de los emisores ANTES del ruido de Poisson (ver ruido.aplicar_camara). Se
muestrea una vez por video y queda fijo en todos sus frames.
"""

import numpy as np

from .config import Config, FONDO_CENTRO_FRAC, FONDO_ANCHO_FRAC


def generar_fondo(cfg: Config, rng):
    """Fondo (H, W) [fotones/px]. Uniforme en cfg.fondo_rango si
    cfg.fondo_no_uniforme es False; si no, ese nivel base mas una mancha
    gaussiana de pico tambien en cfg.fondo_rango, centro y anchos aleatorios
    (FONDO_CENTRO_FRAC, FONDO_ANCHO_FRAC de W y H)."""
    base = rng.uniform(*cfg.fondo_rango)
    if not cfg.fondo_no_uniforme:
        return np.full((cfg.H, cfg.W), base)
    pico = rng.uniform(*cfg.fondo_rango)
    x0, y0 = rng.uniform(*FONDO_CENTRO_FRAC) * cfg.W, rng.uniform(*FONDO_CENTRO_FRAC) * cfg.H
    sx, sy = rng.uniform(*FONDO_ANCHO_FRAC) * cfg.W, rng.uniform(*FONDO_ANCHO_FRAC) * cfg.H
    yy, xx = np.mgrid[0:cfg.H, 0:cfg.W]
    return base + pico * np.exp(-(((xx-x0)/sx)**2 + ((yy-y0)/sy)**2) / 2.0)
