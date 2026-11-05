"""
Muestreo de la poblacion de emisores de un video: cuantos hay, donde empiezan
(x, y, z) y cuantos fotones emite cada uno por frame.

Las funciones consumen el rng que reciben; el orden en que se llaman lo fija
generacion.generar_video (y browniano.generar_trayectorias) y no debe
cambiarse, porque check_e2e.py y figuras/figuras_documento.py lo repiten.
"""

import numpy as np

from .config import Config, MARGEN_PX


def muestrear_n_emisores(cfg: Config, rng):
    """Numero de emisores de un video, entero uniforme en
    cfg.n_emisores_rango (ambos extremos incluidos)."""
    return rng.integers(cfg.n_emisores_rango[0], cfg.n_emisores_rango[1] + 1)


def posiciones_iniciales(cfg: Config, n, rng):
    """Posiciones iniciales (n, 3) de los emisores, TODAS en nm: x, y uniformes
    en el FOV ampliado MARGEN_PX px por lado, z uniforme en cfg.z_rango_nm.
    x, y se pasan de px a nm porque los pasos brownianos se suman en nm."""
    px = cfg.pixel_nm
    return np.stack([rng.uniform(-MARGEN_PX, cfg.W + MARGEN_PX, n) * px,
                     rng.uniform(-MARGEN_PX, cfg.H + MARGEN_PX, n) * px,
                     rng.uniform(*cfg.z_rango_nm, n)], axis=1)   # (n, 3) en nm


def muestrear_fotones(cfg: Config, n, rng):
    """Fotones por emisor. "log": N = exp(U(log Nmin, log Nmax)), cubre por
    igual cada decada del rango; "uniforme": U(Nmin, Nmax). Ambos consumen
    el mismo numero de llamadas al rng."""
    Nmin, Nmax = cfg.fotones_rango
    if cfg.muestreo_fotones == "log":
        return np.exp(rng.uniform(np.log(Nmin), np.log(Nmax), n))
    if cfg.muestreo_fotones == "uniforme":
        return rng.uniform(Nmin, Nmax, n)
    raise ValueError(f"muestreo_fotones desconocido: {cfg.muestreo_fotones}")
