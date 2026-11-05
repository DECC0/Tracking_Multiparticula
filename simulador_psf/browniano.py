"""
Movimiento browniano 3D de los emisores a lo largo de un video.

Cada eje da pasos independientes dx = sqrt(2 D dt) * N(0, 1) por frame, con D
propio de cada particula. z tiene paredes reflectantes en cfg.z_rango_nm
(el banco de PSF solo esta definido ahi); x, y son libres, asi que las
particulas pueden salir y entrar del FOV.

Con cfg.usar_espejo el espejo es una pared reflectante mas: ningun emisor
real lo cruza (espejo.py). Ese caso se integra paso a paso (posicion
anterior + paso, plegada al dominio), porque el plegado de la suma
acumulada solo vale para paredes paralelas; usa los mismos pasos y no
consume el rng.
"""

import numpy as np

from .config import Config
from .posiciones import posiciones_iniciales
from .espejo import Espejo


def reflejar(z, lo, hi):
    """Paredes reflectantes en [lo, hi] (funciona aunque el paso cruce varias veces).
    Desdobla z en un periodo 2 (hi - lo) y lo refleja en la mitad superior."""
    L = hi - lo
    u = np.mod(z - lo, 2 * L)
    return lo + np.where(u > L, 2 * L - u, u)


def generar_trayectorias(cfg: Config, n, rng):
    """Trayectorias 3D de n particulas durante cfg.n_frames frames.

    Consume el rng en este orden: D por particula, posiciones iniciales
    (posiciones.posiciones_iniciales) y pasos.

    Retorna (tray, D): tray (n, T, 3) en [px, px, nm], con el frame 0 igual a
    la posicion inicial; D (n,) en um^2/s, uniforme en cfg.D_rango_um2_s."""
    T, px = cfg.n_frames, cfg.pixel_nm

    D = rng.uniform(*cfg.D_rango_um2_s, n)
    sigma_nm = np.sqrt(2 * D * cfg.dt_s) * 1000.0                # rms por eje por frame (um -> nm)

    pos0 = posiciones_iniciales(cfg, n, rng)                     # (n, 3) en nm

    pasos = rng.normal(size=(n, T, 3)) * sigma_nm[:, None, None]
    pasos[:, 0] = 0.0                                            # frame 0 = pos0
    if cfg.usar_espejo:
        espejo = Espejo(cfg)
        tray = np.empty_like(pasos)
        tray[:, 0] = pos0                                        # ya del lado de la muestra
        for t in range(1, T):                                    # reflexion especular en espejo y paredes
            tray[:, t] = espejo.plegar_nm(tray[:, t - 1] + pasos[:, t])
    else:
        tray = pos0[:, None, :] + np.cumsum(pasos, axis=1)
        tray[..., 2] = reflejar(tray[..., 2], *cfg.z_rango_nm)
    tray[..., :2] /= px                                          # x, y de nm a px
    return tray, D
