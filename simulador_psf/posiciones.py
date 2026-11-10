"""
Muestreo de la poblacion de emisores de un video: cuantos hay, donde empiezan
(x, y, z) y cuantos fotones emite cada uno por frame.

Las funciones consumen el rng que reciben; el orden en que se llaman lo fija
generacion.generar_video (y browniano.generar_trayectorias) y no debe
cambiarse, porque check_e2e.py y figuras/figuras_documento.py lo repiten.
"""

import numpy as np

from .config import Config, MARGEN_PX
from .espejo import Espejo


def muestrear_n_emisores(cfg: Config, rng):
    """Numero de emisores de un video, entero uniforme en
    cfg.n_emisores_rango (ambos extremos incluidos)."""
    return rng.integers(cfg.n_emisores_rango[0], cfg.n_emisores_rango[1] + 1)


def posiciones_iniciales(cfg: Config, n, rng):
    """Posiciones iniciales (n, 3) de los emisores, TODAS en nm: x, y uniformes
    en el FOV ampliado MARGEN_PX px por lado, z uniforme en cfg.z_rango_nm.
    x, y se pasan de px a nm porque los pasos brownianos se suman en nm.

    Con cfg.usar_espejo, los puntos que caen detras del espejo se llevan al
    lado de la muestra con Espejo.plegar_nm (reflexion en el espejo y en las
    paredes en z): mismas llamadas al rng, sin sorteos extra. La densidad
    deja de ser exactamente uniforme cerca del espejo.

    Con cfg.usar_espejo y cfg.dist_max_espejo_um, la posicion lateral se
    sortea en el sistema del espejo en vez de en x, y (ver _cerca_del_espejo):
    mismos tres sorteos, reescalados, z igual que sin dist_max_espejo_um, y
    todos los puntos quedan a distancia [0, dist_max_espejo_um] del plano del
    espejo (no hace falta plegarlos)."""
    px = cfg.pixel_nm
    if cfg.usar_espejo and cfg.dist_max_espejo_um is not None:
        pos = _cerca_del_espejo(cfg, n, rng)
    else:
        pos = np.stack([rng.uniform(-MARGEN_PX, cfg.W + MARGEN_PX, n) * px,
                        rng.uniform(-MARGEN_PX, cfg.H + MARGEN_PX, n) * px,
                        rng.uniform(*cfg.z_rango_nm, n)], axis=1)    # (n, 3) en nm
    if cfg.usar_espejo:
        pos = Espejo(cfg).plegar_nm(pos)
    return pos


def _cerca_del_espejo(cfg: Config, n, rng):
    """Posiciones iniciales (n, 3) [nm] a distancia d en [0, dist_max_espejo_um]
    del PLANO del espejo (perpendicular), con la posicion lateral escrita como
    p0 + s_m m + s_t t (espejo.py):
        z uniforme en cfg.z_rango_nm (el mismo sorteo, sin cambios);
        s_t uniforme en la proyeccion sobre t del FOV ampliado MARGEN_PX px por
            lado (el sorteo de y, reescalado; con orientacion_espejo_deg = 0 es
            el mismo y de siempre);
        s_m uniforme en la franja d en [0, dist_max] a esa z (el sorteo de x,
            reescalado). A altura z, d = s_m sin(alpha) - signo_inclinacion
            signo_z cos(alpha) (z - z_ref), asi que la franja es
                s_m en [s(0), s(dist_max)],  s(d) = (d + signo_inclinacion signo_z
                                                     cos(alpha) (z - z_ref)) / sin(alpha),
            recortada a la proyeccion sobre m del FOV ampliado (el espejo
            inclinado proyecta la franja a un ancho lateral dist_max / sin(alpha),
            que puede pasar del FOV). Si a esa z la franja no toca el FOV
            ampliado, se usa la franja entera (el punto queda fuera del FOV con
            en_fov = False, pero respeta dist_max).
    rng.uniform(0, 1, n) consume lo mismo que rng.uniform(a, b, n): no hay
    sorteos extra. Sin recorte d es uniforme en [0, dist_max]; con recorte,
    uniforme en la parte de la franja dentro del FOV ampliado."""
    e = Espejo(cfg)
    px = cfg.pixel_nm
    u_m = rng.uniform(0.0, 1.0, n)                               # sorteo de x
    u_t = rng.uniform(0.0, 1.0, n)                               # sorteo de y
    z = rng.uniform(*cfg.z_rango_nm, n)                          # sorteo de z
    esq = np.array([[x, y] for x in (-MARGEN_PX, cfg.W + MARGEN_PX)
                    for y in (-MARGEN_PX, cfg.H + MARGEN_PX)]) * px   # FOV ampliado [nm]
    lat = e.lateral_cerca(u_m, u_t, z, esq, cfg.dist_max_espejo_um * 1000.0)   # geometria_espejo.py
    return np.column_stack([lat, z])


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
