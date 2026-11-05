"""
Cadena de ruido de la camara (sCMOS Orca Flash 4.0): de fotones esperados por
pixel a cuentas digitales uint16.

    fotones (senal + fondo) -> Poisson -> / ganancia -> + lectura -> + offset -> uint16

El orden importa y no debe cambiarse: el ruido de disparo (Poisson) es
propio de la luz y ocurre al detectar los fotones, asi que actua sobre la
suma senal + fondo; el ruido de lectura es electronico, se agrega despues
al convertir a ADU, y no depende de la senal. Dominante en bajo SNR: Poisson.

No hay un modulo camara aparte: el muestreo en pixeles ya lo hace la optica
(BancoPSF.colocar deposita la PSF en la rejilla de pixeles), y la ganancia,
el offset y el ruido de lectura son parte de esta misma cadena.
"""

import numpy as np

from .config import Config, ADU_MAX


def aplicar_camara(fotones, cfg: Config, rng):
    """Poisson -> ganancia -> ruido de lectura -> offset -> uint16.

    fotones: array de fotones esperados por pixel (senal + fondo); los
    negativos se tratan como 0. cfg.gain en e-/ADU (los e- se DIVIDEN por
    ella), cfg.read_noise en ADU rms, cfg.offset en ADU.
    Retorna un array uint16 de la misma forma, saturado en [0, ADU_MAX]."""
    e = rng.poisson(np.clip(fotones, 0, None))                   # e- detectados (QE = 1)
    adu = e / cfg.gain + rng.normal(0.0, cfg.read_noise, e.shape) + cfg.offset
    return np.clip(np.round(adu), 0, ADU_MAX).astype(np.uint16)
