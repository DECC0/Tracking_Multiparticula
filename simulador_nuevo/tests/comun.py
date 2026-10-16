"""Utilidades compartidas por los tests: ruta al simulador, bancos de PSF
cacheados (construirlos es caro) y ayudas para los asserts."""
import os
import sys
from functools import lru_cache

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import numpy as np
import simulador as sim

MASCARAS = ("esferica", "ninguna", "corkscrew")


@lru_cache(maxsize=None)
def banco(mascara="esferica", **kw):
    """BancoPSF con la Config por defecto (salvo kw); se construye una sola vez."""
    return sim.BancoPSF(sim.Config(mascara=mascara, **kw))


def espera_error(tipo, fn, *args, **kw):
    """Falla si fn(*args, **kw) NO lanza la excepcion tipo."""
    try:
        fn(*args, **kw)
    except tipo:
        return
    raise AssertionError(f"se esperaba {tipo.__name__} con args={args} kw={kw}")


def centroide(img):
    """(x, y) del centroide de intensidad en coordenadas de pixel (columna, fila)."""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    s = img.sum()
    return (img * xx).sum() / s, (img * yy).sum() / s
