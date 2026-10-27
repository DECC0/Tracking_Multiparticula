"""Utilidades compartidas por los tests: ruta a localizacion/, configuraciones
pequenas, bancos de PSF cacheados (construirlos es caro), checkpoints sin
entrenar y ayudas para asserts."""
import os
import sys
from functools import lru_cache
from pathlib import Path

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from config import ConfigDatos, ConfigTarget, cargar_config  # noqa: E402
from data.simulacion import sim, config_simulador  # noqa: E402

SMOKE = os.path.join(RAIZ, "configs", "smoke.yaml")


def cfg_datos(mascara="ninguna", sim_kw=None, **kw):
    """ConfigDatos pequena, sin randomizacion extra salvo que se pida. H != W a
    proposito: un intercambio de ejes x/y no pasa desapercibido."""
    base = dict(mascara=mascara, H=24, W=40, n_emisores_rango=(2, 6))
    base.update(sim_kw or {})
    return ConfigDatos(simulador=base, **kw)


@lru_cache(maxsize=None)
def banco(mascara="ninguna"):
    """BancoPSF de cfg_datos(mascara); se construye una sola vez."""
    return sim.BancoPSF(config_simulador(cfg_datos(mascara), 0))


def checkpoint_smoke(carpeta, overrides=()):
    """Checkpoint sin entrenar de configs/smoke.yaml (con overrides "seccion.campo=valor")
    en carpeta/modelo.pt; devuelve (ruta, cfg)."""
    import torch
    from models import crear_modelo
    from utils import guardar_checkpoint, rejilla_de
    cfg = cargar_config(SMOKE, overrides)
    torch.manual_seed(0)
    ruta = Path(carpeta) / "modelo.pt"
    guardar_checkpoint(ruta, cfg, crear_modelo(cfg, rejilla_de(cfg)))
    return ruta, cfg


def espera_error(tipo, fn, *args, **kw):
    """Falla si fn(*args, **kw) NO lanza la excepcion tipo."""
    try:
        fn(*args, **kw)
    except tipo:
        return
    raise AssertionError(f"se esperaba {tipo.__name__} con args={args} kw={kw}")
