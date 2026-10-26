"""Smoke tests del entrenamiento con configs/smoke.yaml (CPU, segundos):
corre de punta a punta, escribe config/logs/checkpoints, reanuda desde
ultimo.pt, el checkpoint reconstruye el modelo, entrena con datos
pregenerados y con acumulacion de gradientes, y rechaza datos incompatibles."""
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from comun import espera_error
from config import cargar_config
from data.dataset import pregenerar
from train import entrenar
from utils import cargar_modelo, cargar_checkpoint

SMOKE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "smoke.yaml")
mudo = lambda *a, **k: None


def test_smoke_entrena_reanuda_y_carga():
    cfg = cargar_config(SMOKE)
    with tempfile.TemporaryDirectory() as tmp:
        log = entrenar(cfg, tmp, log=mudo)
        for f in ("config.yaml", "log.csv", "log_iter.csv", "curvas.png", "ultimo.pt", "mejor.pt"):
            assert (Path(tmp) / f).exists(), f
        assert list(log.epoca) == [0, 1]
        assert np.isfinite(log.filter(regex="^(train|val)_").to_numpy()).all()
        # la config guardada reproduce la de la corrida
        assert cargar_config(Path(tmp) / "config.yaml") == cfg

        # reanudar hasta 3 epocas: continua en la epoca 2 con el estado guardado
        paso = cargar_checkpoint(Path(tmp) / "ultimo.pt")["paso"]
        cfg3 = cargar_config(Path(tmp) / "config.yaml", ["entrenamiento.epocas=3"])
        log = entrenar(cfg3, tmp, reanudar=True, log=mudo)
        assert list(log.epoca) == [0, 1, 2]
        ck = cargar_checkpoint(Path(tmp) / "ultimo.pt")
        assert ck["epoca"] == 2 and ck["paso"] == paso + cfg.datos.n_train // cfg.entrenamiento.batch

        # el checkpoint basta para reconstruir el modelo (config incluida)
        modelo, cfg_c, rej = cargar_modelo(Path(tmp) / "mejor.pt")
        assert cfg_c.datos == cfg.datos and not modelo.training
        with torch.no_grad():
            y = modelo(torch.zeros(1, 1, 32, 40))
        assert y.shape == (1, *rej.forma)


def test_smoke_con_datos_pregenerados_y_acumulacion():
    cfg = cargar_config(SMOKE)
    with tempfile.TemporaryDirectory() as tmp:
        pregenerar(cfg.datos, 8, cfg.datos.seed_train, Path(tmp) / "train")
        pregenerar(cfg.datos, 4, cfg.datos.seed_val, Path(tmp) / "val")
        cfg_d = cargar_config(SMOKE, [f"entrenamiento.datos_train={tmp}/train", f"entrenamiento.datos_val={tmp}/val",
                                      "entrenamiento.epocas=1", "entrenamiento.acumulacion=2",
                                      "entrenamiento.batch=2"])
        (Path(tmp) / "corrida").mkdir()
        log = entrenar(cfg_d, Path(tmp) / "corrida", log=mudo)
        assert len(log) == 1 and np.isfinite(log.val_total).all()
        # datos de otro tamano de imagen no se aceptan
        cfg_mal = cargar_config(SMOKE, [f"entrenamiento.datos_train={tmp}/train", "datos.simulador.H=16"])
        espera_error(ValueError, entrenar, cfg_mal, Path(tmp) / "corrida", log=mudo)
