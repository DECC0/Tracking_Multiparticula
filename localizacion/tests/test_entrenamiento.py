"""Smoke tests del entrenamiento con configs/smoke.yaml (CPU, segundos):
corre de punta a punta, escribe config/logs/checkpoints, reanuda desde
ultimo.pt, el checkpoint reconstruye el modelo, entrena con datos
pregenerados y con acumulacion de gradientes, y rechaza datos incompatibles;
checkpoints por epoca (checkpoint_cada), metricas de val por epoca (metricas_val)
y la columna lr de log.csv = la lr usada en la epoca."""
import functools
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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
        # por defecto, nada nuevo: ni checkpoints por epoca ni metricas de deteccion
        assert not list(Path(tmp).glob("epoca_*.pt"))
        assert not [c for c in log.columns if c in ("val_recall", "val_precision", "val_jaccard")]
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


def test_checkpoint_cada_metricas_val_y_lr_de_la_epoca():
    """2 epocas de smoke.yaml con checkpoint_cada=1 y metricas_val: escribe epoca_000.pt y
    epoca_001.pt (que reconstruyen el modelo) y las columnas val_recall... en log.csv; con
    checkpoint_cada=2 solo epoca_001.pt. La columna lr es la usada en la epoca (la de
    log_iter.csv), no la de despues de ReduceLROnPlateau.step: el scheduler se cambia por uno
    que baja la lr al final de cada epoca (umbral imposible), asi que las dos difieren."""
    original = torch.optim.lr_scheduler.ReduceLROnPlateau
    torch.optim.lr_scheduler.ReduceLROnPlateau = functools.partial(original, threshold=10.0)
    try:
        cfg = cargar_config(SMOKE, ["entrenamiento.checkpoint_cada=1", "entrenamiento.metricas_val=true",
                                    "entrenamiento.paciencia_lr=0"])
        with tempfile.TemporaryDirectory() as tmp:
            log = entrenar(cfg, tmp, log=mudo)
            for e in (0, 1):
                ck = cargar_checkpoint(Path(tmp) / f"epoca_{e:03d}.pt")
                assert ck["epoca"] == e
            modelo, cfg_c, _ = cargar_modelo(Path(tmp) / "epoca_000.pt")
            assert cfg_c.entrenamiento.checkpoint_cada == 1 and cfg_c.entrenamiento.metricas_val
            for c in ("val_recall", "val_precision", "val_jaccard", "val_rmse_lat_nm", "val_rmse_z_nm"):
                assert c in log.columns, c
            assert log.val_recall.between(0, 1).all() and log.val_jaccard.between(0, 1).all()
            # lr de la epoca: 1e-3 y luego 5e-4 (con el desfase anterior seria 5e-4 y 2.5e-4)
            lr0 = cfg.entrenamiento.lr
            assert np.allclose(log.lr, [lr0, lr0 * cfg.entrenamiento.factor_lr]), list(log.lr)
            it = pd.read_csv(Path(tmp) / "log_iter.csv")
            assert np.allclose(log.lr, it.groupby("epoca").lr.first().loc[log.epoca].to_numpy())
    finally:
        torch.optim.lr_scheduler.ReduceLROnPlateau = original

    with tempfile.TemporaryDirectory() as tmp:
        log = entrenar(cargar_config(SMOKE, ["entrenamiento.checkpoint_cada=2"]), tmp, log=mudo)
        assert sorted(p.name for p in Path(tmp).glob("epoca_*.pt")) == ["epoca_001.pt"]
        assert "val_recall" not in log.columns

    espera_error(ValueError, cargar_config, SMOKE, ["entrenamiento.checkpoint_cada=-1"])
    espera_error(ValueError, cargar_config, SMOKE, ["entrenamiento.metricas_val=3"])


def test_agregar_fila_con_columnas_nuevas():
    """Al reanudar con metricas_val activado, log.csv gana columnas sin romperse."""
    from train import _agregar_fila
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "log.csv"
        _agregar_fila(ruta, {"epoca": 0, "lr": 1e-3, "val_total": 0.5})
        _agregar_fila(ruta, {"epoca": 1, "lr": 1e-3, "val_total": 0.4, "val_recall": 0.7})
        _agregar_fila(ruta, {"epoca": 2, "lr": 5e-4, "val_total": 0.3, "val_recall": 0.8})
        df = pd.read_csv(ruta)
        assert list(df.epoca) == [0, 1, 2] and np.isnan(df.val_recall[0]) and np.allclose(df.val_recall[1:], [0.7, 0.8])
