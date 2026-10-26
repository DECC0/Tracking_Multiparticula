"""
Utilidades comunes a train.py, infer.py y evaluate.py: dispositivo y AMP,
rejilla de la corrida, carpetas de corridas, checkpoints y curvas de perdida.
"""
import contextlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import a_dict, aplicar_overrides, desde_dict
from data.simulacion import config_simulador
from data.target import RejillaVoxeles

DIR_LOCALIZACION = Path(__file__).resolve().parent


def elegir_dispositivo(nombre="auto"):
    if nombre == "auto":
        nombre = "cuda" if torch.cuda.is_available() else "cpu"
    if nombre == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("se pidio dispositivo cuda pero no hay GPU disponible")
    return torch.device(nombre)


def modo_amp(modo, dispositivo):
    """Modo AMP efectivo: fp16 solo en GPU (en CPU se desactiva)."""
    return "no" if modo == "fp16" and dispositivo.type != "cuda" else modo


def autocast(modo, dispositivo):
    """Contexto de AMP para el forward ("fp16", "bf16" o "no")."""
    if modo == "no":
        return contextlib.nullcontext()
    dtype = torch.float16 if modo == "fp16" else torch.bfloat16
    return torch.autocast(dispositivo.type, dtype=dtype)


def rejilla_de(cfg, H=None, W=None):
    """RejillaVoxeles de una ConfigCorrida (z_rango sale de datos.simulador, y H y W
    tambien salvo que se den: la red es convolucional y sirve para cualquier tamano)."""
    cs = config_simulador(cfg.datos, cfg.datos.seed_train)
    return RejillaVoxeles(cfg.target, H or cs.H, W or cs.W, cs.z_rango_nm)


def con_postproc(cfg, overrides):
    """cfg con los overrides "postproc.campo=valor" aplicados (validados). Solo
    se acepta postproc: el resto define la red ya entrenada."""
    otros = [o for o in overrides if not o.strip().startswith("postproc.")]
    if otros:
        raise ValueError(f"solo se puede cambiar postproc.* sobre un checkpoint: {otros}")
    return desde_dict(aplicar_overrides(a_dict(cfg), overrides))


def carpeta_corridas(cfg_ent):
    """localizacion/corridas/ por defecto; si se fija, relativa al directorio de trabajo."""
    return Path(cfg_ent.carpeta_corridas) if cfg_ent.carpeta_corridas else DIR_LOCALIZACION / "corridas"


def carpeta_nueva(cfg_ent):
    """<carpeta_corridas>/<nombre>_<fecha-hora>/ (con sufijo si ya existe)."""
    base = carpeta_corridas(cfg_ent) / f"{cfg_ent.nombre}_{time.strftime('%Y%m%d-%H%M%S')}"
    carpeta, i = base, 1
    while carpeta.exists():
        carpeta, i = Path(f"{base}_{i}"), i + 1
    carpeta.mkdir(parents=True)
    return carpeta


def guardar_checkpoint(ruta, cfg, modelo, **estado):
    """Guarda pesos, config y el estado extra (optimizador, epoca...) de forma atomica."""
    datos = {"modelo": modelo.state_dict(), "config": a_dict(cfg), **estado}
    tmp = Path(f"{ruta}.tmp")
    torch.save(datos, tmp)
    os.replace(tmp, ruta)


def cargar_checkpoint(ruta, dispositivo="cpu"):
    return torch.load(ruta, map_location=dispositivo, weights_only=True)


def cargar_modelo(ruta, dispositivo="cpu"):
    """(modelo en eval, ConfigCorrida, RejillaVoxeles) desde un checkpoint de train.py."""
    from models import crear_modelo
    ckpt = cargar_checkpoint(ruta, dispositivo)
    cfg = desde_dict(ckpt["config"])
    rejilla = rejilla_de(cfg)
    modelo = crear_modelo(cfg, rejilla).to(dispositivo)
    modelo.load_state_dict(ckpt["modelo"])
    return modelo.eval(), cfg, rejilla


def graficar_log(carpeta):
    """curvas.png: perdida total y cada termino, train (por iteracion y por epoca) y val."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    carpeta = Path(carpeta)
    ep = pd.read_csv(carpeta / "log.csv")
    it = pd.read_csv(carpeta / "log_iter.csv") if (carpeta / "log_iter.csv").exists() else None
    terminos = [c[len("train_"):] for c in ep.columns if c.startswith("train_")]
    fig, axes = plt.subplots(1, len(terminos) + 1, figsize=(5 * (len(terminos) + 1), 3.8))
    colores = {"train_iter": "#86b6ef", "train": "#2a78d6", "val": "#eb6834"}
    for ax, t in zip(axes, terminos):
        if it is not None and len(it):
            ax.plot(it.epoca_continua, it[t], color=colores["train_iter"], lw=1, label="train (iteraciones)")
        ax.plot(ep.epoca + 1, ep[f"train_{t}"], "o-", color=colores["train"], lw=2, ms=5, label="train (epoca)")
        ax.plot(ep.epoca + 1, ep[f"val_{t}"], "o-", color=colores["val"], lw=2, ms=5, label="val")
        ax.set_yscale("log")
        ax.set_xlabel("epoca"); ax.set_title(t)
    axes[0].legend(frameon=False, fontsize=9)
    axes[-1].plot(ep.epoca + 1, ep.lr, "o-", color=colores["train"], lw=2, ms=5)
    axes[-1].set_yscale("log"); axes[-1].set_xlabel("epoca"); axes[-1].set_title("lr")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(carpeta.name)
    fig.tight_layout()
    fig.savefig(carpeta / "curvas.png", dpi=130)
    plt.close(fig)


def resumen_tiempo(segundos):
    m, s = divmod(int(round(segundos)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def media_ponderada(sumas, n):
    return {k: float(v) / max(n, 1) for k, v in sumas.items()}


def es_finito(valores):
    return all(np.isfinite(v) for v in valores.values())
