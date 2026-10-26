"""
Entrenamiento de la red de localizacion 3D.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY localizacion/train.py --config localizacion/configs/base.yaml
    $PY localizacion/train.py --config localizacion/configs/smoke.yaml          # prueba rapida en CPU
    $PY localizacion/train.py --config localizacion/configs/base.yaml --set entrenamiento.epocas=5 datos.n_train=5000
    $PY localizacion/train.py --reanudar localizacion/corridas/base_20261019-153000 --set entrenamiento.epocas=50

Cada corrida escribe en <carpeta_corridas>/<nombre>_<fecha-hora>/:
    config.yaml     configuracion completa usada
    log.csv         una fila por epoca: perdidas de train y val, lr, tiempo
    log_iter.csv    perdida de train cada log_cada iteraciones
    curvas.png      log.csv y log_iter.csv graficados
    ultimo.pt       checkpoint de la ultima epoca (para reanudar)
    mejor.pt        checkpoint con la menor perdida de validacion (para infer.py)
"""
import argparse
import csv
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import agregar_argumentos, cargar_config, config_desde_args, a_dict, guardar_yaml
from data.dataset import DatasetSimulado, DatasetDisco, collate
from models import crear_modelo, crear_perdida, n_parametros
from utils import (elegir_dispositivo, modo_amp, autocast, rejilla_de, carpeta_nueva, guardar_checkpoint,
                   cargar_checkpoint, graficar_log, resumen_tiempo, media_ponderada, es_finito)


def _compatible(ds, rejilla, canales):
    """Un dataset pregenerado debe tener la misma imagen, z_rango y canales que la corrida."""
    cs = ds.cfg_sim
    if (cs.H, cs.W, tuple(cs.z_rango_nm), ds.cfg_datos.canales) != (rejilla.H, rejilla.W,
                                                                     (rejilla.z_min, rejilla.z_max), canales):
        raise ValueError(f"datos pregenerados incompatibles con la corrida: H, W = {cs.H}, {cs.W}, "
                         f"z_rango {cs.z_rango_nm}, canales {ds.cfg_datos.canales}")


def crear_datasets(cfg, rejilla):
    """(train, val): pregenerados si entrenamiento.datos_* apunta a una carpeta, si no on-the-fly."""
    e = cfg.entrenamiento
    train = DatasetDisco(e.datos_train) if e.datos_train else \
        DatasetSimulado(cfg.datos, cfg.datos.n_train, cfg.datos.seed_train)
    banco = getattr(train, "banco", None)
    val = DatasetDisco(e.datos_val) if e.datos_val else \
        DatasetSimulado(cfg.datos, cfg.datos.n_val, cfg.datos.seed_val, banco=banco)
    for ds in (train, val):
        if isinstance(ds, DatasetDisco):
            _compatible(ds, rejilla, cfg.datos.canales)
    return train, val


def _loader(ds, cfg, dispositivo, shuffle, generador=None):
    e = cfg.entrenamiento
    return DataLoader(ds, batch_size=e.batch, shuffle=shuffle, generator=generador, num_workers=e.workers,
                      collate_fn=collate, pin_memory=dispositivo.type == "cuda", drop_last=shuffle)


def _a_dispositivo(lote, dispositivo):
    return (lote[k].to(dispositivo, non_blocking=True) for k in ("imagen", "pos", "valido"))


@torch.no_grad()
def perdida_val(modelo, loader, rejilla, perdida, amp, dispositivo):
    """Perdida media (total y por termino) sobre el set de validacion."""
    modelo.eval()
    sumas, n = {}, 0
    for lote in loader:
        x, pos, val = _a_dispositivo(lote, dispositivo)
        with autocast(amp, dispositivo):
            pred = modelo(x)
        total, v = perdida(pred, rejilla.target(pos, val))
        B = len(x)
        for k, t in {"total": total, **v}.items():
            sumas[k] = sumas.get(k, 0.0) + t * B
        n += B
    return media_ponderada(sumas, n)


def _agregar_fila(ruta, fila):
    nueva = not Path(ruta).exists()
    with open(ruta, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nueva:
            w.writeheader()
        w.writerow(fila)


def _recortar_log(ruta, epoca_inicio):
    """Al reanudar descarta las filas de epocas que no terminaron."""
    if Path(ruta).exists():
        df = pd.read_csv(ruta)
        df[df.epoca < epoca_inicio].to_csv(ruta, index=False)


def _imprimir(msg):
    print(msg, flush=True)                                       # sin buffer: el log se ve aunque stdout vaya a un archivo


def entrenar(cfg, carpeta, reanudar=False, log=_imprimir):
    """Entrena con cfg y escribe todo en carpeta. Con reanudar=True continua
    desde carpeta/ultimo.pt hasta cfg.entrenamiento.epocas. Devuelve log.csv."""
    carpeta = Path(carpeta)
    e = cfg.entrenamiento
    dispositivo = elegir_dispositivo(e.dispositivo)
    amp = modo_amp(e.amp, dispositivo)
    torch.manual_seed(e.semilla)

    rejilla = rejilla_de(cfg)
    train_ds, val_ds = crear_datasets(cfg, rejilla)
    modelo = crear_modelo(cfg, rejilla).to(dispositivo)
    perdida = crear_perdida(cfg, rejilla)
    opt = torch.optim.AdamW(modelo.parameters(), lr=e.lr, weight_decay=e.weight_decay)
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=e.factor_lr, patience=e.paciencia_lr,
                                                      min_lr=e.lr_min)
    escalador = torch.amp.GradScaler(dispositivo.type, enabled=amp == "fp16")
    inicio, mejor_val, sin_mejorar, paso = 0, math.inf, 0, 0
    if reanudar:
        ck = cargar_checkpoint(carpeta / "ultimo.pt", dispositivo)
        modelo.load_state_dict(ck["modelo"])
        opt.load_state_dict(ck["opt"])
        plan.load_state_dict(ck["plan"])
        escalador.load_state_dict(ck["escalador"])
        inicio, mejor_val, sin_mejorar, paso = ck["epoca"] + 1, ck["mejor_val"], ck["sin_mejorar"], ck["paso"]
        for nombre in ("log.csv", "log_iter.csv"):
            _recortar_log(carpeta / nombre, inicio)
    guardar_yaml(a_dict(cfg), carpeta / "config.yaml")

    val_loader = _loader(val_ds, cfg, dispositivo, shuffle=False)
    n_iter = len(train_ds) // e.batch
    log(f"{carpeta}\n  dispositivo {dispositivo}, AMP {amp}; modelo {n_parametros(modelo) / 1e3:.1f} k parametros; "
        f"salida {rejilla.forma}\n  train {len(train_ds)} ({type(train_ds).__name__}), val {len(val_ds)}; "
        f"batch {e.batch} x {e.acumulacion} -> {n_iter} iteraciones por epoca; epocas {inicio + 1}..{e.epocas}")
    if n_iter == 0:
        raise ValueError(f"n_train = {len(train_ds)} es menor que el batch ({e.batch})")

    t_corrida = time.time()
    for epoca in range(inicio, e.epocas):
        if hasattr(train_ds, "fijar_epoca"):
            train_ds.fijar_epoca(epoca)
        gen = torch.Generator().manual_seed(e.semilla * 1_000_003 + epoca)
        modelo.train()
        t_ep = t_log = time.time()
        sumas_ep, sumas_log, n_ep, n_log = {}, {}, 0, 0
        for i, lote in enumerate(_loader(train_ds, cfg, dispositivo, shuffle=True, generador=gen)):
            x, pos, val = _a_dispositivo(lote, dispositivo)
            with autocast(amp, dispositivo):
                pred = modelo(x)
            total, v = perdida(pred, rejilla.target(pos, val))
            escalador.scale(total / e.acumulacion).backward()
            if (i + 1) % e.acumulacion == 0 or i + 1 == n_iter:
                if e.clip_grad:
                    escalador.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), e.clip_grad)
                escalador.step(opt)
                escalador.update()
                opt.zero_grad(set_to_none=True)
            paso += 1
            B = len(x)
            for k, t in {"total": total.detach(), **v}.items():  # suma en el dispositivo: sin sincronizar
                sumas_ep[k] = sumas_ep.get(k, 0.0) + t * B
                sumas_log[k] = sumas_log.get(k, 0.0) + t * B
            n_ep, n_log = n_ep + B, n_log + B
            if (i + 1) % e.log_cada == 0 or i + 1 == n_iter:
                medias = media_ponderada(sumas_log, n_log)
                vel = n_log / (time.time() - t_log)
                _agregar_fila(carpeta / "log_iter.csv", {"paso": paso, "epoca": epoca,
                                                         "epoca_continua": epoca + (i + 1) / n_iter,
                                                         "lr": opt.param_groups[0]["lr"], **medias,
                                                         "muestras_s": vel})
                log(f"  epoca {epoca + 1} it {i + 1}/{n_iter}: " +
                    ", ".join(f"{k} {m:.4f}" for k, m in medias.items()) + f"  ({vel:.0f} muestras/s)")
                if not es_finito(medias):
                    raise FloatingPointError(f"perdida no finita en la epoca {epoca + 1}, iteracion {i + 1}")
                sumas_log, n_log, t_log = {}, 0, time.time()

        tr = media_ponderada(sumas_ep, n_ep)
        va = perdida_val(modelo, val_loader, rejilla, perdida, amp, dispositivo)
        plan.step(va["total"])
        mejor = va["total"] < mejor_val
        mejor_val, sin_mejorar = (va["total"], 0) if mejor else (mejor_val, sin_mejorar + 1)
        estado = dict(opt=opt.state_dict(), plan=plan.state_dict(), escalador=escalador.state_dict(),
                      epoca=epoca, mejor_val=mejor_val, sin_mejorar=sin_mejorar, paso=paso)
        guardar_checkpoint(carpeta / "ultimo.pt", cfg, modelo, **estado)
        if mejor:
            guardar_checkpoint(carpeta / "mejor.pt", cfg, modelo, **estado)
        dt = time.time() - t_ep
        _agregar_fila(carpeta / "log.csv", {"epoca": epoca, "lr": opt.param_groups[0]["lr"],
                                            **{f"train_{k}": m for k, m in tr.items()},
                                            **{f"val_{k}": m for k, m in va.items()},
                                            "mejor": mejor, "t_epoca_s": dt})
        graficar_log(carpeta)
        memoria = (f", memoria pico {torch.cuda.max_memory_allocated() / 2**30:.2f} GB"
                   if dispositivo.type == "cuda" else "")
        log(f"epoca {epoca + 1}/{e.epocas} ({resumen_tiempo(dt)}{memoria}): train {tr['total']:.4f}, "
            f"val {va['total']:.4f} (mse {va.get('mse', float('nan')):.4f}, dice {va.get('dice', float('nan')):.3f})"
            f"{'  <- mejor' if mejor else ''}")
        if e.parada_temprana and sin_mejorar >= e.parada_temprana:
            log(f"parada temprana: {sin_mejorar} epocas sin mejorar en val")
            break
    log(f"listo en {resumen_tiempo(time.time() - t_corrida)}; mejor val {mejor_val:.4f} -> {carpeta / 'mejor.pt'}")
    return pd.read_csv(carpeta / "log.csv")


def main():
    p = agregar_argumentos(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--reanudar", default=None, metavar="CARPETA",
                   help="carpeta de una corrida: continua desde su ultimo.pt con su config.yaml (+ --set)")
    args = p.parse_args()
    if args.reanudar:
        carpeta = Path(args.reanudar)
        cfg = cargar_config(carpeta / "config.yaml", args.set)
    else:
        cfg = config_desde_args(args)
        carpeta = carpeta_nueva(cfg.entrenamiento)
    entrenar(cfg, carpeta, reanudar=bool(args.reanudar))


if __name__ == "__main__":
    main()
