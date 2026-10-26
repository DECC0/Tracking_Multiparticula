"""
Inferencia: videos o imagenes en ADU -> CSV de detecciones, una fila por emisor:
    video_id, frame, x_px, y_px, z_nm, conf
(x_px, y_px con la convencion del simulador; conf segun postproc.confianza).

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY localizacion/infer.py localizacion/corridas/base_.../mejor.pt simulador_nuevo/salida --salida pred.csv
    $PY localizacion/infer.py CHECKPOINT video_000.tif video_001.tif --set postproc.umbral=0.3
    $PY localizacion/infer.py CHECKPOINT simulador_nuevo/salida --figuras 4 --gt simulador_nuevo/salida/ground_truth.csv

Entradas:
    - archivos .tif (T x H x W, como los del simulador): video_id = numero al
      final del nombre (video_007.tif -> 7) o, si no tiene, su orden
    - carpetas con .tif
    - carpetas de pregenerar.py (imagenes.npy): cada imagen es un video de un
      frame con video_id = su indice, como en su ground_truth.csv
La camara (ganancia, offset) para normalizar sale del config.json del
simulador junto a la entrada (config_simulador.json en las de pregenerar.py)
si existe; si no, de la configuracion de entrenamiento. La red es
convolucional: H y W pueden ser distintos a los de entrenamiento.

Escribe <salida>.csv y <salida>.yaml (checkpoint, entradas, postproc usado) y,
con --figuras N, <salida>_figuras/ con las detecciones de los primeros N frames.
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

from config import MODOS_AMP, a_dict, guardar_yaml
from data.dataset import normalizar
from data.simulacion import sim, config_simulador
from postproc import COLUMNAS_PRED, detectar
from utils import elegir_dispositivo, modo_amp, autocast, rejilla_de, cargar_modelo, con_postproc, resumen_tiempo

PIXELES_POR_BATCH = 8 * 64 * 64         # batch por defecto: 8 imagenes de 64 x 64 (~1 GB en la RTX 3060)


# ----------------------------------------------------------------------------
# Entradas
# ----------------------------------------------------------------------------

def _video_id(ruta, por_defecto):
    m = re.search(r"(\d+)$", ruta.stem)
    return int(m.group(1)) if m else por_defecto


def _entrada_tif(ruta, por_defecto):
    imagenes = tifffile.imread(ruta)
    if imagenes.ndim == 2:
        imagenes = imagenes[None]
    if imagenes.ndim != 3:
        raise ValueError(f"{ruta}: se espera un video T x H x W (forma {imagenes.shape})")
    T = len(imagenes)
    vid = _video_id(ruta, por_defecto)
    return dict(nombre=ruta.name, imagenes=imagenes, video_id=np.full(T, vid), frame=np.arange(T),
                config_sim=ruta.parent / "config.json")


def leer_entradas(rutas):
    """Lista de entradas: dicts con nombre, imagenes (T, H, W) en ADU,
    video_id y frame (T,) de cada imagen, y la ruta del config del simulador."""
    entradas = []
    for r in map(Path, rutas):
        if r.is_dir() and (r / "imagenes.npy").exists():
            imagenes = np.load(r / "imagenes.npy", mmap_mode="r")
            if imagenes.shape[1] != 1:
                raise NotImplementedError("entradas con C > 1 canales quedan pendientes del simulador biplano")
            n = len(imagenes)
            entradas.append(dict(nombre=r.name, imagenes=imagenes[:, 0], video_id=np.arange(n),
                                 frame=np.zeros(n, int), config_sim=r / "config_simulador.json"))
        elif r.is_dir():
            tifs = sorted(r.glob("*.tif"))
            if not tifs:
                raise FileNotFoundError(f"{r}: no hay .tif ni imagenes.npy")
            entradas += [_entrada_tif(t, len(entradas) + i) for i, t in enumerate(tifs)]
        elif r.suffix.lower() in (".tif", ".tiff"):
            entradas.append(_entrada_tif(r, len(entradas)))
        else:
            raise ValueError(f"entrada no reconocida (se espera .tif o carpeta): {r}")
    ids = [(e["nombre"], int(v)) for e in entradas for v in np.unique(e["video_id"])]
    vistos = {}
    for nombre, v in ids:
        if v in vistos:
            raise ValueError(f"video_id {v} repetido ({vistos[v]} y {nombre}): renombra o infiere por separado")
        vistos[v] = nombre
    return entradas


def camara_de(entrada, cfg, avisar=print):
    """simulador.Config con la camara de la entrada (su config del simulador) o la de entrenamiento."""
    ruta = entrada["config_sim"]
    entreno = config_simulador(cfg.datos, cfg.datos.seed_train)
    if ruta is None or not Path(ruta).exists():
        return entreno
    cs = sim.cargar_config(ruta)
    distintos = [c for c in ("mascara", "pixel_nm", "NA", "lambda_nm") if getattr(cs, c) != getattr(entreno, c)]
    if distintos:
        avisar(f"  aviso: {entrada['nombre']} difiere de los datos de entrenamiento en " +
               ", ".join(f"{c} ({getattr(cs, c)} vs {getattr(entreno, c)})" for c in distintos))
    return cs


# ----------------------------------------------------------------------------
# Red + post-procesado
# ----------------------------------------------------------------------------

def batch_por_defecto(H, W):
    return max(1, PIXELES_POR_BATCH // (H * W))


@torch.no_grad()
def predecir(modelo, imagenes, cfg, cfg_sim, dispositivo, amp="fp16"):
    """Salida de la red (B, Nz, s H, s W) para imagenes (B, H, W) en ADU."""
    x = normalizar(np.asarray(imagenes), cfg.datos, cfg_sim)[:, None]
    with autocast(modo_amp(amp, dispositivo), dispositivo):
        return modelo(torch.from_numpy(x).to(dispositivo)).float()


def localizar(modelo, imagenes, cfg, cfg_sim, dispositivo, amp="fp16", batch=None):
    """Detecciones de imagenes (T, H, W) en ADU: DataFrame con t (indice de
    la imagen), x_px, y_px, z_nm y conf, con cfg.postproc."""
    if cfg.datos.canales != 1:
        raise NotImplementedError("entradas con C > 1 canales quedan pendientes del simulador biplano")
    T, H, W = imagenes.shape
    rejilla = rejilla_de(cfg, H, W)
    batch = batch or batch_por_defecto(H, W)
    partes = []
    for i0 in range(0, T, batch):
        det = detectar(predecir(modelo, imagenes[i0:i0 + batch], cfg, cfg_sim, dispositivo, amp), rejilla, cfg.postproc)
        det["b"] += i0
        partes.append(det)
    return pd.concat(partes, ignore_index=True).rename(columns={"b": "t"})


def detecciones_de(modelo, entrada, cfg, cfg_sim, dispositivo, amp="fp16", batch=None):
    """Detecciones (COLUMNAS_PRED) de una entrada de leer_entradas."""
    det = localizar(modelo, entrada["imagenes"], cfg, cfg_sim, dispositivo, amp, batch)
    t = det.pop("t").to_numpy()
    det.insert(0, "video_id", entrada["video_id"][t])
    det.insert(1, "frame", entrada["frame"][t])
    return det[COLUMNAS_PRED]


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------

def dibujar_detecciones(axy, axz, imagen, det, z_rango_nm, gt=None):
    """Dibuja en dos ejes la imagen con las detecciones (+, color = z) y el GT
    (circulos), en xy y en xz. Devuelve el scatter de las detecciones (para la
    barra de color)."""
    H, W = imagen.shape
    axy.imshow(imagen, cmap="gray", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    zk = dict(c=det.z_nm, cmap="plasma", vmin=z_rango_nm[0], vmax=z_rango_nm[1], marker="+", s=70, lw=1.5)
    ck = dict(marker="o", facecolors="none", edgecolors="#1baf7a", s=110, lw=1.3)
    if gt is not None:
        axy.scatter(gt.x_px, gt.y_px, label="GT", **ck)
        axz.scatter(gt.x_px, gt.z_nm, label="GT", **ck)
    sc = axy.scatter(det.x_px, det.y_px, label="deteccion", **zk)
    axz.scatter(det.x_px, det.z_nm, label="deteccion", **zk)
    axy.set_xlim(-0.5, W - 0.5); axy.set_ylim(H - 0.5, -0.5)
    axy.set_xlabel("x [px]"); axy.set_ylabel("y [px]")
    margen = 0.04 * (z_rango_nm[1] - z_rango_nm[0])
    axz.set_xlim(-0.5, W - 0.5); axz.set_ylim(z_rango_nm[0] - margen, z_rango_nm[1] + margen)
    axz.set_xlabel("x [px]"); axz.set_ylabel("z [nm]")
    axz.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D                          # leyenda en tinta neutra: el color de las + es z
    marcas = [Line2D([], [], marker="+", ls="none", color="#444444", ms=9, mew=1.5, label="deteccion")]
    if gt is not None:
        marcas.insert(0, Line2D([], [], marker="o", ls="none", mfc="none", mec="#1baf7a", ms=10, mew=1.3, label="GT"))
    axz.legend(handles=marcas, frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    return sc


def graficar_detecciones(imagen, det, z_rango_nm, gt=None, titulo="", ruta=None):
    """Figura de dibujar_detecciones; con ruta la guarda y la cierra."""
    import matplotlib
    if ruta is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axy, axz) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
    sc = dibujar_detecciones(axy, axz, imagen, det, z_rango_nm, gt)
    fig.colorbar(sc, ax=axz, label="z detectado [nm]", pad=0.02)
    fig.suptitle(titulo)
    fig.tight_layout()
    if ruta is not None:
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
    return fig


# ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("checkpoint", help="mejor.pt o ultimo.pt de una corrida de train.py")
    p.add_argument("entradas", nargs="+", help=".tif, carpetas con .tif o carpetas de pregenerar.py")
    p.add_argument("--salida", default="predicciones.csv")
    p.add_argument("--batch", type=int, default=None, help=f"imagenes por batch (por defecto {PIXELES_POR_BATCH} px / (H W))")
    p.add_argument("--dispositivo", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--amp", default="fp16", choices=MODOS_AMP)
    p.add_argument("--set", nargs="*", default=[], metavar="postproc.CAMPO=VALOR",
                   help="cambia el post-procesado, p. ej. --set postproc.umbral=0.3")
    p.add_argument("--figuras", type=int, default=0, metavar="N", help="dibuja las detecciones de los primeros N frames")
    p.add_argument("--gt", default=None, help="ground_truth.csv para dibujar en las figuras")
    args = p.parse_args(argv)

    t0 = time.time()
    dispositivo = elegir_dispositivo(args.dispositivo)
    modelo, cfg, _ = cargar_modelo(args.checkpoint, dispositivo)
    cfg = con_postproc(cfg, args.set)
    entradas = leer_entradas(args.entradas)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.checkpoint}: {len(entradas)} entradas, dispositivo {dispositivo}, postproc {a_dict(cfg.postproc)}")

    tablas, por_dibujar = [], args.figuras
    gt = pd.read_csv(args.gt) if args.gt else None
    for e in entradas:
        t_e = time.time()
        cs = camara_de(e, cfg)
        det = detecciones_de(modelo, e, cfg, cs, dispositivo, args.amp, args.batch)
        tablas.append(det)
        print(f"  {e['nombre']}: {len(e['imagenes'])} frames, {len(det)} detecciones ({resumen_tiempo(time.time() - t_e)})")

        for t_fig in range(min(por_dibujar, len(e["imagenes"]))):
            vid, fr = int(e["video_id"][t_fig]), int(e["frame"][t_fig])
            carpeta_fig = salida.with_name(salida.stem + "_figuras")
            carpeta_fig.mkdir(exist_ok=True)
            g = gt[(gt.video_id == vid) & (gt.frame == fr)] if gt is not None else None
            graficar_detecciones(np.asarray(e["imagenes"][t_fig]), det[(det.video_id == vid) & (det.frame == fr)],
                                 cs.z_rango_nm, g,
                                 f"{e['nombre']}  video {vid}, frame {fr}", carpeta_fig / f"v{vid:03d}_f{fr:04d}.png")
            por_dibujar -= 1

    pred = pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame(columns=COLUMNAS_PRED)
    pred.to_csv(salida, index=False)
    guardar_yaml({"checkpoint": str(args.checkpoint), "entradas": [str(r) for r in args.entradas],
                  "postproc": a_dict(cfg.postproc), "amp": args.amp, "n_detecciones": len(pred)},
                 salida.with_suffix(".yaml"))
    print(f"{len(pred)} detecciones -> {salida} ({resumen_tiempo(time.time() - t0)})")
    return pred


if __name__ == "__main__":
    main()
