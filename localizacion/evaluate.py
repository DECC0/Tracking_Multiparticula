"""
Evaluacion de un checkpoint contra el ground truth: corre la red como
infer.py, empareja las detecciones con el GT (metricas.py) y escribe las
metricas globales, las metricas por fotones, z y SNR del emisor, y figuras.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY localizacion/evaluate.py localizacion/corridas/base_.../mejor.pt simulador_nuevo/salida
    $PY localizacion/evaluate.py CHECKPOINT simulador_nuevo/salida --salida eval_umbral02 --set postproc.umbral=0.2
    $PY localizacion/evaluate.py CHECKPOINT simulador_nuevo/salida/video_003.tif --figuras 10

Entradas como las de infer.py. El ground truth de cada entrada es el
ground_truth.csv de su carpeta (o el de --gt), restringido a los frames
evaluados: se puede evaluar un video suelto con el GT de todo el dataset.
Tolerancia del emparejamiento: --r-xy nm lateral y --r-z nm axial.

Escribe en --salida (por defecto <carpeta del checkpoint>/evaluacion_<primera entrada>/):
    resumen.yaml       metricas globales, checkpoint, entradas, postproc y tolerancias
    por_entrada.csv    metricas de cada entrada (cada video)
    por_fotones.csv    recall, RMSE y sesgo en z por rango de fotones del emisor
    por_z.csv          ... por rango de z
    por_snr.csv        ... por rango de SNR (si el GT trae snr, como el de pregenerar.py)
    metricas.png       las tablas por_*.csv y los histogramas de los errores
    predicciones.csv   detecciones + tp, fuera_fov (emparejada con un emisor fuera del FOV)
    gt_eval.csv        GT + detectado, conf, dx_nm, dy_nm, dz_nm (error de su deteccion)
    figuras/           imagen, detecciones y GT de los primeros --figuras frames
En los videos del simulador cada emisor aparece en varios frames: los conteos
son de pares (emisor, frame), no de emisores independientes.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODOS_AMP, a_dict, guardar_yaml
from infer import PIXELES_POR_BATCH, leer_entradas, camara_de, detecciones_de, graficar_detecciones
from metricas import CLAVES, evaluar, resumen
from utils import elegir_dispositivo, cargar_modelo, con_postproc, resumen_tiempo

N_RANGOS = 8
COLORES = {"lateral": "#2a78d6", "z": "#eb6834", "y": "#1baf7a", "recall": "#444444"}   # paleta categorica
# nombre -> (columna del GT, etiqueta del eje, escala de los rangos)
VARIABLES = {"fotones": ("photons", "fotones del emisor", "log"),
             "z": ("z_nm", "z del emisor [nm]", "lineal"),
             "snr": ("snr", "SNR del pixel pico", "log")}


def ground_truth(entrada, ruta_gt=None, cache=None):
    """GT de los frames de la entrada: el de ruta_gt o el ground_truth.csv de su carpeta."""
    ruta = Path(ruta_gt) if ruta_gt else Path(entrada["config_sim"]).parent / "ground_truth.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"{entrada['nombre']}: no existe {ruta}; pasa el ground truth con --gt")
    cache = {} if cache is None else cache
    if ruta not in cache:
        cache[ruta] = pd.read_csv(ruta)
    evaluados = pd.DataFrame({"video_id": entrada["video_id"], "frame": entrada["frame"]})
    gt = cache[ruta].merge(evaluados, on=CLAVES)
    return gt if "en_fov" in gt else gt.assign(en_fov=True)


# ----------------------------------------------------------------------------
# Metricas por rango
# ----------------------------------------------------------------------------

def _rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float("nan")


def bordes(valores, escala, n=N_RANGOS):
    """n + 1 bordes entre el minimo y el maximo de valores (geometricos con escala log)."""
    lo, hi = float(np.min(valores)), float(np.max(valores))
    if escala == "log" and lo > 0:
        return np.geomspace(lo, hi, n + 1)
    return np.linspace(lo, hi, n + 1)


def por_rangos(gt_eval, columna, b):
    """Recall, RMSE y sesgo en z de los emisores dentro del FOV por rango
    [b_i, b_i+1) de gt_eval[columna]; lo que cae fuera de b va al primer o al
    ultimo rango."""
    g = gt_eval[gt_eval.en_fov]
    k = np.digitize(g[columna].to_numpy(float), b[1:-1])
    filas = []
    for i in range(len(b) - 1):
        s = g[k == i]
        p = s[s.detectado]
        filas.append(dict(desde=b[i], hasta=b[i + 1], n=len(s), detectados=len(p),
                          recall=len(p) / len(s) if len(s) else float("nan"),
                          rmse_lat_nm=_rms(np.hypot(p.dx_nm, p.dy_nm)), rmse_z_nm=_rms(p.dz_nm),
                          sesgo_z_nm=float(p.dz_nm.mean()) if len(p) else float("nan")))
    return pd.DataFrame(filas)


def tablas_por_rango(gt_eval, rejilla):
    """{nombre: tabla de por_rangos} de las VARIABLES que trae el GT; z en
    rangos iguales sobre el z_rango de entrenamiento."""
    g = gt_eval[gt_eval.en_fov]
    tablas = {}
    if len(g) == 0:
        return tablas
    for nombre, (col, _, escala) in VARIABLES.items():
        if col not in g:
            continue
        b = np.linspace(rejilla.z_min, rejilla.z_max, N_RANGOS + 1) if col == "z_nm" else bordes(g[col], escala)
        tablas[nombre] = por_rangos(gt_eval, col, b)
    return tablas


def graficar_metricas(tablas, gt_eval, r, r_xy, r_z, titulo, ruta):
    """metricas.png: recall (arriba) y RMSE (abajo) por rango de cada variable,
    y en la ultima columna los histogramas de los errores lateral y axial."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    n = len(tablas)
    fig, axes = plt.subplots(2, n + 1, figsize=(4.4 * (n + 1), 7.2), squeeze=False)
    for j, (nombre, t) in enumerate(tablas.items()):
        _, etiqueta, escala = VARIABLES[nombre]
        log = escala == "log" and t.desde.iloc[0] > 0
        x = np.sqrt(t.desde * t.hasta) if log else (t.desde + t.hasta) / 2
        axes[0, j].plot(x, t.recall, "o-", color=COLORES["recall"], lw=2, ms=6)
        axes[0, j].set_ylim(0, 1.03)
        axes[0, j].set_ylabel("recall")
        axes[1, j].plot(x, t.rmse_lat_nm, "o-", color=COLORES["lateral"], lw=2, ms=6, label="lateral")
        axes[1, j].plot(x, t.rmse_z_nm, "o-", color=COLORES["z"], lw=2, ms=6, label="axial (z)")
        axes[1, j].set_ylim(bottom=0)
        axes[1, j].set_ylabel("RMSE [nm]")
        for ax in axes[:, j]:
            ax.set_xlabel(etiqueta)
            if log:                                              # marcas en 1, 2 y 5 por decada, sin notacion cientifica
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
    axes[1, 0].legend(frameon=False, fontsize=9)

    p = gt_eval[gt_eval.en_fov & gt_eval.detectado]
    ax = axes[0, n]
    ax.hist(p.dx_nm, np.linspace(-r_xy, r_xy, 61), histtype="step", lw=2, color=COLORES["lateral"], label="x")
    ax.hist(p.dy_nm, np.linspace(-r_xy, r_xy, 61), histtype="step", lw=2, color=COLORES["y"], label="y")
    ax.set_title(f"error lateral: RMSE {r['rmse_lat_nm']:.1f} nm", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1, n]
    ax.hist(p.dz_nm, np.linspace(-r_z, r_z, 61), histtype="step", lw=2, color=COLORES["z"])
    ax.set_title(f"error axial: RMSE {r['rmse_z_nm']:.1f} nm", fontsize=10)
    for ax in axes[:, n]:
        ax.set_xlabel("deteccion - GT [nm]")
        ax.set_ylabel("pares")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def _conteos_frame(gt_eval, pred_eval, vid, fr):
    """(TP, FP, FN) de un frame."""
    g = gt_eval[(gt_eval.video_id == vid) & (gt_eval.frame == fr) & gt_eval.en_fov]
    p = pred_eval[(pred_eval.video_id == vid) & (pred_eval.frame == fr)]
    return int(p.tp.sum()), int((~p.tp & ~p.fuera_fov).sum()), int((~g.detectado).sum())


# ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("checkpoint", help="mejor.pt o ultimo.pt de una corrida de train.py")
    p.add_argument("entradas", nargs="+", help=".tif, carpetas con .tif o carpetas de pregenerar.py")
    p.add_argument("--gt", default=None, help="ground_truth.csv (por defecto el de la carpeta de cada entrada)")
    p.add_argument("--salida", default=None,
                   help="carpeta de salida (por defecto <carpeta del checkpoint>/evaluacion_<primera entrada>)")
    p.add_argument("--r-xy", type=float, default=150.0, help="tolerancia lateral del emparejamiento [nm]")
    p.add_argument("--r-z", type=float, default=300.0, help="tolerancia axial del emparejamiento [nm]")
    p.add_argument("--figuras", type=int, default=4, metavar="N", help="dibuja los primeros N frames (0: ninguno)")
    p.add_argument("--batch", type=int, default=None, help=f"imagenes por batch (por defecto {PIXELES_POR_BATCH} px / (H W))")
    p.add_argument("--dispositivo", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--amp", default="fp16", choices=MODOS_AMP)
    p.add_argument("--set", nargs="*", default=[], metavar="postproc.CAMPO=VALOR",
                   help="cambia el post-procesado, p. ej. --set postproc.umbral=0.3")
    args = p.parse_args(argv)

    t0 = time.time()
    dispositivo = elegir_dispositivo(args.dispositivo)
    modelo, cfg, rejilla = cargar_modelo(args.checkpoint, dispositivo)
    cfg = con_postproc(cfg, args.set)
    entradas = leer_entradas(args.entradas)
    ckpt = Path(args.checkpoint)
    salida = Path(args.salida) if args.salida else ckpt.parent / f"evaluacion_{Path(args.entradas[0]).stem}"
    salida.mkdir(parents=True, exist_ok=True)
    print(f"{ckpt}: {len(entradas)} entradas, dispositivo {dispositivo}, postproc {a_dict(cfg.postproc)}")

    cache, gts, preds, filas, por_dibujar = {}, [], [], [], args.figuras
    for e in entradas:
        t_e = time.time()
        cs = camara_de(e, cfg)
        det = detecciones_de(modelo, e, cfg, cs, dispositivo, args.amp, args.batch)
        ge, pe = evaluar(det, ground_truth(e, args.gt, cache), cs.pixel_nm, args.r_xy, args.r_z)
        r = resumen(ge, pe)
        gts.append(ge)
        preds.append(pe)
        filas.append({"entrada": e["nombre"], **r})
        print(f"  {e['nombre']}: {len(e['imagenes'])} frames, {r['n_gt']} emisores, {len(det)} detecciones, "
              f"Jaccard {r['jaccard']:.3f} ({resumen_tiempo(time.time() - t_e)})")

        for t in range(min(por_dibujar, len(e["imagenes"]))):
            vid, fr = int(e["video_id"][t]), int(e["frame"][t])
            tp, fp, fn = _conteos_frame(ge, pe, vid, fr)
            (salida / "figuras").mkdir(exist_ok=True)
            graficar_detecciones(np.asarray(e["imagenes"][t]), det[(det.video_id == vid) & (det.frame == fr)],
                                 cs.z_rango_nm, ge[(ge.video_id == vid) & (ge.frame == fr)],
                                 f"{e['nombre']}  video {vid}, frame {fr}: {tp} TP, {fp} FP, {fn} FN",
                                 salida / "figuras" / f"v{vid:03d}_f{fr:04d}.png")
            por_dibujar -= 1

    gt_eval = pd.concat(gts, ignore_index=True)
    pred_eval = pd.concat(preds, ignore_index=True)
    r = resumen(gt_eval, pred_eval)
    tablas = tablas_por_rango(gt_eval, rejilla)
    pred_eval.to_csv(salida / "predicciones.csv", index=False)
    gt_eval.to_csv(salida / "gt_eval.csv", index=False)
    pd.DataFrame(filas).to_csv(salida / "por_entrada.csv", index=False)
    for nombre, t in tablas.items():
        t.to_csv(salida / f"por_{nombre}.csv", index=False)
    if tablas:
        graficar_metricas(tablas, gt_eval, r, args.r_xy, args.r_z,
                          f"{ckpt.parent.name}/{ckpt.name}: Jaccard {r['jaccard']:.3f}, recall {r['recall']:.3f}, "
                          f"precision {r['precision']:.3f}", salida / "metricas.png")
    guardar_yaml({"checkpoint": str(ckpt), "entradas": [str(x) for x in args.entradas],
                  "ground_truth": [str(x) for x in cache], "postproc": a_dict(cfg.postproc),
                  "r_xy_nm": args.r_xy, "r_z_nm": args.r_z, "amp": args.amp,
                  "n_imagenes": sum(len(e["imagenes"]) for e in entradas), "metricas": r}, salida / "resumen.yaml")

    print(f"\nJaccard {r['jaccard']:.3f}, recall {r['recall']:.3f}, precision {r['precision']:.3f}  "
          f"(TP {r['tp']}, FP {r['fp']}, FN {r['fn']})")
    print(f"RMSE lateral {r['rmse_lat_nm']:.1f} nm (x {r['rmse_x_nm']:.1f}, y {r['rmse_y_nm']:.1f}), "
          f"axial {r['rmse_z_nm']:.1f} nm; sesgo x {r['sesgo_x_nm']:+.1f}, y {r['sesgo_y_nm']:+.1f}, "
          f"z {r['sesgo_z_nm']:+.1f} nm")
    for nombre, t in tablas.items():
        print(f"\npor {nombre}:\n" + t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n-> {salida}/  ({resumen_tiempo(time.time() - t0)})")
    return r


if __name__ == "__main__":
    main()
