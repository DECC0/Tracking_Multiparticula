"""
Piezas comunes de las figuras de la red de localizacion, usadas por
figuras_localizacion.py y analisis_corridas.ipynb: estilo, paleta, rutas de las
evaluaciones de evaluate.py, distancia de cada emisor al brillante mas cercano,
intervalos de Wilson, recall por bin, recall de los tenues vs distancia al
brillante y el dibujo de un frame con GT y detecciones.

Solo lee lo que ya existe (log.csv, evaluacion_<conjunto>/ de evaluate.py y los
conjuntos de localizacion/datos/ de pregenerar.py); no corre la red.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "localizacion" / "datos"

ESTILO = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
}

# orden fijo por corrida: los dos primeros son los de figuras_documento.py (azul, naranja)
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MARCADORES = ["o", "s", "^", "D", "v", "P"]
CMAP_Z = "RdBu_r"
NORM_Z = TwoSlopeNorm(vmin=-800, vcenter=0, vmax=800)
COLOR_TENUE = "#ffd23f"

# tenue < TENUE fotones, brillante > BRILLANTE fotones (emisores en el FOV)
TENUE, BRILLANTE = 3600.0, 10000.0
BINS_DIST = [(0, 5), (5, 10), (10, 20), (20, 40), (40, np.inf)]   # px al brillante mas cercano

# conjunto de prueba -> (estilo de linea, descripcion)
CONJUNTOS = [("test_snr", "-", "test_snr (brillo homogéneo)"),
             ("test_snr_mezcla", "--", "test_snr_mezcla (tenues junto a brillantes)")]


# ----------------------------------------------------------------------------
# Evaluaciones de evaluate.py
# ----------------------------------------------------------------------------

def ruta_evaluacion(carpeta, conjunto, archivo):
    return Path(carpeta) / f"evaluacion_{conjunto}" / archivo


def comando_evaluate(carpeta, conjunto):
    """Comando que escribe <carpeta>/evaluacion_<conjunto>/ (salida por defecto de evaluate.py)."""
    ckpt = Path(carpeta).resolve() / "mejor.pt"
    try:
        ckpt = ckpt.relative_to(RAIZ)
    except ValueError:
        pass
    return f"$PY localizacion/evaluate.py {ckpt} localizacion/datos/{conjunto}"


def gt_mismo_conjunto(gts):
    """Las evaluaciones de un conjunto deben tener el mismo GT (mismas filas)."""
    clave = ["video_id", "frame", "particle_id", "photons"]
    ref = gts[0][clave].reset_index(drop=True)
    for g in gts[1:]:
        if not g[clave].reset_index(drop=True).equals(ref):
            raise ValueError("las evaluaciones no son sobre el mismo conjunto (GT distinto)")


def conjunto_pregenerado(conjunto):
    """(imagenes (n, C, H, W) uint16 en mmap, config del simulador (dict), um por pixel)
    de un conjunto de pregenerar.py en localizacion/datos/."""
    carpeta = DATOS / conjunto
    cfg_sim = json.loads((carpeta / "config_simulador.json").read_text())["config"]
    px_um = cfg_sim["pixel_camara_um"] / cfg_sim["magnificacion"]
    return np.load(carpeta / "imagenes.npy", mmap_mode="r"), cfg_sim, px_um


# ----------------------------------------------------------------------------
# Estadistica
# ----------------------------------------------------------------------------

def distancia_al_brillante(g):
    """Emisores en el FOV con la distancia (px) al brillante (> BRILLANTE) mas cercano de su
    imagen, sin contarse a si mismo; inf si no hay. Los brillantes pueden estar fuera del FOV
    (dentro del margen del simulador), porque su luz tambien cae en la imagen."""
    g = g.copy()
    dist = np.full(len(g), np.inf)
    pos = {k: i for i, k in enumerate(g.index)}
    for _, d in g.groupby(["video_id", "frame"]):
        b = d[d.photons > BRILLANTE]
        if len(b) == 0:
            continue
        dd = np.hypot(d.x_px.to_numpy()[:, None] - b.x_px.to_numpy()[None],
                      d.y_px.to_numpy()[:, None] - b.y_px.to_numpy()[None])
        dd[d.index.to_numpy()[:, None] == b.index.to_numpy()[None]] = np.inf
        dist[[pos[i] for i in d.index]] = dd.min(axis=1)
    g["dist_brillante"] = dist
    g = g[g.en_fov].copy()
    g["n_frame"] = g.groupby(["video_id", "frame"]).particle_id.transform("size")
    return g


def wilson(k, n, z=1.96):
    """Intervalo de confianza de 95% (Wilson) de una proporcion k/n."""
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    w = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return c - w, c + w


def centros_bin(bordes, log=True):
    """Centro geometrico (ejes log) o aritmetico de cada bin."""
    bordes = np.asarray(bordes, dtype=float)
    return np.sqrt(bordes[:-1] * bordes[1:]) if log else (bordes[:-1] + bordes[1:]) / 2


def recall_por_bin(g, columna, bordes, n_min=30, log=True):
    """Recall de los emisores de g por bin de g[columna] (bordes): DataFrame con el centro
    del bin (centros_bin), n, detectados, recall e IC 95% de Wilson; descarta bins con n < n_min."""
    b = pd.cut(g[columna], bordes, include_lowest=True)
    t = g.groupby(b, observed=False).detectado.agg(["sum", "size"])
    k, n = t["sum"].to_numpy().astype(int), t["size"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        lo, hi = wilson(k, n)
        recall = k / n
    tabla = pd.DataFrame(dict(centro=centros_bin(bordes, log), n=n, detectados=k,
                              recall=recall, ic95_lo=lo, ic95_hi=hi))
    return tabla[tabla.n >= n_min].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Recall de los tenues vs distancia al brillante mas cercano
# ----------------------------------------------------------------------------

def etiqueta_bin(lo, hi):
    return f"≥ {lo:g}" if np.isinf(hi) else f"{lo:g}–{hi:g}"


ETIQUETAS_DIST = [etiqueta_bin(lo, hi) for lo, hi in BINS_DIST] + ["sin\nbrillante"]


def tabla_recall_vs_vecino(gts):
    """gts: {corrida: distancia_al_brillante(gt_eval)} -> una fila por (corrida, bin de
    distancia) con n, detectados, recall e IC 95% de Wilson de los tenues (< TENUE)."""
    filas = []
    for r, g in gts.items():
        t = g[g.photons < TENUE]
        grupos = [t[(t.dist_brillante >= lo) & (t.dist_brillante < hi)] for lo, hi in BINS_DIST]
        grupos.append(t[np.isinf(t.dist_brillante)])
        for et, d in zip(ETIQUETAS_DIST, grupos):
            k, n = int(d.detectado.sum()), len(d)
            lo, hi = wilson(k, n) if n else (np.nan, np.nan)
            filas.append(dict(corrida=r, bin=et.replace("\n", " "), n=n, detectados=k,
                              recall=k / n if n else np.nan, ic95_lo=lo, ic95_hi=hi))
    return pd.DataFrame(filas)


def graficar_recall_vs_vecino(ax, tabla, redes):
    """Recall de los tenues por bin de distancia, una serie por corrida de redes
    [(nombre, carpeta, color, marcador)], con barras de IC 95%."""
    xs = np.r_[np.arange(len(BINS_DIST)), len(BINS_DIST) + 0.6]   # "sin brillante" separado
    desfase = (np.arange(len(redes)) - (len(redes) - 1) / 2) * 0.08
    for (r, _, color, marcador), dx in zip(redes, desfase):
        d = tabla[tabla.corrida == r]
        y = d.recall.to_numpy()
        err = np.vstack([y - d.ic95_lo.to_numpy(), d.ic95_hi.to_numpy() - y])
        nb = len(BINS_DIST)
        ax.plot(xs[:nb] + dx, y[:nb], "-", color=color, lw=2)
        ax.errorbar(xs + dx, y, yerr=err, fmt=marcador, color=color, ms=5, lw=1.2, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.6, label=f"red {r}", zorder=3)
    n_bin = tabla[tabla.corrida == redes[0][0]].n.to_numpy()
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{e}\nn = {n}" for e, n in zip(ETIQUETAS_DIST, n_bin)], fontsize=7.5)
    ax.axvline(len(BINS_DIST) - 0.2, color="0.75", lw=0.6)
    ax.set_xlabel(f"Distancia al emisor brillante (> {BRILLANTE:.0f} fotones) más cercano [px]")
    ax.set_ylabel(f"Recall de los tenues (< {TENUE:.0f} fotones)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(frameon=False, loc="upper left")


# ----------------------------------------------------------------------------
# Frames con GT y detecciones
# ----------------------------------------------------------------------------

def imagen_electrones(imagenes, video_id, cfg_sim):
    """Canal 0 de la muestra video_id de un conjunto pregenerado, en e- (ADU -> e-)."""
    return (imagenes[int(video_id), 0].astype(float) - cfg_sim["offset"]) * cfg_sim["gain"]


def norma_comun(imgs_e):
    """Misma escala de grises para varias imagenes (e-), con raiz cuadrada para que el
    tenue se vea junto al brillante."""
    return PowerNorm(0.5, vmin=max(0.0, min(np.percentile(e, 1) for e in imgs_e)),
                     vmax=max(np.percentile(e, 99.9) for e in imgs_e))


def extension_um(H, W, px_um):
    """extent de imshow en um: pixel (i, j) centrado en (x = j, y = i)."""
    return np.array([-0.5, W - 0.5, H - 0.5, -0.5]) * px_um


def dibujar_frame(ax, img_e, gt_frame, pred_frame, px_um, norma):
    """Imagen (e-) con el GT (circulos) y las detecciones (cruces), ambos con z en color."""
    H, W = img_e.shape
    ax.imshow(img_e, cmap="gray", norm=norma, extent=extension_um(H, W, px_um), interpolation="nearest")
    ax.scatter(gt_frame.x_px * px_um, gt_frame.y_px * px_um, s=90, facecolors="none",
               edgecolors="white", linewidths=2.6, zorder=3)
    ax.scatter(gt_frame.x_px * px_um, gt_frame.y_px * px_um, s=90, facecolors="none",
               edgecolors=plt.get_cmap(CMAP_Z)(NORM_Z(gt_frame.z_nm.to_numpy())), linewidths=1.4, zorder=4)
    cr = ax.scatter(pred_frame.x_px * px_um, pred_frame.y_px * px_um, s=40, marker="x", c=pred_frame.z_nm,
                    cmap=CMAP_Z, norm=NORM_Z, linewidths=1.6, zorder=6)
    cr.set_path_effects([patheffects.withStroke(linewidth=3.2, foreground="black")])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def marcar_emisor(ax, x_px, y_px, px_um):
    """Recuadro discontinuo alrededor de un emisor (el tenue del caso)."""
    lado = 3.2 * px_um
    ax.add_patch(Rectangle((x_px * px_um - lado, y_px * px_um - lado), 2 * lado, 2 * lado,
                           fill=False, edgecolor=COLOR_TENUE, lw=1.1, ls="--", zorder=5))


def barra_escala(ax, ext):
    """Barra de escala de 1 um abajo a la izquierda (como figuras_documento.py); ext en um."""
    x0, yb = ext[0] + 0.25, ext[2] - 0.35
    ax.add_patch(Rectangle((x0 - 0.13, yb - 0.5), 1.36, 0.68, facecolor="black",
                           alpha=0.7, edgecolor="none", zorder=7))
    ax.plot([x0 + 0.05, x0 + 1.05], [yb, yb], color="white", lw=2, solid_capstyle="butt", zorder=8)
    ax.text(x0 + 0.55, yb - 0.12, "1 µm", color="white", fontsize=7, ha="center", va="bottom", zorder=8)


def barra_z(fig, axes):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM_Z, cmap=CMAP_Z), ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("z [nm]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb
