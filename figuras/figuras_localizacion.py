"""
Figuras de la red de localizacion a partir de corridas y evaluaciones ya hechas
(log.csv, gt_eval.csv y predicciones.csv de evaluate.py); no corre la red.

Genera (PNG 300 dpi + PDF vectorial) en figuras/, cada una con 2 o mas corridas
(color y marcador por corrida, en el orden en que se pasan):
  - fig_curvas_entrenamiento<sufijo>  : perdida de val (y train, tenue) por epoca, mejor epoca marcada
  - fig_recall_comparacion<sufijo>    : recall vs fotones y vs SNR sobre test_snr y test_snr_mezcla
  - fig_recall_vs_vecino<sufijo>      : recall de los tenues de test_snr_mezcla vs distancia al
                                        brillante mas cercano de su imagen
  - fig_localizaciones_ejemplo<sufijo>: 3 frames tipicos de test_snr_mezcla (tenue aislado, lejos y
                                        cerca de un brillante) con las detecciones de una corrida

Cada corrida es una carpeta de localizacion/corridas/ con log.csv y las evaluaciones
evaluacion_test_snr/ y evaluacion_test_snr_mezcla/ hechas con evaluate.py; los
ejemplos leen ademas las imagenes de localizacion/datos/test_snr_mezcla/ (pregenerar.py).

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/figuras_localizacion.py localizacion/corridas/hypatia_base_20261026-181641 \\
        localizacion/corridas/hypatia_mezcla_20261026-181651 --nombres base mezcla --ejemplos mezcla
    $PY figuras/figuras_localizacion.py C1 C2 C3 C4 --nombres ... --figuras curvas vecino --sufijo _renovar
"""
import argparse
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

OUT = os.path.dirname(os.path.abspath(__file__))
RAIZ = Path(OUT).parent
DATOS = RAIZ / "localizacion" / "datos"
FIGURAS = ("curvas", "recall", "vecino", "ejemplos")

ap = argparse.ArgumentParser(description="Figuras de la red de localizacion (sin correr la red)")
ap.add_argument("corridas", nargs="+", help="carpetas de corridas (2 o mas), en el orden de color")
ap.add_argument("--nombres", nargs="+", default=None,
                help="nombre de cada corrida en las figuras (por defecto la carpeta sin la fecha)")
ap.add_argument("--figuras", nargs="+", choices=FIGURAS, default=FIGURAS)
ap.add_argument("--ejemplos", default=None,
                help="corrida cuyas detecciones muestra fig_localizaciones_ejemplo (por defecto la ultima)")
ap.add_argument("--sufijo", default="", help="sufijo de los archivos, p. ej. _renovar")
args = ap.parse_args()

plt.rcParams.update({
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
})

# orden fijo por corrida: los dos primeros son los de figuras_documento.py (azul, naranja)
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MARCADORES = ["o", "s", "^", "D", "v", "P"]
CMAP_Z = "RdBu_r"
NORM_Z = TwoSlopeNorm(vmin=-800, vcenter=0, vmax=800)
COLOR_TENUE = "#ffd23f"

# tenue < TENUE fotones, brillante > BRILLANTE fotones (emisores en el FOV)
TENUE, BRILLANTE = 3600.0, 10000.0
BINS_DIST = [(0, 5), (5, 10), (10, 20), (20, 40), (40, np.inf)]   # px al brillante mas cercano


def guardar(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nombre}{args.sufijo}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"OK -> {nombre}{args.sufijo}.png / .pdf")


# ----------------------------------------------------------------------------
# Corridas
# ----------------------------------------------------------------------------

if len(args.corridas) < 2:
    ap.error("se requieren 2 o mas corridas")
if len(args.corridas) > len(COLORES):
    ap.error(f"a lo sumo {len(COLORES)} corridas")
carpetas = [Path(c).resolve() for c in args.corridas]
for c in carpetas:
    if not (c / "log.csv").exists():
        ap.error(f"{c} no es una corrida de train.py (falta log.csv)")
nombres = args.nombres or [re.sub(r"_\d{8}-\d{6}$", "", c.name) for c in carpetas]
if len(nombres) != len(carpetas) or len(set(nombres)) != len(nombres):
    ap.error("--nombres: un nombre distinto por corrida")
REDES = [(n, c, COLORES[i], MARCADORES[i]) for i, (n, c) in enumerate(zip(nombres, carpetas))]
for n, c, _, _ in REDES:
    print(f"{n}: {c}")


def evaluacion(carpeta, conjunto, archivo):
    ruta = carpeta / f"evaluacion_{conjunto}" / archivo
    if not ruta.exists():
        raise SystemExit(f"falta {ruta}: correr localizacion/evaluate.py sobre localizacion/datos/{conjunto}")
    return pd.read_csv(ruta)


def gt_mismo_conjunto(gts):
    """Las evaluaciones de un conjunto deben tener el mismo GT (mismas filas)."""
    clave = ["video_id", "frame", "particle_id", "photons"]
    ref = gts[0][clave].reset_index(drop=True)
    for g in gts[1:]:
        if not g[clave].reset_index(drop=True).equals(ref):
            raise SystemExit("las evaluaciones no son sobre el mismo conjunto (GT distinto)")


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


# ----------------------------------------------------------------------------
# Figura 1: curvas de entrenamiento
# ----------------------------------------------------------------------------

if "curvas" in args.figuras:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    leyenda = []
    ep_max = 0
    for nombre, carpeta, color, _ in REDES:
        log = pd.read_csv(carpeta / "log.csv")
        ep = log.epoca + 1                                  # epocas desde 1, como las imprime train.py
        ep_max = max(ep_max, ep.iloc[-1])
        ax.plot(ep, log.train_total, color=color, lw=1.2, alpha=0.35)
        ax.plot(ep, log.val_total, color=color, lw=2)
        i = int(log.val_total.idxmin())
        ax.plot(ep[i], log.val_total[i], marker="*", ms=12, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        leyenda.append(Line2D([], [], color=color, lw=2,
                              label=f"{nombre}: mejor época {ep[i]} (val {log.val_total[i]:.3f})"))
        print(f"{nombre}: mejor epoca {ep[i]} (val {log.val_total[i]:.4f}); "
              f"ultima epoca {ep.iloc[-1]}: train {log.train_total.iloc[-1]:.4f}, val {log.val_total.iloc[-1]:.4f}")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (MSE + Dice)")
    ax.set_xlim(0.5, ep_max + 0.5)
    ax.grid(alpha=0.25, lw=0.5)
    leyenda += [Line2D([], [], color="0.4", lw=2, label="validación"),
                Line2D([], [], color="0.4", lw=1.2, alpha=0.35, label="entrenamiento"),
                Line2D([], [], color="0.4", lw=0, marker="*", ms=10, label="mejor época (mejor.pt)")]
    ax.legend(handles=leyenda, frameon=False, loc="upper right")
    guardar(fig, "fig_curvas_entrenamiento")


# ----------------------------------------------------------------------------
# Figura 2: recall vs fotones y vs SNR
# ----------------------------------------------------------------------------

CONJUNTOS = [("test_snr", "-", "test_snr (brillo homogéneo)"),
             ("test_snr_mezcla", "--", "test_snr_mezcla (tenues junto a brillantes)")]


def recall_por_bin(g, columna, bordes, n_min=30):
    """Centro geometrico del bin y recall de los emisores en el FOV; descarta bins con n < n_min."""
    b = pd.cut(g[columna], bordes, include_lowest=True)
    t = g.groupby(b, observed=False).detectado.agg(["mean", "size"])
    centros = np.sqrt(bordes[:-1] * bordes[1:])
    ok = t["size"].to_numpy() >= n_min
    return centros[ok], t["mean"].to_numpy()[ok]


if "recall" in args.figuras:
    gts = {(r, c): evaluacion(carpeta, c, "gt_eval.csv").query("en_fov")
           for r, carpeta, _, _ in REDES for c, _, _ in CONJUNTOS}
    for c, _, _ in CONJUNTOS:
        gt_mismo_conjunto([gts[(r, c)] for r, _, _, _ in REDES])
    # mismos bins (log) para todas las curvas de cada panel
    bordes_fot = 1000.0 * 30.0 ** (np.arange(9) / 8)           # 8 bins en [1000, 30000]
    snr = pd.concat([g.snr for g in gts.values()])
    bordes_snr = np.geomspace(min(1.0, snr.min()), snr.max(), 11)   # 10 bins en el rango comun

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.12), sharey=True)
    for ax, (columna, bordes, etiqueta, titulo, ticks) in zip(axes, [
            ("photons", bordes_fot, "Fotones por emisor", "(a) Recall vs fotones", [1000, 3000, 10000, 30000]),
            ("snr", bordes_snr, "SNR del píxel pico", "(b) Recall vs SNR", [1, 2, 3, 5, 10, 20])]):
        for r, _, color, marcador in REDES:
            for c, ls, _ in CONJUNTOS:
                x, y = recall_por_bin(gts[(r, c)], columna, bordes)
                ax.plot(x, y, ls, color=color, lw=2, marker=marcador, ms=5,
                        markerfacecolor=color if ls == "-" else "white", markeredgewidth=1.2)
        ax.set_xscale("log")
        ticks = [t for t in ticks if bordes[0] <= t <= bordes[-1]]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(bordes[0], bordes[-1])
        ax.set_xlabel(etiqueta)
        ax.set_title(titulo, loc="left")
        ax.set_ylim(0, 1.02)
        ax.axhline(0.9, color="0.6", lw=0.6, ls=":", zorder=0)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Recall")
    leyenda = [Line2D([], [], color=c, lw=2, marker=m, ms=5, label=f"red {n}") for n, _, c, m in REDES] + [
        Line2D([], [], color="0.4", lw=2, ls=ls, marker="o", ms=5,
               markerfacecolor="0.4" if ls == "-" else "white", label=e) for _, ls, e in CONJUNTOS]
    fig.legend(handles=leyenda, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    guardar(fig, "fig_recall_comparacion")

    for (r, c), g in gts.items():
        tenues = g[g.photons < TENUE]
        print(f"{r:10s} sobre {c:15s}: recall {g.detectado.mean():.3f} (n {len(g)}), "
              f"recall < {TENUE:.0f} fotones {tenues.detectado.mean():.3f} (n {len(tenues)})")


# ----------------------------------------------------------------------------
# Figura 3: recall de los tenues vs distancia al brillante mas cercano
# ----------------------------------------------------------------------------

def etiqueta_bin(lo, hi):
    return f"≥ {lo:g}" if np.isinf(hi) else f"{lo:g}–{hi:g}"


if "vecino" in args.figuras:
    conjunto = "test_snr_mezcla"
    gts = {r: distancia_al_brillante(evaluacion(carpeta, conjunto, "gt_eval.csv")) for r, carpeta, _, _ in REDES}
    gt_mismo_conjunto(list(gts.values()))
    etiquetas = [etiqueta_bin(lo, hi) for lo, hi in BINS_DIST] + ["sin\nbrillante"]

    filas = []
    for r, g in gts.items():
        t = g[g.photons < TENUE]
        grupos = [t[(t.dist_brillante >= lo) & (t.dist_brillante < hi)] for lo, hi in BINS_DIST]
        grupos.append(t[np.isinf(t.dist_brillante)])
        for et, d in zip(etiquetas, grupos):
            k, n = int(d.detectado.sum()), len(d)
            lo, hi = wilson(k, n) if n else (np.nan, np.nan)
            filas.append(dict(corrida=r, bin=et.replace("\n", " "), n=n, detectados=k,
                              recall=k / n if n else np.nan, ic95_lo=lo, ic95_hi=hi))
    tabla = pd.DataFrame(filas)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = np.r_[np.arange(len(BINS_DIST)), len(BINS_DIST) + 0.6]   # "sin brillante" separado
    desfase = (np.arange(len(REDES)) - (len(REDES) - 1) / 2) * 0.08
    for (r, _, color, marcador), dx in zip(REDES, desfase):
        d = tabla[tabla.corrida == r]
        y = d.recall.to_numpy()
        err = np.vstack([y - d.ic95_lo.to_numpy(), d.ic95_hi.to_numpy() - y])
        nb = len(BINS_DIST)
        ax.plot(xs[:nb] + dx, y[:nb], "-", color=color, lw=2)
        ax.errorbar(xs + dx, y, yerr=err, fmt=marcador, color=color, ms=5, lw=1.2, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.6, label=f"red {r}", zorder=3)
    n_bin = tabla[tabla.corrida == REDES[0][0]].n.to_numpy()
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{e}\nn = {n}" for e, n in zip(etiquetas, n_bin)], fontsize=7.5)
    ax.axvline(len(BINS_DIST) - 0.2, color="0.75", lw=0.6)
    ax.set_xlabel("Distancia al emisor brillante (> 10000 fotones) más cercano [px]")
    ax.set_ylabel(f"Recall de los tenues (< {TENUE:.0f} fotones)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(frameon=False, loc="upper left")
    guardar(fig, "fig_recall_vs_vecino")

    pd.set_option("display.width", 200)
    print(f"recall de los tenues (< {TENUE:.0f} fotones) de {conjunto} vs distancia al brillante (> {BRILLANTE:.0f}):")
    tabla["recall (IC 95%)"] = [f"{r:.3f} ({lo:.3f}-{hi:.3f})" for r, lo, hi in
                                zip(tabla.recall, tabla.ic95_lo, tabla.ic95_hi)]
    print(tabla.pivot(index="bin", columns="corrida", values="recall (IC 95%)").reindex(
        [e.replace("\n", " ") for e in etiquetas])[nombres].assign(
        n=tabla.groupby("bin").n.first()).to_string())


# ----------------------------------------------------------------------------
# Figura 4: ejemplos tipicos de localizacion (una corrida)
# ----------------------------------------------------------------------------

if "ejemplos" in args.figuras:
    r_ej = args.ejemplos or nombres[-1]
    if r_ej not in nombres:
        ap.error(f"--ejemplos {r_ej}: no esta en {nombres}")
    carpeta_ej = dict(zip(nombres, carpetas))[r_ej]
    conjunto = "test_snr_mezcla"
    carpeta_datos = DATOS / conjunto
    cfg_sim = json.loads((carpeta_datos / "config_simulador.json").read_text())["config"]
    px_um = cfg_sim["pixel_camara_um"] / cfg_sim["magnificacion"]
    imagenes = np.load(carpeta_datos / "imagenes.npy", mmap_mode="r")
    H, W = imagenes.shape[-2:]

    g = distancia_al_brillante(evaluacion(carpeta_ej, conjunto, "gt_eval.csv"))
    pred = evaluacion(carpeta_ej, conjunto, "predicciones.csv")
    t = g[g.photons < TENUE]
    legible = (t.x_px > 4) & (t.x_px < W - 5) & (t.y_px > 4) & (t.y_px < H - 5) & (t.n_frame <= 8)
    aislado = np.isinf(t.dist_brillante)
    lejos = ~aislado & (t.dist_brillante >= 20)
    cerca = t.dist_brillante < 10
    # categoria -> (emisores de la categoria, condicion del caso mostrado, descripcion)
    categorias = [("sin brillante en la imagen", aislado, t.detectado, "detectado"),
                  ("a ≥ 20 px de un brillante", lejos, pd.Series(True, index=t.index), None),
                  ("a < 10 px de un brillante", cerca, ~t.detectado, "perdido")]
    # caso tipico: sorteo uniforme (semilla fija) entre los frames legibles de la categoria
    # que cumplen la condicion; no se ordena por nada que favorezca a una red
    rng = np.random.default_rng(0)
    casos = []
    for desc, en_cat, cond, _ in categorias:
        cand = t[en_cat & cond & legible]
        fila = cand.iloc[rng.integers(len(cand))]
        rec = t[en_cat].detectado.mean()
        casos.append((desc, fila, rec, int(en_cat.sum()), len(cand)))

    ext = np.array([-0.5, W - 0.5, H - 0.5, -0.5]) * px_um    # pixel (i, j) centrado en (x = j, y = i)
    # misma escala de grises en los 3 paneles (ADU -> e-), con raiz cuadrada para que el
    # tenue se vea junto al brillante
    img_e = [(imagenes[int(f.video_id), 0].astype(float) - cfg_sim["offset"]) * cfg_sim["gain"]
             for _, f, _, _, _ in casos]
    norma = PowerNorm(0.5, vmin=max(0.0, min(np.percentile(e, 1) for e in img_e)),
                      vmax=max(np.percentile(e, 99.9) for e in img_e))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), gridspec_kw=dict(wspace=0.08))
    for col, (ax, (desc, f, rec, n_cat, n_cand), e) in enumerate(zip(axes, casos, img_e)):
        v, fr = int(f.video_id), int(f.frame)
        ax.imshow(e, cmap="gray", norm=norma, extent=ext, interpolation="nearest")
        gf = g[(g.video_id == v) & (g.frame == fr)]
        ax.scatter(gf.x_px * px_um, gf.y_px * px_um, s=90, facecolors="none",
                   edgecolors="white", linewidths=2.6, zorder=3)
        ax.scatter(gf.x_px * px_um, gf.y_px * px_um, s=90, facecolors="none",
                   edgecolors=plt.get_cmap(CMAP_Z)(NORM_Z(gf.z_nm.to_numpy())), linewidths=1.4, zorder=4)
        p = pred[(pred.video_id == v) & (pred.frame == fr)]
        cr = ax.scatter(p.x_px * px_um, p.y_px * px_um, s=40, marker="x", c=p.z_nm,
                        cmap=CMAP_Z, norm=NORM_Z, linewidths=1.6, zorder=6)
        cr.set_path_effects([patheffects.withStroke(linewidth=3.2, foreground="black")])
        lado = 3.2 * px_um
        ax.add_patch(Rectangle((f.x_px * px_um - lado, f.y_px * px_um - lado), 2 * lado, 2 * lado,
                               fill=False, edgecolor=COLOR_TENUE, lw=1.1, ls="--", zorder=5))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        dist = "" if np.isinf(f.dist_brillante) else f", a {f.dist_brillante:.1f} px"
        ax.set_title(f"({'abc'[col]}) {desc}\ntenue de {f.photons:.0f} fotones{dist}\n"
                     f"{'detectado' if f.detectado else 'perdido'}", fontsize=7.5)
        ax.set_xlabel(f"recall de la categoría\n{rec:.2f} (n = {n_cat})", fontsize=7.5, labelpad=3)
        print(f"({'abc'[col]}) {desc}: muestra {v}, particula {int(f.particle_id)}, {f.photons:.0f} fotones"
              f"{dist}, {'detectado' if f.detectado else 'perdido'}; {int(f.n_frame)} emisores en el frame; "
              f"sorteado entre {n_cand} candidatos; recall de la categoria {rec:.3f} (n {n_cat})")

        # barra de escala de 1 um (como figuras_documento.py)
        x0, yb = ext[0] + 0.25, ext[2] - 0.35
        ax.add_patch(Rectangle((x0 - 0.13, yb - 0.5), 1.36, 0.68, facecolor="black",
                               alpha=0.7, edgecolor="none", zorder=7))
        ax.plot([x0 + 0.05, x0 + 1.05], [yb, yb], color="white", lw=2, solid_capstyle="butt", zorder=8)
        ax.text(x0 + 0.55, yb - 0.12, "1 µm", color="white", fontsize=7, ha="center", va="bottom", zorder=8)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM_Z, cmap=CMAP_Z), ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("z [nm]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    leyenda = [Line2D([], [], lw=0, marker="o", ms=7, markerfacecolor="none", markeredgecolor="0.3",
                      markeredgewidth=1.4, label="ground truth"),
               Line2D([], [], lw=0, marker="x", ms=6, color="0.3", markeredgewidth=1.6, label=f"detección (red {r_ej})"),
               Line2D([], [], lw=1.1, ls="--", color=COLOR_TENUE, label=f"tenue del caso (< {TENUE:.0f} fotones)")]
    fig.legend(handles=leyenda, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.45, 0.02))
    guardar(fig, "fig_localizaciones_ejemplo")
