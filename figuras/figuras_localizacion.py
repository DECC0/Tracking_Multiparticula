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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from comun_localizacion import (ESTILO, COLORES, MARCADORES, COLOR_TENUE, TENUE, BRILLANTE, CONJUNTOS,
                                ETIQUETAS_DIST, ruta_evaluacion, comando_evaluate, gt_mismo_conjunto,
                                conjunto_pregenerado, distancia_al_brillante, recall_por_bin,
                                tabla_recall_vs_vecino, graficar_recall_vs_vecino, imagen_electrones,
                                norma_comun, extension_um, dibujar_frame, marcar_emisor, barra_escala, barra_z)

OUT = os.path.dirname(os.path.abspath(__file__))
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

plt.rcParams.update(ESTILO)


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
    ruta = ruta_evaluacion(carpeta, conjunto, archivo)
    if not ruta.exists():
        raise SystemExit(f"falta {ruta}: correr {comando_evaluate(carpeta, conjunto)}")
    return pd.read_csv(ruta)


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
                t = recall_por_bin(gts[(r, c)], columna, bordes)
                ax.plot(t.centro, t.recall, ls, color=color, lw=2, marker=marcador, ms=5,
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

if "vecino" in args.figuras:
    conjunto = "test_snr_mezcla"
    gts = {r: distancia_al_brillante(evaluacion(carpeta, conjunto, "gt_eval.csv")) for r, carpeta, _, _ in REDES}
    gt_mismo_conjunto(list(gts.values()))
    tabla = tabla_recall_vs_vecino(gts)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    graficar_recall_vs_vecino(ax, tabla, REDES)
    guardar(fig, "fig_recall_vs_vecino")

    pd.set_option("display.width", 200)
    print(f"recall de los tenues (< {TENUE:.0f} fotones) de {conjunto} vs distancia al brillante (> {BRILLANTE:.0f}):")
    tabla["recall (IC 95%)"] = [f"{r:.3f} ({lo:.3f}-{hi:.3f})" for r, lo, hi in
                                zip(tabla.recall, tabla.ic95_lo, tabla.ic95_hi)]
    print(tabla.pivot(index="bin", columns="corrida", values="recall (IC 95%)").reindex(
        [e.replace("\n", " ") for e in ETIQUETAS_DIST])[nombres].assign(
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
    imagenes, cfg_sim, px_um = conjunto_pregenerado(conjunto)
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

    ext = extension_um(H, W, px_um)
    # misma escala de grises en los 3 paneles (ADU -> e-)
    img_e = [imagen_electrones(imagenes, f.video_id, cfg_sim) for _, f, _, _, _ in casos]
    norma = norma_comun(img_e)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), gridspec_kw=dict(wspace=0.08))
    for col, (ax, (desc, f, rec, n_cat, n_cand), e) in enumerate(zip(axes, casos, img_e)):
        v, fr = int(f.video_id), int(f.frame)
        gf = g[(g.video_id == v) & (g.frame == fr)]
        dibujar_frame(ax, e, gf, pred[(pred.video_id == v) & (pred.frame == fr)], px_um, norma)
        marcar_emisor(ax, f.x_px, f.y_px, px_um)
        dist = "" if np.isinf(f.dist_brillante) else f", a {f.dist_brillante:.1f} px"
        ax.set_title(f"({'abc'[col]}) {desc}\ntenue de {f.photons:.0f} fotones{dist}\n"
                     f"{'detectado' if f.detectado else 'perdido'}", fontsize=7.5)
        ax.set_xlabel(f"recall de la categoría\n{rec:.2f} (n = {n_cat})", fontsize=7.5, labelpad=3)
        print(f"({'abc'[col]}) {desc}: muestra {v}, particula {int(f.particle_id)}, {f.photons:.0f} fotones"
              f"{dist}, {'detectado' if f.detectado else 'perdido'}; {int(f.n_frame)} emisores en el frame; "
              f"sorteado entre {n_cand} candidatos; recall de la categoria {rec:.3f} (n {n_cat})")

        barra_escala(ax, ext)

    barra_z(fig, axes)
    leyenda = [Line2D([], [], lw=0, marker="o", ms=7, markerfacecolor="none", markeredgecolor="0.3",
                      markeredgewidth=1.4, label="ground truth"),
               Line2D([], [], lw=0, marker="x", ms=6, color="0.3", markeredgewidth=1.6, label=f"detección (red {r_ej})"),
               Line2D([], [], lw=1.1, ls="--", color=COLOR_TENUE, label=f"tenue del caso (< {TENUE:.0f} fotones)")]
    fig.legend(handles=leyenda, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.45, 0.02))
    guardar(fig, "fig_localizaciones_ejemplo")
