"""
Figuras para el documento a partir de sim_imagenesPSF.py.

Genera (PNG 300 dpi + PDF vectorial) en figuras/:
  - fig_banco_psf_montaje : banco de PSF en z, sin mascara vs esferica (Z4^0)
  - fig_banco_psf_diapo   : version reducida del montaje para diapositiva
  - fig_banco_psf_axial   : cortes XZ de ambos bancos + pico y energia encerrada vs z
  - fig_cuadros_gt        : 3 cuadros simulados de videos distintos y su ground truth
  - fig_cuadros_gt_simple : igual, sin N/SNR en los titulos y escala en fotones

Uso:
    ~/miniforge3/envs/tesis/bin/python figuras/figuras_documento.py [montaje] [diapo] [axial] [cuadros] [cuadros_simple]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sim_imagenesPSF import (Config, BancoPSF, generar_video, energia_encerrada,
                             generar_trayectorias, generar_fondo, muestrear_fotones)

OUT = os.path.dirname(os.path.abspath(__file__))
# figuras a generar: todas por defecto, o las pasadas por argumento
# (montaje, diapo, axial, cuadros, cuadros_simple), p. ej. `python figuras_documento.py cuadros`
FIGURAS = set(sys.argv[1:]) or {"montaje", "diapo", "axial", "cuadros", "cuadros_simple"}

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

COLOR_SIN = "#2a78d6"   # azul  -> sin mascara
COLOR_ESF = "#eb6834"   # naranja -> esferica
CMAP_IMG = "inferno"


def guardar(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nombre}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"OK -> {nombre}.png / .pdf")


def recorte(p, half):
    c = p.shape[0] // 2
    return p[c - half:c + half, c - half:c + half]


# ----------------------------------------------------------------------------
# Bancos de PSF
# ----------------------------------------------------------------------------

cfg_sin = Config(mascara="ninguna")
cfg_esf = Config(mascara="esferica")
banco_sin = BancoPSF(cfg_sin)
banco_esf = BancoPSF(cfg_esf)
px_um = cfg_esf.pixel_nm / 1000.0

bancos = [("Sin máscara", cfg_sin, banco_sin, COLOR_SIN),
          (r"Esférica ($Z_4^0$, $1.5\pi$)", cfg_esf, banco_esf, COLOR_ESF)]


# ----------------------------------------------------------------------------
# Figura 1: montaje del banco en z
# ----------------------------------------------------------------------------

half = 20                                   # recorte de 40 px ~ 3.9 um de lado

if "montaje" in FIGURAS:
    zs = [-800, -600, -400, -200, 0, 200, 400, 600, 800]
    ext = np.array([-half, half, half, -half]) * px_um

    fig, axes = plt.subplots(2, len(zs), figsize=(7.2, 2.25),
                             gridspec_kw=dict(wspace=0.04, hspace=0.08))
    # solo en esta figura (la axial conserva las de bancos); la esferica va en
    # dos lineas para que la etiqueta rotada no invada la fila de arriba
    etiquetas = ["Sin aberración", "Esférica\n" + r"($Z_4^0$, $1.5\pi$)"]
    for fila, ((_, cfg, banco, _), nombre) in enumerate(zip(bancos, etiquetas)):
        p0 = banco.psf_en_z(0).max()            # cada fila normalizada a su pico en foco
        for col, z in enumerate(zs):
            ax = axes[fila, col]
            p = banco.psf_en_z(z)
            im = ax.imshow(recorte(p, half) / p0, cmap=CMAP_IMG, vmin=0, vmax=1,
                           extent=ext, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(0.04, 0.96, f"{p.max()/p0:.2f}", transform=ax.transAxes,
                    color="white", fontsize=7, va="top")
            if fila == 0:
                ax.set_title(f"{z}", fontsize=8)
            if col == 0:
                # p_max en foco (fraccion de fotones en el pixel pico), del banco
                ax.set_ylabel(f"{nombre}\n" + rf"$p_{{\max}}$ = {p0:.3f}", fontsize=8)

    # barra de escala de 1 um en el ultimo panel
    ax = axes[-1, -1]
    x0 = ext[1] - 0.6 - 1.0                     # metida hacia dentro para que "1 um" no se corte
    y0 = ext[2] - 0.35
    ax.plot([x0, x0 + 1.0], [y0, y0], color="white", lw=2, solid_capstyle="butt")
    ax.text(x0 + 0.5, y0 - 0.12, "1 µm", color="white", fontsize=7,
            ha="center", va="bottom")

    fig.suptitle("z [nm]", fontsize=8, y=1.02)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("Intensidad / pico en foco", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    guardar(fig, "fig_banco_psf_montaje")


# ----------------------------------------------------------------------------
# Figura 1b: montaje reducido para diapositiva (fuentes grandes, sin colorbar)
# ----------------------------------------------------------------------------

if "diapo" in FIGURAS:
    zs = [-800, -400, 0, 400, 800]
    ext = np.array([-half, half, half, -half]) * px_um
    # contorno oscuro fino para que los numeros blancos se lean sobre la PSF
    contorno = [patheffects.withStroke(linewidth=2, foreground="black")]

    fig, axes = plt.subplots(2, len(zs), figsize=(10, 4.5), facecolor="white",
                             gridspec_kw=dict(wspace=0.04, hspace=0.06))
    # etiquetas horizontales a la izquierda: rotadas no caben en la altura de la fila
    etiquetas = ["PSF\nestándar", "PSF con\naberración\nesférica"]
    for fila, ((_, cfg, banco, _), nombre) in enumerate(zip(bancos, etiquetas)):
        p0 = banco.psf_en_z(0).max()            # cada fila normalizada a su pico en foco
        for col, z in enumerate(zs):
            ax = axes[fila, col]
            p = banco.psf_en_z(z)
            ax.imshow(recorte(p, half) / p0, cmap=CMAP_IMG, vmin=0, vmax=1,
                      extent=ext, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            # intensidad pico relativa al pico en foco de la fila, del banco
            ax.text(0.05, 0.95, f"{p.max()/p0:.2f}", transform=ax.transAxes,
                    color="white", fontsize=14, fontweight="bold", va="top",
                    path_effects=contorno)
            if fila == 0:
                ax.set_title("0" if z == 0 else f"{z:d} nm".replace("-", "−"),
                             fontsize=16)
            if col == 0:
                ax.set_ylabel(nombre, fontsize=16, rotation=0, ha="right",
                              va="center", labelpad=10)

    # barra de escala de 1 um solo en el ultimo panel de la fila inferior
    ax = axes[-1, -1]
    x0 = ext[1] - 0.45 - 1.0                    # metida hacia dentro para que "1 µm" no se corte
    y0 = ext[2] - 0.35
    ax.plot([x0, x0 + 1.0], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    ax.text(x0 + 0.5, y0 - 0.15, "1 µm", color="white", fontsize=14,
            fontweight="bold", ha="center", va="bottom", path_effects=contorno)

    fig.suptitle("Distancia al foco", fontsize=18, y=1.0)
    guardar(fig, "fig_banco_psf_diapo")


# ----------------------------------------------------------------------------
# Figura 2: cortes XZ + metricas vs z
# ----------------------------------------------------------------------------

if "axial" in FIGURAS:
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.3)

    z_grid = banco_esf.z_grid
    for col, (nombre, cfg, banco, _) in enumerate(bancos):
        ax = fig.add_subplot(gs[0, col])
        K = cfg.K_psf
        xz = banco.stack[:, K // 2, K // 2 - half:K // 2 + half]   # (nz, 2*half)
        xz = xz / xz.max()
        im = ax.imshow(xz, cmap=CMAP_IMG, norm=LogNorm(vmin=1e-3, vmax=1),
                       extent=[-half * px_um, half * px_um, z_grid[0], z_grid[-1]],
                       origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(f"({'ab'[col]}) Corte XZ – {nombre}", loc="left")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("z [nm]")
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Intensidad norm. (log)", fontsize=8)

    z_fino = np.arange(-800, 801, 20)
    ax_p = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    estilos = ["-", "--"]
    for (nombre, cfg, banco, color), ls in zip(bancos, estilos):
        pico = np.array([banco.psf_en_z(z).max() for z in z_fino])
        e2 = np.array([energia_encerrada(banco, z, radio_px=2.0) for z in z_fino])
        ax_p.plot(z_fino, pico / pico.max(), ls, color=color, lw=2, label=nombre)
        ax_e.plot(z_fino, e2, ls, color=color, lw=2, label=nombre)

    ax_p.set_title("(c) Pico relativo vs z", loc="left")
    ax_p.set_ylabel("Pico / pico máx.")
    ax_e.set_title(r"(d) Energía encerrada ($r \leq 2$ px)", loc="left")
    ax_e.set_ylabel("Fracción de energía")
    for ax in (ax_p, ax_e):
        ax.set_xlabel("z [nm]")
        ax.set_xlim(-800, 800)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, lw=0.5)
    ax_p.legend(frameon=False, loc="lower center")
    guardar(fig, "fig_banco_psf_axial")


# ----------------------------------------------------------------------------
# Figura 3: cuadros simulados + ground truth
# ----------------------------------------------------------------------------

def fondo_del_video(cfg):
    """Reproduce el fondo que uso generar_video, repitiendo su orden de
    llamadas al rng (n, trayectorias, fotones, fondo), como en check_e2e."""
    rng = np.random.default_rng(cfg.seed)
    n = rng.integers(cfg.n_emisores_rango[0], cfg.n_emisores_rango[1] + 1)
    generar_trayectorias(cfg, n, rng)
    muestrear_fotones(cfg, n, rng)
    return generar_fondo(cfg, rng)


def snr_emisores(g, fondo, cfg):
    """SNR pico de cada emisor (misma formula que snr_pico de sim_imagenesPSF),
    con p_max en su z y el fondo local en su pixel."""
    p_max = np.array([banco_esf.psf_en_z(z).max() for z in g.z_nm])
    iy = np.clip(g.y_px.astype(int), 0, cfg.H - 1)
    ix = np.clip(g.x_px.astype(int), 0, cfg.W - 1)
    b = fondo[iy, ix]
    senal = g.photons.values * p_max
    return senal / np.sqrt(senal + b + (cfg.read_noise * cfg.gain)**2)


if FIGURAS & {"cuadros", "cuadros_simple"}:
    # videos fijos: semillas elegidas antes por tener la mediana de N (emisores
    # en el FOV) cerca del minimo, del centro geometrico y del maximo de
    # fotones_rango; n_frames y n_emisores_rango deben quedar igual para
    # reproducir los mismos videos
    t_show = 10
    semillas = [49, 89, 113]

    casos = []
    for s in semillas:
        cfg = Config(mascara="esferica", n_frames=t_show + 1,
                     n_emisores_rango=(4, 12), seed=s)
        rng = np.random.default_rng(cfg.seed)
        video, filas = generar_video(cfg, banco_esf, rng, video_id=s)
        casos.append((cfg, video, pd.DataFrame(filas)))

    W, H = casos[0][0].W, casos[0][0].H
    ext_img = [0, W * px_um, H * px_um, 0]
    norm_z = TwoSlopeNorm(vmin=-800, vcenter=0, vmax=800)

    # cuadros: version completa; cuadros_simple: titulos solo (a), (b), (c) y
    # escala de grises en fotones en vez de ADU, mismos videos
    for variante in [v for v in ("cuadros", "cuadros_simple") if v in FIGURAS]:
        simple = variante == "cuadros_simple"
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2),
                                 gridspec_kw=dict(wspace=0.08, hspace=0.1))
        for col, (cfg, video, gt) in enumerate(casos):
            cuadro = video[t_show].astype(float)
            if simple:
                cuadro = (cuadro - cfg.offset) * cfg.gain   # ADU -> e- = fotones detectados (sin QE)
            g_t = gt[gt.frame == t_show]
            g = g_t[g_t.en_fov]
            if not simple:
                snr = snr_emisores(g, fondo_del_video(cfg), cfg)

            # senal sin ruido ni fondo: las PSFs de todos los emisores del cuadro
            # (incluye los de fuera del FOV, que tambien aportan luz en el borde)
            senal = np.zeros((cfg.H, cfg.W))
            for _, r in g_t.iterrows():
                banco_esf.colocar(senal, r.x_px, r.y_px, r.z_nm, r.photons)

            vmin, vmax = np.percentile(cuadro, [1, 99.8])     # recorte por percentiles
            axes[0, col].imshow(cuadro, cmap="gray", vmin=vmin, vmax=vmax,
                                extent=ext_img, interpolation="nearest")
            axes[1, col].imshow(senal, cmap=CMAP_IMG, vmin=0, extent=ext_img,
                                interpolation="nearest")
            for ax in axes[:, col]:
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)

            if simple:
                axes[0, col].set_title(f"({'abc'[col]})", fontsize=8)
            else:
                axes[0, col].set_title(
                    f"({'abc'[col]}) " + rf"$N_{{\mathrm{{med}}}}$ = {g.photons.median():.0f} fotones"
                    f"\nSNR pico {np.median(snr):.1f} ({snr.min():.1f}–{snr.max():.1f})",
                    fontsize=8)
            # (x_px, y_px) -> um, centro de pixel en (i + 0.5); anillo blanco + borde
            # negro para que el marcador se lea sobre fondo claro y oscuro
            xs, ys = (g.x_px + 0.5) * px_um, (g.y_px + 0.5) * px_um
            axes[1, col].scatter(xs, ys, s=60, c="white", zorder=3, linewidths=0)
            sc = axes[1, col].scatter(xs, ys, c=g.z_nm, cmap="RdBu_r", norm=norm_z,
                                      s=28, edgecolors="black", linewidths=0.8, zorder=4)

            # barra de escala de 1 um en la esquina inferior izquierda, sobre un
            # recuadro negro semitransparente para no confundirla con senal
            ax = axes[0, col]
            yb = H * px_um - 0.35                           # eje y invertido: abajo = H
            ax.add_patch(Rectangle((0.12, yb - 0.5), 1.36, 0.68, facecolor="black",
                                   alpha=0.7, edgecolor="none", zorder=5))
            ax.plot([0.3, 1.3], [yb, yb], color="white", lw=2,
                    solid_capstyle="butt", zorder=6)
            ax.text(0.8, yb - 0.12, "1 µm", color="white", fontsize=7,
                    ha="center", va="bottom", zorder=6)

            if not simple:
                print(f"seed {cfg.seed}: {len(g)} emisores en FOV, N = "
                      f"{g.photons.min():.0f}-{g.photons.max():.0f} (mediana {g.photons.median():.0f}), "
                      f"SNR pico mediana {np.median(snr):.2f} rango {snr.min():.2f}-{snr.max():.2f}")

        axes[0, 0].set_ylabel("Cuadro simulado (con ruido)", fontsize=8)
        axes[1, 0].set_ylabel("Señal sin ruido + ground truth", fontsize=8)
        cb = fig.colorbar(sc, ax=axes[1, :], fraction=0.03, pad=0.02)
        cb.set_label("z [nm] (ground truth)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap="gray"), ax=axes[0, :],
                           fraction=0.03, pad=0.02, ticks=[])
        unidad = "fotones" if simple else "ADU"
        cb2.set_label(f"Intensidad [{unidad}]\n(escala por cuadro)", fontsize=8)
        guardar(fig, "fig_cuadros_gt_simple" if simple else "fig_cuadros_gt")
