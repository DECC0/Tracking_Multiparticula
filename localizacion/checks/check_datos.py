"""Check de la etapa 1 (datos + target) con configs/base.yaml: tiempos de
generacion y del target, distribuciones de la randomizacion por muestra
(emisores, fotones, fondo, SNR) y ejemplos de entrada / target con el GT
superpuesto, de bajo a alto SNR. Genera fig_datos.png y fig_datos_distribuciones.png."""
import os
import sys
import time
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import cargar_config
from data.dataset import DatasetSimulado, collate, normalizar
from data.target import RejillaVoxeles

OUT = os.path.dirname(os.path.abspath(__file__))
N_STATS = 2000

cfg = cargar_config(os.path.join(RAIZ, "configs", "base.yaml"))
t0 = time.time()
ds = DatasetSimulado(cfg.datos, N_STATS, cfg.datos.seed_train)
t_banco = time.time() - t0
cs = ds.cfg_sim
rej = RejillaVoxeles.desde_config(cfg.target, cs)
print(cs.resumen())
print(f"\nbanco: {t_banco:.2f} s (mascara {cs.mascara}, K = {ds.banco.K}, {len(ds.banco.z_grid)} planos)")
print(f"rejilla de salida (Nz, sH, sW) = {rej.forma}: voxel {cs.pixel_nm / rej.s:.1f} x "
      f"{cs.pixel_nm / rej.s:.1f} x {rej.dz:.0f} nm, planos z {rej.z_planos[0]:.0f}..{rej.z_planos[-1]:.0f} nm "
      f"({rej.m} de margen a cada lado)")

# --- tiempos ---
t0 = time.time()
items = [ds[i] for i in range(300)]
t_item = (time.time() - t0) / 300
lote = collate(items[:8])
for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    pos, val = lote["pos"].to(dev), lote["valido"].to(dev)
    rej.target(pos, val)                                         # calentamiento
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        t = rej.target(pos, val)
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"target de un batch de 8 en {dev}: {100 * (time.time() - t0):.1f} ms")
mb = np.prod(rej.forma) * 8 * 4 / 2**20
print(f"muestra on-the-fly: {1000 * t_item:.1f} ms ({1 / t_item:.0f} muestras/s por proceso); "
      f"target de un batch de 8: {mb:.0f} MB en float32")

# --- distribuciones ---
gts, metas = [], []
for i in range(N_STATS):
    _, gt, meta = ds.muestra(i)
    gts.append(gt)
    metas.append(meta)
gt = pd.concat(gts, ignore_index=True)
meta = pd.DataFrame(metas)
dentro = gt[gt.en_fov]
print(f"\n{N_STATS} muestras: {meta.n_emisores.mean():.1f} emisores en el FOV por imagen "
      f"(min {meta.n_emisores.min()}, max {meta.n_emisores.max()}), "
      f"{100 * (meta.n_emisores == 0).mean():.1f}% sin emisores, "
      f"{100 * (~gt.en_fov).mean():.0f}% de emisores fuera del FOV")
print(f"fondo uniforme en {100 * (~meta.fondo_no_uniforme).mean():.0f}% de las imagenes; "
      f"fondo medio {meta.fondo_medio.min():.1f}..{meta.fondo_medio.max():.1f} fotones/px")
q = np.percentile(dentro.snr, [5, 25, 50, 75, 95])
print("SNR pico por emisor (percentiles 5/25/50/75/95): " + " / ".join(f"{v:.1f}" for v in q))
print(f"emisores con SNR < 3: {100 * (dentro.snr < 3).mean():.0f}%,  SNR < 5: {100 * (dentro.snr < 5).mean():.0f}%")

fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
color = "#3b6ea8"
ax = axes[0]
ax.hist(dentro.photons, bins=np.geomspace(*cs.fotones_rango, 40), color=color)
ax.set_xscale("log"); ax.set_xlabel("fotones por emisor"); ax.set_ylabel("emisores")
ax = axes[1]
ax.hist(dentro.snr, bins=np.geomspace(max(dentro.snr.min(), 0.1), dentro.snr.max(), 40), color=color)
ax.set_xscale("log"); ax.set_xlabel("SNR pico por emisor")
ax = axes[2]
ax.hist(meta.n_emisores, bins=np.arange(meta.n_emisores.max() + 2) - 0.5, color=color)
ax.set_xlabel("emisores en el FOV por imagen"); ax.set_ylabel("imagenes")
ax = axes[3]
ax.hist(meta.fondo_medio, bins=40, color=color)
ax.set_xlabel("fondo medio [fotones/px]")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Randomizacion por muestra ({N_STATS} muestras, config base)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_datos_distribuciones.png", dpi=130)
plt.close(fig)

# --- ejemplos de bajo a alto SNR (imagenes con >= 3 emisores en el FOV) ---
snr_med = dentro.groupby("video_id").snr.median()
cand = snr_med[meta.set_index("video_id").n_emisores.reindex(snr_med.index) >= 3]
elegidos = [int((cand - cand.quantile(p)).abs().idxmin()) for p in (0.05, 0.35, 0.65, 0.95)]

fig, axes = plt.subplots(3, 4, figsize=(18, 11), gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
lote = collate([ds[i] for i in elegidos])
T = rej.target(lote["pos"], lote["valido"]).numpy()
ext_xy = [-0.5, cs.W - 0.5, cs.H - 0.5, -0.5]
ext_xz = [-0.5, cs.W - 0.5, rej.z_planos[0] - rej.dz / 2, rej.z_planos[-1] + rej.dz / 2]
for c, idx in enumerate(elegidos):
    img, g, m = ds.muestra(idx)
    g = g[g.en_fov]
    x = normalizar(img, cfg.datos, cs)[0]
    ax = axes[0, c]
    ax.imshow(x, cmap="gray", extent=ext_xy)
    ax.scatter(g.x_px, g.y_px, s=110, facecolors="none", edgecolors="cyan", lw=1.0)
    ax.set_title(f"muestra {idx}: {len(g)} emisores, SNR mediana {g.snr.median():.1f}\n"
                 f"fotones {m['fotones_min']:.0f}-{m['fotones_max']:.0f}, fondo {m['fondo_medio']:.0f}/px",
                 fontsize=10)
    ax = axes[1, c]
    ax.imshow(T[c].max(axis=0), cmap="inferno", extent=ext_xy, interpolation="nearest")
    ax.scatter(g.x_px, g.y_px, s=40, marker="+", color="cyan", lw=0.8)
    ax.set_title("target: proyeccion maxima en z", fontsize=10)
    ax = axes[2, c]
    ax.imshow(T[c].max(axis=1), cmap="inferno", extent=ext_xz, origin="lower", aspect="auto",
              interpolation="nearest")
    ax.scatter(g.x_px, g.z_nm, s=40, marker="+", color="cyan", lw=0.8)
    for zl in cs.z_rango_nm:
        ax.axhline(zl, color="white", ls="--", lw=0.6)
    ax.set_xlabel("x [px]")
    ax.set_title("target: proyeccion maxima en y (x-z)", fontsize=10)
axes[0, 0].set_ylabel("y [px]")
axes[1, 0].set_ylabel("y [px]")
axes[2, 0].set_ylabel("z [nm]")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_datos.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_datos.png, fig_datos_distribuciones.png")
