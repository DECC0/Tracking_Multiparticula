"""Check de extremo a extremo con la Config por defecto: un video con el GT
superpuesto, imagen limpia vs con ruido, residuo normalizado contra N(0, 1),
trayectorias y tiempo de generacion. Genera fig_e2e.png."""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulador as sim

OUT = os.path.dirname(os.path.abspath(__file__))
cfg = sim.Config(n_emisores_rango=(8, 8), seed=7)
t0 = time.time()
b = sim.BancoPSF(cfg)
t_banco = time.time() - t0
t0 = time.time()
video, gt, ex = sim.generar_video(cfg, b, 0, extra=True)
t_video = time.time() - t0
print(f"banco: {t_banco:.2f} s (K = {b.K}),  video {video.shape}: {t_video:.2f} s")
print(f"rango ADU [{video.min()}, {video.max()}], {gt.particle_id.nunique()} particulas, "
      f"{100*gt.en_fov.mean():.0f}% de filas en el FOV")

mu = ex["limpio"]
sigma = np.sqrt(mu + (cfg.read_noise**2 + 1 / 12) * cfg.gain**2)
r = ((sim.adu_a_electrones(video, cfg) - mu) / sigma).ravel()
print(f"residuo normalizado: media {r.mean():+.4f}, varianza {r.var():.4f} (esperado 0, 1)")

fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
g0 = gt[gt.frame == 0]
for ax, img, t in [(axes[0], video[0], "Frame 0 simulado (ADU) + GT"),
                   (axes[1], mu[0], "Imagen esperada sin ruido (fotones)")]:
    ax.imshow(img, cmap="inferno")
    ax.scatter(g0.x_px, g0.y_px, s=90, facecolors="none", edgecolors="cyan", lw=1.2)
    ax.set_xlim(-0.5, cfg.W - 0.5); ax.set_ylim(cfg.H - 0.5, -0.5)
    ax.set_title(t)
axes[2].hist(r, bins=100, density=True, alpha=0.7)
xs = np.linspace(-5, 5, 200)
axes[2].plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "r-")
axes[2].set_yscale("log"); axes[2].set_ylim(1e-5, 1)
axes[2].set_title("Residuo normalizado vs N(0, 1)")
axes[3].imshow(video.max(axis=0), cmap="gray")
for pid, s in gt.groupby("particle_id"):
    axes[3].plot(s.x_px, s.y_px, "-", lw=1)
axes[3].set_xlim(-0.5, cfg.W - 0.5); axes[3].set_ylim(cfg.H - 0.5, -0.5)
axes[3].set_title("Proyeccion maxima + trayectorias GT")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_e2e.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_e2e.png")
