"""Check 5: consistencia ground truth <-> imagen generada (end-to-end),
usando generar_video de sim_imagenesPSF.py (mascara='esferica' por default).

API real de sim_imagenesPSF.py (video con difusion 3D, SIN fotofisica on/off):
  generar_video(cfg, banco, rng, video_id) -> video, filas
  filas: video_id, frame, particle_id, x_px, y_px, z_nm, photons, D_um2_s, en_fov
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_imagenesPSF import Config, BancoPSF, generar_video, generar_fondo, generar_trayectorias

OUT = os.path.dirname(os.path.abspath(__file__))

cfg = Config(n_emisores_rango=(6, 6), n_frames=30, seed=7)
banco = BancoPSF(cfg)
rng = np.random.default_rng(cfg.seed)
video, filas = generar_video(cfg, banco, rng, video_id=0)
gt = pd.DataFrame(filas)

print(f"video shape: {video.shape}  dtype: {video.dtype}  rango=[{video.min()},{video.max()}]")
print(f"filas de gt: {len(gt)}  (esperado {cfg.n_frames} frames * {gt.particle_id.nunique()} particulas = "
      f"{cfg.n_frames * gt.particle_id.nunique()})")
print(f"en_fov: {gt.en_fov.mean()*100:.1f}% de las filas dentro del FOV")

# --- reproducir exactamente el mismo fondo que uso generar_video internamente ---
# (misma secuencia de llamadas al rng: n, trayectorias, fotones, fondo)
rng_repro = np.random.default_rng(cfg.seed)
n0 = rng_repro.integers(cfg.n_emisores_rango[0], cfg.n_emisores_rango[1] + 1)
_tray, _D = generar_trayectorias(cfg, n0, rng_repro)
_Ns = rng_repro.uniform(*cfg.fotones_rango, n0)
fondo = generar_fondo(cfg, rng_repro)

frame0_gt = gt[gt.frame == 0]
img_limpia = np.zeros((cfg.H, cfg.W))
for _, r in frame0_gt.iterrows():
    banco.colocar(img_limpia, r.x_px, r.y_px, r.z_nm, r.photons)
img_limpia_total = img_limpia + fondo

fotones_gt_frame0 = frame0_gt.photons.sum()
fotones_colocados = img_limpia.sum()
print(f"\nframe 0: fotones GT (suma photons declarados) = {fotones_gt_frame0:.1f}")
print(f"frame 0: fotones realmente colocados en imagen (banco.colocar)  = {fotones_colocados:.1f} "
      f"(diff esperada por PSF cortada en bordes/recorte K_psf)")

adu_menos_offset = (video[0].astype(float) - cfg.offset) * cfg.gain
total_obs = adu_menos_offset.sum()
total_esperado = img_limpia_total.sum()
sigma = np.sqrt(total_esperado)
print(f"\ntotal fotones esperados (colocados+fondo) = {total_esperado:.1f}")
print(f"total (ADU-offset)*gain observado          = {total_obs:.1f}")
print(f"diferencia = {total_obs - total_esperado:.1f}  ({(total_obs-total_esperado)/sigma:.2f} sigma, "
      f"deberia ser O(1) porque el total es la suma de ~{cfg.H*cfg.W} pixeles Poisson independientes)")

# --- visual: frame real vs reconstruccion limpia, con posiciones GT superpuestas ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].imshow(video[0], cmap="inferno")
axes[0].scatter(frame0_gt.x_px, frame0_gt.y_px, facecolors="none", edgecolors="cyan", s=80, linewidths=1.2)
axes[0].set_title("frame 0 real (con ruido) + posiciones GT")

axes[1].imshow(img_limpia_total, cmap="inferno")
axes[1].scatter(frame0_gt.x_px, frame0_gt.y_px, facecolors="none", edgecolors="cyan", s=80, linewidths=1.2)
axes[1].set_title("reconstruccion limpia (GT -> PSF + fondo, sin ruido)")

pid = 0
sub = gt[gt.particle_id == pid].sort_values("frame")
axes[2].plot(sub.frame, sub.x_px, label="x_px")
axes[2].plot(sub.frame, sub.y_px, label="y_px")
ax2b = axes[2].twinx()
ax2b.plot(sub.frame, sub.z_nm, "g--", alpha=0.6, label="z_nm")
axes[2].set_title(f"trayectoria particle_id={pid} (D={sub.D_um2_s.iloc[0]:.3f} um2/s)")
axes[2].set_xlabel("frame")
axes[2].legend(loc="upper left", fontsize=8)
ax2b.legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUT}/fig_e2e_gt_vs_imagen.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_e2e_gt_vs_imagen.png")
