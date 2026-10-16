"""Check colocar: error del desplazamiento subpixel (fase lineal en Fourier
sobre el recorte) contra la PSF exacta desplazada en la pupila, en una rejilla
de offsets; conservacion de fotones y sesgo de centroide respecto a la PSF
exacta (el centroide de toda la imagen lo dominan las colas truncadas, incluso
en la PSF exacta, asi que se compara contra ella y no contra (x, y)).
Genera fig_colocar_subpixel.png."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulador as sim

OUT = os.path.dirname(os.path.abspath(__file__))
b = sim.BancoPSF(sim.Config())                       # esferica, la mascara por defecto
cfg, K, c = b.cfg, b.K, b.cfg.M_pupila // 2
z = 0.0
offs = np.linspace(0.0, 0.95, 11)

err = np.zeros((len(offs), len(offs)))
energia = np.zeros_like(err)
cent = np.zeros_like(err)
yy, xx = np.mgrid[0:K, 0:K]
centroide = lambda p: np.array([(p * xx).sum(), (p * yy).sum()]) / p.sum()
for i, dy in enumerate(offs):
    for j, dx in enumerate(offs):
        img = np.zeros((K, K))
        b.colocar(img, K // 2 + dx, K // 2 + dy, z, 1.0)
        ref = sim.psf_directa(cfg, z, dx, dy)[c - K // 2: c + K // 2, c - K // 2: c + K // 2]
        err[i, j] = np.abs(img - ref).max() / ref.max()
        energia[i, j] = img.sum()
        cent[i, j] = np.hypot(*(centroide(img) - centroide(ref)))

print(f"mascara {cfg.mascara}, K = {K}, z = {z} nm")
print(f"error max |colocar - exacta| / pico : {err.max():.2e}")
print(f"energia colocada                    : {energia.min():.6f} .. {energia.max():.6f} "
      f"(1 - recortada = {1 - b.energia_recortada[np.argmin(np.abs(b.z_grid - z))]:.6f})")
print(f"sesgo de centroide vs PSF exacta    : max {cent.max():.2e} px")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
ext = [offs[0], offs[-1], offs[0], offs[-1]]
for ax, M, t in zip(axes, [err, energia, cent],
                    ["|colocar - PSF exacta|max / pico", "fotones colocados / N",
                     "|centroide colocar - centroide exacta| [px]"]):
    im = ax.imshow(M, origin="lower", extent=ext, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("offset subpixel en x"); ax.set_ylabel("offset subpixel en y"); ax.set_title(t)
fig.suptitle(f"Colocacion subpixel ({cfg.mascara}, K = {K}, z = {z:.0f} nm)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_colocar_subpixel.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_colocar_subpixel.png")
