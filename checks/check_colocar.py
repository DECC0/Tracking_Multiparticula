"""Check 2: conservacion de energia y precision subpixel de BancoPSF.colocar()."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_imagenesPSF import Config, BancoPSF

OUT = os.path.dirname(os.path.abspath(__file__))

cfg = Config()
banco = BancoPSF(cfg)

# imagen grande para que colocar() nunca recorte contra el borde
H = W = 400
N = 1000.0
z = 0.0

rng = np.random.default_rng(0)
offsets = np.linspace(0, 0.999, 25)
energias = []
centroides_err = []
for off in offsets:
    img = np.zeros((H, W))
    x, y = W/2 + off, H/2 + off
    banco.colocar(img, x, y, z, N)
    total = img.sum()
    energias.append(total / N)
    yy, xx = np.mgrid[0:H, 0:W]
    cx = (img * xx).sum() / total
    cy = (img * yy).sum() / total
    centroides_err.append(np.hypot(cx - x, cy - y))

energias = np.array(energias)
centroides_err = np.array(centroides_err)

print("=== conservacion de energia vs offset subpixel (sin recorte de borde) ===")
print(f"energia/N: min={energias.min():.6f} max={energias.max():.6f} (deberia ser ~1.0, "
      f"la perdida = energia_recortada de esta PSF)")
print(f"error de centroide [px]: min={centroides_err.min():.2e} max={centroides_err.max():.2e}")

# --- caso con recorte de borde (emisor cerca del borde de una imagen 64x64) ---
Hs = Ws = cfg.H
img_borde = np.zeros((Hs, Ws))
x_borde, y_borde = 2.3, 2.7   # cerca de la esquina, con margen=3 tipico del script
banco.colocar(img_borde, x_borde, y_borde, z, N)
frac_borde = img_borde.sum() / N
print(f"\nemisor cerca del borde (x={x_borde}, y={y_borde}) en imagen {Hs}x{Ws}: "
      f"fraccion de energia capturada = {frac_borde:.3f} (se pierde el resto por salir del FOV, esperado)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].plot(offsets, energias, "o-")
axes[0].axhline(1 - banco.energia_recortada.max(), color="red", ls="--", lw=1,
                label="1 - energia_recortada(z=0)")
axes[0].set_xlabel("offset subpixel (x=y=W/2+offset)")
axes[0].set_ylabel("energia total / N")
axes[0].set_title("Conservacion de energia vs shift subpixel")
axes[0].legend()

axes[1].plot(offsets, centroides_err, "o-", color="tab:orange")
axes[1].set_xlabel("offset subpixel")
axes[1].set_ylabel("|centroide - (x,y) real| [px]")
axes[1].set_title("Precision del centroide vs shift subpixel")

half = 20
c = Hs  # not used
crop = img_borde
axes[2].imshow(crop, cmap="inferno")
axes[2].scatter([x_borde], [y_borde], marker="+", color="cyan", s=120)
axes[2].set_title(f"Emisor cerca del borde\n(energia capturada={frac_borde:.2f})")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_colocar_checks.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_colocar_checks.png")
