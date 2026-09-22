"""
Checks de la PSF esferica (Config default ahora es mascara='esferica')
en sim_imagenesPSF.py. Genera:
  - reporte de texto (energia recortada, compacidad vs z)
  - fig_psf_montage.png     : PSF a distintos z
  - fig_psf_metrics_vs_z.png: pico relativo y energia encerrada vs z
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_imagenesPSF import Config, BancoPSF, angulo_lobulo, energia_encerrada

OUT = os.path.dirname(os.path.abspath(__file__))

cfg = Config()
print("mascara:", cfg.mascara, " K_psf:", cfg.K_psf)
print(cfg.resumen())

banco = BancoPSF(cfg)
print(f"\nenergia fuera del recorte K_psf={cfg.K_psf}px: max {100*banco.energia_recortada.max():.3f}%  "
      f"min {100*banco.energia_recortada.min():.3f}%")

zs = np.array([-800, -600, -400, -200, -100, 0, 100, 200, 400, 600, 800], dtype=float)
p0 = banco.psf_en_z(0).max()

picos = []
enc2 = []
enc4 = []
enc8 = []
for z in zs:
    p = banco.psf_en_z(z)
    picos.append(p.max() / p0)
    enc2.append(energia_encerrada(banco, z, radio_px=2.0))
    enc4.append(energia_encerrada(banco, z, radio_px=4.0))
    enc8.append(energia_encerrada(banco, z, radio_px=8.0))

print("\n   z[nm]   pico_rel   E(r<=2px)   E(r<=4px)   E(r<=8px)")
for z, pk, e2, e4, e8 in zip(zs, picos, enc2, enc4, enc8):
    print(f"  {z:6.0f}   {pk:8.3f}   {e2:9.3f}   {e4:9.3f}   {e8:9.3f}")

# --- montaje visual de la PSF a distintos z ---
zs_montage = [-800, -400, -200, 0, 200, 400, 800]
fig, axes = plt.subplots(1, len(zs_montage), figsize=(3*len(zs_montage), 3.2))
K = cfg.K_psf
c = K // 2
half = 24  # recorte visual alrededor del centro para ver detalle
for ax, z in zip(axes, zs_montage):
    p = banco.psf_en_z(z)
    crop = p[c-half:c+half, c-half:c+half]
    im = ax.imshow(crop, cmap="inferno")
    ax.set_title(f"z = {z:.0f} nm\npico={p.max()/p0:.2f}")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"PSF esferica (Alex) vs z  |  NA={cfg.NA}, amp={cfg.amp_esferica/np.pi:.2f}*pi")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_psf_montage.png", dpi=130)
plt.close(fig)

# --- metricas vs z ---
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(zs, picos, "o-", color="tab:red", label="pico relativo")
ax1.set_xlabel("z [nm]")
ax1.set_ylabel("pico relativo (I_max(z)/I_max(0))", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")
ax1.axvline(0, color="gray", lw=0.5)

ax2 = ax1.twinx()
ax2.plot(zs, enc2, "s--", color="tab:blue", label="E(r<=2px)")
ax2.plot(zs, enc4, "^--", color="tab:green", label="E(r<=4px)")
ax2.set_ylabel("fraccion de energia encerrada", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center")
ax1.set_title("Compacidad de la PSF esferica vs z (deberia degradarse poco -> DOF extendido)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_psf_metrics_vs_z.png", dpi=130)
plt.close(fig)

print("\nOK -> fig_psf_montage.png, fig_psf_metrics_vs_z.png")
