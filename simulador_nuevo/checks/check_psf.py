"""Check PSF: montaje de las tres mascaras vs z, metricas vs z (pico relativo,
energia encerrada, angulo del corkscrew) y perfil en foco contra Airy.
Genera fig_psf_montaje.png, fig_psf_metricas_z.png y fig_psf_airy.png."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j1

import simulador as sim

OUT = os.path.dirname(os.path.abspath(__file__))
MASCARAS = ("ninguna", "esferica", "corkscrew")

bancos = {m: sim.BancoPSF(sim.Config(mascara=m)) for m in MASCARAS}
print(sim.Config().resumen())
print("\n mascara     K    energia fuera del recorte (max / min)")
for m, b in bancos.items():
    print(f" {m:<10} {b.K:4d}    {100*b.energia_recortada.max():.2f}% / {100*b.energia_recortada.min():.2f}%")

# --- montaje ---
zs = [-800, -400, -200, 0, 200, 400, 800]
h = 12
fig, axes = plt.subplots(len(MASCARAS), len(zs), figsize=(2.2 * len(zs), 2.4 * len(MASCARAS)))
for fila, (m, b) in zip(axes, bancos.items()):
    c = b.K // 2
    vmax = b.stack[:, c - h:c + h, c - h:c + h].max()
    for ax, z in zip(fila, zs):
        ax.imshow(b.psf_en_z(z)[c - h:c + h, c - h:c + h], cmap="inferno", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if m == MASCARAS[0]:
            ax.set_title(f"z = {z} nm")
    fila[0].set_ylabel(m)
fig.suptitle("PSF por mascara y profundidad (misma escala de color por fila)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_psf_montaje.png", dpi=130)
plt.close(fig)

# --- metricas vs z ---
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
print("\n mascara     pico min/max   E(r<=2px) min..max")
for m, b in bancos.items():
    pico = b.stack.max(axis=(1, 2))
    ee = np.array([sim.energia_encerrada(b, z, 2.0) for z in b.z_grid])
    a1.plot(b.z_grid, pico / pico.max(), label=m)
    a2.plot(b.z_grid, ee, label=m)
    print(f" {m:<10}  {pico.min()/pico.max():10.3f}   {ee.min():.3f}..{ee.max():.3f}")
a1.set_xlabel("z [nm]"); a1.set_ylabel("pico / max_z(pico)"); a1.set_title("Pico relativo vs z"); a1.legend()
a2.set_xlabel("z [nm]"); a2.set_ylabel("E(r <= 2 px)"); a2.set_title("Energia encerrada vs z"); a2.legend()
bc = bancos["corkscrew"]
zz = np.arange(-800.0, 801.0, 20.0)
ang = np.degrees(np.unwrap(np.radians([sim.angulo_lobulo(bc, z)[0] for z in zz])))
a3.plot(zz, ang - ang[len(ang) // 2], color="tab:green")
a3.set_xlabel("z [nm]"); a3.set_ylabel("angulo del lobulo - angulo(z=0) [grados]")
a3.set_title(f"Corkscrew: rotacion total {abs(ang[-1]-ang[0]):.0f} grados")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_psf_metricas_z.png", dpi=130)
plt.close(fig)

# --- perfil en foco contra Airy ---
cfg = sim.Config(mascara="ninguna", blur_extra_px=0.0, integrar_pixel=False)
p = sim.psf_directa(cfg, 0.0)
c = cfg.M_pupila // 2
r_px = np.arange(0, 13)
perfil = p[c, c:c + 13] / p[c, c]
r_fino = np.linspace(1e-6, 12, 600)
v = 2 * np.pi * cfg.NA * r_fino * cfg.pixel_nm / cfg.lambda_nm
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogy(r_fino, (2 * j1(v) / v)**2, "k-", lw=1, label="Airy teorico [2 J1(v)/v]^2")
ax.semilogy(r_px, perfil, "o", color="tab:red", label="simulador (sin blur, muestreo puntual)")
for cfg_b, lab in [(sim.Config(mascara="ninguna"), "simulador (blur 0.6 px + integracion de pixel)")]:
    pb = sim.psf_directa(cfg_b, 0.0)
    ax.semilogy(r_px, pb[c, c:c + 13] / pb[c, c], "s--", color="tab:blue", ms=4, label=lab)
ax.set_ylim(1e-5, 1.5)
ax.set_xlabel("r [px]"); ax.set_ylabel("I(r) / I(0)")
ax.set_title(f"PSF en foco vs Airy (NA {cfg.NA}, {cfg.lambda_nm:.0f} nm, {cfg.pixel_nm:.1f} nm/px)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_psf_airy.png", dpi=130)
plt.close(fig)
err = np.abs(perfil - np.where(r_px == 0, 1.0, (2 * j1(2*np.pi*cfg.NA*r_px*cfg.pixel_nm/cfg.lambda_nm)
                                               / (2*np.pi*cfg.NA*np.maximum(r_px, 1e-9)*cfg.pixel_nm/cfg.lambda_nm))**2)).max()
print(f"\nerror max del perfil en foco vs Airy: {err:.2e} (relativo al pico)")
print("\nOK -> fig_psf_montaje.png, fig_psf_metricas_z.png, fig_psf_airy.png")
