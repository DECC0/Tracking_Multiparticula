"""Check camara: histogramas de ADU contra la distribucion teorica
(Poisson convolucionada con el ruido de lectura) y curva de transferencia de
fotones (PTC) con la ganancia y el ruido de lectura estimados.
Genera fig_camara.png."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import simulador as sim

OUT = os.path.dirname(os.path.abspath(__file__))
cfg = sim.Config()
rng = np.random.default_rng(0)
print(f"gain {cfg.gain} e-/ADU, offset {cfg.offset} ADU, read noise {cfg.read_noise} ADU")

niveles = [0.0, 3.0, 30.0, 1000.0]
fig, axes = plt.subplots(1, len(niveles) + 1, figsize=(4 * (len(niveles) + 1), 3.8))
for ax, F in zip(axes, niveles):
    adu = sim.aplicar_camara(np.full(300_000, F), cfg, rng)
    vals, cuentas = np.unique(adu, return_counts=True)
    ax.bar(vals, cuentas / cuentas.sum(), width=1.0, color="tab:blue", alpha=0.6, label="simulado")
    # teorica: P(ADU = a) = sum_e Pois(e; F) * P(redondeo de e/g + offset + N(0, sr) = a)
    e = np.arange(stats.poisson.ppf(1e-9, F), stats.poisson.ppf(1 - 1e-9, F) + 1) if F > 0 else np.array([0.0])
    pe = stats.poisson.pmf(e, F) if F > 0 else np.array([1.0])
    a = np.arange(vals.min(), vals.max() + 1)
    teo = (pe[:, None] * (stats.norm.cdf(a[None, :] + 0.5, e[:, None] / cfg.gain + cfg.offset, cfg.read_noise)
                          - stats.norm.cdf(a[None, :] - 0.5, e[:, None] / cfg.gain + cfg.offset, cfg.read_noise))).sum(0)
    ax.plot(a, teo, "r-", lw=1.2, label="teorica")
    ax.set_title(f"{F:g} fotones / px")
    ax.set_xlabel("ADU")
    ax.legend(fontsize=8)

# --- PTC (hasta 20000 fotones: con gain 0.46, 30000 fotones ya rozan la saturacion
# de uint16 y el recorte achica la varianza) ---
medias, varianzas = [], []
for F in np.geomspace(1, 20000, 20):
    adu = sim.aplicar_camara(np.full(50_000, F), cfg, rng).astype(float)
    medias.append(adu.mean()); varianzas.append(adu.var())
medias, varianzas = np.array(medias), np.array(varianzas)
pend, inter = np.polyfit(medias - cfg.offset, varianzas, 1)
oscuro = sim.aplicar_camara(np.zeros(200_000), cfg, rng).astype(float)
print(f"PTC: ganancia estimada {1/pend:.4f} e-/ADU (real {cfg.gain}), "
      f"ruido de lectura de frames oscuros {np.sqrt(oscuro.var() - 1/12):.3f} ADU (real {cfg.read_noise})")
ax = axes[-1]
ax.loglog(medias - cfg.offset, varianzas, "o", ms=4, label="simulado")
m = np.geomspace(1, medias.max() - cfg.offset, 100)
ax.loglog(m, m / cfg.gain + cfg.read_noise**2 + 1 / 12, "r-", lw=1, label="(media - offset)/g + sr^2")
ax.set_xlabel("media - offset [ADU]"); ax.set_ylabel("varianza [ADU^2]")
ax.set_title(f"PTC: g estimada = {1/pend:.3f} e-/ADU")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_camara.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_camara.png")
