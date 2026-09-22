"""Check 3: modelo de camara (Poisson -> ganancia -> ruido lectura -> offset -> uint16)
y reproducibilidad con seed."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_imagenesPSF import Config, aplicar_camara

OUT = os.path.dirname(os.path.abspath(__file__))
cfg = Config()

# --- distribucion empirica vs teorica para flujo de fotones constante ---
fotones_prueba = [5, 50, 500, 2000]
fig, axes = plt.subplots(1, len(fotones_prueba), figsize=(4*len(fotones_prueba), 3.6))
rng = np.random.default_rng(123)
print("=== estadisticos ADU vs teoria, para fotones constantes (N muestras=200000) ===")
print(" fotones   media_emp   media_teo   var_emp    var_teo")
for ax, f in zip(axes, fotones_prueba):
    muestras = aplicar_camara(np.full(200_000, float(f)), cfg, rng).astype(float)
    media_emp, var_emp = muestras.mean(), muestras.var()
    media_teo = f / cfg.gain + cfg.offset
    var_teo = f / cfg.gain**2 + cfg.read_noise**2
    print(f" {f:7.0f}   {media_emp:9.3f}   {media_teo:9.3f}   {var_emp:8.3f}   {var_teo:8.3f}")
    ax.hist(muestras, bins=60, density=True, alpha=0.7, label="empirico")
    xs = np.linspace(muestras.min(), muestras.max(), 200)
    gauss = np.exp(-(xs-media_teo)**2/(2*var_teo)) / np.sqrt(2*np.pi*var_teo)
    ax.plot(xs, gauss, "r-", lw=1.5, label="N(media_teo, var_teo)")
    ax.set_title(f"fotones={f}")
    ax.legend(fontsize=8)
fig.suptitle("aplicar_camara: histograma ADU vs prediccion Poisson(+lectura) aproximada por Normal")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_camara_dist.png", dpi=130)
plt.close(fig)

# --- reproducibilidad con seed ---
rng1 = np.random.default_rng(cfg.seed)
rng2 = np.random.default_rng(cfg.seed)
a = aplicar_camara(np.full((10, 10), 300.0), cfg, rng1)
b = aplicar_camara(np.full((10, 10), 300.0), cfg, rng2)
print(f"\nreproducibilidad (misma seed): identico = {np.array_equal(a, b)}")

# --- saturacion: fotones extremos ---
extremo = aplicar_camara(np.full(1000, 1_000_000.0), cfg, np.random.default_rng(0))
print(f"saturacion con 1e6 fotones: max={extremo.max()} (limite uint16=65535), "
      f"fraccion saturada={np.mean(extremo==65535):.3f}")

print("\nOK -> fig_camara_dist.png")
