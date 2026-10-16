"""Check difusion: MSD vs teoria con y sin desenfoque de movimiento,
histograma de incrementos normalizados y distribucion de z con paredes.
Genera fig_difusion.png."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulador as sim

OUT = os.path.dirname(os.path.abspath(__file__))
D, n = 0.2, 20000
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))

for n_sub, col in [(1, "tab:blue"), (10, "tab:orange")]:
    cfg = sim.Config(n_frames=21, D_rango_um2_s=(D, D), n_subpasos=n_sub, z_rango_nm=(-1e6, 1e6))
    tray, _, _ = sim.generar_trayectorias(cfg, n, np.random.default_rng(0))
    lags = np.arange(1, cfg.n_frames)
    msd = np.array([((tray[:, m:, :2] - tray[:, :-m, :2])**2).sum(-1).mean() for m in lags]) * cfg.pixel_nm**2 / 1e6
    t = lags * cfg.dt_s
    corr = cfg.t_exp * (n_sub**2 - 1) / (3 * n_sub**2)
    a1.plot(t, msd, "o", ms=3, color=col, label=f"simulado, n_subpasos={n_sub}")
    a1.plot(t, 4 * D * (t - corr), "-", lw=1, color=col, label=f"4 D (t - {corr:.3f} s)")
    pend, inter = np.polyfit(t[:5], msd[:5], 1)                  # lags cortos: los mas precisos
    print(f"n_subpasos={n_sub:2d}: D ajustado = {pend/4:.4f} um2/s (real {D}), "
          f"intercepto = {inter:.4f} um2 (teoria {-4*D*corr:.4f})")
a1.set_xlabel("t [s]"); a1.set_ylabel("MSD_xy [um^2]"); a1.set_title("MSD vs teoria (desenfoque de movimiento)")
a1.legend(fontsize=8)

cfg = sim.Config(n_frames=200)
tray, _, Ds = sim.generar_trayectorias(cfg, 3000, np.random.default_rng(1))
u = (np.diff(tray[..., 0], axis=1) / (np.sqrt(2 * Ds * cfg.dt_s) * 1000 / cfg.pixel_nm)[:, None]).ravel()
a2.hist(u, bins=100, density=True, alpha=0.7)
xs = np.linspace(-5, 5, 200)
a2.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "r-")
a2.set_title(f"Incrementos en x normalizados (media {u.mean():+.3f}, var {u.var():.3f})")

lo, hi = cfg.z_rango_nm
a3.hist(tray[..., 2].ravel(), bins=80, density=True, alpha=0.7, color="tab:purple")
a3.axhline(1 / (hi - lo), color="r")
a3.set_xlabel("z [nm]"); a3.set_title("z con paredes reflectantes (uniforme esperado)")
print(f"z fuera de [{lo}, {hi}]: {np.mean((tray[..., 2] < lo) | (tray[..., 2] > hi)):.4f}")

fig.tight_layout()
fig.savefig(f"{OUT}/fig_difusion.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_difusion.png")
