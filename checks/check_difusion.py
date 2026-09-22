"""Check 4: difusion browniana (MSD ~ 2*D*t) y paredes reflectantes en z."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_imagenesPSF import Config, generar_trayectorias

OUT = os.path.dirname(os.path.abspath(__file__))
cfg = Config(n_frames=200)
rng = np.random.default_rng(1)

n_particulas = 3000
tray, D = generar_trayectorias(cfg, n_particulas, rng)  # (n, T, 3) en px,px,nm
px = cfg.pixel_nm

x_nm = tray[..., 0] * px
y_nm = tray[..., 1] * px
z_nm = tray[..., 2]

t = np.arange(cfg.n_frames) * cfg.dt_s

# elegimos 3 grupos de D distintos (bins) para comparar pendientes
order = np.argsort(D)
grupos = [order[:n_particulas//3], order[n_particulas//3:2*n_particulas//3], order[2*n_particulas//3:]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
colors = ["tab:blue", "tab:orange", "tab:green"]
for g, c in zip(grupos, colors):
    msd_x = ((x_nm[g] - x_nm[g, :1])**2).mean(axis=0)
    msd_y = ((y_nm[g] - y_nm[g, :1])**2).mean(axis=0)
    msd = msd_x + msd_y  # 2D MSD = suma de MSD por eje
    D_mean = D[g].mean()
    ax1.plot(t, msd, color=c, label=f"empirico D~{D_mean:.3f} um2/s")
    ax1.plot(t, 4 * D_mean * t * 1e6, "--", color=c, lw=1)  # 4Dt en nm^2 (2 ejes), D en um2/s -> nm2/s *1e6

ax1.set_xlabel("t [s]")
ax1.set_ylabel("MSD_xy [nm^2]")
ax1.set_title("MSD 2D empirico (solido) vs 4*D*t teorico (punteado)")
ax1.legend(fontsize=8)

# ajuste lineal para estimar D efectivo de todo el ensemble
msd_x_all = ((x_nm - x_nm[:, :1])**2).mean(axis=0)
msd_y_all = ((y_nm - y_nm[:, :1])**2).mean(axis=0)
msd_all = msd_x_all + msd_y_all
slope = np.polyfit(t[1:], msd_all[1:], 1)[0]  # nm^2/s
D_ajustado = slope / 4 / 1e6  # um2/s
print(f"D esperado (rango uniforme {cfg.D_rango_um2_s}): media teorica = {np.mean(cfg.D_rango_um2_s):.4f} um2/s")
print(f"D ajustado del MSD global (pendiente/4): {D_ajustado:.4f} um2/s")

# --- verificacion de paredes reflectantes en z ---
zlo, zhi = cfg.z_rango_nm
print(f"\nz fuera de [{zlo},{zhi}]: {np.mean((z_nm < zlo) | (z_nm > zhi))*100:.4f}% de las muestras (debe ser 0)")

ax2.hist(z_nm.ravel(), bins=80, color="tab:purple", alpha=0.8)
ax2.axvline(zlo, color="red", ls="--"); ax2.axvline(zhi, color="red", ls="--")
ax2.set_xlabel("z [nm]")
ax2.set_title("Distribucion de z (reflejada en los bordes)")

fig.tight_layout()
fig.savefig(f"{OUT}/fig_difusion.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_difusion.png")
