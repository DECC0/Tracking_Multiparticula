import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_imagenesPSF import Config, BancoPSF, energia_encerrada

OUT = os.path.dirname(os.path.abspath(__file__))

cfg = Config()
banco = BancoPSF(cfg)
zs = banco.z_grid  # rejilla completa, paso 20nm, -800..800
p0 = banco.psf_en_z(0).max()

picos = np.array([banco.stack[i].max() for i in range(len(zs))]) / p0
enc2 = np.array([energia_encerrada(banco, z, 2.0) for z in zs])
enc4 = np.array([energia_encerrada(banco, z, 4.0) for z in zs])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax1.plot(zs, picos, "-", color="tab:red")
ax1.axhline(1.0, color="gray", lw=0.7, ls=":")
ax1.axvline(0, color="gray", lw=0.5)
ax1.set_ylabel("pico relativo I_max(z)/I_max(0)")
ax1.set_title("Barrido completo (paso 20 nm) -- esferica, amp=1.5*pi")

ax2.plot(zs, enc2, "-", color="tab:blue", label="E(r<=2px)")
ax2.plot(zs, enc4, "-", color="tab:green", label="E(r<=4px)")
ax2.axvline(0, color="gray", lw=0.5)
ax2.set_xlabel("z [nm]")
ax2.set_ylabel("energia encerrada")
ax2.legend()

fig.tight_layout()
fig.savefig(f"{OUT}/fig_psf_full_sweep.png", dpi=130)
plt.close(fig)

imax = np.argmax(picos)
print(f"pico maximo global: {picos[imax]:.3f} en z={zs[imax]:.0f} nm (vs z=0 -> 1.000 por definicion)")
print(f"pico en z=-800: {picos[0]:.3f} | z=+800: {picos[-1]:.3f}")
print(f"simetria pico(z) vs pico(-z), muestras:")
for z in [200, 400, 600, 800]:
    i_pos = np.argmin(np.abs(zs - z))
    i_neg = np.argmin(np.abs(zs + z))
    print(f"  z=+{z}: {picos[i_pos]:.3f}   z=-{z}: {picos[i_neg]:.3f}   diff={picos[i_pos]-picos[i_neg]:+.3f}")
