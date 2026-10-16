"""Check CRLB: precision de un ajuste de maxima verosimilitud (MLE) sobre
imagenes simuladas contra la cota de Cramer-Rao calculada con el mismo modelo.
Si PSF, fotones y ruido estan bien encadenados, el MLE alcanza la cota
(razon ~1, errores gaussianos) en el regimen de SNR alto. A SNR bajo (con la
esferica, N ~ 1000 sobre 20 fotones/px) el MLE no es eficiente: aparecen colas
pesadas (curtosis > 0) y la razon sube; eso es esperado, no un error del
simulador. Genera fig_crlb.png. Tarda ~2 min."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import simulador as sim

OUT = os.path.dirname(os.path.abspath(__file__))
b = sim.BancoPSF(sim.Config())                       # esferica
cfg = b.cfg
L, z, fondo = 15, 0.0, 20.0
var_e2 = (cfg.read_noise**2 + 1 / 12) * cfg.gain**2
rng = np.random.default_rng(0)
n_ajustes = 200


def nll(theta, datos):
    """-log L con la aproximacion sCMOS de Huang et al. (2013): datos + var ~ Poisson(mu + var)."""
    x, y, N, bg = theta
    if N <= 0 or bg <= 0:
        return 1e30
    mu = sim.modelo_roi(b, x, y, z, N, bg, L) + var_e2
    return np.sum(mu - (datos + var_e2) * np.log(mu))


Ns = [1000.0, 3000.0, 10000.0, 30000.0]
res = []
print(f"mascara {cfg.mascara}, ROI {L}x{L}, z = {z} nm, fondo = {fondo} fotones/px, {n_ajustes} ajustes por N")
print(f"(incertidumbre de la razon ~ +-{1/np.sqrt(2*n_ajustes):.2f})")
print("      N   CRLB_x [nm]   std MLE [nm]   razon   sesgo [nm]   curtosis exceso")
for N in Ns:
    x0 = y0 = L // 2
    est = []
    for _ in range(n_ajustes):
        dx, dy = rng.uniform(-0.5, 0.5, 2)
        mu = sim.modelo_roi(b, x0 + dx, y0 + dy, z, N, fondo, L)
        datos = sim.adu_a_electrones(sim.aplicar_camara(mu, cfg, rng), cfg)
        r = minimize(nll, [x0, y0, max(datos.sum() - fondo * L * L, 50.0), fondo], args=(datos,),
                     method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000))
        est.append([r.x[0] - (x0 + dx), r.x[1] - (y0 + dy)])
    est = np.array(est) * cfg.pixel_nm
    cota = sim.crlb(b, N, fondo, z, tam_roi=L)["x"] * cfg.pixel_nm
    e = est[:, 0]
    std = e.std()
    curt = ((e - e.mean())**4).mean() / e.var()**2 - 3
    res.append((N, cota, std, e.mean()))
    print(f"  {N:6.0f}   {cota:10.2f}   {std:12.2f}   {std/cota:5.2f}   {e.mean():+9.2f}   {curt:14.2f}")

res = np.array(res)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(res[:, 0], res[:, 1], "k-", label="CRLB (modelo)")
ax.loglog(res[:, 0], res[:, 2], "o", color="tab:red", label=f"std del MLE ({n_ajustes} ajustes)")
ax.set_xlabel("fotones por emisor N"); ax.set_ylabel("precision en x [nm]")
ax.set_title(f"MLE vs cota de Cramer-Rao ({cfg.mascara}, fondo {fondo:g} fotones/px)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/fig_crlb.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_crlb.png")
