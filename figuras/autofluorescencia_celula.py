"""
Autofluorescencia de la celula en Yeast6_1: contraste del fondo dentro y
fuera de la celula, para el fondo del simulador v2
(Config.fondo_celula_contraste_rango, configs/celula_v2.yaml).

La celula no se ve aislada: alrededor del real esta el halo de su PSF, que
cae con el fotoblanqueo (x2.3 en 100 cuadros). Por cuadro t y por anillo de
4 px alrededor del real (centrado en su posicion de ese cuadro, en el sector
de +-60 grados que se aleja del espejo, lejos de la franja), el exceso sobre
el fondo lejano (mediana a 60-120 px en ese sector, dentro de la apertura)
se ajusta por minimos cuadrados como
    exceso(r, t) = a(r) + b(r) N(t),
con N(t) los fotones del nucleo (r < 6 px). b(r) N es el halo del real y a(r),
el exceso que queda con N = 0: la autofluorescencia (si se blanquea al mismo
ritmo que el real, a la subestima: cota inferior). El contraste es
a(r) / fondo lejano en el plateau (r = 8-20 px), y sobre a(r) se ajusta el
perfil del simulador (cuerda de la esfera con blur gaussiano, perfil_celula):
amplitud, radio y sigma.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/autofluorescencia_celula.py [--tif RUTA] [--salida figuras]   # ~30 s
Escribe fig_autofluorescencia_celula.png (300 dpi) y autofluorescencia_celula.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import least_squares

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import analisis_datos_reales as adr  # noqa: E402

G = 0.45                                   # e-/ADU (psf_real.py, transferencia de fotones)
ANILLOS = np.arange(8, 64, 4)              # bordes internos de los anillos de 4 px
SECTOR_DEG = 60.0
LEJOS = (60, 120)                          # [px] fondo lejano
PLATEAU = (8, 20)                          # [px] anillos promediados para el contraste
# posiciones medias (coordenadas locales de cada mitad; psf_real.py / resultados_psf_real.md)
PUNTOS = {"der": ((327.4, 442.6), (373.7, 450.8)), "izq": ((435.9, 524.6), (431.9, 480.0))}
COLOR = {"der": "#eb6834", "izq": "#2a78d6"}


def seguir(e, p0):
    """Posicion del real por cuadro: maximo de la imagen suavizada cerca de la
    anterior y centroide (fondo local restado) en 5 px."""
    T, H, W = e.shape
    yy, xx = np.mgrid[0:H, 0:W]
    p, pos = np.array(p0, float), []
    for t in range(T):
        s = ndi.gaussian_filter(e[t], 2.0)
        w = np.hypot(xx - p[0], yy - p[1]) < 10
        i = np.argmax(np.where(w, s, -np.inf))
        p = np.array([xx.flat[i], yy.flat[i]], float)
        for _ in range(5):
            r = np.hypot(xx - p[0], yy - p[1])
            q = np.clip(s - np.percentile(s[r < 12], 20), 0, None) * (r < 5)
            p = np.array([(q * xx).sum(), (q * yy).sum()]) / q.sum()
        pos.append(p.copy())
    return np.array(pos)


def medir(e, ap, pos, m):
    """Por cuadro: nucleo N(t), fondo lejano y exceso medio por anillo."""
    T, H, W = e.shape
    yy, xx = np.mgrid[0:H, 0:W]
    filas = []
    for t in range(T):
        dx, dy = xx - pos[t, 0], yy - pos[t, 1]
        r = np.hypot(dx, dy)
        sector = -(dx * m[0] + dy * m[1]) / np.maximum(r, 1e-9) > np.cos(np.radians(SECTOR_DEG))
        lejos = np.median(e[t][ap & sector & (r > LEJOS[0]) & (r < LEJOS[1])])
        nucleo = (e[t][r < 6] - lejos).sum()
        filas.append([nucleo, lejos] + [np.mean(e[t][sector & (r >= a) & (r < a + 4)]) - lejos for a in ANILLOS])
    return np.array(filas)


def regresion(f):
    """a(r), ee(a), b(r) de exceso = a + b N por anillo."""
    T = len(f)
    A = np.c_[np.ones(T), f[:, 0]]
    out = []
    for k in range(len(ANILLOS)):
        y = f[:, 2 + k]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        ee = np.sqrt(res @ res / (T - 2) * np.linalg.inv(A.T @ A)[0, 0])
        out.append((coef[0], ee, coef[1]))
    return np.array(out)


def perfil_modelo(r, amp, R, sigma, c):
    """Perfil radial de perfil_celula (cuerda de la esfera con blur gaussiano)
    promediado en anillos de 4 px, en una rejilla 2D con el centro en el real."""
    n = 161
    yy, xx = np.mgrid[-n // 2 + 1:n // 2 + 1, -n // 2 + 1:n // 2 + 1]
    rr = np.hypot(xx, yy)
    p = ndi.gaussian_filter(np.sqrt(np.clip(1 - rr ** 2 / R ** 2, 0, None)), max(sigma, 1e-3))
    return np.array([amp * p[(rr >= a) & (rr < a + 4)].mean() + c for a in r])


def main():
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap_.add_argument("--tif", default=adr.TIF_DEF)
    ap_.add_argument("--salida", default=AQUI)
    a = ap_.parse_args()
    v, _ = adr.cargar(a.tif)
    T = v.shape[0]
    c = adr.buscar_corte(v.mean(0))
    corte = c["corte"]
    filas, perfiles = [], {}
    for nom, sl, ap in (("der", slice(corte, None), c["ap_der"]), ("izq", slice(0, corte), c["ap_izq"])):
        vid = v[:, :, sl].astype(np.float32)
        osc = np.array([np.median(vid[t][~ap]) for t in range(T)])
        e = (vid - osc[:, None, None]) * G
        real, refl = (np.array(p) for p in PUNTOS[nom])
        m = (refl - real) / np.linalg.norm(refl - real)               # real -> reflejo (hacia el espejo)
        pos = seguir(e, real)
        f = medir(e, ap, pos, m)
        reg = regresion(f)
        lejos = float(f[:, 1].mean())
        sel = (ANILLOS >= PLATEAU[0]) & (ANILLOS + 4 <= PLATEAU[1])
        a_pl = float(reg[sel, 0].mean())
        sol = least_squares(lambda q: (perfil_modelo(ANILLOS, *q) - reg[:, 0]) / reg[:, 1],
                            x0=[a_pl, 28.0, 4.0, 0.3], bounds=([0, 10, 0, -2], [50, 60, 15, 5]))
        amp, R, sg, cte = sol.x
        exceso_ini = f[:10, 2:][:, sel].mean()                          # con el halo, cuadros 0-9
        perfiles[nom] = (reg, f, sol.x, lejos)
        filas.append(dict(canal=nom, fondo_lejos_e=lejos, nucleo_ini_e=f[:10, 0].mean(), nucleo_fin_e=f[-10:, 0].mean(),
                          exceso_plateau_N0_e=a_pl, contraste_plateau=a_pl / lejos,
                          exceso_plateau_con_halo_ini_e=exceso_ini, contraste_con_halo_ini=exceso_ini / lejos,
                          ajuste_amp_e=amp, ajuste_contraste=amp / lejos, ajuste_R_px=R, ajuste_R_um=R * adr.PIXEL_NM / 1000,
                          ajuste_sigma_px=sg, ajuste_cte_e=cte, deriva_real_px=float(np.ptp(pos, axis=0).max())))
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(a.salida, "autofluorescencia_celula.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        for nom, (reg, f, q, lejos) in perfiles.items():
            print(f"\n{nom}: anillo [px]  a (N = 0) +- ee   b x 1e4   exceso medio")
            for k, r0 in enumerate(ANILLOS):
                print(f"   {r0:2d}-{r0 + 4:2d}   {reg[k, 0]:6.2f} +- {reg[k, 1]:.2f}   {reg[k, 2] * 1e4:7.2f}   {f[:, 2 + k].mean():6.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    rr = ANILLOS + 2
    for nom, (reg, f, q, lejos) in perfiles.items():
        col = COLOR[nom]
        axs[0].plot(rr, f[:10, 2:].mean(0), ":", color=col, lw=1.4, label=f"{nom}: exceso, cuadros 0-9 (con halo)")
        axs[0].errorbar(rr, reg[:, 0], 1.96 * reg[:, 1], fmt="o", color=col, ms=4, label=f"{nom}: exceso con N = 0")
        rf = np.linspace(8, 62, 28)
        axs[0].plot(rr, perfil_modelo(ANILLOS, *q), "-", color=col, lw=1.6,
                    label=f"{nom}: cuerda + blur (R {q[1]:.0f} px, sigma {q[2]:.1f} px)")
        axs[1].plot(f[:, 0], f[:, 2 + 2], "o", color=col, ms=3, alpha=0.6, label=f"{nom}: anillo 16-20 px")
        xx = np.array([0, f[:, 0].max()])
        axs[1].plot(xx, reg[2, 0] + reg[2, 2] * xx, "-", color=col, lw=1.4)
    axs[0].axhline(0, color="#8a8984", lw=0.7)
    axs[0].set_xlabel("distancia al real [px] (sector que se aleja del espejo)")
    axs[0].set_ylabel("exceso sobre el fondo lejano [e-/px]")
    axs[0].set_title("perfil del exceso: halo del real y celula", fontsize=9)
    axs[1].set_xlabel("fotones del nucleo del real (r < 6 px) [e-]")
    axs[1].set_ylabel("exceso en el anillo [e-/px]")
    axs[1].set_title("exceso = a + b N por cuadro: a es la celula", fontsize=9)
    for ax in axs:
        ax.legend(frameon=False, fontsize=6.5)
        ax.grid(True, color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(a.salida, "fig_autofluorescencia_celula.png"), dpi=300)


if __name__ == "__main__":
    main()
