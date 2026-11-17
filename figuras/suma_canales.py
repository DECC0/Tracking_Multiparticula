"""
Suma de los dos canales de polarizacion de Yeast6_1: se parece mas a la PSF
escalar del simulador que cada canal por separado? (continuacion de
resultados_psf_real.md; sin la red)

1. Registro del canal izquierdo sobre la rejilla del derecho, cuadro a cuadro,
   con la similitud impropia ya estimada (x_der = s Q x_izq + t, s = 1.049, Q =
   Rot(-85.2) diag(1, -1); la misma de psf_real.py): en e- (ADU menos el nivel
   oscuro de cada canal y cuadro, por 0.46 e-/ADU), spline cubica (orden 3, con
   prefiltro) y factor 1 / s^2 (jacobiano), de modo que se conserva el flujo. Se
   verifica con los centroides del real y del reflejo cuadro a cuadro (en el
   izquierdo registrado contra el derecho y contra la transformacion de los
   centroides nativos) y con la fotometria (disco nativo de radio r / s contra
   disco registrado de radio r). Suma = derecho + izquierdo registrado; suma
   equilibrada (suma_eq) = derecho + k izquierdo registrado, con k el cociente de
   flujos del real (disco de 2.5 um): los dos canales con el mismo peso, como si la
   diferencia de brillo (~2.8x) fuera de transmision.
2. Para el real y el reflejo en derecho, izquierdo registrado, suma y suma equilibrada: imagen media
   de los cuadros 0..N-1 alineados en subpixel (recortes_alineados de psf_real.py);
   eje y fuerza de los lobulos con medir() de psf_real.py (dos gaussianas y
   momentos, armonicos); ajuste contra PSFRecortada de cerca y de lejos (espejo a
   45 grados con phi = m de los datos, recorte de pupila del real y del reflejo)
   con x, y, z, fotones y fondo libres en el disco de 25 px: rejilla en z de -8 a
   8 um cada 100 nm (fotones y fondo lineales) y refinamiento no lineal desde los
   3 mejores minimos; el real tambien con |z| <= 3 um (sin acotar se va a |z| ~ 9 um,
   un disco difuso sin anillos). chi2 con la varianza entre cuadros de la imagen media (la
   suma interpolada no es Poisson: chi2/nu sirve para comparar casos, no como
   verosimilitud), residuo relativo |datos - modelo| / |datos - fondo| (y sin la
   contribucion del ruido, restando sum(var) del numerador) y
   armonicos 1, 2 y 4 del residuo en el nucleo (1.5-7 px) y el anillo (6-18 px).
3. Fotones del real y del reflejo (discos de 1.5 y 2.5 um, fondo local, cuadros
   0..N-1) y del ajuste, por canal y en la suma; cociente reflejo / real.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/suma_canales.py [--tif RUTA] [--cuadros 30] [--salida figuras/suma_canales]
Escribe en --salida: suma_canales_registro.csv, suma_canales_lobulos.csv,
suma_canales_ajustes.csv, suma_canales_fotones.csv, suma_canales_cociente.csv, fig_suma_canales_imagenes.png,
fig_suma_canales_residuos.png y fig_suma_canales_medidas.png (300 dpi).
Resultados en resultados_suma_canales.md.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import least_squares

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(1, str(AQUI.parent / "simulador_nuevo"))
import analisis_datos_reales as adr  # noqa: E402
import psf_real as pr  # noqa: E402
import simulador as sim  # noqa: E402

PX, G = adr.PIXEL_NM, adr.GANANCIA_E_ADU
R_REG = 90                      # medio lado de la region registrada alrededor del par [px, marco derecho]
R_AJ = 25.0                     # radio del ajuste de PSF [px]
K_PSF = 128
Z_GRID = np.arange(-8000.0, 8000.1, 100.0)
Z_ACOTADA = 3000.0              # ajuste del real tambien con |z| <= 3 um
FUENTES = ("der", "izq_reg", "suma", "suma_eq")
PUNTOS = ("real", "reflejo")
VARIANTES = {"cerca": True, "lejos": False}
COLOR = {"der": "#eb6834", "izq_reg": "#2a78d6", "suma": "#1baf7a", "suma_eq": "#6250d6", "cerca": "#2a78d6", "lejos": "#eb6834"}
RADIOS_UM = (1.5, 2.5)


# ---------------------------------------------------------------- registro

def nivel_oscuro(video, cort, corte):
    """Media recortada fuera de las aperturas, por canal y cuadro (como psf_real.py)."""
    T, H, W = video.shape
    union = np.zeros((H, W), bool)
    union[:, :corte] = cort["ap_izq"]
    union[:, corte:] = cort["ap_der"]
    fuera = ndi.binary_erosion(~ndi.binary_dilation(union, iterations=15), iterations=5)
    fuera[:, corte - 20:corte + 20] = False
    fl, fd = fuera[:, :corte], fuera[:, corte:]
    return (np.array([pr.media_recortada(video[t, :, :corte][fl]) for t in range(T)]),
            np.array([pr.media_recortada(video[t, :, corte:][fd]) for t in range(T)]))


def registrar(img_izq, tr, x0, y0, forma):
    """Imagen del canal izquierdo (e-) llevada a la region del derecho con origen (x0, y0)
    y forma (alto, ancho): x_izq = M^-1 (x_der - t); spline cubica y jacobiano 1/|det M|."""
    S = np.array([[0.0, 1.0], [1.0, 0.0]])
    Minv = np.linalg.inv(tr["M"])
    mat = S @ Minv @ S
    off = S @ Minv @ (np.array([x0, y0], float) - tr["t"])
    out = ndi.affine_transform(img_izq, mat, offset=off, output_shape=forma, order=3, mode="constant", cval=0.0)
    return out / abs(np.linalg.det(tr["M"]))


def centroides(stack, pos, r=10):
    """Centroide iterativo de analisis_datos_reales en cada cuadro desde pos[t]."""
    out = np.full((len(stack), 2), np.nan)
    for t in range(len(stack)):
        if np.isfinite(pos[t]).all():
            x, y, _ = adr.centroide(stack[t], float(pos[t, 0]), float(pos[t, 1]), r=r)
            out[t] = x, y
    return out


# ---------------------------------------------------------------- ajuste de PSF

class Ajustador:
    """PSFRecortada de una variante con la geometria de los datos; ajusta b + N P(x, y, z)."""

    def __init__(self, anillos, phi):
        cfg = sim.Config(anillos_hacia_objetivo=anillos, usar_espejo=True, angulo_espejo_eje_deg=45.0,
                         orientacion_espejo_deg=phi, recorte_pupila_real=True, recorte_pupila_reflejo=True)
        self.psf = sim.PSFRecortada(cfg, K_PSF)
        self.cache = {}

    def unitaria(self, x, y, z, n):
        img = np.zeros((n, n))
        self.psf.colocar(img, x, y, z, 1.0)
        return img

    def ajustar(self, datos, var, z_lim=None):
        """z_lim: |z| <= z_lim [nm] (rejilla y cotas); None: libre en Z_GRID."""
        n = datos.shape[0]
        c = (n - 1) / 2
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        sel = np.hypot(X - c, Y - c) <= R_AJ
        d, w = datos[sel], 1 / np.sqrt(var[sel])

        def lineal(P):                                   # N, b por minimos cuadrados ponderados
            A = np.c_[P[sel], np.ones(sel.sum())] * w[:, None]
            coef, *_ = np.linalg.lstsq(A, d * w, rcond=None)
            return coef, float(((A @ coef - d * w) ** 2).sum())

        chi_z = []
        for z in Z_GRID:
            key = (n, z)
            if key not in self.cache:
                self.cache[key] = self.unitaria(c, c, z, n)
            coef, chi = lineal(self.cache[key])
            chi_z.append(chi if coef[0] > 0 else np.inf)
        chi_z = np.array(chi_z)
        if z_lim is not None:
            chi_z[np.abs(Z_GRID) > z_lim] = np.inf
        loc = [i for i in range(len(Z_GRID)) if np.isfinite(chi_z[i]) and chi_z[i] <= chi_z[max(i - 1, 0)]
               and chi_z[i] <= chi_z[min(i + 1, len(Z_GRID) - 1)]]
        inicios = sorted(loc, key=lambda i: chi_z[i])[:3]
        mejor = None
        for i in inicios:
            coef, _ = lineal(self.cache[(n, Z_GRID[i])])

            def res(q):
                return ((q[4] + q[3] * self.unitaria(q[0], q[1], q[2], n)[sel]) - d) * w
            q0 = np.array([c, c, Z_GRID[i], coef[0], coef[1]])
            zl = 9000.0 if z_lim is None else z_lim
            s_ = least_squares(res, q0, x_scale=np.array([0.3, 0.3, 50.0, abs(coef[0]) * 0.1 + 1, 1.0]),
                               bounds=([c - 4, c - 4, -zl, 0, -np.inf], [c + 4, c + 4, zl, np.inf, np.inf]))
            if mejor is None or s_.cost < mejor.cost:
                mejor = s_
        q = mejor.x
        modelo = q[4] + q[3] * self.unitaria(q[0], q[1], q[2], n)
        chi2 = 2 * mejor.cost
        nu = int(sel.sum()) - 5
        sr2, sd2 = ((datos - modelo)[sel] ** 2).sum(), ((datos - q[4])[sel] ** 2).sum()
        rel = float(np.sqrt(sr2 / sd2))
        # sin la contribucion esperada del ruido de la imagen media (sum var): solo la falta de modelo
        rel_mod = float(np.sqrt(max(sr2 - var[sel].sum(), 0.0) / sd2))
        return dict(x=q[0], y=q[1], z_nm=q[2], N=q[3], b=q[4], chi2=chi2, nu=nu, chi2_nu=chi2 / nu, residuo_rel=rel,
                    residuo_rel_sin_ruido=rel_mod, z_acotada=z_lim is not None,
                    fraccion_pupila=self.psf.fraccion, modelo=modelo, chi_z=chi_z)


def armonicos_residuo(res, datos, centro, b):
    """Amplitud del armonico k (1, 2, 4) del perfil angular del residuo, relativa al nivel
    medio de los datos (menos el fondo) en la misma corona, y su direccion; nucleo y anillo."""
    out = {}
    for nombre, corona in (("nucleo", pr.CRUZ), ("anillo", pr.ANILLO)):
        th, p_res = pr.perfil_angular(res, centro, 0.0, corona)
        _, p_dat = pr.perfil_angular(datos, centro, b, corona)
        ref = p_dat.mean()
        for k in (1, 2, 4):
            ck = (p_res * np.exp(-1j * k * np.radians(th))).mean()
            out[f"res_{nombre}_a{k}"] = float(2 * abs(ck) / ref)
            out[f"res_{nombre}_a{k}_dir"] = float(pr.envolver(-np.degrees(np.angle(ck)) / k, 360.0 / k))
        out[f"res_{nombre}_rms_rel"] = float(np.sqrt(np.mean(p_res ** 2)) / ref)
        out[f"res_{nombre}_media_rel"] = float(p_res.mean() / ref)
    return out


# ---------------------------------------------------------------- figuras

def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def figura_imagenes(medias, lob, phi, ruta):
    plt = _estilo()
    fig, axes = plt.subplots(2, len(FUENTES), figsize=(3.5 * len(FUENTES), 8.0))
    for i, pto in enumerate(PUNTOS):
        for j, f in enumerate(FUENTES):
            ax = axes[i, j]
            im = medias[f, pto]
            n = im.shape[0]
            c = (n - 1) / 2
            z = im[int(c) - 20:int(c) + 21, int(c) - 20:int(c) + 21]
            ax.imshow(z, cmap="magma", vmin=np.percentile(z, 1), vmax=z.max(), extent=(-20.5, 20.5, 20.5, -20.5))
            g = lob[f, pto]["g2"]
            x1, y1, x2, y2 = (np.array(g["lob"]) - c)
            ax.plot([x1, x2], [y1, y2], "-", color="white", lw=1)
            ax.plot([x1, x2], [y1, y2], "o", ms=3, mfc="none", mec="white", mew=0.8)
            u = pr.vec(phi)
            ax.annotate("", xy=(14 + 6 * u[0], 15 + 6 * u[1]), xytext=(14, 15),
                        arrowprops=dict(arrowstyle="->", color="#1baf7a", lw=1))
            ax.set_title(f"{f}, {pto}: eje {g['eje']:+.0f}, flujo 2/1 {g['flujo']:.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Imagen media (cuadros 0-29, alineados) en el marco del canal derecho; blanco: lobulos (dos gaussianas); "
                 "verde: m (reflejo -> real)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def figura_residuos(medias, ajustes, ruta):
    plt = _estilo()
    filas = [("real", "cerca"), ("real", "lejos"), ("real", "cerca, |z| <= 3 um"), ("real", "lejos, |z| <= 3 um"),
             ("reflejo", "cerca"), ("reflejo", "lejos")]
    fig, axes = plt.subplots(len(filas), 2 * len(FUENTES), figsize=(20, 2.65 * len(filas)))
    for i, (pto, var) in enumerate(filas):
        for j, f in enumerate(FUENTES):
            d = medias[f, pto]
            c = int((d.shape[0] - 1) / 2)
            v = slice(c - 25, c + 26)
            a = ajustes[f, pto, var]
            lim = np.percentile(np.abs(d[v, v] - a["b"]), 99.5)
            axes[i, 2 * j].imshow(a["modelo"][v, v], cmap="magma", vmin=a["b"], vmax=a["b"] + lim)
            axes[i, 2 * j].set_title(f"{f} {pto}: {var}, z {a['z_nm'] / 1000:+.2f} um", fontsize=7)
            axes[i, 2 * j + 1].imshow((d - a["modelo"])[v, v], cmap="RdBu_r", vmin=-0.25 * lim, vmax=0.25 * lim)
            axes[i, 2 * j + 1].set_title(f"residuo: chi2/nu {a['chi2_nu']:.1f}, rel {a['residuo_rel']:.2f} "
                                         f"({a['residuo_rel_sin_ruido']:.2f})", fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Ajuste contra PSFRecortada (x, y, z, fotones y fondo libres, disco de 25 px; el real tambien con |z| <= 3 um). "
                 "Residuo en +-25 % del pico; rel = |datos - modelo| / |datos - fondo| (entre parentesis, sin el ruido)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def figura_medidas(reg, aj, lb, fot, ruta):
    plt = _estilo()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))
    ax = axes[0, 0]
    for pto, ls in (("real", "-"), ("reflejo", "--")):
        s = reg[reg.punto == pto]
        ax.plot(s.cuadro, s.dist_der_nm, ls, color=COLOR["izq_reg"], lw=1.2, label=f"{pto}: izq registrado - der")
        ax.plot(s.cuadro, s.dist_tr_nm, ls, color="#52514e", lw=1, label=f"{pto}: izq registrado - T(izq nativo)")
    ax.set_xlabel("cuadro")
    ax.set_ylabel("distancia entre centroides [nm]")
    ax.set_title("verificacion del registro", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[0, 1]
    x = np.arange(len(FUENTES))
    for k, (pto, desp) in enumerate((("real", -0.2), ("reflejo", 0.2))):
        s = lb[lb.punto == pto].set_index("fuente").loc[list(FUENTES)]
        ax.bar(x + desp, s.flujo_2_1, 0.38, color=[COLOR[f] for f in FUENTES], alpha=1 if pto == "real" else 0.55,
               label=pto)
        for xi, (e, f) in enumerate(zip(s.eje_2g, s.flujo_2_1)):
            ax.text(xi + desp, f + 0.01, f"{e:+.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, FUENTES)
    ax.set_ylabel("flujo del lobulo 2 / lobulo 1 (dos gaussianas)")
    ax.set_title("fuerza de los lobulos (numero: eje en grados); claro: reflejo", fontsize=9)
    ax = axes[0, 2]
    for k, (pto, desp) in enumerate((("real", -0.2), ("reflejo", 0.2))):
        s = lb[lb.punto == pto].set_index("fuente").loc[list(FUENTES)]
        ax.bar(x + desp, s.cociente_ejes, 0.38, color=[COLOR[f] for f in FUENTES], alpha=1 if pto == "real" else 0.55)
    ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_xticks(x, FUENTES)
    ax.set_ylabel("cociente de ejes del nucleo (momentos, r 4 px)")
    ax.set_title("elongacion del nucleo; claro: reflejo", fontsize=9)
    for j, pto in enumerate(PUNTOS):
        ax = axes[1, j]
        for var, ls in (("cerca", "-"), ("lejos", "--")):
            s = aj[(aj.punto == pto) & (aj.variante == var)].set_index("fuente").loc[list(FUENTES)]
            ax.plot(x, s.chi2_nu, ls, marker="o", color=COLOR[var], label=f"{var}: chi2/nu")
            ax2 = ax
        ax.set_xticks(x, FUENTES)
        if pto == "real":
            for var, ls in (("cerca, |z| <= 3 um", ":"), ("lejos, |z| <= 3 um", "-.")):
                s = aj[(aj.punto == pto) & (aj.variante == var)].set_index("fuente").loc[list(FUENTES)]
                ax.plot(x, s.chi2_nu, ls, marker="o", ms=3, color=COLOR[var.split(",")[0]], alpha=0.7, label=f"{var}: chi2/nu")
        ax.set_ylabel("chi2/nu (varianza entre cuadros)")
        ax.set_title(f"{pto}: ajuste de PSF", fontsize=9)
        axb = ax.twinx()
        for var, ls in (("cerca", "-"), ("lejos", "--")):
            s = aj[(aj.punto == pto) & (aj.variante == var)].set_index("fuente").loc[list(FUENTES)]
            axb.plot(x, s.residuo_rel_sin_ruido, ls, marker="s", ms=3, color=COLOR[var], alpha=0.5)
        axb.set_ylabel("residuo relativo sin ruido (cuadrados)", fontsize=7)
        axb.spines["right"].set_visible(True)
        ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 2]
    for pto, desp in (("real", -0.2), ("reflejo", 0.2)):
        s = fot[(fot.punto == pto) & (fot.radio_um == 2.5)].set_index("fuente").loc[list(FUENTES)]
        ax.bar(x + desp, s.e_media, 0.38, yerr=s.e_std, color=[COLOR[f] for f in FUENTES],
               alpha=1 if pto == "real" else 0.55)
    ax.set_xticks(x, FUENTES)
    ax.set_ylabel("e- en el disco de 2.5 um (cuadros 0-29)")
    ax.set_title("fotones; claro: reflejo", fontsize=9)
    for a_ in axes.flat:
        a_.grid(True, color="#e6e5e0", lw=0.6)
    fig.suptitle("Suma de canales: registro, lobulos, ajuste de PSF y fotones", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- principal

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--tif", default=adr.TIF_DEF)
    ap.add_argument("--cuadros", type=int, default=30)
    ap.add_argument("--salida", default=str(AQUI / "suma_canales"))
    a = ap.parse_args()
    N = a.cuadros
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    video, _ = adr.cargar(a.tif)
    T = len(video)
    cort = adr.buscar_corte(video.mean(0))
    corte = cort["corte"]
    oscL, oscR = nivel_oscuro(video, cort, corte)
    canL, canR = video[:, :, :corte].astype(np.float32), video[:, :, corte:].astype(np.float32)
    del video
    pares = []
    for k, (vid, apm) in enumerate(((canL, cort["ap_izq"]), (canR, cort["ap_der"]))):
        pk = adr.pares_real_reflejo(adr.enlazar([adr.detectar(vid[t], apm) for t in range(T)], T), T)
        assert len(pk) == 1
        pares.append(pk[0])
    pL, pR = pares
    ok = np.isfinite(pL["real"][:, 0] + pL["refl"][:, 0] + pR["real"][:, 0] + pR["refl"][:, 0])
    tr = adr.similitud(np.r_[pL["real"][ok], pL["refl"][ok]], np.r_[pR["real"][ok], pR["refl"][ok]], True)
    print(f"transformacion izq -> der: s = {tr['s']:.4f}, angulo de Q = {adr.angulo_Q(tr['Q']):+.2f} deg, "
          f"det M = {np.linalg.det(tr['M']):+.4f}; nivel oscuro izq {oscL.mean():.2f}, der {oscR.mean():.2f} ADU")
    assert abs(tr["s"] - 1.049) < 0.002 and abs(adr.angulo_Q(tr["Q"]) + 85.2) < 0.2, "no es la transformacion estimada"

    # e- y region del derecho alrededor del par
    eL = (canL - oscL[:, None, None]) * G
    eR = (canR - oscR[:, None, None]) * G
    del canL, canR
    cen = np.nanmean(np.r_[pR["real"], pR["refl"]], 0)
    x0, y0 = int(round(cen[0])) - R_REG, int(round(cen[1])) - R_REG
    forma = (2 * R_REG + 1, 2 * R_REG + 1)
    assert x0 >= 0 and y0 >= 0 and y0 + forma[0] <= eR.shape[1] and x0 + forma[1] <= eR.shape[2]
    R = eR[:, y0:y0 + forma[0], x0:x0 + forma[1]].astype(float)
    Lr = np.array([registrar(eL[t], tr, x0, y0, forma) for t in range(T)])
    Sm = R + Lr
    off = np.array([x0, y0])
    pos = {"der": {"real": pR["real"] - off, "reflejo": pR["refl"] - off}}
    T_real, T_refl = adr.aplicar(tr, pL["real"]) - off, adr.aplicar(tr, pL["refl"]) - off

    # 1. verificacion del registro: centroides en el izquierdo registrado
    filas_reg = []
    pos["izq_reg"], pos["suma"] = {}, {}
    for pto, p_der, p_tr in (("real", pos["der"]["real"], T_real), ("reflejo", pos["der"]["reflejo"], T_refl)):
        c_reg = centroides(Lr, p_tr)
        c_der = centroides(R, p_der)
        c_sum = centroides(Sm, p_der)
        pos["izq_reg"][pto], pos["suma"][pto] = c_reg, c_sum
        for t in range(T):
            filas_reg.append(dict(punto=pto, cuadro=t, dist_der_nm=np.hypot(*(c_reg[t] - c_der[t])) * PX,
                                  dist_tr_nm=np.hypot(*(c_reg[t] - p_tr[t])) * PX,
                                  dx_der_nm=(c_reg[t, 0] - c_der[t, 0]) * PX, dy_der_nm=(c_reg[t, 1] - c_der[t, 1]) * PX))
    reg = pd.DataFrame(filas_reg)
    # suma equilibrada: el izquierdo escalado por el cociente de flujos del real (disco de 2.5 um, cuadros 0..N-1),
    # como si los dos canales tuvieran la misma transmision
    r25 = 2.5 * 1000 / PX
    f_der = np.mean([pr.disco(R[t], pos["der"]["real"][t], r25, pos["der"]["reflejo"][t])[0] for t in range(N)])
    f_izq = np.mean([pr.disco(Lr[t], pos["izq_reg"]["real"][t], r25, pos["izq_reg"]["reflejo"][t])[0] for t in range(N)])
    k_eq = f_der / f_izq
    Seq = R + k_eq * Lr
    pos["suma_eq"] = {pto: centroides(Seq, pos["der"][pto]) for pto in PUNTOS}
    print(f"    suma equilibrada: izquierdo x {k_eq:.3f} (flujo del real en 2.5 um: der {f_der:.0f}, izq reg {f_izq:.0f} e-)")
    reg.to_csv(salida / "suma_canales_registro.csv", index=False)
    print("\n[1] registro: centroide del izquierdo registrado frente al derecho y frente a T(centroide izq nativo)")
    for pto, s in reg.groupby("punto"):
        print(f"    {pto:7s}: izq reg - der: mediana {s.dist_der_nm.median():.0f} nm, rms {np.sqrt((s.dist_der_nm ** 2).mean()):.0f} nm "
              f"(sesgo x {s.dx_der_nm.mean():+.0f}, y {s.dy_der_nm.mean():+.0f} nm); izq reg - T(izq nativo): mediana "
              f"{s.dist_tr_nm.median():.0f} nm, rms {np.sqrt((s.dist_tr_nm ** 2).mean()):.0f} nm")

    # 3 (antes, para el flujo). fotometria por cuadro, y conservacion del flujo frente al izquierdo nativo
    filas_f, por_cuadro = [], {}
    s_ = tr["s"]
    for pto, otro in (("real", "reflejo"), ("reflejo", "real")):
        for rum in RADIOS_UM:
            r = rum * 1000 / PX
            arr = {f: np.array([pr.disco(img[t], pos[f2][pto][t], r, pos[f2][otro][t])[0] for t in range(N)])
                   for f, img, f2 in (("der", R, "der"), ("izq_reg", Lr, "izq_reg"), ("suma", Sm, "suma"),
                                         ("suma_eq", Seq, "suma_eq"))}
            ladr = pL["real"] if pto == "real" else pL["refl"]
            lotro = pL["refl"] if pto == "real" else pL["real"]
            arr["izq_nativo"] = np.array([pr.disco(eL[t], ladr[t], r / s_, lotro[t])[0] for t in range(N)])
            for f, v in arr.items():
                filas_f.append(dict(punto=pto, radio_um=rum, fuente=f, e_media=v.mean(), e_std=v.std(ddof=1)))
                por_cuadro[f, pto, rum] = v
    fot = pd.DataFrame(filas_f)
    # cociente reflejo / real cuadro a cuadro (los dos puntos del mismo cuadro: el fotoblanqueo se cancela)
    filas_c = []
    for rum in RADIOS_UM:
        for f in ("der", "izq_nativo", "izq_reg", "suma", "suma_eq"):
            q = por_cuadro[f, "reflejo", rum] / por_cuadro[f, "real", rum]
            filas_c.append(dict(fuente=f, radio_um=rum, cociente_medio=q.mean(), ee=q.std(ddof=1) / np.sqrt(len(q))))
        d = por_cuadro["izq_reg", "reflejo", rum] / por_cuadro["izq_reg", "real", rum] - \
            por_cuadro["der", "reflejo", rum] / por_cuadro["der", "real", rum]
        filas_c.append(dict(fuente="izq_reg - der (pareado)", radio_um=rum, cociente_medio=d.mean(),
                            ee=d.std(ddof=1) / np.sqrt(len(d))))
    coc = pd.DataFrame(filas_c)
    coc.to_csv(salida / "suma_canales_cociente.csv", index=False)
    print("\n[3] cociente reflejo / real por cuadro (media +- ee, cuadros 0..N-1)")
    print(coc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 2. imagenes medias alineadas y lobulos
    medias, varis, lob = {}, {}, {}
    stacks = {"der": R, "izq_reg": Lr, "suma": Sm, "suma_eq": Seq}
    phi = pr.ang_vec(np.nanmean(pR["real"] - pR["refl"], 0))          # m: reflejo -> real
    dir_otro = {"real": phi + 180.0, "reflejo": phi}
    for f, st in stacks.items():
        for pto in PUNTOS:
            rec, _ = pr.recortes_alineados(st, pos[f][pto], N)
            medias[f, pto], varis[f, pto] = rec.mean(0), rec.var(0, ddof=1) / N
            lob[f, pto] = pr.medir(medias[f, pto], dir_otro[pto], 4.0)
    filas_l = []
    print(f"\n[2] lobulos (marco del derecho; m = {phi:+.1f}, t = {pr.ang_eje(phi + 90):+.1f} deg)")
    for (f, pto), r in lob.items():
        g = r["g2"]
        filas_l.append(dict(fuente=f, punto=pto, eje_2g=g["eje"], eje_vs_m=pr.ang_eje(g["eje"] - phi), sep_nm=g["sep"] * PX,
                            flujo_2_1=g["flujo"], eje_momentos=r["mom"], cociente_ejes=r["ejes"], a4_nucleo=r["a4"][0],
                            a2_nucleo=r["a2c"][0], a1_anillo=r["a1"][0], a1_anillo_dir=r["a1"][1], a2_anillo=r["a2"][0],
                            a2_anillo_eje=r["a2"][1]))
        print(f"    {f:7s} {pto:7s}: dos gaussianas eje {g['eje']:+6.1f} (vs m {pr.ang_eje(g['eje'] - phi):+6.1f}), sep "
              f"{g['sep'] * PX:4.0f} nm, flujo 2/1 {g['flujo']:.2f} | momentos eje {r['mom']:+6.1f}, cociente {r['ejes']:.2f} | "
              f"nucleo a4 {r['a4'][0]:.2f}, a2 {r['a2c'][0]:.2f} | anillo a1 {r['a1'][0]:.2f} hacia {r['a1'][1]:+.0f}, a2 "
              f"{r['a2'][0]:.2f} eje {r['a2'][1]:+.0f}")
    lb = pd.DataFrame(filas_l)
    lb.to_csv(salida / "suma_canales_lobulos.csv", index=False)

    # ajuste de PSF
    ajus = {v: Ajustador(an, phi) for v, an in VARIANTES.items()}
    filas_a, ajustes = [], {}
    print("\n[2] ajuste contra PSFRecortada (disco de 25 px; chi2 con la varianza entre cuadros)")
    for f in FUENTES:
        for pto in PUNTOS:
            casos = [(var, aj, None) for var, aj in ajus.items()]
            if pto == "real":                                # rango plausible del real (z triangulada -0.84 um)
                casos += [(f"{var}, |z| <= 3 um", aj, Z_ACOTADA) for var, aj in ajus.items()]
            for var, aj, zl in casos:
                r = aj.ajustar(medias[f, pto], varis[f, pto], zl)
                ajustes[f, pto, var] = r
                c = ((medias[f, pto].shape[0] - 1) / 2,) * 2
                h = armonicos_residuo(medias[f, pto] - r["modelo"], medias[f, pto], (r["x"], r["y"]), r["b"])
                filas_a.append(dict(fuente=f, punto=pto, variante=var, **{k: v for k, v in r.items()
                                                                          if k not in ("modelo", "chi_z")}, **h))
                print(f"    {f:7s} {pto:7s} {var}: z {r['z_nm']:+6.0f} nm, N {r['N']:7.0f} e-, b {r['b']:.2f} e-/px, chi2/nu "
                      f"{r['chi2_nu']:6.1f}, residuo rel {r['residuo_rel']:.3f} (sin ruido {r['residuo_rel_sin_ruido']:.3f}); residuo nucleo a1/a2/a4 {h['res_nucleo_a1']:.2f}/"
                      f"{h['res_nucleo_a2']:.2f}/{h['res_nucleo_a4']:.2f}, anillo {h['res_anillo_a1']:.2f}/{h['res_anillo_a2']:.2f}/"
                      f"{h['res_anillo_a4']:.2f}; media del residuo nucleo {h['res_nucleo_media_rel']:+.2f}, anillo "
                      f"{h['res_anillo_media_rel']:+.2f}")
    aj = pd.DataFrame(filas_a)
    aj.to_csv(salida / "suma_canales_ajustes.csv", index=False)

    # 3. fotones y cocientes
    for (pto, var), s in aj.groupby(["punto", "variante"]):
        for _, row in s.iterrows():
            filas_f.append(dict(punto=pto, radio_um=np.nan, fuente=row.fuente, e_media=row.N, e_std=np.nan,
                                metodo=f"ajuste {var}"))
    fot = pd.DataFrame(filas_f)
    fot["metodo"] = fot.get("metodo").fillna("disco") if "metodo" in fot else "disco"
    fot.to_csv(salida / "suma_canales_fotones.csv", index=False)
    print("\n[3] fotones (e-, cuadros 0..N-1)")
    for metodo, s in fot.groupby(["metodo", "radio_um"], dropna=False):
        t = s.pivot_table(index="fuente", columns="punto", values="e_media")
        if {"real", "reflejo"} <= set(t.columns):
            t["reflejo/real"] = t["reflejo"] / t["real"]
        print(f"  {metodo}:\n" + t.round(3).to_string())
    figura_imagenes(medias, lob, phi, salida / "fig_suma_canales_imagenes.png")
    figura_residuos(medias, ajustes, salida / "fig_suma_canales_residuos.png")
    figura_medidas(reg, aj, lb, fot[fot.metodo == "disco"], salida / "fig_suma_canales_medidas.png")


if __name__ == "__main__":
    main()
