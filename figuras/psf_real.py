"""
PSF real de Yeast6_1 en los dos canales de polarizacion, sin la red.

Pregunta: los dos lobulos del nucleo de la PSF real (resultados_celula.md,
seccion 4) son una aberracion del camino comun (mismo eje en los dos canales
una vez llevados al mismo marco) o un efecto de polarizacion (eje rotado ~90
grados entre canales)? Ademas, fotones y fondo en e- frente a los rangos de
localizacion/configs/celula.yaml.

Pasos:
  1. deteccion, pistas y par real/reflejo por canal (analisis_datos_reales.py);
     transformacion izquierdo -> derecho (similitud impropia: deshace el giro de
     ~90 grados y el espejado entre las dos mitades del chip) validada con las
     posiciones del real y del reflejo (residuos, prediccion cruzada, intercambio,
     direccion del segmento, pasos cuadro a cuadro);
  2. por canal y por punto (real, reflejo): recortes de los cuadros 0..N-1
     alineados en subpixel (desplazamiento de Fourier al centroide y despues
     correlacion cruzada con la media, sobremuestreada x20) e imagen media; el
     canal izquierdo se lleva al marco del derecho con la parte ortogonal Q de la
     transformacion. Eje de los lobulos del nucleo por dos estimadores: ajuste de
     dos gaussianas (eje = recta entre los dos centros) y momentos de segundo
     orden; simetria de cruz del nucleo (armonico 4 del perfil angular a 1.5-7
     px, en el marco nativo de cada canal); perfil angular del anillo (6-18 px)
     alrededor del centro del nucleo y sus armonicos 1 y 2;
  3. angulos con IC por bootstrap de cuadros (los mismos indices en los dos
     canales: son simultaneos);
  4. fotones del real y del reflejo (disco de 1.5 um con fondo local, por
     cuadro; curva de crecimiento de la imagen media hasta 25 px y correccion
     por la energia encerrada de la PSF esferica simulada) y fondo dentro y fuera
     de la apertura, en e- (0.46 e-/ADU; nivel oscuro = media recortada fuera de
     las aperturas); ruido de lectura y ganancia por transferencia de fotones
     (diferencias de cuadros consecutivos).

Angulos en el marco del canal derecho (coordenadas locales: x = columna - corte,
y = fila), de +x hacia +y; ejes modulo 180. Figuras PNG 300 dpi en --salida:
fig_psf_real_imagenes.png y fig_psf_real_medidas.png. Resultados en
resultados_psf_real.md. Importa analisis_datos_reales.py (deteccion) y, solo
para la energia encerrada, simulador_nuevo/simulador.py.

Uso:
    $PY figuras/psf_real.py [--tif RUTA] [--cuadros 30] [--r-nucleo 4] [--n-boot 2000] [--salida figuras]
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from skimage.registration import phase_cross_correlation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import analisis_datos_reales as adr  # noqa: E402

PX = adr.PIXEL_NM
G = adr.GANANCIA_E_ADU
R_REC = 30                     # medio lado de los recortes [px]
ANILLO = (6.0, 18.0)           # anillo de la PSF [px] (resultados_celula.md, seccion 4)
CRUZ = (1.5, 7.0)              # corona del nucleo para el armonico 4 [px]
FONDO_REC = (22.0, 29.0)       # anillo del fondo dentro del recorte [px]
R_AJUSTE = 7.0                 # radio de la region del ajuste de dos gaussianas [px]
SEP = (1.0, 6.0)               # separacion admisible entre los dos lobulos [px]
R_CG = 25                      # radio maximo de la curva de crecimiento [px]
N_BINS = 36                    # perfiles angulares: bins de 10 grados
# z del ajuste del perfil radial de resultados_celula.md (seccion 4): (anillos_hacia_objetivo, z real, z reflejo)
Z_SECCION4 = {"cerca": (True, -1200.0, -5800.0), "lejos": (False, 5800.0, 1300.0)}
COLOR = {"izq": "#2a78d6", "der": "#eb6834", "gris": "#52514e", "rejilla": "#e6e5e0"}
PUNTOS = ("real", "reflejo")


# ---------------------------------------------------------------- utilidades

def envolver(a, mod):
    """Angulo en (-mod/2, mod/2]."""
    h = mod / 2
    return -((-np.asarray(a, dtype=float) + h) % mod - h)


def ang_eje(a):
    return envolver(a, 180.0)


def ang_dir(a):
    return envolver(a, 360.0)


def ang_vec(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def vec(a_deg):
    return np.array([np.cos(np.radians(a_deg)), np.sin(np.radians(a_deg))])


def desplazar(img, dy, dx):
    m = np.median(img)
    return np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(img - m), (dy, dx))).real + m


def a_marco_der(img, Q):
    """Lleva una imagen centrada del canal izquierdo al marco del derecho: el pixel
    de salida a o_xy del centro viene de Q^T o_xy en la entrada (Q ortogonal)."""
    S = np.array([[0.0, 1.0], [1.0, 0.0]])
    A = S @ Q.T @ S
    c = (np.array(img.shape) - 1) / 2
    return ndi.affine_transform(img, A, offset=c - A @ c, order=3, mode="nearest")


def malla(n):
    Y, X = np.mgrid[0:n, 0:n]
    return X.astype(float), Y.astype(float)


# ---------------------------------------------------------------- 2. recortes alineados

def recortes_alineados(vid, pos, n, r=R_REC, iters=2, r_reg=12):
    """Recortes (2r+1) de los cuadros 0..n-1 centrados en pos[t] por desplazamiento de
    Fourier y refinados por correlacion cruzada (sobremuestreo x20) con la media, en
    la ventana central de medio lado r_reg (con ventana de Hann). Devuelve
    (n, 2r+1, 2r+1) y los desplazamientos de refinamiento [px]."""
    p = 6
    crud = []
    for t in range(n):
        x, y = pos[t]
        ix, iy = int(round(x)), int(round(y))
        crud.append((vid[t, iy - r - p: iy + r + p + 1, ix - r - p: ix + r + p + 1].astype(float), iy - y, ix - x))
    extra = np.zeros((n, 2))
    w = np.outer(np.hanning(2 * r_reg + 1), np.hanning(2 * r_reg + 1))
    sl = slice(r + p - r_reg, r + p + r_reg + 1)
    for it in range(iters + 1):
        rec = np.array([desplazar(c, dy + extra[t, 0], dx + extra[t, 1]) for t, (c, dy, dx) in enumerate(crud)])
        if it == iters:
            break
        media = rec.mean(0)
        ref = (media[sl, sl] - np.median(media)) * w
        for t in range(n):
            mov = (rec[t][sl, sl] - np.median(rec[t])) * w
            extra[t] += phase_cross_correlation(ref, mov, upsample_factor=20, normalization=None)[0]
    return rec[:, p:-p, p:-p], extra


def fondo_recorte(img, dir_otro=None, sector_deg=70.0):
    """Mediana del anillo FONDO_REC sin el sector de +-sector_deg hacia el otro punto."""
    n = img.shape[0]
    X, Y = malla(n)
    c = (n - 1) / 2
    X, Y = X - c, Y - c
    rr = np.hypot(X, Y)
    sel = (rr >= FONDO_REC[0]) & (rr <= FONDO_REC[1])
    if dir_otro is not None:
        sel &= np.abs(ang_dir(np.degrees(np.arctan2(Y, X)) - dir_otro)) > sector_deg
    return float(np.median(img[sel]))


def centro_nucleo(img, fondo, r_n):
    """Maximo de la imagen suavizada (sigma 1 px) a <= 6 px del centro del recorte y
    luego centroide iterativo de (img - fondo) en el disco r_n."""
    n = img.shape[0]
    c = (n - 1) / 2
    X, Y = malla(n)
    lisa = ndi.gaussian_filter(img, 1.0)
    lisa[np.hypot(X - c, Y - c) > 6] = -np.inf
    yi, xi = np.unravel_index(np.argmax(lisa), lisa.shape)
    x, y = float(xi), float(yi)
    for _ in range(30):
        w = np.clip(img - fondo, 0, None) * (np.hypot(X - x, Y - y) <= r_n)
        xn, yn = (w * X).sum() / w.sum(), (w * Y).sum() / w.sum()
        if np.hypot(xn - x, yn - y) < 1e-4:
            break
        x, y = xn, yn
    return x, y


def momentos_nucleo(img, centro, r_n):
    """Eje mayor del nucleo por momentos de segundo orden: pesos = img - nivel en el
    disco r_n, con nivel = mediana del anillo (r_n, r_n + 2] (el halo justo fuera
    del nucleo). Devuelve el angulo del eje [deg, mod 180] y sqrt(l1/l2)."""
    x, y = centro
    X, Y = malla(img.shape[0])
    d = np.hypot(X - x, Y - y)
    nivel = np.median(img[(d > r_n) & (d <= r_n + 2)])
    w = np.clip(img - nivel, 0, None) * (d <= r_n)
    s = w.sum()
    dx, dy = X - x, Y - y
    mxx, myy, mxy = (w * dx * dx).sum() / s, (w * dy * dy).sum() / s, (w * dx * dy).sum() / s
    ang = 0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy))
    raiz = np.sqrt(max((mxx - myy) ** 2 / 4 + mxy ** 2, 0))
    l1, l2 = (mxx + myy) / 2 + raiz, (mxx + myy) / 2 - raiz
    return float(ang_eje(ang)), float(np.sqrt(l1 / max(l2, 1e-12)))


def _modelo2g(q, X, Y):
    """Constante + dos gaussianas isotropicas; q = (c, a1, x1, y1, s1, a2, d, ang, s2):
    el lobulo 2 esta a distancia d del 1 en la direccion ang [rad]."""
    c, a1, x1, y1, s1, a2, d, ang, s2 = q
    x2, y2 = x1 + d * np.cos(ang), y1 + d * np.sin(ang)
    return (c + a1 * np.exp(-((X - x1) ** 2 + (Y - y1) ** 2) / (2 * s1 * s1))
            + a2 * np.exp(-((X - x2) ** 2 + (Y - y2) ** 2) / (2 * s2 * s2)))


def _datos_ajuste(img, centro, r, var=None):
    X, Y = malla(img.shape[0])
    sel = np.hypot(X - centro[0], Y - centro[1]) <= r
    w = 1 / np.sqrt(var[sel]) if var is not None else np.ones(sel.sum())
    return X[sel], Y[sel], img[sel], w


def _cotas(centro, r):
    lo = [-np.inf, 0, centro[0] - r, centro[1] - r, 0.5, 0, SEP[0], -np.inf, 0.5]
    hi = [np.inf, np.inf, centro[0] + r, centro[1] + r, 3.5, np.inf, SEP[1], np.inf, 3.5]
    return np.array(lo), np.array(hi)


def _resultado(q, costo):
    q = q.copy()
    f1, f2 = q[1] * q[4] ** 2, q[5] * q[8] ** 2
    if f2 > f1:                                   # lobulo 1 = el de mas flujo
        x2, y2 = q[2] + q[6] * np.cos(q[7]), q[3] + q[6] * np.sin(q[7])
        q = np.array([q[0], q[5], x2, y2, q[8], q[1], q[6], q[7] + np.pi, q[4]])
        f1, f2 = f2, f1
    q[7] = np.radians(ang_dir(np.degrees(q[7])))
    x2, y2 = q[2] + q[6] * np.cos(q[7]), q[3] + q[6] * np.sin(q[7])
    return dict(eje=float(ang_eje(np.degrees(q[7]))), sep=float(q[6]), flujo=float(f2 / f1), q=q,
                lob=(q[2], q[3], x2, y2), costo=float(costo))


def dos_gaussianas(img, centro, iniciales=None, r=R_AJUSTE, var=None):
    """Ajuste por minimos cuadrados de constante + dos gaussianas isotropicas en el
    disco r alrededor del centro del nucleo; la separacion entre lobulos esta acotada
    a SEP px (los lobulos del nucleo, no el anillo). Arranques: lobulo 1 en el maximo y
    el 2 a 3 px en 8 direcciones, mas los iniciales dados. Lobulo 1 = el de mas flujo.
    Devuelve eje [deg, mod 180], separacion [px], cociente de flujos 2/1 y parametros."""
    xs, ys, zs, ws = _datos_ajuste(img, centro, r, var)
    X, Y = malla(img.shape[0])
    lisa = ndi.gaussian_filter(img, 1.0)
    lisa[np.hypot(X - centro[0], Y - centro[1]) > r] = -np.inf
    yi, xi = np.unravel_index(np.argmax(lisa), lisa.shape)
    c0 = np.percentile(zs, 10)
    amp = zs.max() - c0
    arranques = [np.array([c0, amp, xi, yi, 1.2, amp / 2, 3.0, a, 1.2]) for a in np.radians(np.arange(0, 360, 45))]
    if iniciales is not None:
        arranques.insert(0, np.asarray(iniciales, dtype=float))
    lo, hi = _cotas(centro, r)
    mejor = None
    for q0 in arranques:
        q0 = np.clip(q0, lo + 1e-6, hi - 1e-6)
        sol = least_squares(lambda q: (_modelo2g(q, xs, ys) - zs) * ws, q0, bounds=(lo, hi), x_scale="jac")
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol
    return _resultado(mejor.x, mejor.cost)


def perfil_eje(img, var, centro, sol, r=R_AJUSTE, paso=5.0):
    """Perfil de chi2 del eje de los lobulos: el mismo ajuste con la direccion lobulo 1
    -> lobulo 2 fija en a (0..360, paso) y lo demas libre, arrancando de la solucion
    libre sol; chi2 con la varianza por pixel de la imagen media (var). Devuelve los
    ejes (mod 180), chi2(eje) = min(chi2(a), chi2(a + 180)) y el numero de pixeles."""
    xs, ys, zs, ws = _datos_ajuste(img, centro, r, var)
    lo, hi = _cotas(centro, r)
    q0 = sol["q"]
    angs = np.arange(0, 360, paso)
    chi = []
    for a in angs:
        ar = np.radians(a)

        def res(q8):
            return (_modelo2g(np.r_[q8[:7], ar, q8[7]], xs, ys) - zs) * ws
        x2, y2 = sol["lob"][2], sol["lob"][3]
        mejor = None
        for ini in (np.r_[q0[:7], q0[8]], np.array([q0[0], q0[5], x2, y2, q0[8], q0[1], q0[6], q0[4]])):
            ini = np.clip(ini, np.r_[lo[:7], lo[8]] + 1e-6, np.r_[hi[:7], hi[8]] - 1e-6)
            s_ = least_squares(res, ini, bounds=(np.r_[lo[:7], lo[8]], np.r_[hi[:7], hi[8]]), x_scale="jac")
            if mejor is None or s_.cost < mejor.cost:
                mejor = s_
        chi.append(2 * mejor.cost)
    chi = np.array(chi)
    n = len(angs) // 2
    ejes = ang_eje(angs[:n])
    o = np.argsort(ejes)
    return ejes[o], np.minimum(chi[:n], chi[n:])[o], len(zs)


def chi2_en(ejes, chi, a):
    """chi2 interpolado (circular, mod 180) en el eje a."""
    e = np.r_[ejes - 180, ejes, ejes + 180]
    return float(np.interp(ang_eje(a), e, np.r_[chi, chi, chi]))


def perfil_angular(img, centro, fondo, corona, n_bins=N_BINS):
    """Intensidad media (menos el fondo) por sector angular en la corona dada; angulos
    de +x hacia +y, centros de bin."""
    X, Y = malla(img.shape[0])
    dx, dy = X - centro[0], Y - centro[1]
    rr = np.hypot(dx, dy)
    th = np.degrees(np.arctan2(dy, dx)) % 360
    sel = (rr >= corona[0]) & (rr < corona[1])
    b = (th[sel] / (360 / n_bins)).astype(int)
    p = np.bincount(b, weights=img[sel] - fondo, minlength=n_bins) / np.bincount(b, minlength=n_bins)
    return (np.arange(n_bins) + 0.5) * 360 / n_bins, p


def armonico(th, p, k):
    """Modulacion relativa 2|c_k| y direccion del maximo (mod 360/k) del armonico k."""
    c = (p * np.exp(-1j * k * np.radians(th))).sum() / p.sum()
    return float(2 * abs(c)), float(envolver(-np.degrees(np.angle(c)) / k, 360.0 / k))


def mejor_rotacion(p_a, p_b):
    """Rotacion (grados, multiplo del bin) que lleva el perfil p_a sobre p_b (corr. maxima)."""
    a = (p_a - p_a.mean()) / p_a.std()
    b = (p_b - p_b.mean()) / p_b.std()
    cc = np.array([np.mean(np.roll(a, k) * b) for k in range(len(a))])
    return float(ang_dir(np.argmax(cc) * 360 / len(a))), cc


# ---------------------------------------------------------------- medidas de una imagen

def medir(img, dir_otro, r_n, iniciales=None, con_perfiles=True):
    f = fondo_recorte(img, dir_otro)
    cen = centro_nucleo(img, f, r_n)
    ang, ejes = momentos_nucleo(img, cen, r_n)
    g2 = dos_gaussianas(img, cen, iniciales)
    out = dict(fondo=f, centro=cen, mom=ang, ejes=ejes, g2=g2)
    if con_perfiles:
        th, p = perfil_angular(img, cen, f, ANILLO)
        out.update(th=th, perfil=p, a1=armonico(th, p, 1), a2=armonico(th, p, 2))
        thc, pc = perfil_angular(img, cen, f, CRUZ)
        out.update(th_c=thc, perfil_c=pc, a4=armonico(thc, pc, 4), a2c=armonico(thc, pc, 2))
    return out


def armonico4_nativo(img, dir_otro, r_n):
    f = fondo_recorte(img, dir_otro)
    cen = centro_nucleo(img, f, r_n)
    th, p = perfil_angular(img, cen, f, CRUZ)
    return armonico(th, p, 4)


# ---------------------------------------------------------------- 4. fotometria

def disco(img, c, r, otro):
    """Suma de (img - fondo) en el disco r [px] de c; fondo = mediana del anillo
    [r + 3, r + 9] sin los pixeles a menos de r + 3 del otro punto."""
    H, W = img.shape
    m = int(r + 10)
    x0, y0 = int(round(c[0])), int(round(c[1]))
    ys, xs = slice(max(y0 - m, 0), min(y0 + m + 1, H)), slice(max(x0 - m, 0), min(x0 + m + 1, W))
    sub = img[ys, xs]
    Y, X = np.mgrid[ys, xs]
    d = np.hypot(X - c[0], Y - c[1])
    fondo = np.median(sub[(d >= r + 3) & (d <= r + 9) & (np.hypot(X - otro[0], Y - otro[1]) > r + 3)])
    return float((sub[d <= r] - fondo).sum()), float(fondo)


def media_recortada(v):
    lo, hi = np.percentile(v, (1, 99))
    return float(v[(v >= lo) & (v <= hi)].mean())


def energia_encerrada(radios_px):
    """Fraccion de la energia de la PSF esferica simulada (simulador_nuevo.psf_directa,
    sin recorte de pupila) dentro de cada radio, en las z de Z_SECCION4."""
    sys.path.insert(0, os.path.join(os.path.dirname(AQUI), "simulador_nuevo"))
    import simulador as sim
    out = {}
    for var, (an, z_real, z_refl) in Z_SECCION4.items():
        cfg = sim.Config(mascara="esferica", anillos_hacia_objetivo=an)
        for pto, z in (("real", z_real), ("reflejo", z_refl)):
            psf = sim.psf_directa(cfg, z)
            n = psf.shape[0]
            X, Y = malla(n)
            d = np.hypot(X - n // 2, Y - n // 2)
            out[var, pto] = np.array([psf[d <= r].sum() / psf.sum() for r in radios_px])
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tif", default=adr.TIF_DEF)
    ap.add_argument("--cuadros", type=int, default=30, help="cuadros promediados (0..N-1)")
    ap.add_argument("--r-nucleo", type=float, default=4.0, help="radio del nucleo para los momentos [px]")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--salida", default=AQUI)
    args = ap.parse_args()
    N, r_n = args.cuadros, args.r_nucleo
    rng = np.random.default_rng(0)

    video, info = adr.cargar(args.tif)
    T, H, W = video.shape
    print(f"video {video.shape}, metadatos {info}")
    cort = adr.buscar_corte(video.mean(0))
    corte = cort["corte"]
    can = [video[:, :, :corte].astype(np.float32), video[:, :, corte:].astype(np.float32)]
    aps = [cort["ap_izq"], cort["ap_der"]]
    del video

    # ------------------------------------------------------------ 1. pares y transformacion
    pares = []
    for k in range(2):
        dets = [adr.detectar(can[k][t], aps[k]) for t in range(T)]
        pk = adr.pares_real_reflejo(adr.enlazar(dets, T), T)
        if len(pk) != 1:
            raise SystemExit(f"se esperaba un par real/reflejo en el canal {adr.NOMBRES[k]}; hay {len(pk)}")
        p = pk[0]
        pares.append(p)
        print(f"canal {adr.NOMBRES[k]}: real en ({np.nanmean(p['real'][:, 0]):.1f}, {np.nanmean(p['real'][:, 1]):.1f}), "
              f"reflejo en ({np.nanmean(p['refl'][:, 0]):.1f}, {np.nanmean(p['refl'][:, 1]):.1f}) px locales; cuadros con "
              f"los dos {np.isfinite(p['real'][:, 0] + p['refl'][:, 0]).sum()}")
    pL, pR = pares
    ok = np.isfinite(pL["real"][:, 0] + pL["refl"][:, 0] + pR["real"][:, 0] + pR["refl"][:, 0])
    n = int(ok.sum())
    A = np.r_[pL["real"][ok], pL["refl"][ok]]
    B = np.r_[pR["real"][ok], pR["refl"][ok]]
    tr = adr.similitud(A, B, True)
    Q0 = np.array([[0.0, -1.0], [-1.0, 0.0]])   # antidiagonal pura: x_der = -y_izq + tx, y_der = -x_izq + ty
    trans = {"similitud impropia (todos)": tr,
             "similitud impropia (solo real)": adr.similitud(A[:n], B[:n], True),
             "similitud impropia (solo reflejo)": adr.similitud(A[n:], B[n:], True),
             "antidiagonal pura + traslacion": dict(M=Q0, t=(B - A @ Q0.T).mean(0), Q=Q0, s=1.0)}
    print(f"\n[1] transformacion izq -> der, {n} cuadros x 2 puntos (real, reflejo)")
    print(f"    similitud impropia: s = {tr['s']:.4f}, angulo de Q = {adr.angulo_Q(tr['Q']):+.2f} deg (espejado respecto "
          f"de la recta a {adr.angulo_Q(tr['Q']) / 2:+.2f}; la antidiagonal es -45), t = ({tr['t'][0]:.1f}, {tr['t'][1]:.1f}) px")
    val = {}
    for nombre, tv in trans.items():
        res = (B - adr.aplicar(tv, A)) * PX
        val[nombre] = (adr.rms2d(res[:n]), adr.rms2d(res[n:]), adr.angulo_Q(tv["Q"]), tv["s"])
        print(f"    {nombre:34s}: RMS real {val[nombre][0]:4.0f} nm, reflejo {val[nombre][1]:4.0f} nm; angulo Q "
              f"{val[nombre][2]:+.1f} deg, s {val[nombre][3]:.3f}")
    LrT, LfT = adr.aplicar(tr, pL["real"][ok]), adr.aplicar(tr, pL["refl"][ok])
    cruces = {(a, b): float(np.median(np.hypot(*(u - v).T)) * PX)
              for a, u in (("real", LrT), ("reflejo", LfT)) for b, v in (("real", pR["real"][ok]), ("reflejo", pR["refl"][ok]))}
    print("    izq transformado -> der (distancia mediana): " + ", ".join(f"{a} -> {b} {v:.0f} nm" for (a, b), v in cruces.items()))
    segL = np.nanmean(pL["real"] - pL["refl"], 0)
    segR = np.nanmean(pR["real"] - pR["refl"], 0)
    phiR, phiL_T = ang_vec(segR), ang_vec(tr["Q"] @ segL)
    print(f"    direccion reflejo -> real (m): izq {ang_vec(segL):+.1f} deg local, transformada {phiL_T:+.1f}; der {phiR:+.1f} "
          f"(diferencia {ang_dir(phiL_T - phiR):+.1f} deg); |segmento| izq x s {np.hypot(*segL) * tr['s'] * PX:.0f} nm, "
          f"der {np.hypot(*segR) * PX:.0f} nm")
    m_dir = vec(phiR)
    t_dir = vec(phiR + 90)
    dL, dR = np.diff(LrT, axis=0), np.diff(pR["real"][ok], axis=0)
    corr_pasos = {e: float(np.corrcoef(dL @ v, dR @ v)[0, 1]) for e, v in (("m", m_dir), ("t", t_dir))}
    print(f"    pasos cuadro a cuadro del real izq transformado vs der: corr a lo largo de m {corr_pasos['m']:+.2f}, "
          f"de t {corr_pasos['t']:+.2f} (con el signo cambiado: {-corr_pasos['m']:+.2f}, {-corr_pasos['t']:+.2f})")

    # ------------------------------------------------------------ 2. imagenes medias
    print(f"\n[2] imagenes medias alineadas, cuadros 0..{N - 1}; radio del nucleo (momentos) {r_n:g} px")
    rec = {}
    for k, nom in enumerate(adr.NOMBRES):
        p = pares[k]
        for pto, pos in zip(PUNTOS, (p["real"], p["refl"])):
            if not np.isfinite(pos[:N]).all():
                raise SystemExit(f"faltan posiciones en los cuadros 0..{N - 1} ({nom}, {pto})")
            rec[nom, pto], extra = recortes_alineados(can[k], pos, N)
            e = np.hypot(*extra.T)
            print(f"    {nom} {pto}: refinamiento por correlacion respecto del centroide: mediana {np.median(e):.2f} px, "
                  f"max {e.max():.2f} px")
    Q = tr["Q"]
    # direccion hacia el otro punto, en el marco nativo de cada canal
    dir_nat = {("der", "real"): ang_vec(-segR), ("der", "reflejo"): ang_vec(segR),
               ("izq", "real"): ang_vec(-segL), ("izq", "reflejo"): ang_vec(segL)}

    def dir_der(nom, pto, Qm):
        return ang_vec(Qm @ vec(dir_nat[nom, pto])) if nom == "izq" else dir_nat[nom, pto]

    def varianzas():
        v = {}
        for (nom, pto), r in rec.items():
            vn = r.var(0, ddof=1) / len(r)
            v[nom, pto] = np.clip(a_marco_der(vn, Q), vn.min(), None) if nom == "izq" else vn
        return v

    def imagenes(idx, Qm=Q):
        im, nat = {}, {}
        for (nom, pto), r in rec.items():
            nat[nom, pto] = r[idx].mean(0)
            im[nom, pto] = a_marco_der(nat[nom, pto], Qm) if nom == "izq" else nat[nom, pto]
        return im, nat

    todos = np.arange(N)
    im, nat = imagenes(todos)
    me = {k: medir(v, dir_der(*k, Q), r_n) for k, v in im.items()}
    a4n = {k: armonico4_nativo(v, dir_nat[k], r_n) for k, v in nat.items()}
    m_nat = {"izq": ang_vec(segL), "der": phiR}
    print(f"    marco del canal derecho: m = {phiR:+.1f} deg (reflejo -> real), t = {ang_eje(phiR + 90):+.1f}; "
          f"ejes de la camara izq llevados a este marco: {mapear(0.0, Q):+.1f} y {mapear(90.0, Q):+.1f}")
    for nom in adr.NOMBRES:
        for pto in PUNTOS:
            r = me[nom, pto]
            g = r["g2"]
            print(f"    {nom:3s} {pto:7s}: dos gaussianas: eje {g['eje']:+6.1f} deg (vs m {ang_eje(g['eje'] - phiR):+6.1f}), "
                  f"separacion {g['sep']:.2f} px ({g['sep'] * PX:.0f} nm), flujo 2/1 {g['flujo']:.2f}, sigmas "
                  f"{g['q'][4]:.2f}/{g['q'][8]:.2f} px | momentos (r {r_n:g}): eje {r['mom']:+6.1f} (vs m "
                  f"{ang_eje(r['mom'] - phiR):+6.1f}), cociente de ejes {r['ejes']:.2f}")
            print(f"                 nucleo {CRUZ[0]:g}-{CRUZ[1]:g} px: armonico 2 = {r['a2c'][0]:.2f} (eje {r['a2c'][1]:+6.1f}), "
                  f"armonico 4 = {r['a4'][0]:.2f} (brazos a {r['a4'][1]:+5.1f} mod 90); en el marco nativo {a4n[nom, pto][0]:.2f}, "
                  f"brazos a {a4n[nom, pto][1]:+5.1f} mod 90 (camara 0, m nativo {envolver(m_nat[nom], 90):+.1f})")
            print(f"                 anillo {ANILLO[0]:g}-{ANILLO[1]:g} px: armonico 1 = {r['a1'][0]:.2f} hacia {r['a1'][1]:+6.1f} "
                  f"(vs m {ang_dir(r['a1'][1] - phiR):+6.1f}), armonico 2 = {r['a2'][0]:.2f} con eje {r['a2'][1]:+6.1f}; "
                  f"fondo del recorte {r['fondo']:.1f} ADU")
    print("    sensibilidad: eje por momentos segun el radio del nucleo, y dos gaussianas segun el radio del ajuste")
    for rr in (3.0, 4.0, 5.0, 6.0):
        f = [momentos_nucleo(im[k], me[k]["centro"], rr)[0] for k in ((n_, p_) for n_ in adr.NOMBRES for p_ in PUNTOS)]
        print(f"      momentos r = {rr:g}: izq real {f[0]:+6.1f}, izq reflejo {f[1]:+6.1f}, der real {f[2]:+6.1f}, der reflejo "
              f"{f[3]:+6.1f}; izq - der: real {ang_eje(f[0] - f[2]):+6.1f}, reflejo {ang_eje(f[1] - f[3]):+6.1f}")
    for rr in (5.0, 6.0, 7.0, 9.0):
        f = [dos_gaussianas(im[k], me[k]["centro"], r=rr)["eje"] for k in ((n_, p_) for n_ in adr.NOMBRES for p_ in PUNTOS)]
        print(f"      dos gaussianas r = {rr:g}: izq real {f[0]:+6.1f}, izq reflejo {f[1]:+6.1f}, der real {f[2]:+6.1f}, "
              f"der reflejo {f[3]:+6.1f}; izq - der: real {ang_eje(f[0] - f[2]):+6.1f}, reflejo {ang_eje(f[1] - f[3]):+6.1f}")
    print("    sistematico del registro (izq - der, dos gaussianas / momentos):")
    sist = {}
    for nombre, tv in trans.items():
        im2, _ = imagenes(todos, tv["Q"])
        m2 = {k: medir(v, dir_der(*k, tv["Q"]), r_n, con_perfiles=False) for k, v in im2.items()}
        sist[nombre] = {p_: (ang_eje(m2["izq", p_]["g2"]["eje"] - m2["der", p_]["g2"]["eje"]),
                             ang_eje(m2["izq", p_]["mom"] - m2["der", p_]["mom"])) for p_ in PUNTOS}
        print(f"      {nombre:34s}: real {sist[nombre]['real'][0]:+6.1f} / {sist[nombre]['real'][1]:+6.1f}, "
              f"reflejo {sist[nombre]['reflejo'][0]:+6.1f} / {sist[nombre]['reflejo'][1]:+6.1f} deg")
    # perfil de chi2 del eje: compara "mismo eje que el otro canal" (camino comun) con "rotado 90" (polarizacion).
    # El modelo (dos gaussianas) no describe todo el nucleo (chi2/nu > 1): delta chi2 se escala por chi2_min/nu.
    var = varianzas()
    perfiles, hip = {}, {}
    print("    perfil de chi2 del eje (dos gaussianas con el eje fijo; varianza por pixel de la imagen media; "
          "delta chi2 escalado por chi2_min/nu):")
    for k in ((n_, p_) for n_ in adr.NOMBRES for p_ in PUNTOS):
        sol = dos_gaussianas(im[k], me[k]["centro"], iniciales=me[k]["g2"]["q"], var=var[k])
        ejes, chi, npx = perfil_eje(im[k], var[k], me[k]["centro"], sol)
        nu = npx - 9
        esc = chi.min() / nu
        perfiles[k] = (ejes, (chi - chi.min()) / esc)
        hip[k] = dict(min=float(ejes[np.argmin(chi)]), chi2nu=float(esc), libre=sol["eje"])
    for p_ in PUNTOS:
        for nom, otro in (("izq", "der"), ("der", "izq")):
            ejes, dchi = perfiles[nom, p_]
            a_otro = hip[otro, p_]["min"]
            d0, d90 = chi2_en(ejes, dchi, a_otro), chi2_en(ejes, dchi, a_otro + 90)
            hip[nom, p_].update(d0=d0, d90=d90, a_otro=a_otro)
            ic1 = ejes[dchi <= 1.0]
            ic4 = ejes[dchi <= 4.0]
            print(f"      {nom} {p_:7s}: minimo en {hip[nom, p_]['min']:+5.0f} deg (libre ponderado {hip[nom, p_]['libre']:+.0f}; "
                  f"chi2/nu {hip[nom, p_]['chi2nu']:.1f}); delta <= 1 en {len(ic1)} de {len(ejes)} ejes, <= 4 en {len(ic4)}; "
                  f"en el eje del {otro} ({a_otro:+.0f}): {d0:.1f}; en el del {otro} + 90 ({ang_eje(a_otro + 90):+.0f}): {d90:.1f}; "
                  f"rango {dchi.max():.1f}")
    rot = {p_: mejor_rotacion(me["izq", p_]["perfil"], me["der", p_]["perfil"]) for p_ in PUNTOS}
    for p_ in PUNTOS:
        al, cc = rot[p_]
        print(f"    {p_}: rotacion del perfil del anillo izq -> der con correlacion maxima {al:+.0f} deg (corr {cc.max():.2f}; "
              f"sin rotar {cc[0]:.2f}, +90 {cc[9]:.2f}, -90 {cc[27]:.2f})")
    print(f"    eje real - eje reflejo (dos gaussianas): izq {ang_eje(me['izq', 'real']['g2']['eje'] - me['izq', 'reflejo']['g2']['eje']):+.1f}, "
          f"der {ang_eje(me['der', 'real']['g2']['eje'] - me['der', 'reflejo']['g2']['eje']):+.1f} deg")

    # ------------------------------------------------------------ 3. bootstrap
    B_ = args.n_boot
    print(f"\n[3] bootstrap de cuadros ({B_} remuestreos de {N} cuadros, mismos indices en los dos canales)")
    t0 = time.time()
    claves = [(n_, p_) for n_ in adr.NOMBRES for p_ in PUNTOS]
    bs = {c: [] for c in ("g2", "mom", "a4n", "a1", "rot")}
    for b in range(B_):
        idx = rng.integers(0, N, N)
        imb, natb = imagenes(idx)
        mb = {k: medir(imb[k], dir_der(*k, Q), r_n, iniciales=me[k]["g2"]["q"]) for k in claves}
        bs["g2"].append([mb[k]["g2"]["eje"] for k in claves])
        bs["mom"].append([mb[k]["mom"] for k in claves])
        bs["a4n"].append([armonico4_nativo(natb[k], dir_nat[k], r_n)[1] for k in claves])
        bs["a1"].append([mb[k]["a1"][1] for k in claves])
        bs["rot"].append([mejor_rotacion(mb["izq", p_]["perfil"], mb["der", p_]["perfil"])[0] for p_ in PUNTOS])
    bs = {c: np.array(v) for c, v in bs.items()}
    print(f"    ({time.time() - t0:.0f} s)")

    def ic(v, c0, mod):
        d = envolver(np.asarray(v) - c0, mod)
        return (float(c0), float(c0 + np.percentile(d, 2.5)), float(c0 + np.percentile(d, 97.5)), float(d.std()))

    ics = {}
    for est, nombre in (("g2", "dos gaussianas"), ("mom", "momentos")):
        for i, k in enumerate(claves):
            c0 = me[k]["g2"]["eje"] if est == "g2" else me[k]["mom"]
            ics[est, k] = ic(bs[est][:, i], c0, 180)
        for j, p_ in enumerate(PUNTOS):
            iL, iR = claves.index(("izq", p_)), claves.index(("der", p_))
            dif = ang_eje(bs[est][:, iL] - bs[est][:, iR])
            c0 = ang_eje(ics[est, ("izq", p_)][0] - ics[est, ("der", p_)][0])
            ics[est, "dif", p_] = ic(dif, c0, 180)
            ics[est, "dif", p_] += (float(np.mean(np.abs(dif) > 45)),)
        for i, k in enumerate(claves):
            v = ics[est, k]
            print(f"    {nombre:15s} {k[0]} {k[1]:7s}: {v[0]:+6.1f} deg, IC95 [{v[1]:+6.1f}, {v[2]:+6.1f}], std {v[3]:.1f}")
        for p_ in PUNTOS:
            v = ics[est, "dif", p_]
            print(f"    {nombre:15s} izq - der, {p_:7s}: {v[0]:+6.1f} deg, IC95 [{v[1]:+6.1f}, {v[2]:+6.1f}], std {v[3]:.1f}; "
                  f"fraccion de remuestreos con |dif| > 45: {v[4]:.3f}")
    for i, k in enumerate(claves):
        v = ic(bs["a4n"][:, i], a4n[k][1], 90)
        ics["a4n", k] = v
        v1 = ic(bs["a1"][:, i], me[k]["a1"][1], 360)
        ics["a1", k] = v1
        print(f"    {k[0]} {k[1]:7s}: brazos de la cruz (nativo, mod 90) {v[0]:+5.1f} [{v[1]:+5.1f}, {v[2]:+5.1f}]; "
              f"armonico 1 del anillo hacia {v1[0]:+6.1f} [{v1[1]:+6.1f}, {v1[2]:+6.1f}]")
    for j, p_ in enumerate(PUNTOS):
        v = ic(bs["rot"][:, j], rot[p_][0], 360)
        ics["rot", p_] = v
        print(f"    rotacion del anillo izq -> der, {p_}: {v[0]:+.0f} deg [{v[1]:+.0f}, {v[2]:+.0f}]")

    # ------------------------------------------------------------ 4. fotones y fondo
    print(f"\n[4] fotometria en e- ({G} e-/ADU)")
    union = np.zeros((H, W), bool)
    union[:, :corte] = cort["ap_izq"]
    union[:, corte:] = cort["ap_der"]
    fuera_m = ndi.binary_erosion(~ndi.binary_dilation(union, iterations=15), iterations=5)
    fuera_m[:, corte - 20:corte + 20] = False
    fot, fondo_info = {}, {}
    radios = (1.5, 2.2)
    for k, nom in enumerate(adr.NOMBRES):
        vid, p = can[k], pares[k]
        fuera_k = fuera_m[:, :corte] if k == 0 else fuera_m[:, corte:]
        dentro_k = ndi.binary_erosion(aps[k], iterations=10)
        X, Y = np.meshgrid(np.arange(vid.shape[2]), np.arange(vid.shape[1]))
        for pos in (p["real"], p["refl"]):
            c = np.nanmean(pos, 0)
            dentro_k &= np.hypot(X - c[0], Y - c[1]) > 40
        fuera_t = np.array([media_recortada(vid[t][fuera_k]) for t in range(T)])
        dentro_t = np.array([media_recortada(vid[t][dentro_k]) for t in range(T)])
        dif = np.diff(vid, axis=0)
        var_f = np.var(dif[:, fuera_k], axis=0) / 2
        var_d = np.var(dif[:, dentro_k], axis=0) / 2
        g_est = (dentro_t.mean() - fuera_t.mean()) / (np.median(var_d) - np.median(var_f))
        fi = dict(fuera=fuera_t, dentro=dentro_t, rn_med=float(np.sqrt(np.median(var_f))),
                  rn_rms=float(np.sqrt(np.mean(var_f))), g=float(g_est), n_fuera=int(fuera_k.sum()),
                  n_dentro=int(dentro_k.sum()))
        fondo_info[nom] = fi
        fot[nom] = {}
        for pto, pos, otro in (("real", p["real"], p["refl"]), ("reflejo", p["refl"], p["real"])):
            arr = np.full((T, len(radios), 2), np.nan)
            for t in range(T):
                if np.isfinite(pos[t, 0] + otro[t, 0]):
                    for i, rum in enumerate(radios):
                        arr[t, i] = disco(vid[t], pos[t], rum * 1000 / PX, otro[t])
            fot[nom][pto] = arr
        b_e = (dentro_t - fuera_t) * G
        print(f"  canal {nom}: nivel oscuro (fuera de las aperturas, {fi['n_fuera']} px) {fuera_t.mean():.2f} ADU (std entre "
              f"cuadros {fuera_t.std():.3f}); dentro, a > 40 px de las particulas ({fi['n_dentro']} px) {dentro_t.mean():.2f} ADU "
              f"-> fondo {b_e.mean():.2f} e-/px (cuadros 0-{N - 1} {b_e[:N].mean():.2f}; 0-4 {b_e[:5].mean():.2f}, 95-99 {b_e[-5:].mean():.2f})")
        print(f"    ruido de lectura (fuera, diferencias de cuadros / sqrt 2): mediana {fi['rn_med']:.2f} ADU, rms {fi['rn_rms']:.2f} ADU "
              f"= {fi['rn_med'] * G:.2f} / {fi['rn_rms'] * G:.2f} e- (simulador 3.26 ADU = 1.50 e-); ganancia por transferencia "
              f"de fotones (exceso de varianza dentro vs fuera) {g_est:.2f} e-/ADU (supuesta {G})")
        for pto in PUNTOS:
            arr = fot[nom][pto]
            print(f"    {pto:7s}: fondo local (anillo del disco de 1.5 um, 18-24 px) {(np.nanmean(arr[:N, 0, 1]) - fuera_t[:N].mean()) * G:.2f} "
                  f"e-/px; fondo del recorte medio ({FONDO_REC[0]:g}-{FONDO_REC[1]:g} px) {(me[nom, pto]['fondo'] - fuera_t[:N].mean()) * G:.2f} e-/px")
            for i, rum in enumerate(radios):
                e = arr[:, i, 0] * G
                print(f"      disco {rum} um: cuadros 0-{N - 1} {np.nanmean(e[:N]):.0f} +- {np.nanstd(e[:N]):.0f} e-, 0-4 "
                      f"{np.nanmean(e[:5]):.0f}, 95-99 {np.nanmean(e[-5:]):.0f} (x{np.nanmean(e[:5]) / np.nanmean(e[-5:]):.1f})")
    rad_cg = np.arange(2, R_CG + 1)
    cg = {}
    for k, m in im.items():
        X, Y = malla(m.shape[0])
        d = np.hypot(X - me[k]["centro"][0], Y - me[k]["centro"][1])
        cg[k] = np.array([((m - me[k]["fondo"])[d <= r]).sum() * G for r in rad_cg])
    ee = energia_encerrada(list(rad_cg))
    print(f"    curva de crecimiento de la imagen media (fondo {FONDO_REC[0]:g}-{FONDO_REC[1]:g} px) y fotones totales = "
          f"e-(r = {R_CG} px) / energia encerrada simulada a {R_CG} px en las z de la seccion 4:")
    totales = {}
    for k in claves:
        v = cg[k]
        tot = {var: v[-1] / ee[var, k[1]][-1] for var in Z_SECCION4}
        totales[k] = tot
        print(f"      {k[0]} {k[1]:7s}: r = 10/15/20/25 px: " + "/".join(f"{v[rad_cg == r][0]:.0f}" for r in (10, 15, 20, 25))
              + " e-; " + ", ".join(f"{var} (z {Z_SECCION4[var][1 if k[1] == 'real' else 2]:+.0f}, EE {ee[var, k[1]][-1]:.2f}) "
                                     f"-> {tot[var]:.0f}" for var in Z_SECCION4))
    for nom in adr.NOMBRES:
        print(f"    {nom}: reflejo / real (r = {R_CG} px, imagen media) {cg[nom, 'reflejo'][-1] / cg[nom, 'real'][-1]:.2f}")
    print(f"    izq / der (r = {R_CG} px): real {cg['izq', 'real'][-1] / cg['der', 'real'][-1]:.2f}, reflejo "
          f"{cg['izq', 'reflejo'][-1] / cg['der', 'reflejo'][-1]:.2f}")
    print("    celula.yaml: fotones_rango [1000, 30000] (log-uniforme, por emisor, el real; reflejo = ratio x fotones, "
          "ratio en [0.45, 1]), fondo_rango [5, 40] fotones/px, read_noise 3.26 ADU, offset 99 ADU, gain 0.46 e-/ADU")

    figura_imagenes(im, me, phiR, os.path.join(args.salida, "fig_psf_real_imagenes.png"))
    figura_medidas(pL, pR, tr, ok, me, bs, claves, ics, fot, fondo_info, cg, rad_cg, ee, N, perfiles,
                   os.path.join(args.salida, "fig_psf_real_medidas.png"))


def mapear(a_deg, Q):
    return float(ang_eje(ang_vec(Q @ vec(a_deg))))


# ---------------------------------------------------------------- figuras

def _limpio(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=COLOR["rejilla"], lw=0.6)
    ax.set_axisbelow(True)


def figura_imagenes(im, me, phiR, ruta):
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.8))
    z = 9
    mv = vec(phiR)
    for i, pto in enumerate(PUNTOS):
        for j, nom in enumerate(adr.NOMBRES):
            m, r = im[nom, pto], me[nom, pto]
            c, g = r["centro"], r["g2"]
            ax = axs[i, j]
            ax.imshow(m - r["fondo"], cmap="magma", vmin=0)
            for rr in ANILLO:
                ax.add_patch(plt.Circle(c, rr, fill=False, color="w", lw=0.5, ls=":"))
            ax.annotate("", xy=(10 + 8 * mv[0], 52 + 8 * mv[1]), xytext=(10, 52),
                        arrowprops=dict(arrowstyle="->", color="w", lw=1))
            ax.text(3, 47, "m", color="w", fontsize=8)
            ax.set_title(f"{pto}, {'izquierdo (al marco del der)' if nom == 'izq' else 'derecho'}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax2 = axs[i, 2 + j]
            ax2.imshow(m - r["fondo"], cmap="magma", vmin=0)
            ax2.contour(ndi.gaussian_filter(m - r["fondo"], 0.6), 6, colors="w", linewidths=0.35, alpha=0.6)
            ax2.set_xlim(c[0] - z, c[0] + z); ax2.set_ylim(c[1] + z, c[1] - z)
            x1, y1, x2, y2 = g["lob"]
            ax2.plot([x1, x2], [y1, y2], "-", color="#1baf7a", lw=1.2)
            ax2.plot([x1], [y1], "o", ms=6, mfc="none", mec="#1baf7a", mew=1.4)
            ax2.plot([x2], [y2], "o", ms=4, mfc="none", mec="#1baf7a", mew=1.2)
            u = vec(r["mom"]) * 6
            ax2.plot([c[0] - u[0], c[0] + u[0]], [c[1] - u[1], c[1] + u[1]], "--", color="w", lw=0.9)
            ax2.set_title(f"nucleo {nom}: 2 gaus. {g['eje']:+.0f}, momentos {r['mom']:+.0f} deg", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
        ax3 = axs[i, 4]
        for nom in adr.NOMBRES:
            r = me[nom, pto]
            th, p = r["th"], r["perfil"]
            ax3.plot(np.r_[th, th[0] + 360], np.r_[p, p[0]] / p.mean(), "-", color=COLOR[nom], lw=1.6,
                     label=f"{nom} (armonico 1 hacia {r['a1'][1]:+.0f})")
        ax3.axvline(phiR % 360, color=COLOR["gris"], lw=0.8, ls=":")
        ax3.text(phiR % 360, 0.03, " m", transform=ax3.get_xaxis_transform(), fontsize=8, color=COLOR["gris"])
        ax3.set_xlim(0, 360); ax3.set_xticks(range(0, 361, 90))
        ax3.set_xlabel("angulo en el marco del der [deg]")
        ax3.set_ylabel(f"anillo {ANILLO[0]:g}-{ANILLO[1]:g} px / media")
        ax3.set_title(f"perfil angular del anillo, {pto}", fontsize=9)
        ax3.legend(fontsize=7, frameon=False)
        _limpio(ax3)
    fig.suptitle("Yeast6_1: PSF media alineada (cuadros 0-29) en el marco del canal derecho. Nucleo: verde = dos gaussianas "
                 "(circulo grande = lobulo de mas flujo), blanco discontinuo = momentos", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)
    print(f"figura: {ruta}")


def figura_medidas(pL, pR, tr, ok, me, bs, claves, ics, fot, fondo_info, cg, rad_cg, ee, N, perfiles, ruta):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    # (a) validacion de la transformacion
    ax = axs[0, 0]
    for pto, pl, pr, col in (("real", pL["real"], pR["real"], COLOR["der"]), ("reflejo", pL["refl"], pR["refl"], COLOR["izq"])):
        res = (adr.aplicar(tr, pl[ok]) - pr[ok]) * PX
        ax.plot(res[:, 0], res[:, 1], "o", ms=3, color=col, alpha=0.8,
                label=f"{pto}: RMS {adr.rms2d(res):.0f} nm")
    ax.axhline(0, color=COLOR["gris"], lw=0.6); ax.axvline(0, color=COLOR["gris"], lw=0.6)
    ax.set_aspect("equal")
    lim = 650
    ax.set_xlim(-lim, lim); ax.set_ylim(lim, -lim)
    ax.set_xlabel("izq transformado - der, x [nm]"); ax.set_ylabel("y [nm]")
    ax.set_title("(a) residuos de la transformacion izq -> der por cuadro\n(similitud impropia; el real va al real y el reflejo al reflejo)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _limpio(ax)
    # (b) bootstrap de la diferencia de ejes
    ax = axs[0, 1]
    bins = np.arange(-60, 121, 3)
    for p_, col in (("real", COLOR["der"]), ("reflejo", COLOR["izq"])):
        iL, iR = claves.index(("izq", p_)), claves.index(("der", p_))
        for est, ls, nombre in (("g2", "-", "2 gaus."), ("mom", ":", "momentos")):
            v = ics[est, "dif", p_]
            ax.hist(envolver(bs[est][:, iL] - bs[est][:, iR] - 30, 180) + 30, bins=bins, histtype="step", lw=1.5, ls=ls, color=col,
                    label=f"{p_}, {nombre}: {v[0]:+.0f} [{v[1]:+.0f}, {v[2]:+.0f}]")
    for v in (0, 90):
        ax.axvline(v, color=COLOR["gris"], lw=0.8, ls="--")
    ax.set_xlim(-60, 120); ax.set_xticks(range(-60, 121, 30))
    ax.set_xlabel("eje izq - eje der en el marco comun [deg, mod 180]\n(0: camino comun; 90: polarizacion)")
    ax.set_ylabel("remuestreos")
    ax.set_title("(b) eje de los lobulos del nucleo: bootstrap de cuadros", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    _limpio(ax)
    # (c) perfil angular del nucleo (la cruz)
    ax = axs[0, 2]
    for nom in adr.NOMBRES:
        for p_, ls in (("real", "-"), ("reflejo", "--")):
            ejes, chi = perfiles[nom, p_]
            ax.plot(ejes, chi - chi.min(), ls, color=COLOR[nom], lw=1.4, label=f"{nom}, {p_}")
            ax.plot([me[nom, p_]["g2"]["eje"]], [0], "v", color=COLOR[nom], ms=6, mfc="none" if p_ == "reflejo" else COLOR[nom])
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlim(-90, 90); ax.set_xticks(range(-90, 91, 30))
    ax.set_xlabel("eje de los lobulos en el marco del der [deg]")
    ax.set_ylabel("chi2 - minimo (dos gaussianas con el eje fijo)")
    ax.set_title("(c) perfil de chi2 del eje (triangulos: ajuste libre)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _limpio(ax)
    # (d) fotones por cuadro
    ax = axs[1, 0]
    for nom in adr.NOMBRES:
        for p_, ls in (("real", "-"), ("reflejo", "--")):
            ax.plot(fot[nom][p_][:, 0, 0] * G, ls, color=COLOR[nom], lw=1.2, label=f"{nom}, {p_}")
    ax.axvspan(0, N - 1, color=COLOR["rejilla"], alpha=0.6, lw=0)
    ax.axhspan(1000, 30000, color="#1baf7a", alpha=0.07, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("cuadro"); ax.set_ylabel("e- en el disco de 1.5 um (fondo local restado)")
    ax.set_title("(d) fotones por cuadro (franja verde: fotones_rango de celula.yaml)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _limpio(ax)
    # (e) curva de crecimiento vs energia encerrada simulada
    ax = axs[1, 1]
    r_um = rad_cg * PX / 1000
    for nom in adr.NOMBRES:
        for p_, ls in (("real", "-"), ("reflejo", "--")):
            v = cg[nom, p_]
            ax.plot(r_um, v / v[-1], ls, color=COLOR[nom], lw=1.4, label=f"{nom}, {p_}")
    for (var, p_), v in ee.items():
        ax.plot(r_um, np.asarray(v) / v[-1], ":", color=COLOR["gris"], lw=1.0)
    ax.plot([], [], ":", color=COLOR["gris"], label="PSF esferica simulada (z de la seccion 4)")
    ax.set_xlabel("radio [um]"); ax.set_ylabel(f"e- encerrados / e- en r = {R_CG} px")
    ax.set_title("(e) curva de crecimiento (imagen media)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _limpio(ax)
    # (f) fondo por cuadro
    ax = axs[1, 2]
    for nom in adr.NOMBRES:
        fi = fondo_info[nom]
        ax.plot((fi["dentro"] - fi["fuera"]) * G, "-", color=COLOR[nom], lw=1.5, label=f"{nom}: apertura, lejos de la particula")
        ax.plot((fot[nom]["real"][:, 0, 1] - fi["fuera"]) * G, "--", color=COLOR[nom], lw=1.0, label=f"{nom}: anillo 18-24 px del real")
    ax.axhspan(5, 40, color=COLOR["rejilla"], alpha=0.8, lw=0)
    ax.text(1, 13.2, "fondo_rango de celula.yaml: 5-40", fontsize=7, color=COLOR["gris"])
    ax.set_ylim(0, 14)
    ax.set_xlabel("cuadro"); ax.set_ylabel("fondo sobre el nivel oscuro [e-/px]")
    ax.set_title("(f) fondo por cuadro", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    _limpio(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)
    print(f"figura: {ruta}")


if __name__ == "__main__":
    main()
