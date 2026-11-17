"""
Analisis de los datos reales del montaje anterior con espejo (adquisicion de
Alex, Micro-Manager): un video de 100 cuadros, 1696 x 900 px, con dos canales
de polarizacion en las dos mitades del chip. En cada canal se ve la particula
real (la mas brillante, con anillos) y su reflejo en el espejo.

No usa ni modifica el simulador ni localizacion: solo mide en los datos para
validar el modelo del espejo (geometria_espejo.py) y decidir como llevar los
dos canales al simulador y a la red. Resultados en resultados_datos_reales.md.

Convencion del repo: +z hacia el objetivo, beta = angulo del espejo desde el
eje optico, pixel (fila i, columna j) centrado en (x = j, y = i), phi = azimut
de m (direccion lateral linea del espejo -> muestra, de +x hacia +y). La
separacion lateral real - reflejo es 2 d cos(beta) a lo largo de +m, y la
coordenada a lo largo de la linea (t) es la misma para los dos.

Pasos:
  1. corte entre las dos mitades (columnas vacias entre las dos aperturas de
     campo), deteccion por cuadro (diferencia de gaussianas + maximos locales
     + centroide iterativo), enlace en pistas, pares real/reflejo por canal;
  2. transformacion izquierdo -> derecho por minimos cuadrados (similitud
     propia e impropia, y afin general con bootstrap por cuadros);
  3. por canal: separacion, orientacion, distancia al espejo, ratio
     reflejo/real (fotometria de apertura, dos estimaciones del fondo) y
     ratio de brillo entre canales;
  4. acople a lo largo de la linea del espejo (pasos del reflejo vs del real);
  5. figuras fig_datos_reales_canales.png y fig_datos_reales_medidas.png.

El fotoblanqueo (~2.6x en 100 cuadros) no se modela (trabajo futuro): los
ratios son cocientes por cuadro, asi que no lo arrastran.

Uso:
    $PY figuras/analisis_datos_reales.py [--tif RUTA] [--beta-deg 56] [--radio-um 1.5] [--salida figuras/datos_reales]
"""

import argparse
import os

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIF_DEF = "/home/decar/Datos Alex/Yeast6_1_MMStack_Pos0.ome.tif"
PIXEL_NM = 98.48          # el de los metadatos (0.4 um) esta mal
GANANCIA_E_ADU = 0.46     # Orca Flash 4.0 de simulador_psf; solo para fotones aproximados
RADIOS_UM = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2)
NOMBRES = ("izq", "der")

# deteccion
SIGMA_DOG = 3.0           # [px] gaussiana chica de la diferencia de gaussianas
UMBRAL_K = 10.0           # umbral en MAD robustas de la DoG
DIST_FUSION = 16          # [px] picos mas cercanos se fusionan (lobulos/anillos de una PSF)
R_CENTROIDE = 10          # [px] radio del centroide iterativo
SALTO_MAX = 8.0           # [px] salto maximo entre cuadros al enlazar
FRAC_PISTA = 0.8          # fraccion minima de cuadros de una pista
SEP_PAR = (15.0, 120.0)   # [px] separacion admisible real - reflejo


# ---------------------------------------------------------------- utilidades

def cargar(ruta):
    with tifffile.TiffFile(ruta) as t:
        video = t.asarray()
        md = t.pages[0].tags.get("MicroManagerMetadata")
        md = md.value if md is not None else {}
    info = {k: md.get(k) for k in ("Exposure-ms", "PixelSizeUm", "ROI", "Binning", "Camera")}
    return video, info


def ventana(img, x, y, r):
    """Recorte entero alrededor de (x, y) con medio lado r y sus grillas X, Y."""
    H, W = img.shape
    x0, x1 = max(int(round(x)) - r, 0), min(int(round(x)) + r + 1, W)
    y0, y1 = max(int(round(y)) - r, 0), min(int(round(y)) + r + 1, H)
    Y, X = np.mgrid[y0:y1, x0:x1]
    return img[y0:y1, x0:x1], X, Y


def centroide(img, x, y, r=R_CENTROIDE, iters=20):
    """Centroide iterativo con fondo = mediana del anillo [r+3, r+8]."""
    for _ in range(iters):
        sub, X, Y = ventana(img, x, y, r + 9)
        d2 = (X - x) ** 2 + (Y - y) ** 2
        fondo = np.median(sub[(d2 > (r + 3) ** 2) & (d2 <= (r + 8) ** 2)])
        w = np.clip(sub - fondo, 0, None) * (d2 <= r * r)
        s = w.sum()
        if s <= 0:
            break
        xn, yn = (w * X).sum() / s, (w * Y).sum() / s
        hecho = np.hypot(xn - x, yn - y) < 1e-3
        x, y = xn, yn
        if hecho:
            break
    return x, y, s


# ---------------------------------------------------------------- 1. corte

def buscar_corte(media):
    """Corte entre las dos aperturas de campo. Las aperturas se tocan cerca
    del centro (no hay columnas vacias en todas las filas), asi que el corte
    es el valle del perfil de columnas promediado en las filas donde se tocan
    (las que estan "dentro" en W/2 +- 10). Devuelve el corte, el valle por
    bloques de 20 filas (verticalidad), las filas de contacto, la extension
    de cada apertura y sus mascaras."""
    H, W = media.shape
    lisa = ndi.gaussian_filter(media.astype(float), 4)
    fuera, dentro = np.percentile(lisa, 3), np.percentile(lisa, 60)
    mascara = ndi.binary_opening(lisa > fuera + 0.35 * (dentro - fuera), iterations=3)
    z0 = W // 2 - 70
    filas = np.flatnonzero(mascara[:, W // 2 - 10:W // 2 + 11].any(1))
    sel = filas if len(filas) >= 10 else np.arange(H)
    perfil = ndi.gaussian_filter1d(media[sel].mean(0), 3)
    corte = int(z0 + np.argmin(perfil[z0:W // 2 + 71]))
    valles = []
    for f0 in range(int(sel.min()), int(sel.max()) - 18, 20):
        pr = ndi.gaussian_filter1d(media[f0:f0 + 20].mean(0), 3)
        valles.append((f0 + 10, int(z0 + np.argmin(pr[z0:W // 2 + 71]))))
    ap = []
    for lado in (slice(0, corte), slice(corte, W)):
        mk = np.zeros_like(mascara)
        mk[:, lado] = mascara[:, lado]
        lab, n = ndi.label(mk)
        areas_k = ndi.sum(mk, lab, range(1, n + 1))
        ap.append(ndi.binary_fill_holes(lab == np.argmax(areas_k) + 1))
    return dict(corte=corte, valles=valles, filas_contacto=(int(sel.min()), int(sel.max())),
                valle_adu=float(perfil[corte]), fuera_adu=float(fuera),
                ap_izq=ap[0][:, :corte], ap_der=ap[1][:, corte:], areas=(int(ap[0].sum()), int(ap[1].sum())))


# ---------------------------------------------------------------- deteccion y pistas

def detectar(cuadro, apertura):
    f = ndi.gaussian_filter(cuadro, SIGMA_DOG) - ndi.gaussian_filter(cuadro, 6 * SIGMA_DOG)
    vals = f[apertura]
    ruido = 1.4826 * np.median(np.abs(vals - np.median(vals)))
    util = ndi.binary_erosion(apertura, iterations=25)
    mx = ndi.maximum_filter(f, size=2 * 6 + 1)
    pk = np.argwhere((f == mx) & (f > UMBRAL_K * ruido) & util)
    pk = pk[np.argsort(-f[pk[:, 0], pk[:, 1]])]
    elegidos = []
    for y, x in pk:                                   # fusion: el mas alto se queda
        if all(np.hypot(x - xe, y - ye) > DIST_FUSION for xe, ye in elegidos):
            elegidos.append((x, y))
    dets = []
    for x, y in elegidos:
        xc, yc, s = centroide(cuadro, float(x), float(y))
        dets.append((xc, yc, s))
    # una PSF ancha puede dejar dos picos cuyo centroide converge al mismo punto
    final = []
    for d in sorted(dets, key=lambda d: -d[2]):
        if all(np.hypot(d[0] - e[0], d[1] - e[1]) > DIST_FUSION for e in final):
            final.append(d)
    return final


def enlazar(dets, T):
    """Pistas por asignacion hungara cuadro a cuadro; hueco de hasta 2 cuadros."""
    pistas = []
    for t in range(T):
        act = [p for p in pistas if t - max(p) <= 3]
        if act and dets[t]:
            a = np.array([p[max(p)][:2] for p in act])
            b = np.array([d[:2] for d in dets[t]])
            c = np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1))
            fi, co = linear_sum_assignment(c)
            usados = set()
            for i, j in zip(fi, co):
                if c[i, j] <= SALTO_MAX * (t - max(act[i])):
                    act[i][t] = dets[t][j]
                    usados.add(j)
        else:
            usados = set()
        for j, d in enumerate(dets[t]):
            if j not in usados:
                pistas.append({t: d})
    return [p for p in pistas if len(p) >= FRAC_PISTA * T]


def a_matriz(pista, T):
    m = np.full((T, 3), np.nan)
    for t, d in pista.items():
        m[t] = d
    return m


def pares_real_reflejo(pistas, T):
    """Empareja pistas en pares de separacion estable; real = la de mas senal."""
    mats = [a_matriz(p, T) for p in pistas]
    cand = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            d = np.hypot(*(mats[i][:, :2] - mats[j][:, :2]).T)
            ok = np.isfinite(d)
            if ok.sum() < FRAC_PISTA * T:
                continue
            dm, ds = np.nanmean(d), np.nanstd(d)
            if SEP_PAR[0] <= dm <= SEP_PAR[1]:
                cand.append((ds / dm, i, j))
    usados, pares = set(), []
    for _, i, j in sorted(cand):
        if i in usados or j in usados:
            continue
        usados |= {i, j}
        si, sj = np.nanmedian(mats[i][:, 2]), np.nanmedian(mats[j][:, 2])
        real, refl = (mats[i], mats[j]) if si >= sj else (mats[j], mats[i])
        pares.append(dict(real=real[:, :2], refl=refl[:, :2], s_real=real[:, 2], s_refl=refl[:, 2]))
    return pares


# ---------------------------------------------------------------- 2. transformacion

def similitud(A, B, impropia):
    """B ~ s Q A + t (Umeyama), con det Q = -1 si impropia, +1 si no."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, D, Vt = np.linalg.svd(b.T @ a / len(A))
    sg = np.sign(np.linalg.det(U @ Vt))
    S = np.diag([1.0, -sg if impropia else sg])
    Q = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (a ** 2).sum(1).mean()
    t = mb - s * Q @ ma
    return dict(M=s * Q, t=t, s=s, Q=Q)


def afin(A, B):
    X = np.c_[A, np.ones(len(A))]
    P, *_ = np.linalg.lstsq(X, B, rcond=None)
    return dict(M=P[:2].T, t=P[2])


def aplicar(tr, A):
    return A @ tr["M"].T + tr["t"]


def angulo_Q(Q):
    """Propia: Q = Rot(th). Impropia: Q = Rot(th) diag(1, -1), un espejado
    respecto de la recta a th/2."""
    return np.degrees(np.arctan2(Q[1, 0], Q[0, 0]))


def describir_afin(M):
    """Descomposicion polar M = Q P (Q ortogonal): angulo, escalas, det."""
    U, S, Vt = np.linalg.svd(M)
    Q = U @ Vt
    return dict(det=np.linalg.det(M), escalas=S, angulo=angulo_Q(Q), impropia=np.linalg.det(Q) < 0)


# ---------------------------------------------------------------- 3. fotometria

def fotometria(img, c, otro, r, metodo="anillo", fondo_perfil=None, fondo_global=None):
    """Suma de (img - fondo) en el disco de radio r [px] centrado en c.
    'anillo': fondo = mediana del anillo [r+3, r+10] sin los pixeles a menos
        de r+3 del otro punto;
    'franja': resta antes el perfil de la franja del espejo (fondo_perfil,
        funcion (X, Y) -> fondo) y luego la mediana del anillo del residuo;
    'global': fondo = constante (mediana de la apertura lejos de las
        particulas y de la franja), sin fondo local."""
    sub, X, Y = ventana(img, c[0], c[1], int(r) + 12)
    d2 = (X - c[0]) ** 2 + (Y - c[1]) ** 2
    disco = d2 <= r * r
    if metodo == "global":
        return (sub[disco] - fondo_global).sum()
    lejos = (X - otro[0]) ** 2 + (Y - otro[1]) ** 2 > (r + 3) ** 2
    anillo = (d2 > (r + 3) ** 2) & (d2 <= (r + 10) ** 2) & lejos
    if metodo == "franja":
        sub = sub - fondo_perfil(X, Y)
    return (sub[disco] - np.median(sub[anillo])).sum()


def _coords(shape, centro, m_dir, t_dir):
    H, W = shape
    Y, X = np.mgrid[0:H, 0:W]
    v = (X - centro[0]) * m_dir[0] + (Y - centro[1]) * m_dir[1]
    u = (X - centro[0]) * t_dir[0] + (Y - centro[1]) * t_dir[1]
    return X, Y, u, v


def _perfil_v(img, sel, v, v_lo, v_hi):
    """Perfil (mediana por bin de 1 px) de img a lo largo de v en [v_lo, v_hi]."""
    vb = np.arange(v_lo, v_hi + 1)
    idx = np.round(v[sel]).astype(int) - v_lo
    ok = (idx >= 0) & (idx < len(vb))
    vals, idx = img[sel][ok], idx[ok]
    pr = np.array([np.median(vals[idx == k]) if np.any(idx == k) else np.nan for k in range(len(vb))])
    pr = np.where(np.isfinite(pr), pr, np.nanmedian(pr))
    return vb, pr


def _borde(vb, pr, v_pico):
    """Borde nitido de la franja del lado de la muestra (+m): maxima caida de
    la derivada del perfil suavizado en (v_pico, v_pico + 40]."""
    d = np.gradient(ndi.gaussian_filter1d(pr, 1.5))
    sel = (vb > v_pico) & (vb <= v_pico + 40)
    return float(vb[sel][np.argmin(d[sel])])


def perfil_franja(media, centro, m_dir, t_dir, apertura, excluir):
    """La franja del espejo es paralela a t y asimetrica: borde nitido del
    lado de la muestra (+m) y cola ancha del otro. Fondo como funcion de v
    (coordenada a lo largo de m respecto de centro): mediana por bin de 1 px
    de los pixeles con |u| en [40, 250] px, dentro de la apertura y lejos de
    las particulas. Devuelve (v_bins, fondo_v, funcion de fondo, franja) con
    el pico, el borde y el angulo del borde respecto de t (borde por tramos
    de u)."""
    X, Y, u, v = _coords(media.shape, centro, m_dir, t_dir)
    lejos = np.ones(media.shape, bool)
    for e in excluir:
        lejos &= (X - e[0]) ** 2 + (Y - e[1]) ** 2 > 35 ** 2
    sel = apertura & lejos & (np.abs(u) >= 40) & (np.abs(u) <= 250)
    vb, fondo_v = _perfil_v(media, sel, v, -150, 150)

    def f(Xq, Yq):
        vq = (Xq - centro[0]) * m_dir[0] + (Yq - centro[1]) * m_dir[1]
        return np.interp(vq, vb, fondo_v)

    lisa = ndi.gaussian_filter1d(fondo_v, 3)
    v_pico = float(vb[np.argmax(lisa)])
    v_borde = _borde(vb, fondo_v, v_pico)
    # angulo del borde respecto de t: borde en tramos de 50 px de u
    us, vs = [], []
    for u0 in np.arange(-225, 226, 50):
        if abs(u0) < 50:
            continue
        s2 = apertura & lejos & (np.abs(u - u0) < 25)
        if s2.sum() < 2000:
            continue
        vb2, pr2 = _perfil_v(media, s2, v, int(v_pico) - 40, int(v_pico) + 50)
        p2 = float(vb2[np.argmax(ndi.gaussian_filter1d(pr2, 3))])
        us.append(u0)
        vs.append(_borde(vb2, pr2, p2))
    pend = np.polyfit(us, vs, 1)[0] if len(us) >= 4 else np.nan
    fondo_glob = float(np.median(media[apertura & lejos & (np.abs(v - v_pico) > 120)]))
    return vb, fondo_v, f, dict(v_pico=v_pico, v_borde=v_borde, amplitud=float(lisa.max() - fondo_glob),
                                angulo_vs_t_deg=float(np.degrees(np.arctan(pend))), fondo_global=fondo_glob,
                                us=us, vs=vs)


# ---------------------------------------------------------------- 4. acople

def acople(real, refl, eje, instrumento=None, n_boot=2000):
    """Pasos a lo largo de 'eje' (vector unitario) del reflejo vs del real.
    La pendiente OLS (reflejo ~ real) se atenua por el ruido de localizacion
    del real; la de variable instrumental usa los pasos del MISMO real medido
    en el otro canal (ruido independiente): cov(reflejo, inst) / cov(real, inst)."""
    pr, pf = real @ eje, refl @ eje
    dr, df = np.diff(pr), np.diff(pf)
    di = np.diff(instrumento @ eje) if instrumento is not None else np.zeros_like(dr)
    ok = np.isfinite(dr) & np.isfinite(df) & np.isfinite(di)
    dr, df, di = dr[ok], df[ok], di[ok]

    def ols(a, b):
        a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
        return (a * b).sum(-1) / (a * a).sum(-1)

    def iv(a, b, z):
        a, b, z = (x - x.mean(-1, keepdims=True) for x in (a, b, z))
        return (b * z).sum(-1) / (a * z).sum(-1)

    rng = np.random.default_rng(0)
    k = rng.integers(0, len(dr), (n_boot, len(dr)))
    res = dict(n=len(dr), corr=np.corrcoef(dr, df)[0, 1], pend_ols=ols(dr, df),
               pend_ols_err=np.std(ols(dr[k], df[k])), pend_inv=1.0 / ols(df, dr),
               std_paso_real_nm=dr.std() * PIXEL_NM, std_paso_refl_nm=df.std() * PIXEL_NM, dr=dr, df=df)
    if instrumento is not None:
        b = iv(dr[k], df[k], di[k])
        res.update(pend_iv=iv(dr, df, di), pend_iv_err=np.std(b), pend_iv_ic=np.percentile(b, [2.5, 97.5]),
                   corr_inst=np.corrcoef(dr, di)[0, 1], boot_iv=b,
                   ruido_dif_nm=np.std(dr - di) / np.sqrt(2) * PIXEL_NM)
    return res


# ---------------------------------------------------------------- principal

def regresion(x, y):
    """y = rho x + c por minimos cuadrados: (rho, error de rho, c / media(y))."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    X = np.c_[x, np.ones_like(x)]
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = ((y - X @ coef) ** 2).sum() / (len(x) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return coef[0], np.sqrt(cov[0, 0]), coef[1] / y.mean()


def similitud_fija(A, B, impropia):
    """Como similitud pero con escala 1 (rigida)."""
    tr = similitud(A, B, impropia)
    t = B.mean(0) - tr["Q"] @ A.mean(0)
    return dict(M=tr["Q"], t=t, s=1.0, Q=tr["Q"])


def rms2d(res):
    return float(np.sqrt((res ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tif", default=TIF_DEF)
    ap.add_argument("--beta-deg", type=float, default=56.0, help="angulo del espejo desde el eje optico")
    ap.add_argument("--radio-um", type=float, default=1.5, help="radio de la fotometria principal")
    ap.add_argument("--salida", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_reales"))
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    beta = np.radians(args.beta_deg)
    r_px = args.radio_um * 1000 / PIXEL_NM

    video, info = cargar(args.tif)
    T, H, W = video.shape
    print(f"video {video.shape} {video.dtype}, metadatos: {info}")
    media = video.mean(0)

    # ------------------------------------------------------------ 1. corte y deteccion
    c = buscar_corte(media)
    corte = c["corte"]
    print(f"\n[1] las aperturas se tocan en las filas {c['filas_contacto'][0]}..{c['filas_contacto'][1]}; valle del perfil "
          f"en x = {corte} ({c['valle_adu']:.1f} ADU vs {c['fuera_adu']:.1f} fuera de las aperturas); W/2 = {W // 2}")
    print(f"    valle por bloques de 20 filas (fila, x): {c['valles']}")
    print(f"    corte en x = {corte}: izq columnas 0..{corte - 1} ({corte} px), der {corte}..{W - 1} ({W - corte} px); "
          f"las x del canal derecho son locales (columna - {corte})")
    print(f"    areas de apertura izq/der = {c['areas'][0]} / {c['areas'][1]} px; "
          f"sqrt(cociente der/izq) = {np.sqrt(c['areas'][1] / c['areas'][0]):.4f}")
    canales = [video[:, :, :corte].astype(np.float32), video[:, :, corte:].astype(np.float32)]
    aperturas = [c["ap_izq"], c["ap_der"]]
    del video

    pares = []
    for k, (vid, apk) in enumerate(zip(canales, aperturas)):
        dets = [detectar(vid[t], apk) for t in range(T)]
        pistas = enlazar(dets, T)
        pk = pares_real_reflejo(pistas, T)
        nd, cu = np.unique([len(d) for d in dets], return_counts=True)
        print(f"    canal {NOMBRES[k]}: detecciones por cuadro (n: cuadros) "
              f"{ {int(a): int(b) for a, b in zip(nd, cu)} }, {len(pistas)} pistas en >= "
              f"{FRAC_PISTA:.0%} de los cuadros, {len(pk)} par(es) real/reflejo")
        for p in pk:
            print(f"      par: real en ({np.nanmean(p['real'][:, 0]):.1f}, {np.nanmean(p['real'][:, 1]):.1f}), "
                  f"reflejo en ({np.nanmean(p['refl'][:, 0]):.1f}, {np.nanmean(p['refl'][:, 1]):.1f}) px, "
                  f"cuadros con los dos {np.isfinite(p['real'][:, 0] + p['refl'][:, 0]).sum()}, "
                  f"senal (centroide, mediana) real/reflejo {np.nanmedian(p['s_real']):.0f}/{np.nanmedian(p['s_refl']):.0f} ADU")
        pares.append(pk)
    if not pares[0] or not pares[1]:
        raise SystemExit("no hay pares real/reflejo en algun canal")

    # ------------------------------------------------------------ 2. transformacion
    mejor = None
    for i, pi in enumerate(pares[0]):
        for j, pj in enumerate(pares[1]):
            ok = np.isfinite(pi["real"][:, 0] + pi["refl"][:, 0] + pj["real"][:, 0] + pj["refl"][:, 0])
            A, B = np.r_[pi["real"][ok], pi["refl"][ok]], np.r_[pj["real"][ok], pj["refl"][ok]]
            for imp in (False, True):
                r = rms2d(aplicar(similitud(A, B, imp), A) - B)
                if mejor is None or r < mejor[0]:
                    mejor = (r, i, j)
    _, i, j = mejor
    n_part = 1   # con mas pares se emparejarian los demas con la transformacion ya fijada
    pL, pR = pares[0][i], pares[1][j]
    ok = np.isfinite(pL["real"][:, 0] + pL["refl"][:, 0] + pR["real"][:, 0] + pR["refl"][:, 0])
    n = int(ok.sum())
    A, B = np.r_[pL["real"][ok], pL["refl"][ok]], np.r_[pR["real"][ok], pR["refl"][ok]]
    print(f"\n[2] transformacion izq -> der (coordenadas locales de cada mitad), {n} cuadros x 2 puntos")
    ajustes = {}
    variantes = (("similitud propia", lambda a, b: similitud(a, b, False), "todos"),
                 ("similitud impropia", lambda a, b: similitud(a, b, True), "todos"),
                 ("rigida impropia (s = 1)", lambda a, b: similitud_fija(a, b, True), "todos"),
                 ("afin general", afin, "todos"),
                 ("similitud impropia solo real", lambda a, b: similitud(a, b, True), "real"))
    for nombre, fn, puntos in variantes:
        tr_v = fn(A, B) if puntos == "todos" else fn(A[:n], B[:n])
        res = (B - aplicar(tr_v, A)) * PIXEL_NM
        d = describir_afin(tr_v["M"])
        ajustes[nombre] = dict(tr=tr_v, res=res, **d)
        sesgo_f = res[n:].mean(0)
        print(f"    {nombre:29s}: RMS 2D real {rms2d(res[:n]):5.1f} nm, reflejo {rms2d(res[n:]):5.1f} nm "
              f"(sesgo medio del reflejo ({sesgo_f[0]:+.0f}, {sesgo_f[1]:+.0f}) nm); det {d['det']:+.4f}, "
              f"escalas {d['escalas'][0]:.4f}/{d['escalas'][1]:.4f}, angulo {d['angulo']:+.2f} deg")
        print(f"      M = [[{tr_v['M'][0, 0]:+.4f}, {tr_v['M'][0, 1]:+.4f}], [{tr_v['M'][1, 0]:+.4f}, {tr_v['M'][1, 1]:+.4f}]], "
              f"t = ({tr_v['t'][0]:.2f}, {tr_v['t'][1]:.2f}) px")
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        k = rng.integers(0, n, n)
        Ak, Bk = np.r_[A[:n][k], A[n:][k]], np.r_[B[:n][k], B[n:][k]]
        da = describir_afin(afin(Ak, Bk)["M"])
        si = similitud(Ak, Bk, True)
        sr = similitud(A[:n][k], B[:n][k], True)
        bs.append((da["det"], da["escalas"][0], da["escalas"][1], si["s"], angulo_Q(si["Q"]), sr["s"], angulo_Q(sr["Q"])))
    bs = np.array(bs)
    print(f"    bootstrap por cuadros (2000): det afin {bs[:, 0].mean():+.3f} +- {bs[:, 0].std():.3f} "
          f"(fraccion con det > 0: {(bs[:, 0] > 0).mean():.3f}); escalas afin {bs[:, 1].mean():.3f} +- {bs[:, 1].std():.3f} "
          f"y {bs[:, 2].mean():.3f} +- {bs[:, 2].std():.3f}")
    print(f"      similitud impropia: s {bs[:, 3].mean():.4f} +- {bs[:, 3].std():.4f}, angulo {bs[:, 4].mean():+.2f} +- "
          f"{bs[:, 4].std():.2f} deg; solo real: s {bs[:, 5].mean():.4f} +- {bs[:, 5].std():.4f}, angulo "
          f"{bs[:, 6].mean():+.2f} +- {bs[:, 6].std():.2f} deg")
    tr = ajustes["similitud impropia"]["tr"]
    rms_p, rms_i = rms2d(ajustes["similitud propia"]["res"]), rms2d(ajustes["similitud impropia"]["res"])
    th = angulo_Q(tr["Q"])
    print(f"    => espejado: {'SI' if rms_i < rms_p else 'NO'} (RMS impropia {rms_i:.0f} nm vs propia {rms_p:.0f} nm). "
          f"Q = Rot({th:+.2f}) diag(1,-1): un solo espejado respecto de la recta a {th / 2:+.2f} deg "
          f"(la antidiagonal y = -x es -45)")
    segL = np.nanmean(pL["real"] - pL["refl"], 0)
    tL = np.array([-segL[1], segL[0]]) / np.hypot(*segL)
    print(f"    palanca: |segmento| izq {np.hypot(*segL):.1f} px; recorrido del real perpendicular al segmento "
          f"{np.ptp(pL['real'][ok] @ tL):.1f} px (std {np.std(pL['real'][ok] @ tL):.2f} px) y a lo largo "
          f"{np.ptp(pL['real'][ok] @ (segL / np.hypot(*segL))):.1f} px")

    # ------------------------------------------------------------ 3. por canal
    print(f"\n[3] medidas por canal (pixel {PIXEL_NM} nm, radio {args.radio_um} um = {r_px:.1f} px)")
    medidas = []
    for k, p in enumerate((pL, pR)):
        vid = canales[k]
        seg = p["real"] - p["refl"]
        sep = np.hypot(*seg.T)
        sm_ = np.nanmean(seg, 0)
        m_dir = sm_ / np.hypot(*sm_)
        t_dir = np.array([-m_dir[1], m_dir[0]])
        phi = np.degrees(np.arctan2(m_dir[1], m_dir[0]))
        sep_nm = np.nanmean(sep) * PIXEL_NM
        print(f"  canal {NOMBRES[k]}: separacion {sep_nm:.0f} nm (std entre cuadros {np.nanstd(sep) * PIXEL_NM:.0f}, rango "
              f"{np.nanmin(sep) * PIXEL_NM:.0f}..{np.nanmax(sep) * PIXEL_NM:.0f}); orientacion reflejo -> real "
              f"phi = {phi:+.1f} deg (+x hacia +y; ImageJ {-phi:+.1f} deg)")
        for b_deg in (args.beta_deg, 45.0):
            bb = np.radians(b_deg)
            dd = sep_nm / (2 * np.cos(bb)) / 1000
            print(f"    beta {b_deg:.0f} deg: d = sep / (2 cos beta) = {dd:.2f} um, z_reflejo - z_real = {-2 * dd * np.sin(bb):.2f} um")
        centro = np.nanmean(np.r_[p["real"], p["refl"]], 0)
        excl = [np.nanmean(p["real"], 0), np.nanmean(p["refl"], 0)]
        vb, fv, ffondo, fr = perfil_franja(vid.mean(0), centro, m_dir, t_dir, aperturas[k], excl)
        s_real = (excl[0] - centro) @ m_dir - fr["v_borde"]
        s_refl = (excl[1] - centro) @ m_dir - fr["v_borde"]
        print(f"    franja del espejo (paralela a t): pico a v = {fr['v_pico']:+.0f} px y borde nitido (lado de la muestra) "
              f"a v = {fr['v_borde']:+.0f} px del punto medio del par; amplitud {fr['amplitud']:.1f} ADU; angulo del borde "
              f"con t {fr['angulo_vs_t_deg']:+.2f} deg")
        print(f"    respecto del borde: real a s_m = {s_real * PIXEL_NM / 1000:+.2f} um, reflejo a {s_refl * PIXEL_NM / 1000:+.2f} um")
        for b_deg in (args.beta_deg, 45.0):
            bb = np.radians(b_deg)
            dd = sep_nm / (2 * np.cos(bb)) / 1000
            z = (dd - s_real * PIXEL_NM / 1000 * np.cos(bb)) / np.sin(bb)
            print(f"      si el borde es la linea espejo/plano focal y beta = {b_deg:.0f}: z_real = {z:+.2f} um, "
                  f"z_reflejo = {z - 2 * dd * np.sin(bb):+.2f} um")
        metodos = ("anillo", "franja", "global")
        fot = {mt: np.full((len(RADIOS_UM), T, 2), np.nan) for mt in metodos}
        for t in range(T):
            if not np.isfinite(p["real"][t, 0] + p["refl"][t, 0]):
                continue
            for ir, rum in enumerate(RADIOS_UM):
                rr = rum * 1000 / PIXEL_NM
                for mt in metodos:
                    kw = dict(metodo=mt, fondo_perfil=ffondo, fondo_global=fr["fondo_global"])
                    fot[mt][ir, t, 0] = fotometria(vid[t], p["real"][t], p["refl"][t], rr, **kw)
                    fot[mt][ir, t, 1] = fotometria(vid[t], p["refl"][t], p["real"][t], rr, **kw)
        ratio = {mt: fot[mt][..., 1] / fot[mt][..., 0] for mt in fot}
        # ratio por regresion a lo largo del fotoblanqueo: I_refl = rho I_real + c; la ordenada c absorbe un
        # error aditivo constante del fondo (p. ej. la franja bajo el reflejo), que el cociente por cuadro no
        reg = {mt: np.array([regresion(fot[mt][ir, :, 0], fot[mt][ir, :, 1]) for ir in range(len(RADIOS_UM))])
               for mt in metodos}
        print(f"    ratio reflejo/real por radio. Cociente por cuadro, media +- std entre cuadros (fondo: anillo | perfil "
              f"de franja + anillo | global {fr['fondo_global']:.1f} ADU) || regresion reflejo ~ real a lo largo del "
              f"fotoblanqueo, pendiente +- error (ordenada / reflejo medio) con fondo anillo | global:")
        for ir, rum in enumerate(RADIOS_UM):
            print(f"      r = {rum:4.2f} um: " + " | ".join(f"{np.nanmean(ratio[mt][ir]):.3f} +- {np.nanstd(ratio[mt][ir]):.3f}"
                                                         for mt in metodos)
                  + " || " + " | ".join(f"{reg[mt][ir, 0]:.3f} +- {reg[mt][ir, 1]:.3f} ({reg[mt][ir, 2]:+.2f})"
                                        for mt in ("anillo", "global")))
        ir0 = int(np.argmin(np.abs(np.array(RADIOS_UM) - args.radio_um)))
        rr = ratio["anillo"][ir0]
        q = np.flatnonzero(np.isfinite(rr))
        print(f"    ratio (r {RADIOS_UM[ir0]} um, anillo): mediana {np.nanmedian(rr):.3f}, cociente de sumas "
              f"{np.nansum(fot['anillo'][ir0, :, 1]) / np.nansum(fot['anillo'][ir0, :, 0]):.3f}, primera/segunda mitad "
              f"{np.nanmean(rr[:T // 2]):.3f}/{np.nanmean(rr[T // 2:]):.3f}, pendiente {np.polyfit(q, rr[q], 1)[0] * 100:+.3f} / 100 cuadros")
        I = fot["anillo"][ir0]
        print(f"    intensidad integrada (r {RADIOS_UM[ir0]} um): real {np.nanmean(I[:5, 0]):.0f} -> {np.nanmean(I[-5:, 0]):.0f} ADU "
              f"(cuadros 0-4 -> 95-99, x{np.nanmean(I[:5, 0]) / np.nanmean(I[-5:, 0]):.1f}; ~{np.nanmean(I[:5, 0]) * GANANCIA_E_ADU:.0f} -> "
              f"{np.nanmean(I[-5:, 0]) * GANANCIA_E_ADU:.0f} fotones con {GANANCIA_E_ADU} e-/ADU), reflejo "
              f"{np.nanmean(I[:5, 1]):.0f} -> {np.nanmean(I[-5:, 1]):.0f} ADU")
        medidas.append(dict(sep=sep, m=m_dir, t=t_dir, phi=phi, fot=fot, ratio=ratio, reg=reg, vb=vb, fv=fv, fr=fr,
                            centro=centro, sep_nm=sep_nm))
    # coherencia de las franjas con la transformacion (estimacion independiente del angulo)
    aL = np.degrees(np.arctan2(medidas[0]["t"][1], medidas[0]["t"][0])) + medidas[0]["fr"]["angulo_vs_t_deg"]
    aR = np.degrees(np.arctan2(medidas[1]["t"][1], medidas[1]["t"][0])) + medidas[1]["fr"]["angulo_vs_t_deg"]
    th_fr = (aL + aR + 90) % 180 - 90
    print(f"    angulo de la transformacion impropia desde las franjas (independiente de las particulas): {th_fr:+.2f} deg "
          f"(mod 180) vs {th:+.2f} de los puntos")

    ir0 = int(np.argmin(np.abs(np.array(RADIOS_UM) - args.radio_um)))
    IL, IR = medidas[0]["fot"]["anillo"][ir0], medidas[1]["fot"]["anillo"][ir0]
    print(f"\n    brillo izq / der (r {RADIOS_UM[ir0]} um, fondo anillo): media +- std entre cuadros (CV), cociente de sumas")
    cocientes = {}
    for nombre, a_, b_ in (("real", IL[:, 0], IR[:, 0]), ("reflejo", IL[:, 1], IR[:, 1]),
                           ("real + reflejo", IL.sum(1), IR.sum(1))):
        q = a_ / b_
        cocientes[nombre] = q
        tt = np.flatnonzero(np.isfinite(q))
        rg = regresion(b_, a_)
        print(f"      {nombre:15s}: {np.nanmean(q):.3f} +- {np.nanstd(q):.3f} (CV {np.nanstd(q) / np.nanmean(q):.1%}), "
              f"{np.nansum(a_) / np.nansum(b_):.3f}; tendencia {np.polyfit(tt, q[tt], 1)[0] * 100:+.3f} / 100 cuadros; "
              f"regresion izq ~ der {rg[0]:.3f} +- {rg[1]:.3f} (ordenada / izq medio {rg[2]:+.2f})")
    print(f"    entre particulas: {n_part} particula emparejada en ambos canales -> la dispersion entre particulas no es estimable")

    # ------------------------------------------------------------ 4. acople
    Minv = np.linalg.inv(tr["M"])
    inst = [(pR["real"] - tr["t"]) @ Minv.T, aplicar(tr, pL["real"])]   # real del otro canal en las coordenadas de cada uno
    print(f"\n[4] acople: pasos (cuadro a cuadro) del reflejo vs del real; VI = variable instrumental con el real del otro canal")
    acoples = {}
    for k, p in enumerate((pL, pR)):
        for eje_n in ("t", "m"):
            a = acople(p["real"], p["refl"], medidas[k][eje_n], instrumento=inst[k])
            acoples[(k, eje_n)] = a
            print(f"  canal {NOMBRES[k]}, eje {eje_n}: n {a['n']}, corr {a['corr']:+.3f}, pendiente OLS {a['pend_ols']:+.3f} +- "
                  f"{a['pend_ols_err']:.3f}, 1/OLS inversa {a['pend_inv']:+.3f}, VI {a['pend_iv']:+.3f} +- {a['pend_iv_err']:.3f} "
                  f"(IC95 {a['pend_iv_ic'][0]:+.2f}..{a['pend_iv_ic'][1]:+.2f}; corr real-instrumento {a['corr_inst']:+.2f})")
            print(f"      std de los pasos real/reflejo {a['std_paso_real_nm']:.0f}/{a['std_paso_refl_nm']:.0f} nm; "
                  f"std(paso real - paso real del otro canal)/sqrt2 = {a['ruido_dif_nm']:.0f} nm")
    realC = 0.5 * (aplicar(tr, pL["real"]) + pR["real"])
    reflC = 0.5 * (aplicar(tr, pL["refl"]) + pR["refl"])
    for eje_n in ("t", "m"):
        a = acople(realC, reflC, medidas[1][eje_n])
        acoples[(2, eje_n)] = a
        print(f"  canales combinados (izq transformado + der, ejes del der), {eje_n}: corr {a['corr']:+.3f}, "
              f"pendiente OLS {a['pend_ols']:+.3f} +- {a['pend_ols_err']:.3f}, 1/OLS inversa {a['pend_inv']:+.3f}")
    print(f"    esperado a lo largo de t: 1 (particula o deriva). A lo largo de m: dm' = -cos(2 beta) dm - sin(2 beta) dz, "
          f"o sea {-np.cos(2 * beta):+.3f} con {args.beta_deg:.0f} deg y 0 con 45 deg si dz no se correlaciona con dm; 1 si es deriva")
    for k in (0, 1):
        b = acoples[(k, "m")]["boot_iv"]
        bet = 0.5 * np.degrees(np.arccos(np.clip(-b, -1, 1)))
        print(f"    beta implicito por la pendiente VI en m, canal {NOMBRES[k]}: {np.median(bet):.1f} deg "
              f"(IC95 {np.percentile(bet, 2.5):.1f}..{np.percentile(bet, 97.5):.1f})")

    figuras(canales, corte, pL, pR, tr, medidas, acoples, cocientes, ajustes, args.salida)


# ---------------------------------------------------------------- 5. figuras

def recorte(img, c, r):
    x0, y0 = int(round(c[0])) - r, int(round(c[1])) - r
    return img[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1], x0, y0


def norm(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.8)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def figuras(canales, corte, pL, pR, tr, medidas, acoples, cocientes, ajustes, salida):
    plt.rcParams.update({"font.size": 9})
    rango = slice(0, 10)
    mL, mR = canales[0][rango].mean(0), canales[1][rango].mean(0)
    fig = plt.figure(figsize=(13, 8.8))
    g = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax = fig.add_subplot(g[0, :])
    full = np.c_[mL, mR]
    ax.imshow(full, cmap="gray", vmin=np.percentile(full, 1), vmax=np.percentile(full, 99.95))
    ax.axvline(corte - 0.5, color="y", lw=0.8, ls="--")
    for p, off in ((pL, 0), (pR, corte)):
        c = 0.5 * (np.nanmean(p["real"][rango], 0) + np.nanmean(p["refl"][rango], 0))
        ax.add_patch(plt.Rectangle((c[0] + off - 60, c[1] - 60), 121, 121, fill=False, ec="c", lw=0.8))
    ax.set_title(f"media de los cuadros 0-9; corte en x = {corte} (amarillo); recuadros = zonas ampliadas abajo")
    ax.set_xticks([]), ax.set_yticks([])
    r = 60
    paneles = []
    for k, (img, p) in enumerate(((mL, pL), (mR, pR))):
        c = 0.5 * (np.nanmean(p["real"][rango], 0) + np.nanmean(p["refl"][rango], 0))
        sub, x0, y0 = recorte(img, c, r)
        a = fig.add_subplot(g[1, k])
        a.imshow(sub, cmap="magma", vmin=np.percentile(sub, 1), vmax=np.percentile(sub, 99.9),
                 extent=(x0 - 0.5, x0 + 2 * r + 0.5, y0 + 2 * r + 0.5, y0 - 0.5))
        re, rf = np.nanmean(p["real"][rango], 0), np.nanmean(p["refl"][rango], 0)
        a.plot(*re, "o", mfc="none", mec="c", ms=14, mew=1.5, label="real")
        a.plot(*rf, "s", mfc="none", mec="lime", ms=12, mew=1.5, label="reflejo")
        a.annotate("", xy=re, xytext=rf, arrowprops=dict(arrowstyle="->", color="w", lw=1))
        md = medidas[k]
        # borde de la franja del espejo
        cen, m_dir, t_dir = md["centro"], md["m"], md["t"]
        pb = cen + md["fr"]["v_borde"] * m_dir
        ang = np.radians(md["fr"]["angulo_vs_t_deg"])
        dl = np.cos(ang) * t_dir + np.sin(ang) * m_dir
        a.plot([pb[0] - 80 * dl[0], pb[0] + 80 * dl[0]], [pb[1] - 80 * dl[1], pb[1] + 80 * dl[1]], color="w", ls=":", lw=1,
               label="borde de la franja")
        a.set_xlim(x0 - 0.5, x0 + 2 * r + 0.5), a.set_ylim(y0 + 2 * r + 0.5, y0 - 0.5)
        a.set_title(f"canal {NOMBRES[k]} (x local): sep {md['sep_nm']:.0f} nm, phi {md['phi']:+.0f} deg")
        a.legend(loc="lower left", fontsize=7, framealpha=0.5)
        paneles.append((c, x0, y0))
    cR, x0R, y0R = paneles[1]
    Minv = np.linalg.inv(tr["M"])
    YR, XR = np.mgrid[y0R:y0R + 2 * r + 1, x0R:x0R + 2 * r + 1]
    src = (np.c_[XR.ravel(), YR.ravel()] - tr["t"]) @ Minv.T
    Lw = ndi.map_coordinates(mL, [src[:, 1], src[:, 0]], order=1).reshape(XR.shape)
    Rw = mR[y0R:y0R + 2 * r + 1, x0R:x0R + 2 * r + 1]
    a = fig.add_subplot(g[1, 2])
    a.imshow(np.dstack([norm(Lw), norm(Rw), norm(Lw)]), extent=(x0R - 0.5, x0R + 2 * r + 0.5, y0R + 2 * r + 0.5, y0R - 0.5))
    for p, trans, col, nom in ((pL, True, "m", "izq transf."), (pR, False, "lime", "der")):
        re, rf = np.nanmean(p["real"][rango], 0), np.nanmean(p["refl"][rango], 0)
        if trans:
            re, rf = aplicar(tr, re[None])[0], aplicar(tr, rf[None])[0]
        a.plot(*re, "o", mfc="none", mec=col, ms=14, mew=1.2, label=f"real {nom}")
        a.plot(*rf, "s", mfc="none", mec=col, ms=12, mew=1.2, label=f"reflejo {nom}")
    a.set_title(f"izq transformado (magenta) sobre der (verde)\nsimilitud impropia, RMS {rms2d(ajustes['similitud impropia']['res']):.0f} nm")
    a.legend(loc="lower left", fontsize=6.5, framealpha=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(salida, f"fig_datos_reales_canales.{ext}"), dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(17, 8))
    a = axs[0, 0]
    for k, col in ((0, "C0"), (1, "C3")):
        for mt, ls in (("anillo", "-"), ("franja", "--"), ("global", ":")):
            rt = medidas[k]["ratio"][mt]
            a.errorbar(RADIOS_UM, np.nanmean(rt, 1), np.nanstd(rt, 1) if mt == "anillo" else None, color=col, ls=ls,
                       marker="o", ms=3, capsize=2, label=f"{NOMBRES[k]}, fondo {mt}")
    a.axvline(1.5, color="0.6", lw=0.8, ls=":")
    a.set_xlabel("radio de la apertura [um]"), a.set_ylabel("reflejo / real"), a.legend(fontsize=6.5)
    a.set_title("ratio reflejo/real vs radio\n(barras: std entre cuadros)")
    for col_ax, eje_n, titulo in ((1, "t", "a lo largo de la linea (t)"), (2, "m", "perpendicular (m)")):
        a = axs[0, col_ax]
        for k, col in ((0, "C0"), (1, "C3")):
            ac = acoples[(k, eje_n)]
            a.plot(ac["dr"] * PIXEL_NM, ac["df"] * PIXEL_NM, ".", color=col, ms=4, alpha=0.6,
                   label=f"{NOMBRES[k]}: corr {ac['corr']:.2f}, OLS {ac['pend_ols']:.2f}, VI {ac['pend_iv']:.2f}")
            lim = np.nanmax(np.abs(ac["dr"])) * PIXEL_NM
            a.plot([-lim, lim], [-lim * ac["pend_iv"], lim * ac["pend_iv"]], color=col, lw=1)
        lim = a.get_xlim()
        a.plot(lim, lim, color="0.5", ls=":", lw=1, label="pendiente 1")
        a.set_xlabel(f"paso del real [nm]"), a.set_ylabel(f"paso del reflejo [nm]"), a.legend(fontsize=6.5)
        a.set_title(f"acople de los pasos, {titulo}\n(rectas: pendiente VI)")
    a = axs[0, 3]
    for k, col in ((0, "C0"), (1, "C3")):
        md = medidas[k]
        a.plot(md["vb"] * PIXEL_NM / 1000, md["fv"], color=col, lw=1, label=f"{NOMBRES[k]}")
        for v, ls in ((md["fr"]["v_borde"], "--"),):
            a.axvline(v * PIXEL_NM / 1000, color=col, ls=ls, lw=0.8)
        for nm_, mk in (("real", "o"), ("refl", "s")):
            vv = (np.nanmean(pL[nm_] if k == 0 else pR[nm_], 0) - md["centro"]) @ md["m"]
            a.plot(vv * PIXEL_NM / 1000, np.interp(vv, md["vb"], md["fv"]), mk, color=col, mfc="none")
    a.set_xlabel("v a lo largo de m desde el punto medio del par [um]"), a.set_ylabel("fondo [ADU]")
    a.set_title("perfil de la franja del espejo (lejos del par)\n(-- borde; o real, cuadrado reflejo)")
    a.legend(fontsize=7)
    a = axs[1, 0]
    # el espejado entre canales invierte t: el izquierdo se dibuja con el signo que coincide con el derecho
    signo = np.sign((tr["Q"] @ medidas[0]["t"]) @ medidas[1]["t"])
    for k, (p, col) in enumerate(((pL, "C0"), (pR, "C3"))):
        t_dir = medidas[k]["t"] * (signo if k == 0 else 1)
        ur, uf = p["real"] @ t_dir, p["refl"] @ t_dir
        a.plot((ur - np.nanmean(ur)) * PIXEL_NM, color=col, lw=1, label=f"{NOMBRES[k]} real")
        a.plot((uf - np.nanmean(uf)) * PIXEL_NM, color=col, lw=1, ls="--", label=f"{NOMBRES[k]} reflejo")
    a.set_xlabel("cuadro (10 Hz)"), a.set_ylabel("posicion a lo largo de t [nm]"), a.legend(fontsize=7, ncol=2)
    a.set_title("trayectorias a lo largo de la linea del espejo\n(izq con el signo de t del der)")
    a = axs[1, 1]
    for nombre, col in (("real", "C2"), ("reflejo", "C1"), ("real + reflejo", "k")):
        a.plot(cocientes[nombre], color=col, lw=1, label=f"{nombre}: {np.nanmean(cocientes[nombre]):.3f}")
    a.set_xlabel("cuadro"), a.set_ylabel("brillo izq / der"), a.legend(fontsize=7)
    a.set_title("ratio de brillo entre canales (r = 1.5 um, anillo)")
    a = axs[1, 2]
    ir0 = list(RADIOS_UM).index(1.5)
    for k, col in ((0, "C0"), (1, "C3")):
        I = medidas[k]["fot"]["anillo"][ir0]
        a.plot(I[:, 0] * GANANCIA_E_ADU, color=col, lw=1, label=f"{NOMBRES[k]} real")
        a.plot(I[:, 1] * GANANCIA_E_ADU, color=col, lw=1, ls="--", label=f"{NOMBRES[k]} reflejo")
        a.plot(medidas[k]["ratio"]["anillo"][ir0] * 0, alpha=0)
    a.set_xlabel("cuadro"), a.set_ylabel(f"e- en r = 1.5 um (x {GANANCIA_E_ADU} e-/ADU)")
    a.set_title("intensidad integrada (fotoblanqueo, no modelado)"), a.legend(fontsize=7)
    a = axs[1, 3]
    for nombre, col in (("similitud propia", "C7"), ("similitud impropia", "C2"), ("similitud impropia solo real", "C4")):
        res = ajustes[nombre]["res"]
        nn = len(res) // 2
        a.plot(res[:nn, 0], res[:nn, 1], "o", color=col, ms=2.5, alpha=0.6, label=f"{nombre}: real {rms2d(res[:nn]):.0f} nm")
        a.plot(res[nn:, 0], res[nn:, 1], "s", color=col, ms=2.5, alpha=0.6, mfc="none",
               label=f"   reflejo {rms2d(res[nn:]):.0f} nm")
    a.axhline(0, color="0.7", lw=0.5), a.axvline(0, color="0.7", lw=0.5)
    a.set_aspect("equal", "datalim")
    a.set_xlabel("residuo x [nm]"), a.set_ylabel("residuo y [nm]"), a.legend(fontsize=6)
    a.set_title("residuos de la transformacion izq -> der")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(salida, f"fig_datos_reales_medidas.{ext}"), dpi=200)
    plt.close(fig)
    print(f"\nfiguras en {salida}: fig_datos_reales_canales.png/.pdf, fig_datos_reales_medidas.png/.pdf")


if __name__ == "__main__":
    main()
