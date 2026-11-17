"""
Pipeline de tracking 3D de tracking_3d.py aplicado a Yeast6_1 (canal derecho,
una particula, 100 cuadros a 10 Hz). Ver resultados/resultados_tracking_yeast6_1.md.

Partes (en orden; analisis lee lo que escribio localizar):
    localizar  deteccion clasica del par real / reflejo (analisis_datos_reales.py),
               geometria (phi del segmento reflejo -> real, linea espejo/plano
               focal en el borde de la franja, como en celula_v2.py), la red s1
               en la ventana de 96 px centrada en el punto medio del par (umbral
               0.2: superconjunto del 0.4) y las manchas candidatas a reflejo
               (tracking/reflejos.py) en la misma ventana -> CACHE (no versionado)
    analisis   1. modelo del desfase real - reflejo a lo largo de la linea: dt lineal en
                  la separacion L (un giro delta de m), ajustado en los pares del
                  primer pase y aplicado a todos los cuadros; sigma_dt = residuo
               2. pipeline final (umbral 0.2, anillos de la PSF real suprimidos en
                  16 px, sin las detecciones que parecen el reflejo, enlace 2D, D
                  lateral auto-calibrado) en dos pases, la variante tal_cual de
                  tracking_3d y ablaciones de cada eleccion; comparacion con el par
                  clasico de analisis_datos_reales.py
               3. x, y: red y clasico, ruido de localizacion (ordenada del MSD y
                  pasos - 2 D dt) y sesgo red - clasico en el tiempo
               4. z triangulada (beta 40-50), atipicos por consistencia temporal
                  (k = 3 fijo; 2.5 y 3.5 de sensibilidad) y D por MSD (lags 1-4 con
                  ordenada) en 3D, lateral y z con IC por bootstrap de bloques
               -> figuras/tracking_yeast6_1/ (CSV y PNG 300 dpi)

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    LD_PRELOAD=$HOME/miniforge3/envs/tesis/lib/libstdc++.so.6 $PY figuras/tracking_yeast6_1.py localizar [--tif RUTA]   # ~1 min
    $PY figuras/tracking_yeast6_1.py analisis
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
sys.path.insert(1, str(RAIZ / "tracking"))
sys.path.insert(2, str(RAIZ / "localizacion"))

CORRIDA = RAIZ / "localizacion" / "corridas" / "hypatia_celula_v2_b45_cerca_s1_20261114-090158"
CACHE = RAIZ / "localizacion" / "datos" / "yeast6_1"
SALIDA = AQUI / "tracking_yeast6_1"
TIF_DEF = "/home/decar/Datos Alex/Yeast6_1_MMStack_Pos0.ome.tif"
DT = 0.1                          # s (10 Hz)
T_E = 0.1                         # s, exposicion (= periodo: desenfoque de movimiento en todo el cuadro)
VENTANA = 96                      # px, la de entrenamiento
UMBRAL_MIN = 0.2
UMBRAL = 0.4                      # el del checkpoint y el elegido en tracking_3d (B1)
GAIN, OFFSET = 0.45, 97.75        # camara de celula_v2.yaml: solo para pasar a e- en el detector de manchas
K_REFL = 5.0                      # umbral del DoG de las manchas (el de tracking_3d)


# ----------------------------------------------------------------------------
# localizar
# ----------------------------------------------------------------------------

def localizar(a):
    import analisis_datos_reales as adr
    import red_celula as rc
    from infer import localizar as loc_red
    from utils import elegir_dispositivo
    from reflejos import detectar_manchas
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    px = adr.PIXEL_NM
    video, _ = adr.cargar(a.tif)
    T = len(video)
    c = adr.buscar_corte(video.mean(0))
    corte = c["corte"]
    crudo = np.ascontiguousarray(video[:, :, corte:])
    del video
    vid = crudo.astype(np.float32)
    ap = c["ap_der"]
    dets = [adr.detectar(vid[t], ap) for t in range(T)]
    pares = adr.pares_real_reflejo(adr.enlazar(dets, T), T)
    assert len(pares) == 1, f"se esperaba un par real/reflejo; hay {len(pares)}"
    real, refl = pares[0]["real"], pares[0]["refl"]
    seg = np.nanmean(real - refl, 0)
    m = seg / np.hypot(*seg)
    t_dir = np.array([-m[1], m[0]])
    centro = np.nanmean(np.r_[real, refl], 0)
    _, _, _, fr = adr.perfil_franja(vid.mean(0), centro, m, t_dir, ap, [np.nanmean(real, 0), np.nanmean(refl, 0)])
    q = centro + fr["v_borde"] * m
    phi = float(np.degrees(np.arctan2(m[1], m[0])))
    medio = np.nanmean((real + refl) / 2, 0)
    x0, y0 = int(round(medio[0])) - VENTANA // 2, int(round(medio[1])) - VENTANA // 2
    assert ap[y0:y0 + VENTANA, x0:x0 + VENTANA].all(), "la ventana se sale de la apertura"
    ims = np.ascontiguousarray(crudo[:, y0:y0 + VENTANA, x0:x0 + VENTANA])
    np.save(CACHE / "media_ventana.npy", ims[:10].mean(0))
    pd.DataFrame(dict(frame=np.arange(T), x_real=real[:, 0], y_real=real[:, 1], x_refl=refl[:, 0], y_refl=refl[:, 1],
                      senal_real=pares[0]["s_real"], senal_refl=pares[0]["s_refl"])).to_csv(CACHE / "clasico.csv", index=False)
    geo = dict(tif=str(a.tif), T=T, corte=int(corte), px=px, phi_deg=phi, q_px=q.tolist(), m=m.tolist(),
               v_borde_px=float(fr["v_borde"]), angulo_borde_deg=float(fr["angulo_vs_t_deg"]),
               ventana_x0=x0, ventana_y0=y0, ventana=VENTANA)
    (CACHE / "geometria.json").write_text(json.dumps(geo, indent=1))

    disp = elegir_dispositivo("auto")
    modelo, cfg, cs = rc.cargar_red("cerca", disp, CORRIDA / "mejor.pt")
    cfg = replace(cfg, postproc=replace(cfg.postproc, umbral=UMBRAL_MIN))
    qv = q - np.array([x0, y0])
    det = loc_red(modelo, ims, cfg, replace(cs, H=VENTANA, W=VENTANA), disp, geometria=rc.geometria(45.0, phi, qv[0], qv[1], T))
    det = det.rename(columns={"t": "frame"})
    det["x_px"] += x0
    det["y_px"] += y0
    det[["frame", "x_px", "y_px", "z_nm", "conf"]].to_csv(CACHE / "detecciones_red.csv", index=False)

    partes = []
    for f in range(T):
        e = (ims[f].astype(float) - OFFSET) * GAIN
        s = pd.DataFrame(detectar_manchas(e, k=K_REFL)).assign(frame=f)
        s["x_px"] += x0
        s["y_px"] += y0
        partes.append(s)
    man = pd.concat(partes, ignore_index=True)
    man.to_csv(CACHE / "manchas.csv", index=False)
    print(f"Yeast6_1 canal derecho: {T} cuadros, corte {corte}, phi {phi:+.1f} deg, linea a {fr['v_borde']:+.1f} px "
          f"del punto medio, ventana ({x0}, {y0}) de {VENTANA} px; {len(det)} detecciones de la red (conf >= {UMBRAL_MIN}), "
          f"{len(man)} manchas ({time.time() - t0:.0f} s) -> {CACHE}")


# ----------------------------------------------------------------------------
# analisis: utilidades
# ----------------------------------------------------------------------------

R_ASOC_PX = 3.0                   # deteccion de la red "es el real" si cae a <= 3 px del real clasico (red_celula.py)
R_REFL_PX = 2.0                   # mancha "es el reflejo clasico" si cae a <= 2 px
BETAS = np.round(np.arange(40.0, 50.01, 0.5), 2)
BETA_REF = 45.0
VENT_MED = 9                      # cuadros de la mediana movil (centrada)
K_ATIP = 3.0                      # atipico: |z - mediana movil| > K_ATIP sigma (sigma = 1.4826 MAD de los residuos)
KS_ATIP = (2.5, 3.0, 3.5)       # k = 3 fijo; 2.5 y 3.5 solo como sensibilidad
N_LAGS = 4
BLOQUE = 10                       # cuadros por bloque del bootstrap
N_BOOT = 2000
D_INI = 0.2                       # um^2/s: D del primer enlace (despues se usa el medido)
D_MIN = 0.01                      # um^2/s: piso del D medido para el enlace
SIG_DT_INI = 200.0                # nm: sigma_dt del primer emparejamiento (despues el residuo del modelo del desfase)
SIG_Z_RED = np.nan                # nm: sigma_z de la red (tracking_3d) para la ablacion con enlace 3D; se lee en analisis
COLOR = {"red": "#2a78d6", "clasico": "#52514e", "reflejo": "#eda100", "tri": "#1baf7a", "atipico": "#d62a4a",
         "lat": "#2a78d6", "z": "#1baf7a", "3d": "#52514e", "gris": "#8a8984"}


def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def mas_cercano(det, x, y, r):
    """(T,) indice (en det) de la deteccion mas cercana a (x[f], y[f]) en cada cuadro si esta a <= r px; -1 si no."""
    idx = -np.ones(len(x), int)
    for f, g in det.groupby("frame"):
        d = np.hypot(g.x_px.to_numpy() - x[f], g.y_px.to_numpy() - y[f])
        if len(d) and np.nanmin(d) <= r:
            idx[f] = g.index[np.nanargmin(d)]
    return idx


def estad(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return dict(n=len(e), media=e.mean(), std=e.std(ddof=1), rms=np.sqrt(np.mean(e ** 2)),
                std_robusta=1.4826 * np.median(np.abs(e - np.median(e))))


def mediana_movil(z, w=VENT_MED):
    """Mediana de los z finitos en [f - w//2, f + w//2] (nan si hay menos de 3)."""
    T, h = len(z), w // 2
    out = np.full(T, np.nan)
    for f in range(T):
        v = z[max(0, f - h):f + h + 1]
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            out[f] = np.median(v)
    return out


def atipicos(z, k=K_ATIP):
    """Mascara (T,) de atipicos por consistencia temporal: |z - mediana movil| > k sigma, sigma = 1.4826 MAD
    de los residuos de todo el video. Devuelve tambien la mediana movil y sigma."""
    med = mediana_movil(z)
    r = z - med
    rr = r[np.isfinite(r)]
    sig = 1.4826 * np.median(np.abs(rr - np.median(rr)))
    return np.isfinite(r) & (np.abs(r) > k * sig), med, sig


def sumas_msd(r, ini=None, largo=None, n_lags=N_LAGS):
    """Sumas de desplazamientos cuadraticos (n_lags, d) y conteos (n_lags,) de la trayectoria r (T, d) con nan.
    Con ini, solo pares dentro de cada bloque [i, i + largo) (con repeticion)."""
    T, d = r.shape
    S, N = np.zeros((n_lags, d)), np.zeros(n_lags)
    tramos = [(0, T)] if ini is None else [(i, i + largo) for i in ini]
    for a, b in tramos:
        x = r[a:b]
        for k in range(1, n_lags + 1):
            dd = x[k:] - x[:-k]
            ok = np.isfinite(dd).all(1)
            S[k - 1] += (dd[ok] ** 2).sum(0)
            N[k - 1] += ok.sum()
    return S, N


def ajuste_msd(S, N, dims, dt=DT):
    """D [um^2/s] y ordenada [nm^2] del ajuste lineal (con ordenada) de MSD(tau) = sum de los ejes dims, lags 1..n."""
    msd = S[:, dims].sum(1) / np.maximum(N, 1)
    if (N < 3).any():
        return np.nan, np.nan, msd
    tau = dt * np.arange(1, len(N) + 1)
    p, b = np.polyfit(tau, msd, 1)
    return p / (2 * len(dims)) * 1e-6, b, msd


EJES = {"3d": [0, 1, 2], "lateral": [0, 1], "z": [2], "x": [0], "y": [1]}


def d_por_msd(r, rng, ejes=EJES, n_boot=N_BOOT, bloque=BLOQUE):
    """D por MSD de r (T, 3) [nm] con IC 95 % por bootstrap de bloques de cuadros consecutivos (bloques moviles,
    T / bloque por remuestreo; solo pares dentro de un bloque)."""
    T = len(r)
    S, N = sumas_msd(r)
    out = {}
    for nombre, dims in ejes.items():
        D, b, msd = ajuste_msd(S, N, dims)
        out[nombre] = dict(D=D, ordenada_nm2=b, msd=msd, pares=N.copy())
    nb = int(np.ceil(T / bloque))
    bs = {k: [] for k in ejes}
    for _ in range(n_boot):
        ini = rng.integers(0, T - bloque + 1, nb)
        Sb, Nb = sumas_msd(r, ini, bloque)
        for nombre, dims in ejes.items():
            bs[nombre].append(ajuste_msd(Sb, Nb, dims)[0])
    for nombre in ejes:
        v = np.array(bs[nombre])
        v = v[np.isfinite(v)]
        out[nombre]["lo"], out[nombre]["hi"] = np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
    return out


# ----------------------------------------------------------------------------
# analisis
# ----------------------------------------------------------------------------

R_ANILLO_PX = 16.0                # variante adaptada: radio de los anillos de la PSF real (detecciones 6-20 px)


def suprimir_anillos(det, r=R_ANILLO_PX):
    """Una deteccion por grupo de radio r, por cuadro: desde la de mas conf, el grupo son las detecciones a <= r
    px de ella y se queda la mas cercana al centro ponderado por conf (los anillos rodean al real); se repite con
    las que quedan."""
    keep = []
    for _, g in det.groupby("frame"):
        while len(g):
            j = g.conf.idxmax()
            c = g[np.hypot(g.x_px - g.x_px[j], g.y_px - g.y_px[j]) <= r]
            cx, cy = (c.x_px * c.conf).sum() / c.conf.sum(), (c.y_px * c.conf).sum() / c.conf.sum()
            keep.append(c.index[np.argmin(np.hypot(c.x_px - cx, c.y_px - cy))])
            g = g.drop(c.index)
    return det.loc[sorted(keep)].reset_index(drop=True)


def lejos_de(man, det, r):
    """Mascara de las manchas a > r px de toda deteccion de su cuadro."""
    ok = np.ones(len(man), bool)
    dd = {f: g for f, g in det.groupby("frame")}
    for f, g in man.groupby("frame"):
        d = dd.get(f)
        if d is not None and len(d):
            dist = np.hypot(g.x_px.to_numpy()[:, None] - d.x_px.to_numpy()[None], g.y_px.to_numpy()[:, None] - d.y_px.to_numpy()[None])
            ok[g.index.to_numpy()] = dist.min(1) > r
    return ok


def parece_reflejo(det, phi, q, px, t3):
    """Mascara de las detecciones de la red que estan del lado del espejo de otra del mismo cuadro, a mas de
    R_ANILLO_PX a lo largo de m y a < TOL_T_NM a lo largo de t: pueden ser el reflejo detectado como real."""
    from espejo import coordenadas_espejo
    s_, t_ = coordenadas_espejo(det.x_px, det.y_px, px, phi, q)
    out = np.zeros(len(det), bool)
    for _, g in det.groupby("frame"):
        i = g.index.to_numpy()
        ds = s_[i][None, :] - s_[i][:, None]                     # s_otra - s_esta
        dt = np.abs(t_[i][None, :] - t_[i][:, None])
        out[i] = ((ds > R_ANILLO_PX * px) & (dt < t3.TOL_T_NM)).any(1)
    return out


def ajuste_robusto(x, y, iters=10, k=3.0):
    """y ~ a + b x por minimos cuadrados con recorte iterativo (|residuo| > k 1.4826 MAD). Devuelve a, b, mascara de
    los que quedan y sigma robusta del residuo."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    dentro = ok.copy()
    for _ in range(iters):
        if dentro.sum() < 5:
            break
        b, a = np.polyfit(x[dentro], y[dentro], 1)
        r = y - (a + b * x)
        sg = 1.4826 * np.median(np.abs(r[dentro] - np.median(r[dentro])))
        nuevo = ok & (np.abs(r) <= k * sg)
        if (nuevo == dentro).all():
            break
        dentro = nuevo
    b, a = np.polyfit(x[dentro], y[dentro], 1)
    r = y - (a + b * x)
    return a, b, dentro, 1.4826 * np.median(np.abs(r[dentro] - np.median(r[dentro])))


def bootstrap_bloques(T, rng, n_boot, bloque=BLOQUE):
    """Indices de cuadros de cada remuestreo (bloques moviles de `bloque` cuadros, T / bloque bloques)."""
    nb = int(np.ceil(T / bloque))
    for _ in range(n_boot):
        ini = rng.integers(0, T - bloque + 1, nb)
        yield (ini[:, None] + np.arange(bloque)).ravel()


def ic(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return tuple(np.percentile(v, [2.5, 97.5])) if len(v) > 10 else (np.nan, np.nan)


# ----------------------------------------------------------------------------
# 1. modelo del desfase real - reflejo a lo largo de la linea
# ----------------------------------------------------------------------------

def modelo_desfase(tipo, L, dt, sF, rng=None, n_boot=0):
    """Modelo del desfase dt = t_real - t_reflejo de un conjunto de cuadros (arreglos (T,) con nan).

    'L'       dt = a + b L: el segmento real -> reflejo no es paralelo a m sino a m + b t, es decir la
              direccion de m esta girada delta = atan(b) (hacia t). Se corrige girando el sistema (s, t)
              en delta (phi + delta) y restando la mediana del dt girado (una constante para todo el video).
    'sF'      dt = a + c s_reflejo (diagnostico, no es una geometria del espejo).
    'cte'     dt = constante (la mediana; sin giro): separa el efecto de la constante del de la pendiente.
    'ninguno' dt = 0.
    sigma_dt = 1.4826 MAD del residuo (con recorte iterativo de los que se apartan mas de 3 sigma)."""
    ok = np.isfinite(L) & np.isfinite(dt)
    out = dict(tipo=tipo, n=int(ok.sum()), delta_deg=0.0, a=0.0, b=0.0, c=0.0, constante=0.0)
    if tipo in ("ninguno", "cte"):
        r = dt[ok]
        cte = float(np.median(r)) if tipo == "cte" else 0.0
        out.update(constante=cte, sigma_dt=1.4826 * np.median(np.abs(r - np.median(r))),
                   corr=lambda s, c_=cte: np.full_like(s, c_))
        return out
    x = L if tipo == "L" else sF
    a, b, dentro, sg = ajuste_robusto(x, dt)
    out.update(a=a, n_dentro=int(dentro.sum()), r2=1 - np.var((dt - a - b * x)[dentro]) / np.var(dt[dentro]),
               sigma_sin_modelo=1.4826 * np.median(np.abs(dt[ok] - np.median(dt[ok]))))
    if tipo == "L":
        d = np.arctan(b)
        dtg = dt * np.cos(d) - L * np.sin(d)               # dt en el sistema girado en d
        cte = float(np.median(dtg[dentro]))
        r = dtg[dentro] - cte
        out.update(b=b, delta_deg=float(np.degrees(d)), constante=cte,
                   sigma_dt=1.4826 * np.median(np.abs(r - np.median(r))), corr=lambda s, c_=cte: np.full_like(s, c_))
    else:
        out.update(c=b, sigma_dt=sg, corr=lambda s, a_=a, c_=b: a_ + c_ * s)
    if n_boot:
        T = len(L)
        bs = []
        for idx in bootstrap_bloques(T, rng, n_boot):
            xs, ys = x[idx], dt[idx]
            if (np.isfinite(xs) & np.isfinite(ys)).sum() < 10:
                continue
            bb = ajuste_robusto(xs, ys)[1]
            bs.append(np.degrees(np.arctan(bb)) if tipo == "L" else bb)
        out["ic_lo"], out["ic_hi"] = ic(bs)
    return out


# ----------------------------------------------------------------------------
# 2. pipeline (variantes)
# ----------------------------------------------------------------------------

FINAL = dict(umbral=UMBRAL_MIN, suprimir=True, descartar=True, excluir=True, enlace="2d", D_fijo=None, desfase="L")
VARIANTES = {"tal_cual": dict(umbral=UMBRAL, suprimir=False, descartar=False, excluir=False, enlace="2d", D_fijo=None,
                              desfase="ninguno"),
             "final": FINAL}
# cada una cambia una sola eleccion de la final (seccion "Elecciones hechas despues de ver los datos")
ABLACIONES = {"sin_modelo_desfase": dict(desfase="ninguno"), "desfase_solo_constante": dict(desfase="cte"),
              "desfase_en_s_reflejo": dict(desfase="sF"),
              "sin_supresion_anillos": dict(suprimir=False), "sin_descartar_reflejos": dict(descartar=False),
              "sin_exclusion_manchas_16px": dict(excluir=False), "umbral_0.4": dict(umbral=UMBRAL),
              "enlace_3d": dict(enlace="3d"), "D_fijo_0.02": dict(D_fijo=0.02), "D_fijo_0.2": dict(D_fijo=0.2),
              "D_fijo_1.5": dict(D_fijo=1.5)}
NOMBRE_VAR = {"tal_cual": "tal cual (tracking_3d)", "final": "pipeline final", "clasico": "par clasico"}


def preparar(cfg, det_todo, man, geo, t3):
    """Detecciones candidatas a real y manchas candidatas a reflejo de una configuracion."""
    px, phi, q = geo["px"], geo["phi_deg"], np.array(geo["q_px"])
    det = det_todo[det_todo.conf >= cfg["umbral"]].reset_index(drop=True)
    if cfg["suprimir"]:
        det = suprimir_anillos(det)
    n_refl = 0
    if cfg["descartar"]:
        pr_ = parece_reflejo(det, phi, q, px, t3)       # el reflejo detectado como real: no es candidato a real
        n_refl = int(pr_.sum())
        det = det[~pr_].reset_index(drop=True)
    man = man.reset_index(drop=True)
    km = t3.filtrar_manchas(man, det, phi, q, px)
    if cfg["excluir"]:
        km &= lejos_de(man, det, R_ANILLO_PX)
    return det, man[km].reset_index(drop=True), n_refl


def correr(det, cand, geo, cl, t3, D_um2, sig_dt, enlace, phi_, corr=None):
    """Enlace (real y reflejo), emparejamiento en el sistema (s, t) de phi_ (corr(s_F) se suma a t_F) y
    trayectoria de la particula: el componente conexo de pistas unidas por pares con mas cuadros triangulados."""
    from enlace import CostoNormalizado, enlazar, filtrar_cortas, _renumerar
    from espejo import coordenadas_espejo, triangular_z, emparejar_pistas
    from metricas import a_arreglo
    T, px, q = geo["T"], geo["px"], np.array(geo["q_px"])
    st = lambda x, y: coordenadas_espejo(x, y, px, phi_, q)  # noqa: E731
    s_lat = sig_dt / np.sqrt(2)                     # reparto igual entre real y reflejo (solo fija el costo)
    c2 = CostoNormalizado([s_lat, s_lat], D_um2 * 1e6, DT, t3.R_CHI, c_loc=2.0)
    if enlace == "3d":
        c = CostoNormalizado([s_lat, s_lat, SIG_Z_RED], D_um2 * 1e6, DT, t3.R_CHI, c_loc=2.0)
        pos = np.c_[det.x_px * px, det.y_px * px, det.z_nm]
    else:
        c, pos = c2, np.c_[det.x_px * px, det.y_px * px]

    def pistas(d, p_, costo):
        if len(d) == 0:
            return d, np.zeros(0, int)
        p = enlazar(d.frame.to_numpy(), p_, max_hueco=HUECO, costo=costo)
        k = filtrar_cortas(p, LARGO)
        return d[k].reset_index(drop=True), (_renumerar(p[k]) if k.any() else np.zeros(0, int))
    dr, p = pistas(det, pos, c)
    dfc, pf = pistas(cand, np.c_[cand.x_px * px, cand.y_px * px], c2)
    vacio = lambda d: np.zeros((0, T, d))  # noqa: E731
    sR, tR = st(dr.x_px, dr.y_px)
    sF, tF = st(dfc.x_px, dfc.y_px)
    if corr is not None and len(dfc):
        tF = tF + corr(np.asarray(sF))
    Rr = a_arreglo(p, dr.frame.to_numpy(), np.c_[sR, tR], T)[0] if len(dr) else vacio(2)
    Ff = a_arreglo(pf, dfc.frame.to_numpy(), np.c_[sF, tF], T)[0] if len(dfc) else vacio(2)
    X = a_arreglo(p, dr.frame.to_numpy(), np.c_[dr.x_px, dr.y_px], T)[0] if len(dr) else vacio(2)
    Xf = a_arreglo(pf, dfc.frame.to_numpy(), np.c_[dfc.x_px, dfc.y_px], T)[0] if len(dfc) else vacio(2)

    def z_par(i, j, ok):
        return t3.Z_POSIBLE[0] <= np.median(triangular_z(Rr[i, ok, 0], Ff[j, ok, 0], BETA_REF)) <= t3.Z_POSIBLE[1]
    pares = emparejar_pistas(Rr, Ff, sig_dt, z_par=z_par)[0] if len(Rr) and len(Ff) else []
    padre = {}

    def raiz(u):
        while padre.setdefault(u, u) != u:
            u = padre[u]
        return u
    for pr in pares:
        padre[raiz(("r", pr["i"]))] = raiz(("f", pr["j"]))
    peso = {}
    for pr in pares:
        k_ = raiz(("r", pr["i"]))
        peso[k_] = peso.get(k_, 0) + int(pr["cuadros"].sum())
    comp = max(peso, key=peso.get) if peso else None
    reales_c = sorted({pr["i"] for pr in pares if raiz(("r", pr["i"])) == comp})
    tray = {k_: np.full(T, np.nan) for k_ in ("x_px", "y_px", "x_tri_px", "y_tri_px", "x_refl_px", "y_refl_px")}
    for i in reales_c:                                  # lateral: todos los cuadros de las pistas reales
        ok = np.isfinite(X[i, :, 0]) & np.isnan(tray["x_px"])
        tray["x_px"][ok], tray["y_px"][ok] = X[i, ok, 0], X[i, ok, 1]
    for pr in pares:
        pr["particula"] = raiz(("r", pr["i"])) == comp
        if pr["particula"]:
            ok = pr["cuadros"]
            for c_, A, i in (("tri", X, pr["i"]), ("refl", Xf, pr["j"])):
                tray[f"x_{c_}_px"][ok], tray[f"y_{c_}_px"][ok] = A[i, ok, 0], A[i, ok, 1]
            tray["x_px"][ok], tray["y_px"][ok] = X[pr["i"], ok, 0], X[pr["i"], ok, 1]
    return dict(dr=dr, dfc=dfc, X=X, Xf=Xf, R=Rr, F=Ff, pares=pares, reales=reales_c, tray=pd.DataFrame(tray))


def datos_desfase(o, geo, phi_):
    """(L, dt, s_F) por cuadro (T,) en todos los cuadros comunes de los pares de la particula (sin corregir)."""
    from espejo import coordenadas_espejo
    T, px, q = geo["T"], geo["px"], np.array(geo["q_px"])
    L, dt, sF = (np.full(T, np.nan) for _ in range(3))
    for pr in o["pares"]:
        if not pr["particula"]:
            continue
        xr, xf = o["X"][pr["i"]], o["Xf"][pr["j"]]
        ok = np.isfinite(xr[:, 0]) & np.isfinite(xf[:, 0]) & np.isnan(L)
        sr, tr = coordenadas_espejo(xr[ok, 0], xr[ok, 1], px, phi_, q)
        sf, tf = coordenadas_espejo(xf[ok, 0], xf[ok, 1], px, phi_, q)
        L[ok], dt[ok], sF[ok] = sr - sf, tr - tf, sf
    return L, dt, sF


def d_lateral(o, geo):
    tr = o["tray"]
    xy = np.c_[tr.x_px, tr.y_px] * geo["px"]
    if np.isfinite(xy[:, 0]).sum() <= 5:
        return np.nan
    at = atipicos(xy[:, 0])[0] | atipicos(xy[:, 1])[0]
    D = ajuste_msd(*sumas_msd(np.where(at[:, None], np.nan, xy)), [0, 1])[0]
    return max(D, D_MIN) if np.isfinite(D) else np.nan


def pipeline(nombre, cfg, det_todo, man, cl, geo, t3, rng, n_boot_desfase=0):
    """Dos pases: el primero con D_INI (o D fijo), SIG_DT_INI y la geometria de la franja; de sus pares se miden
    el D lateral y el modelo del desfase (que da phi, la correccion de t_F y sigma_dt), y el segundo pase es el
    resultado."""
    phi0 = geo["phi_deg"]
    det, cand, n_refl = preparar(cfg, det_todo, man, geo, t3)
    D0 = cfg["D_fijo"] or D_INI
    o0 = correr(det, cand, geo, cl, t3, D0, SIG_DT_INI, cfg["enlace"], phi0)
    D1 = cfg["D_fijo"] or d_lateral(o0, geo)
    L, dt, sF = datos_desfase(o0, geo, phi0)
    md = modelo_desfase(cfg["desfase"], L, dt, sF, rng, n_boot_desfase) if np.isfinite(L).sum() > 5 else None
    if md is None or not np.isfinite(D1):
        return dict(nombre=nombre, cfg=cfg, det=det, cand=cand, n_refl=n_refl, o=o0, D=D0, sig_dt=SIG_DT_INI, phi=phi0,
                    modelo=md, datos_desfase=(L, dt, sF), corr=None)
    phi1 = phi0 + md["delta_deg"]
    o1 = correr(det, cand, geo, cl, t3, D1, md["sigma_dt"], cfg["enlace"], phi1, md["corr"])
    return dict(nombre=nombre, cfg=cfg, det=det, cand=cand, n_refl=n_refl, o=o1, o0=o0, D=D1, sig_dt=md["sigma_dt"],
                phi=phi1, modelo=md, datos_desfase=(L, dt, sF), corr=md["corr"])


def z_de(r, geo, beta, phi_=None):
    """z triangulada (T,) de la trayectoria de r (dict de pipeline) en su sistema (s, t)."""
    from espejo import coordenadas_espejo, triangular_z
    tr = r["o"]["tray"]
    phi_ = r["phi"] if phi_ is None else phi_
    sR, _ = coordenadas_espejo(tr.x_tri_px, tr.y_tri_px, geo["px"], phi_, np.array(geo["q_px"]))
    sF, _ = coordenadas_espejo(tr.x_refl_px, tr.y_refl_px, geo["px"], phi_, np.array(geo["q_px"]))
    return triangular_z(np.asarray(sR), np.asarray(sF), beta)


def evaluar(r, cl, geo, z_cl45):
    """Cuadros triangulados y correctos frente al par clasico y error de z (beta 45) frente a la z clasica."""
    tr = r["o"]["tray"]
    con = tr.x_refl_px.notna().to_numpy()
    real_ok = (np.hypot(tr.x_tri_px - cl.x_real, tr.y_tri_px - cl.y_real) <= R_ASOC_PX).to_numpy() & con
    refl_ok = (np.hypot(tr.x_refl_px - cl.x_refl, tr.y_refl_px - cl.y_refl) <= R_REFL_PX).to_numpy() & con
    dz = z_de(r, geo, BETA_REF) - z_cl45
    md = r["modelo"] or {}
    return dict(variante=r["nombre"], D_enlace=r["D"], sigma_dt=r["sig_dt"], delta_deg=md.get("delta_deg", np.nan),
                constante_nm=md.get("constante", np.nan), c_sF=md.get("c", np.nan),
                detecciones_por_cuadro=len(r["det"]) / geo["T"], descartadas_reflejo=r["n_refl"] / geo["T"],
                manchas_por_cuadro=len(r["cand"]) / geo["T"], pistas_real=len(r["o"]["X"]), pistas_reflejo=len(r["o"]["Xf"]),
                pares=len(r["o"]["pares"]), pistas_particula=len(r["o"]["reales"]),
                cuadros_lateral=int(tr.x_px.notna().sum()), cuadros_triangulados=int(con.sum()),
                reflejo_correcto=int(refl_ok.sum()), real_correcto=int(real_ok.sum()),
                cuadros_sin_triangular=" ".join(map(str, np.flatnonzero(~con))) or "-",
                dz45_media=np.nanmean(dz) if np.isfinite(dz).any() else np.nan,
                dz45_rms=np.sqrt(np.nanmean(dz ** 2)) if np.isfinite(dz).any() else np.nan)


# ----------------------------------------------------------------------------
# 3. ruido de localizacion y sesgo red - clasico
# ----------------------------------------------------------------------------

def ruido(x, rng, n_boot=N_BOOT):
    """Ruido de localizacion de una coordenada x (T,) [nm] con nan.
    msd:   ordenada b del ajuste MSD(tau) = 2 D tau + b, tau = 1..4: b = 2 sigma^2 - 2 D t_E / 3
    pasos: var(paso entre cuadros consecutivos) - 2 D dt = 2 sigma^2 - 2 D t_E / 3
    (desenfoque de movimiento con exposicion t_E = dt; D de la misma pendiente). Devuelve sigma con y sin el
    termino de desenfoque e IC por bootstrap de bloques."""
    def una(idx_bloques=None):
        r = x[:, None]
        if idx_bloques is None:
            S, N = sumas_msd(r)
            d = np.diff(x)
        else:
            S, N = sumas_msd(r, idx_bloques, BLOQUE)
            d = np.concatenate([np.diff(x[i:i + BLOQUE]) for i in idx_bloques])
        D, b, _ = ajuste_msd(S, N, [0])
        d = d[np.isfinite(d)]
        if not np.isfinite(D) or len(d) < 5:
            return [np.nan] * 5
        D_nm = D * 1e6
        blur = 2 * D_nm * T_E / 3
        v = np.var(d, ddof=1)
        return [D, (b + blur) / 2, (v - 2 * D_nm * DT + blur) / 2, b / 2, (v - 2 * D_nm * DT) / 2]
    D, s2m, s2p, s2m0, s2p0 = una()
    T = len(x)
    nb = int(np.ceil(T / BLOQUE))
    bs = np.array([una(rng.integers(0, T - BLOQUE + 1, nb)) for _ in range(n_boot)])
    raiz = lambda v: np.sign(v) * np.sqrt(np.abs(v))  # noqa: E731  (sigma con signo: negativo = varianza < 0)
    out = dict(cuadros=int(np.isfinite(x).sum()), D_um2_s=D, sigma_msd=raiz(s2m), sigma_pasos=raiz(s2p),
               sigma_msd_sin_desenfoque=raiz(s2m0), sigma_pasos_sin_desenfoque=raiz(s2p0))
    for k, j in (("sigma_msd", 1), ("sigma_pasos", 2)):
        out[f"{k}_lo"], out[f"{k}_hi"] = (raiz(v) for v in ic(bs[:, j]))
    return out


def sesgo_en_el_tiempo(b, z, senal, rng, n_boot=N_BOOT):
    """Sesgo b (T,) [nm] con nan: medias por bloque de 10 cuadros, pendiente con el cuadro (nm / 100 cuadros) y
    correlaciones con z y con la senal del real, con IC por bootstrap de bloques."""
    T = len(b)
    fr = np.arange(T)
    ok = np.isfinite(b)

    def stats(idx):
        o = ok[idx]
        bb, ff, zz, ss = b[idx][o], fr[idx][o], z[idx][o], senal[idx][o]
        if len(bb) < 10:
            return [np.nan] * 3
        return [np.polyfit(ff, bb, 1)[0] * 100, np.corrcoef(bb, zz)[0, 1], np.corrcoef(bb, ss)[0, 1]]
    p, rz, rs = stats(fr)
    bs = np.array([stats(idx) for idx in bootstrap_bloques(T, rng, n_boot)])
    bloques = [b[i:i + BLOQUE][np.isfinite(b[i:i + BLOQUE])] for i in range(0, T, BLOQUE)]
    medias = np.array([v.mean() if len(v) else np.nan for v in bloques])
    # medias por bloque frente a una constante: chi^2 con el error estandar de cada bloque (supone cuadros
    # independientes dentro del bloque; la autocorrelacion de paso 1 dice si lo son)
    from scipy.stats import chi2
    se = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 2 else np.nan for v in bloques])
    okb = np.isfinite(medias) & np.isfinite(se)
    mp = np.average(medias[okb], weights=1 / se[okb] ** 2)
    x2 = float(np.sum(((medias[okb] - mp) / se[okb]) ** 2))
    par = ok[:-1] & ok[1:]
    out = dict(media=np.nanmean(b), std=np.nanstd(b, ddof=1), pendiente_nm_100c=p, corr_z=rz, corr_senal=rs,
               medias_bloque=" ".join(f"{m:+.0f}" for m in medias), rango_medias_bloque=np.nanmax(medias) - np.nanmin(medias),
               std_medias_bloque=np.nanstd(medias, ddof=1), se_medio_bloque=np.nanmean(se), chi2_bloques=x2,
               gl=int(okb.sum() - 1), p_chi2=float(chi2.sf(x2, okb.sum() - 1)),
               autocorr_paso1=float(np.corrcoef(b[:-1][par], b[1:][par])[0, 1]))
    for k, j in (("pendiente_nm_100c", 0), ("corr_z", 1), ("corr_senal", 2)):
        out[f"{k}_lo"], out[f"{k}_hi"] = ic(bs[:, j])
    return out, medias


# ----------------------------------------------------------------------------
# analisis
# ----------------------------------------------------------------------------

def analisis(a):
    import tracking_3d as t3
    from espejo import coordenadas_espejo, triangular_z
    global HUECO, LARGO, SIG_Z_RED
    rng = np.random.default_rng(a.seed)
    SALIDA.mkdir(parents=True, exist_ok=True)
    geo = json.loads((CACHE / "geometria.json").read_text())
    T, px, phi, q = geo["T"], geo["px"], geo["phi_deg"], np.array(geo["q_px"])
    cl = pd.read_csv(CACHE / "clasico.csv")
    det_todo = pd.read_csv(CACHE / "detecciones_red.csv")
    man = pd.read_csv(CACHE / "manchas.csv")
    mejor = pd.read_csv(t3.SALIDA / "tracking_3d_enlace_mejor.csv").iloc[0]
    assert mejor.costo == "norm2" and float(mejor.umbral) == UMBRAL
    HUECO, LARGO = int(mejor.hueco), int(mejor.largo)
    SIG_Z_RED = float(pd.read_csv(t3.SALIDA / "tracking_3d_enlace_sigmas.csv").set_index("umbral").loc[UMBRAL_MIN, "sigma_z"])
    st0 = lambda x, y: coordenadas_espejo(x, y, px, phi, q)  # noqa: E731
    sR_c, tR_c = st0(cl.x_real, cl.y_real)
    sF_c, tF_c = st0(cl.x_refl, cl.y_refl)
    sR_c, tR_c, sF_c, tF_c = (np.asarray(v) for v in (sR_c, tR_c, sF_c, tF_c))
    z_cl = {b: triangular_z(sR_c, sF_c, b) for b in BETAS}

    # ---- desfase del par clasico (referencia del modelo; no lo usa el pipeline)
    L_c, dt_c = sR_c - sF_c, tR_c - tF_c
    filas_des = []
    for tipo in ("L", "sF"):
        md = modelo_desfase(tipo, L_c, dt_c, sF_c, rng, N_BOOT)
        filas_des.append(dict(fuente="par clasico", **{k: v for k, v in md.items() if k != "corr"}))

    # ---- pipeline: variantes y ablaciones
    res = {}
    for v, cfg in VARIANTES.items():
        res[v] = pipeline(v, cfg, det_todo, man, cl, geo, t3, rng, n_boot_desfase=N_BOOT if v == "final" else 0)
    for v, cambio in ABLACIONES.items():
        res[v] = pipeline(v, {**FINAL, **cambio}, det_todo, man, cl, geo, t3, rng)
    for v in ("final", "desfase_solo_constante", "desfase_en_s_reflejo", "sin_modelo_desfase"):
        md = res[v]["modelo"]
        filas_des.append(dict(fuente=f"pipeline ({v}), pares del primer pase", **{k: x for k, x in md.items() if k != "corr"}))
    tdes = pd.DataFrame(filas_des)
    tdes["angulo_borde_franja_deg"] = geo["angulo_borde_deg"]
    tdes["delta_predicho_por_el_borde_deg"] = -geo["angulo_borde_deg"]
    tdes.to_csv(SALIDA / "tracking_yeast6_1_desfase.csv", index=False)
    ev = pd.DataFrame([evaluar(r, cl, geo, z_cl[BETA_REF]) for r in res.values()])

    # ---- x, y: red por cuadro frente al clasico; ruido de localizacion y sesgo
    fuentes_xy = {"clasico": np.c_[cl.x_real, cl.y_real]}
    for nombre, u in (("red umbral 0.4", UMBRAL), ("red umbral 0.2", UMBRAL_MIN)):
        d = det_todo[det_todo.conf >= u].reset_index(drop=True)
        i = mas_cercano(d, cl.x_real.to_numpy(), cl.y_real.to_numpy(), R_ASOC_PX)
        xy = np.full((T, 2), np.nan)
        xy[i >= 0] = np.c_[d.x_px.to_numpy()[i[i >= 0]], d.y_px.to_numpy()[i[i >= 0]]]
        fuentes_xy[nombre] = xy
    tr_f = res["final"]["o"]["tray"]
    fuentes_xy["pipeline final (pista)"] = np.c_[tr_f.x_px, tr_f.y_px]
    comp, rui, ses, medias_sesgo = [], [], [], {}
    for nombre, xy in fuentes_xy.items():
        for e_, k in (("x", 0), ("y", 1)):
            rui.append(dict(fuente=nombre, eje=e_, **ruido(xy[:, k] * px, rng)))
        if nombre == "clasico":
            continue
        ex, ey = (xy[:, 0] - cl.x_real.to_numpy()) * px, (xy[:, 1] - cl.y_real.to_numpy()) * px
        for e_, v in (("x", ex), ("y", ey), ("radial", np.hypot(ex, ey))):
            comp.append(dict(fuente=nombre, eje=e_, **estad(v)))
        if nombre == "red umbral 0.4":
            for e_, v in (("x", ex), ("y", ey)):
                o_, m_ = sesgo_en_el_tiempo(v, z_cl[BETA_REF], cl.senal_real.to_numpy(), rng)
                ses.append(dict(fuente=nombre, eje=e_, **o_))
                medias_sesgo[e_] = (v, m_)
    pd.DataFrame(comp).to_csv(SALIDA / "tracking_yeast6_1_xy_red_vs_clasico.csv", index=False)
    rui = pd.DataFrame(rui)
    rui.to_csv(SALIDA / "tracking_yeast6_1_ruido_localizacion.csv", index=False)
    ses = pd.DataFrame(ses)
    ses.to_csv(SALIDA / "tracking_yeast6_1_sesgo_tiempo.csv", index=False)

    # ---- z triangulada, atipicos (k = 3 fijo, sensibilidad 2.5 y 3.5) y D por MSD
    trays = {"clasico": (np.c_[cl.x_real, cl.y_real], z_cl, np.c_[cl.x_real, cl.y_real])}
    for v in ("final", "tal_cual"):
        tr = res[v]["o"]["tray"]
        trays[v] = (np.c_[tr.x_tri_px, tr.y_tri_px], {b: z_de(res[v], geo, b) for b in BETAS}, np.c_[tr.x_px, tr.y_px])
    zf, dfil, msd_t, pc = [], [], [], dict(frame=np.arange(T))
    for fuente, (pr, zb, plat) in trays.items():
        xy, xy_lat = pr * px, plat * px
        for b in BETAS:
            z = zb[b]
            n = int(np.isfinite(z).sum())
            fila = dict(fuente=fuente, beta=b, cuadros=n, z_media=np.nanmean(z), z_std=np.nanstd(z))
            for k in KS_ATIP:
                at, _, sg = atipicos(z, k)
                fila[f"atipicos_k{k:g}"] = int(at.sum())
            fila["sigma_residuo"] = sg
            zf.append(fila)
            pc[f"z_{fuente}_b{b:g}_nm"] = z
            pc[f"atipico_{fuente}_b{b:g}"] = atipicos(z, K_ATIP)[0]
            for k in ((None,) + KS_ATIP if b == BETA_REF else (K_ATIP,)):
                zz = z if k is None else np.where(atipicos(z, k)[0], np.nan, z)
                r3 = np.where(np.isfinite(zz)[:, None], np.c_[xy, zz], np.nan)
                rb = d_por_msd(r3, rng, ejes={"3d": [0, 1, 2], "z": [2]})
                if b == BETA_REF and k == K_ATIP:
                    rb.update(d_por_msd(np.c_[xy_lat, np.zeros(T)], rng, ejes={"lateral": [0, 1], "x": [0], "y": [1]}))
                for e_, val in rb.items():
                    dfil.append(dict(fuente=fuente, beta=b, k_atipicos=k if k is not None else np.inf, eje=e_,
                                     D_um2_s=val["D"], lo=val["lo"], hi=val["hi"], ordenada_nm2=val["ordenada_nm2"],
                                     pares_lag1=int(val["pares"][0])))
                    if b == BETA_REF and k == K_ATIP:
                        for j in range(N_LAGS):
                            msd_t.append(dict(fuente=fuente, eje=e_, lag=j + 1, msd_nm2=val["msd"][j], pares=int(val["pares"][j])))
    zt = pd.DataFrame(zf)
    zt.to_csv(SALIDA / "tracking_yeast6_1_z_beta.csv", index=False)
    td = pd.DataFrame(dfil)
    td.to_csv(SALIDA / "tracking_yeast6_1_D.csv", index=False)
    tm = pd.DataFrame(msd_t)
    tm.to_csv(SALIDA / "tracking_yeast6_1_msd.csv", index=False)
    # efecto de girar phi sobre la z del pipeline final (misma trayectoria, sistema de la franja)
    zg = z_de(res["final"], geo, BETA_REF, phi_=phi)
    efecto_giro = float(np.sqrt(np.nanmean((trays["final"][1][BETA_REF] - zg) ** 2)))

    # ---- ablaciones: D de cada una (beta 45, k = 3, menos remuestreos)
    for i, r in enumerate(res.values()):
        tr = r["o"]["tray"]
        z = z_de(r, geo, BETA_REF)
        zz = np.where(atipicos(z, K_ATIP)[0], np.nan, z) if np.isfinite(z).sum() > 5 else z
        xy = np.c_[tr.x_tri_px, tr.y_tri_px] * px
        r3 = np.where(np.isfinite(zz)[:, None], np.c_[xy, zz], np.nan)
        rb = d_por_msd(r3, rng, ejes={"3d": [0, 1, 2], "z": [2]}, n_boot=500)
        rb.update(d_por_msd(np.c_[np.c_[tr.x_px, tr.y_px] * px, np.zeros(T)], rng, ejes={"lateral": [0, 1]}, n_boot=500))
        for e_ in ("lateral", "z", "3d"):
            ev.loc[i, f"D_{e_}"] = rb[e_]["D"]
            ev.loc[i, f"D_{e_}_lo"], ev.loc[i, f"D_{e_}_hi"] = rb[e_]["lo"], rb[e_]["hi"]
    ev.to_csv(SALIDA / "tracking_yeast6_1_variantes.csv", index=False)
    for v in VARIANTES:
        tr = res[v]["o"]["tray"]
        for c_ in tr:
            pc[f"{c_}_{v}"] = tr[c_]
    pd.DataFrame(pc).to_csv(SALIDA / "tracking_yeast6_1_por_cuadro.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 400, "display.max_colwidth", 60):
        f1 = lambda v: f"{v:.4g}"  # noqa: E731
        print(f"geometria: phi {phi:+.2f} deg, borde de la franja a {geo['angulo_borde_deg']:+.2f} deg de t; "
              f"sigma_z de la red para el enlace 3D {SIG_Z_RED:.1f} nm; efecto del giro de phi sobre z (beta 45): "
              f"{efecto_giro:.1f} nm RMS")
        for t_ in (tdes, ev.drop(columns=["cuadros_sin_triangular"]), ev[["variante", "cuadros_sin_triangular"]],
                   pd.DataFrame(comp), rui, ses, zt[zt.beta.isin([40, 42.5, 45, 47.5, 50])],
                   td[td.beta.isin([40, 42.5, 45, 47.5, 50])]):
            print(t_.to_string(index=False, float_format=f1))
            print()
    (SALIDA / "tracking_yeast6_1_geometria.json").write_text(json.dumps(dict(
        phi_franja_deg=phi, angulo_borde_franja_deg=geo["angulo_borde_deg"], phi_final_deg=res["final"]["phi"],
        sigma_dt_final_nm=res["final"]["sig_dt"], D_enlace_final=res["final"]["D"], sigma_z_red_enlace_3d=SIG_Z_RED,
        efecto_giro_z45_rms_nm=efecto_giro), indent=1))
    figuras(pd.DataFrame(pc), zt, td, tm, trays, geo, res, cl, (L_c, dt_c, sF_c, z_cl[BETA_REF]), rui, medias_sesgo)


# ----------------------------------------------------------------------------
# figuras
# ----------------------------------------------------------------------------

def figuras(pc, zt, td, tm, trays, geo, res, cl, des_cl, rui, medias_sesgo):
    from espejo import coordenadas_espejo
    plt = _estilo()
    px, phi, q, T = geo["px"], geo["phi_deg"], np.array(geo["q_px"]), geo["T"]
    x0, y0 = geo["ventana_x0"], geo["ventana_y0"]
    st = lambda x, y, p=phi: coordenadas_espejo(x, y, px, p, q)  # noqa: E731
    fr = np.arange(T)
    grid = dict(color="#e6e5e0", lw=0.6)
    fin = res["final"]
    ad = fin["o"]["tray"]

    # -- deteccion, emparejamiento y desfase
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1.3, 1, 1]))
    ax = axs[0]
    med = np.load(CACHE / "media_ventana.npy")
    ax.imshow(med, cmap="gray", vmin=np.percentile(med, 1), vmax=np.percentile(med, 99.7),
              extent=(x0 - 0.5, x0 + VENTANA - 0.5, y0 + VENTANA - 0.5, y0 - 0.5))
    det = pd.read_csv(CACHE / "detecciones_red.csv")
    d2 = det[det.conf >= UMBRAL_MIN]
    ax.scatter(d2.x_px, d2.y_px, s=3, color="#c9c8c2", lw=0, label="red, conf >= 0.2 (todos los cuadros)")
    ax.scatter(fin["det"].x_px, fin["det"].y_px, s=4, color=COLOR["red"], lw=0, label="candidatas a real (final)")
    ax.scatter(fin["cand"].x_px, fin["cand"].y_px, s=4, color=COLOR["reflejo"], lw=0, label="manchas candidatas a reflejo")
    ax.plot(cl.x_real, cl.y_real, "-", color="white", lw=0.8)
    ax.plot(cl.x_refl, cl.y_refl, "-", color="white", lw=0.8, label="par clasico")
    ok = ad.x_refl_px.notna()
    for xx, yy in ((ad.x_tri_px, ad.y_tri_px), (ad.x_refl_px, ad.y_refl_px)):
        ax.scatter(xx[ok], yy[ok], s=7, marker="o", facecolor="none", edgecolor=COLOR["tri"], lw=0.7,
                   label="pipeline final: cuadros triangulados" if xx is ad.x_tri_px else None)
    ax.set_xlim(x0 - 0.5, x0 + VENTANA - 0.5); ax.set_ylim(y0 + VENTANA - 0.5, y0 - 0.5)
    ax.set_xlabel("x [px, canal derecho]"); ax.set_ylabel("y [px]")
    ax.set_title("ventana de la red (media de los cuadros 0-9)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    ax = axs[1]
    sRc, _ = st(cl.x_real, cl.y_real)
    for v, nombre, mk, c in (("tal_cual", "tal cual (umbral 0.4)", "x", COLOR["gris"]), ("final", "pipeline final", "o", COLOR["red"])):
        trr = res[v]["o"]["tray"]
        s_, _ = st(trr.x_px, trr.y_px)
        ax.plot(fr, s_ - sRc, mk, ms=3, mfc="none", color=c, lw=0, label=f"{nombre}: s - s clasico")
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_xlabel("cuadro (10 Hz)"); ax.set_ylabel("real de la pista - real clasico, a lo largo de m [nm]")
    ax.set_ylim(-1600, 1600); ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7)
    ax.set_title("real de la pista de la particula (los saltos de 600-1500 nm son anillos)", fontsize=8.5)
    L_c, dt_c, sF_c, z45 = des_cl
    md_c = modelo_desfase("L", L_c, dt_c, sF_c)
    md_p = fin["modelo"]
    Lp, dtp, _ = fin["datos_desfase"]
    ax = axs[2]
    ax.scatter(L_c, dt_c, s=9, color=COLOR["clasico"], label="par clasico")
    ax.scatter(Lp, dtp, s=12, marker="o", facecolor="none", edgecolor=COLOR["tri"], label="pipeline, pares del primer pase")
    xx = np.linspace(np.nanmin(np.r_[L_c, Lp]), np.nanmax(np.r_[L_c, Lp]), 2)
    ax.plot(xx, md_c["a"] + np.tan(np.radians(md_c["delta_deg"])) * xx, "-", color=COLOR["clasico"], lw=1.2,
            label=f"ajuste clasico: delta = {md_c['delta_deg']:+.1f} deg")
    ax.plot(xx, md_p["a"] + np.tan(np.radians(md_p["delta_deg"])) * xx, "-", color=COLOR["tri"], lw=1.2,
            label=f"ajuste del pipeline: delta = {md_p['delta_deg']:+.1f} deg")
    db = -geo["angulo_borde_deg"]
    ax.plot(xx, np.tan(np.radians(db)) * (xx - np.nanmean(L_c)), ":", color=COLOR["atipico"], lw=1.2,
            label=f"giro que pide el borde de la franja ({db:+.1f} deg)")
    ax.set_xlabel("L = s real - s reflejo [nm]"); ax.set_ylabel("dt = t real - t reflejo [nm] (sistema de la franja)")
    ax.set_ylim(-500, 500); ax.grid(True, **grid); ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("modelo pedido: dt lineal en L (giro de m)", fontsize=8.5)
    ax = axs[3]
    ax.scatter(z45, dt_c, s=9, color=COLOR["clasico"], label="par clasico")
    p_ = np.polyfit(z45, dt_c, 1)
    zz = np.array([z45.min(), z45.max()])
    ax.plot(zz, np.polyval(p_, zz), "-", color=COLOR["clasico"], lw=1.2,
            label=f"pendiente {p_[0]:+.2f}, r = {np.corrcoef(z45, dt_c)[0, 1]:+.2f}")
    ax.set_xlabel("z triangulada, beta 45 [nm] (= -s reflejo)"); ax.set_ylabel("dt [nm]")
    ax.set_ylim(-500, 500); ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7)
    ax.set_title("donde esta la varianza de dt: en z, no en L", fontsize=8.5)
    fig.suptitle("Yeast6_1, canal derecho: red s1 frente a la deteccion clasica, emparejamiento y desfase real - reflejo", fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_yeast6_1_deteccion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -- ruido de localizacion y sesgo en el tiempo
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    for ax, e_ in zip(axs[:2], ("x", "y")):
        v, m_ = medias_sesgo[e_]
        ax.plot(fr, v, "o", ms=3, color=COLOR["red"], alpha=0.6, label="por cuadro")
        cen = np.arange(BLOQUE / 2 - 0.5, T, BLOQUE)
        ax.plot(cen, m_, "s-", color=COLOR["clasico"], ms=5, lw=1.2, label="media por bloque de 10 cuadros")
        ax.axhline(np.nanmean(v), color="#8a8984", lw=0.8, ls=":")
        ax.set_xlabel("cuadro (10 Hz)"); ax.set_ylabel(f"{e_} red (umbral 0.4) - {e_} clasico [nm]")
        ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"sesgo red - clasico en {e_}", fontsize=8.5)
    ax = axs[2]
    fuentes = list(dict.fromkeys(rui.fuente))
    w = 0.18
    for k, (met, c) in enumerate((("sigma_msd", COLOR["red"]), ("sigma_pasos", COLOR["reflejo"]))):
        for j, e_ in enumerate(("x", "y")):
            s_ = rui[rui.eje == e_].set_index("fuente").loc[fuentes]
            xs = np.arange(len(fuentes)) + (2 * k + j - 1.5) * w
            ax.errorbar(xs, s_[met], yerr=[s_[met] - s_[f"{met}_lo"], s_[f"{met}_hi"] - s_[met]], fmt="o" if j == 0 else "s",
                        color=c, ms=4, lw=1, capsize=2, label=f"{'ordenada del MSD' if k == 0 else 'pasos - 2 D dt'}, {e_}")
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_xticks(np.arange(len(fuentes)))
    ax.set_xticklabels([f.replace("pipeline final (pista)", "pipeline\nfinal").replace("red umbral", "red\numbral") for f in fuentes],
                       fontsize=7)
    ax.set_ylabel("sigma de localizacion [nm] (negativo: varianza < 0)")
    ax.grid(True, axis="y", **grid); ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("ruido de localizacion (con el termino de desenfoque; IC 95 %)", fontsize=8.5)
    fig.suptitle("Yeast6_1: sesgo entre la red y la deteccion clasica y ruido de localizacion de cada una", fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_yeast6_1_ruido.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -- z(t) con atipicos (k = 3)
    zc = trays["clasico"][1]
    zf_ = trays["final"][1]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[2.4, 1]))
    ax = axs[0]
    ax.fill_between(fr, zc[40.0], zc[50.0], color=COLOR["clasico"], alpha=0.12, lw=0, label="par clasico, beta 40-50")
    zcl = zc[BETA_REF]
    atc = atipicos(zcl, K_ATIP)[0]
    ax.plot(fr, zcl, "-", color=COLOR["clasico"], lw=1, label="par clasico, beta 45")
    ax.plot(fr[atc], zcl[atc], "x", color=COLOR["clasico"], ms=5, mew=1, label=f"atipicos del clasico (k = {K_ATIP:g})")
    za = zf_[BETA_REF]
    ata, meda, _ = atipicos(za, K_ATIP)
    ax.plot(fr, za, "o", ms=3.5, color=COLOR["tri"], label="pipeline final, beta 45")
    ax.plot(fr, meda, "-", color=COLOR["tri"], lw=0.9, alpha=0.7, label=f"mediana movil ({VENT_MED} cuadros)")
    ax.plot(fr[ata], za[ata], "o", ms=8, mfc="none", mec=COLOR["atipico"], mew=1.3, label=f"atipicos del pipeline (k = {K_ATIP:g})")
    ax.set_xlabel("cuadro (10 Hz)"); ax.set_ylabel("z del real [nm] (+z hacia el objetivo; linea en el borde de la franja)")
    ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = axs[1]
    for fuente, c, mk in (("clasico", COLOR["clasico"], "s"), ("final", COLOR["tri"], "o")):
        z_ = zt[zt.fuente == fuente]
        for k, ls in zip(KS_ATIP, (":", "-", "--")):
            ax.plot(z_.beta, 100 * z_[f"atipicos_k{k:g}"] / z_.cuadros, ls + mk, color=c, ms=3, lw=1.1 if k == K_ATIP else 0.8,
                    mfc="none", label=f"{NOMBRE_VAR[fuente]}, k = {k:g} ({int(z_.cuadros.iloc[0])} cuadros)")
    ax.set_xlabel("beta [grados]"); ax.set_ylabel("cuadros rechazados [%]")
    ax.set_ylim(bottom=0); ax.grid(True, **grid); ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Yeast6_1: z triangulada por cuadro y atipicos por consistencia temporal (k = 3; no por la z de la red)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_yeast6_1_z.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -- trayectoria 3D
    zw_r = np.where(ata, np.nan, za) / 1000
    zcl_r = np.where(atc, np.nan, zcl) / 1000
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1.2, 0.04], wspace=0.5)
    xc, yc = cl.x_real.mean(), cl.y_real.mean()
    Xc, Yc = (cl.x_real - xc) * px / 1000, (cl.y_real - yc) * px / 1000
    Xa, Ya = (ad.x_tri_px - xc) * px / 1000, (ad.y_tri_px - yc) * px / 1000
    et = "pipeline final"
    ax = fig.add_subplot(gs[0], projection="3d")
    ax.plot(Xc, Yc, zcl_r, "-", color=COLOR["clasico"], lw=0.7, alpha=0.6, label="par clasico")
    sc = ax.scatter(Xa, Ya, zw_r, c=fr, cmap="viridis", s=10, vmin=0, vmax=T - 1, label=et)
    ax.set_xlabel("x [um]", labelpad=2); ax.set_ylabel("y [um]", labelpad=2); ax.set_zlabel("z [um]", labelpad=2)
    ax.tick_params(labelsize=6, pad=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("trayectoria 3D", fontsize=8.5)
    ax = fig.add_subplot(gs[1])
    ax.plot(Xc, Yc, "-", color=COLOR["clasico"], lw=0.7, alpha=0.6, label="par clasico")
    ax.scatter(Xa, Ya, c=fr, cmap="viridis", s=10, vmin=0, vmax=T - 1, label=et)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]"); ax.grid(True, **grid); ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("proyeccion lateral (los puntos sueltos son anillos)", fontsize=8.5)
    ax = fig.add_subplot(gs[2])
    ax.plot(fr, zcl_r, "-", color=COLOR["clasico"], lw=0.8, alpha=0.7, label="par clasico")
    ax.scatter(fr, zw_r, c=fr, cmap="viridis", s=10, vmin=0, vmax=T - 1, label=et)
    ax.set_xlabel("cuadro (10 Hz)"); ax.set_ylabel("z [um]"); ax.grid(True, **grid); ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("z en el tiempo (sin atipicos)", fontsize=8.5)
    fig.colorbar(sc, cax=fig.add_subplot(gs[3]), label="cuadro")
    fig.suptitle("Yeast6_1: trayectoria 3D de la particula (x, y de la red; z triangulada con beta 45, linea en el borde de la franja)",
                 fontsize=10)
    fig.savefig(SALIDA / "fig_tracking_yeast6_1_trayectoria.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -- MSD por eje y D_z(beta)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    dims = {"lateral": 2, "z": 1, "3d": 3}
    cols = {"lateral": COLOR["lat"], "z": COLOR["z"], "3d": COLOR["3d"]}
    tau = DT * np.arange(1, N_LAGS + 1)
    sel = lambda f, e: td[(td.fuente == f) & (td.beta == BETA_REF) & (td.k_atipicos == K_ATIP) & (td.eje == e)]  # noqa: E731
    for ax, fuente in zip(axs[:2], ("final", "clasico")):
        for e_, d_ in dims.items():
            m_ = tm[(tm.fuente == fuente) & (tm.eje == e_)].sort_values("lag")
            f_ = sel(fuente, e_).iloc[0]
            ax.plot(tau, m_.msd_nm2 / d_ / 1e3, "o", color=cols[e_], ms=5)
            ax.plot(tau, (2 * f_.D_um2_s * 1e6 * tau + f_.ordenada_nm2 / d_) / 1e3, "-", color=cols[e_], lw=1.5,
                    label=f"{e_}: D = {f_.D_um2_s:.3f} [{f_.lo:.3f}, {f_.hi:.3f}] um^2/s")
        ax.set_xlabel("lag [s]"); ax.set_ylabel("MSD por eje [10^3 nm^2]")
        ax.set_xlim(0, tau[-1] * 1.08); ax.set_ylim(bottom=0)
        ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_title(f"{NOMBRE_VAR[fuente]}\n(beta 45, atipicos con k = 3; IC 95 % bootstrap de bloques)", fontsize=8)
    ax = axs[2]
    for fuente, c, mk in (("clasico", COLOR["clasico"], "s"), ("final", COLOR["tri"], "o")):
        f_ = td[(td.fuente == fuente) & (td.k_atipicos == K_ATIP) & (td.eje == "z")].sort_values("beta")
        ax.fill_between(f_.beta, f_.lo, f_.hi, color=c, alpha=0.12, lw=0)
        ax.plot(f_.beta, f_.D_um2_s, "-" + mk, color=c, ms=3.5, lw=1.2, label=f"D_z, {NOMBRE_VAR[fuente]}")
        fl = sel(fuente, "lateral").iloc[0]
        ax.axhline(fl.D_um2_s, color=c, ls=":", lw=1.2, label=f"D lateral, {NOMBRE_VAR[fuente]}: {fl.D_um2_s:.3f}")
    ax.set_xlabel("beta [grados]"); ax.set_ylabel("D [um^2/s]")
    ax.set_ylim(bottom=0); ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7)
    ax.set_title("D_z en funcion de beta (banda: IC 95 %)", fontsize=8.5)
    fig.suptitle("Yeast6_1: D por MSD (lags 1-4 con ordenada) en 3D, lateral y z", fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_yeast6_1_msd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("partes", nargs="+", choices=("localizar", "analisis"))
    p.add_argument("--tif", default=TIF_DEF)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if "localizar" in a.partes:
        localizar(a)
    if "analisis" in a.partes:
        analisis(a)


if __name__ == "__main__":
    main()
