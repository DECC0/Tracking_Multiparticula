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
    analisis   1. x, y del real: red frente a la deteccion clasica, por cuadro
               2. pistas de la red (enlace 2D elegido en tracking_3d), pistas de
                  reflejo y emparejamiento real <-> reflejo, con D y sigma_dt
                  auto-calibrados en dos pases, en tres variantes: tal_cual (la de
                  tracking_3d), adaptada (umbral 0.2, anillos de la PSF real
                  suprimidos, sin las detecciones que parecen el reflejo) y
                  adaptada_ancha (sigma_dt x 2); comparacion con el par clasico
               3. z triangulada por cuadro (beta 45 y 40-50) y rechazo de atipicos
                  por consistencia temporal (mediana movil), no por la z de la red
               4. trayectoria 3D y D por MSD (lags 1-4 con ordenada) en 3D, lateral
                  y z, IC por bootstrap de bloques de cuadros; D_z(beta)
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
DT = 0.1                          # s (10 Hz, exposicion 100 ms)
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
               v_borde_px=float(fr["v_borde"]), ventana_x0=x0, ventana_y0=y0, ventana=VENTANA)
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
KS_ATIP = (2.5, 3.0, 4.0)
N_LAGS = 4
BLOQUE = 10                       # cuadros por bloque del bootstrap
N_BOOT = 2000
D_INI = 0.2                       # um^2/s: D del primer enlace (despues se usa el medido)
D_MIN = 0.01                      # um^2/s: piso del D medido para el enlace
N_ITER = 6                        # pases de la auto-calibracion (se usa el segundo; el resto es diagnostico)
SIG_DT_INI = 200.0                # nm: sigma_dt del primer emparejamiento (despues el medido)
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
VARIANTES = {"tal_cual": dict(umbral=UMBRAL, anillos=False),
             "adaptada": dict(umbral=UMBRAL_MIN, anillos=True),
             "adaptada_ancha": dict(umbral=UMBRAL_MIN, anillos=True, factor_sigma=2.0)}   # sigma_dt x 2 (sensibilidad)
NOMBRE_VAR = {"tal_cual": "tal cual (tracking_3d)", "adaptada": "adaptada (anillos suprimidos)",
              "adaptada_ancha": "adaptada, sigma_dt x 2", "clasico": "par clasico"}


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


def pipeline(var, det_todo, man, cl, geo, t3, filas):
    """Pistas de la red y de reflejos, emparejamiento y trayectoria de la particula para una variante."""
    from enlace import CostoNormalizado, enlazar, filtrar_cortas, _renumerar
    from espejo import coordenadas_espejo, triangular_z, emparejar_pistas
    from metricas import a_arreglo
    T, px, phi, q = geo["T"], geo["px"], geo["phi_deg"], np.array(geo["q_px"])
    cfg = VARIANTES[var]
    R = lambda k, v: filas.append(dict(variante=var, seccion="2", magnitud=k, valor=v))  # noqa: E731
    det = det_todo[det_todo.conf >= cfg["umbral"]].reset_index(drop=True)
    n_refl = 0
    if cfg["anillos"]:
        det = suprimir_anillos(det)
        pr_ = parece_reflejo(det, phi, q, px, t3)       # el reflejo detectado como real: no es candidato a real
        n_refl = int(pr_.sum())
        det = det[~pr_].reset_index(drop=True)
    man = man.reset_index(drop=True)
    km = t3.filtrar_manchas(man, det, phi, q, px)
    if cfg["anillos"]:
        km &= lejos_de(man, det, R_ANILLO_PX)
    cand = man[km].reset_index(drop=True)
    st = lambda x, y: coordenadas_espejo(x, y, px, phi, q)  # noqa: E731
    xr, yr = cl.x_real.to_numpy(), cl.y_real.to_numpy()

    def correr(D_um2, sig_dt):
        s_lat = sig_dt / np.sqrt(2)                 # reparto igual entre real y reflejo (solo fija el costo)
        c = CostoNormalizado([s_lat, s_lat], D_um2 * 1e6, DT, t3.R_CHI, c_loc=2.0)
        p = enlazar(det.frame.to_numpy(), np.c_[det.x_px * px, det.y_px * px], max_hueco=HUECO, costo=c)
        k = filtrar_cortas(p, LARGO)
        dr, p = det[k].reset_index(drop=True), _renumerar(p[k])
        pf = enlazar(cand.frame.to_numpy(), np.c_[cand.x_px * px, cand.y_px * px], max_hueco=HUECO, costo=c)
        kf = filtrar_cortas(pf, LARGO)
        dfc, pf = cand[kf].reset_index(drop=True), _renumerar(pf[kf])
        Rr, _ = a_arreglo(p, dr.frame.to_numpy(), np.c_[st(dr.x_px, dr.y_px)], T)
        Ff, _ = a_arreglo(pf, dfc.frame.to_numpy(), np.c_[st(dfc.x_px, dfc.y_px)], T)
        X, _ = a_arreglo(p, dr.frame.to_numpy(), np.c_[dr.x_px, dr.y_px], T)
        Xf, _ = a_arreglo(pf, dfc.frame.to_numpy(), np.c_[dfc.x_px, dfc.y_px], T)

        def z_par(i, j, ok):
            return t3.Z_POSIBLE[0] <= np.median(triangular_z(Rr[i, ok, 0], Ff[j, ok, 0], BETA_REF)) <= t3.Z_POSIBLE[1]
        pares, _ = emparejar_pistas(Rr, Ff, sig_dt, z_par=z_par)
        # particula = componente conexo de pistas reales y de reflejo unidas por pares con mas cuadros
        # triangulados (criterio del pipeline, sin el par clasico)
        padre = {}

        def raiz(u):
            while padre.setdefault(u, u) != u:
                u = padre[u]
            return u
        for pr in pares:
            padre[raiz(("r", pr["i"]))] = raiz(("f", pr["j"]))
        peso = {}
        for pr in pares:
            peso[raiz(("r", pr["i"]))] = peso.get(raiz(("r", pr["i"])), 0) + int(pr["cuadros"].sum())
        comp = max(peso, key=peso.get) if peso else None
        reales_c = sorted({pr["i"] for pr in pares if raiz(("r", pr["i"])) == comp})
        tray = {c: np.full(T, np.nan) for c in ("x_px", "y_px", "x_tri_px", "y_tri_px", "x_refl_px", "y_refl_px")}
        for i in reales_c:                                  # lateral: todos los cuadros de las pistas reales
            ok = np.isfinite(X[i, :, 0]) & np.isnan(tray["x_px"])
            tray["x_px"][ok], tray["y_px"][ok] = X[i, ok, 0], X[i, ok, 1]
        for pr in pares:
            if raiz(("r", pr["i"])) == comp:
                ok = pr["cuadros"]
                tray["x_tri_px"][ok], tray["y_tri_px"][ok] = X[pr["i"], ok, 0], X[pr["i"], ok, 1]
                tray["x_px"][ok], tray["y_px"][ok] = X[pr["i"], ok, 0], X[pr["i"], ok, 1]
                tray["x_refl_px"][ok], tray["y_refl_px"][ok] = Xf[pr["j"], ok, 0], Xf[pr["j"], ok, 1]
        for pr in pares:
            pr["particula"] = raiz(("r", pr["i"])) == comp
        cerca = np.hypot(X[..., 0] - xr, X[..., 1] - yr) <= R_ASOC_PX
        i_p = reales_c
        return dict(dr=dr, p=p, dfc=dfc, pf=pf, X=X, Xf=Xf, R=Rr, F=Ff, pares=pares, i_p=i_p, cerca=cerca,
                    tray=pd.DataFrame(tray))

    def medir(o):
        """D lateral de la particula (sin atipicos laterales, >= D_MIN) y sigma_dt = 1.4826 MAD de t_R - t_F en
        todos los cuadros comunes de sus pares: no solo los consistentes (el corte |dt| < 3 sigma lo encogeria en
        cada iteracion) y robusta (los cuadros en que la pista real esta sobre un anillo lo inflarian)."""
        tr = o["tray"]
        xy = np.c_[tr.x_px, tr.y_px] * px
        at = atipicos(xy[:, 0])[0] | atipicos(xy[:, 1])[0] if np.isfinite(xy[:, 0]).sum() > 5 else np.zeros(T, bool)
        D = ajuste_msd(*sumas_msd(np.where(at[:, None], np.nan, xy)), [0, 1])[0]
        dts = []
        for pr in o["pares"]:
            if pr["particula"]:
                ok = np.isfinite(o["R"][pr["i"], :, 1]) & np.isfinite(o["F"][pr["j"], :, 1])
                dts.append(o["R"][pr["i"], ok, 1] - o["F"][pr["j"], ok, 1])
        dt = np.concatenate(dts) if dts else np.zeros(0)
        return (max(D, D_MIN) if np.isfinite(D) else np.nan), (estad(dt)["std_robusta"] if len(dt) > 5 else np.nan)

    # auto-calibracion sin el par clasico, en dos pases: el primero con D_INI y SIG_DT_INI, el segundo con el D
    # lateral y el sigma_dt medidos en el primero (es el que se usa). Las iteraciones siguientes son diagnostico:
    # cada emparejamiento elige los cuadros con dt chico, asi que sigma_dt se estrecha o oscila.
    it_f, D_e, sig_dt, usado = [], D_INI, SIG_DT_INI, None
    for it in range(N_ITER):
        o = correr(D_e, sig_dt)
        D_n, s_n = medir(o)
        it_f.append(dict(variante=var, iteracion=it, usada=it == 1, D_enlace=D_e, sigma_dt_enlace=sig_dt, pistas_real=len(o["X"]),
                         pistas_reflejo=len(o["Xf"]), pares=len(o["pares"]), cuadros_pista=int(o["tray"].x_px.notna().sum()),
                         cuadros_con_reflejo=int(o["tray"].x_refl_px.notna().sum()), D_lat_medido=D_n, sigma_dt_medido=s_n))
        if it == 1:
            usado = (o, D_e, sig_dt)
        if not (np.isfinite(s_n) and np.isfinite(D_n)):
            break
        D_e, sig_dt = D_n, s_n
    o, D_e, sig_dt = usado
    if cfg.get("factor_sigma"):
        sig_dt *= cfg["factor_sigma"]
        o = correr(D_e, sig_dt)
    sens = []
    for Dx in (0.02, 0.2, 1.5):
        for sx in (0.5 * sig_dt, sig_dt, 2 * sig_dt):
            oo = correr(Dx, sx)
            sens.append(dict(variante=var, D_enlace=Dx, sigma_dt=sx, pistas_real=len(oo["X"]), pistas_reflejo=len(oo["Xf"]),
                             pares=len(oo["pares"]), cuadros_pista=int(oo["tray"].x_px.notna().sum()),
                             cuadros_con_reflejo=int(oo["tray"].x_refl_px.notna().sum()),
                             reflejo_sobre_clasico=int((np.hypot(oo["tray"].x_refl_px - cl.x_refl, oo["tray"].y_refl_px - cl.y_refl)
                                                        <= R_REFL_PX).sum()),
                             real_tri_sobre_clasico=int((np.hypot(oo["tray"].x_tri_px - xr, oo["tray"].y_tri_px - yr)
                                                         <= R_ASOC_PX).sum())))
    tr = o["tray"]
    R("detecciones usadas por cuadro", len(det) / T)
    R("detecciones descartadas por parecer el reflejo, por cuadro", n_refl / T)
    R("cuadros con una deteccion a <= 3 px del real clasico", int((mas_cercano(det, xr, yr, R_ASOC_PX) >= 0).sum()))
    R("manchas candidatas por cuadro", len(cand) / T)
    R("candidatas a <= 2 px del reflejo clasico: cuadros",
      int((mas_cercano(cand, cl.x_refl.to_numpy(), cl.y_refl.to_numpy(), R_REFL_PX) >= 0).sum()))
    R("D del enlace final [um^2/s]", D_e)
    R("sigma_dt final [nm]", sig_dt)
    R("pistas de la red (>= largo)", len(o["X"]))
    R("pistas de reflejo (>= largo)", len(o["Xf"]))
    R("pares", len(o["pares"]))
    R("pistas reales de la particula (componente)", len(o["i_p"]))
    R("cuadros de la particula (lateral)", int(tr.x_px.notna().sum()))
    R("de ellos, a <= 3 px del real clasico", int((np.hypot(tr.x_px - xr, tr.y_px - yr) <= R_ASOC_PX).sum()))
    con = tr.x_refl_px.notna()
    R("cuadros triangulados (con reflejo emparejado)", int(con.sum()))
    R("de ellos, real a <= 3 px del real clasico", int((np.hypot(tr.x_tri_px - xr, tr.y_tri_px - yr)[con] <= R_ASOC_PX).sum()))
    R("de ellos, reflejo a <= 2 px del reflejo clasico",
      int((np.hypot(tr.x_refl_px - cl.x_refl, tr.y_refl_px - cl.y_refl)[con] <= R_REFL_PX).sum()))
    R("cuadros sin triangular", " ".join(map(str, np.flatnonzero(~con))) or "-")
    pistas = []
    for k in range(len(o["X"])):
        ok = np.isfinite(o["X"][k, :, 0])
        pistas.append(dict(variante=var, tipo="real", pista=k, cuadros=int(ok.sum()), es_particula=k in o["i_p"],
                           dist_real_px=np.nanmedian(np.hypot(o["X"][k, ok, 0] - xr[ok], o["X"][k, ok, 1] - yr[ok])),
                           dist_reflejo_px=np.nanmedian(np.hypot(o["X"][k, ok, 0] - cl.x_refl[ok], o["X"][k, ok, 1] - cl.y_refl[ok])),
                           emparejada=any(pr["i"] == k for pr in o["pares"])))
    for k in range(len(o["Xf"])):
        ok = np.isfinite(o["Xf"][k, :, 0])
        pistas.append(dict(variante=var, tipo="reflejo", pista=k, cuadros=int(ok.sum()),
                           es_particula=any(pr["j"] == k and pr["particula"] for pr in o["pares"]),
                           dist_real_px=np.nanmedian(np.hypot(o["Xf"][k, ok, 0] - xr[ok], o["Xf"][k, ok, 1] - yr[ok])),
                           dist_reflejo_px=np.nanmedian(np.hypot(o["Xf"][k, ok, 0] - cl.x_refl[ok], o["Xf"][k, ok, 1] - cl.y_refl[ok])),
                           emparejada=any(pr["j"] == k for pr in o["pares"])))
    pares = pd.DataFrame([dict(variante=var, **{k: v for k, v in pr.items() if k != "cuadros"}) for pr in o["pares"]])
    return dict(o=o, det=det, cand=cand, iter=pd.DataFrame(it_f), sens=pd.DataFrame(sens), pistas=pd.DataFrame(pistas),
                pares=pares, sig_dt=sig_dt, D=D_e)


def analisis(a):
    import tracking_3d as t3
    from espejo import coordenadas_espejo, triangular_z, sigma_z_triangulada
    global HUECO, LARGO
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
    st = lambda x, y: coordenadas_espejo(x, y, px, phi, q)  # noqa: E731  (s, t) [nm]
    sR_c, tR_c = st(cl.x_real, cl.y_real)
    sF_c, tF_c = st(cl.x_refl, cl.y_refl)
    filas = []

    # ---- 1. x, y del real: red por cuadro (umbral 0.4 y 0.2, y tras suprimir anillos) frente al clasico
    comp = []
    entradas = [(f"umbral {UMBRAL:g}", det_todo[det_todo.conf >= UMBRAL]), (f"umbral {UMBRAL_MIN:g}", det_todo[det_todo.conf >= UMBRAL_MIN]),
                (f"umbral {UMBRAL_MIN:g}, anillos suprimidos", suprimir_anillos(det_todo[det_todo.conf >= UMBRAL_MIN].reset_index(drop=True)))]
    for nombre, d in entradas:
        d = d.reset_index(drop=True)
        i = mas_cercano(d, cl.x_real.to_numpy(), cl.y_real.to_numpy(), R_ASOC_PX)
        ok = i >= 0
        xn, yn = d.x_px.to_numpy()[i[ok]], d.y_px.to_numpy()[i[ok]]
        sN, tN = st(xn, yn)
        ex, ey = (xn - cl.x_real[ok].to_numpy()) * px, (yn - cl.y_real[ok].to_numpy()) * px
        R = lambda k, v: filas.append(dict(variante=nombre, seccion="1", magnitud=k, valor=v))  # noqa: E731
        R("detecciones por cuadro", len(d) / T)
        R("cuadros con el real detectado (<= 3 px)", int(ok.sum()))
        R("cuadros sin el real", " ".join(map(str, np.flatnonzero(~ok))) or "-")
        R("otras detecciones por cuadro", (len(d) - ok.sum()) / T)
        R("otras a 3-20 px del real (anillos) por cuadro",
          float(sum(((np.hypot(g.x_px - cl.x_real[f], g.y_px - cl.y_real[f]) > R_ASOC_PX) &
                     (np.hypot(g.x_px - cl.x_real[f], g.y_px - cl.y_real[f]) <= 20)).sum() for f, g in d.groupby("frame")) / T))
        R("otras a <= 3 px del reflejo clasico por cuadro",
          float((mas_cercano(d, cl.x_refl.to_numpy(), cl.y_refl.to_numpy(), R_ASOC_PX) >= 0).mean()))
        for e_, v in (("x", ex), ("y", ey), ("s", sN - sR_c[ok]), ("t", tN - tR_c[ok]), ("radial", np.hypot(ex, ey))):
            comp.append(dict(entrada=nombre, eje=e_, **estad(v)))
        # ruido a lo largo de t con el reflejo clasico como referencia (t' = t): std de dt y de su paso
        dt = tN - tF_c[ok]
        comp.append(dict(entrada=nombre, eje="dt con el reflejo clasico", **estad(dt)))
    dtc = tR_c - tF_c
    z45 = triangular_z(sR_c, sF_c, BETA_REF)
    pend_dt_z = np.polyfit(z45, dtc, 1)[0]
    for nombre, v in (("corr(dt, z triangulada 45)", np.corrcoef(dtc, z45)[0, 1]), ("pendiente d(dt)/dz", pend_dt_z),
                      ("corr(dt, s del real)", np.corrcoef(dtc, sR_c)[0, 1]), ("corr(dt, cuadro)", np.corrcoef(dtc, np.arange(T))[0, 1]),
                      ("std de dt sin la parte lineal en z [nm]", np.std(dtc - np.polyval(np.polyfit(z45, dtc, 1), z45), ddof=1))):
        filas.append(dict(variante="clasico", seccion="2", magnitud=nombre, valor=v))
    comp.append(dict(entrada="clasico", eje="dt con el reflejo clasico", **estad(dtc)))
    comp.append(dict(entrada="clasico", eje="paso de dt / raiz 2", **estad(np.diff(dtc) / np.sqrt(2))))
    comp = pd.DataFrame(comp)
    comp.to_csv(SALIDA / "tracking_yeast6_1_xy_red_vs_clasico.csv", index=False)

    # ---- 2. pipeline: pistas, reflejos y emparejamiento (dos variantes)
    res = {v: pipeline(v, det_todo, man, cl, geo, t3, filas) for v in VARIANTES}
    for k, nombre in (("iter", "autocalibracion"), ("sens", "sensibilidad_enlace"), ("pistas", "pistas"), ("pares", "pares")):
        pd.concat([r[k] for r in res.values()], ignore_index=True).to_csv(SALIDA / f"tracking_yeast6_1_{nombre}.csv", index=False)
    comp_refl = []
    for v, r in res.items():
        tr = r["o"]["tray"]
        sFp, tFp = st(tr.x_refl_px, tr.y_refl_px)
        sRp, tRp = st(tr.x_tri_px, tr.y_tri_px)
        for e_, val in (("x", (tr.x_refl_px - cl.x_refl) * px), ("y", (tr.y_refl_px - cl.y_refl) * px), ("s", sFp - sF_c),
                        ("t", tFp - tF_c), ("radial", np.hypot(tr.x_refl_px - cl.x_refl, tr.y_refl_px - cl.y_refl) * px),
                        ("real de la pista - clasico, s", sRp - sR_c), ("real de la pista - clasico, t", tRp - tR_c),
                        ("dt de la pista", tRp - tFp)):
            comp_refl.append(dict(variante=v, eje=e_, **estad(val)))
    comp_refl = pd.DataFrame(comp_refl)
    comp_refl.to_csv(SALIDA / "tracking_yeast6_1_reflejo_vs_clasico.csv", index=False)

    # ---- 3. z triangulada por cuadro, barrido de beta, atipicos por consistencia temporal
    fuentes = {"clasico": (np.c_[cl.x_real, cl.y_real], np.c_[cl.x_refl, cl.y_refl], np.c_[cl.x_real, cl.y_real])}
    for v, r in res.items():
        tr = r["o"]["tray"]
        fuentes[v] = (np.c_[tr.x_tri_px, tr.y_tri_px], np.c_[tr.x_refl_px, tr.y_refl_px], np.c_[tr.x_px, tr.y_px])
    pc = dict(frame=np.arange(T), x_real_clasico_px=cl.x_real, y_real_clasico_px=cl.y_real, x_refl_clasico_px=cl.x_refl,
              y_refl_clasico_px=cl.y_refl)
    for v in VARIANTES:
        tr = res[v]["o"]["tray"]
        for c in tr:
            pc[f"{c}_{v}"] = tr[c]
    zf, tray = [], {}
    for fuente, (pr, pf, _) in fuentes.items():
        sr, _ = st(pr[:, 0], pr[:, 1])
        sf, _ = st(pf[:, 0], pf[:, 1])
        sig_s = res["adaptada"]["sig_dt"] / np.sqrt(2)
        for b in BETAS:
            z = triangular_z(sr, sf, b)
            n = int(np.isfinite(z).sum())
            fila = dict(fuente=fuente, beta=b, cuadros=n, z_media=np.nanmean(z) if n else np.nan,
                        z_std=np.nanstd(z) if n else np.nan, sigma_z_propagado=float(sigma_z_triangulada(sig_s, sig_s, b)))
            for k in KS_ATIP:
                fila[f"atipicos_k{k:g}"] = int(atipicos(z, k)[0].sum()) if n > 5 else 0
            at, med, sg = atipicos(z, K_ATIP) if n > 5 else (np.zeros(T, bool), np.full(T, np.nan), np.nan)
            fila["sigma_residuo"] = sg
            fila["max_paso"] = float(np.nanmax(np.abs(np.diff(z)))) if n > 1 else np.nan
            zf.append(fila)
            tray[fuente, b] = (z, at, med)
            pc[f"z_{fuente}_b{b:g}_nm"] = z
            pc[f"atipico_{fuente}_b{b:g}"] = at
    zt = pd.DataFrame(zf)
    zt.to_csv(SALIDA / "tracking_yeast6_1_z_beta.csv", index=False)
    # sensibilidad del rechazo a la ventana y a k (beta 45)
    sv = []
    for fuente in ("adaptada", "adaptada_ancha", "tal_cual", "clasico"):
        z = tray[fuente, BETA_REF][0]
        for w in (5, 9, 15):
            for k in KS_ATIP:
                med = mediana_movil(z, w)
                rr = z - med
                rv = rr[np.isfinite(rr)]
                sg = 1.4826 * np.median(np.abs(rv - np.median(rv)))
                sv.append(dict(fuente=fuente, ventana=w, k=k, sigma_residuo=sg, atipicos=int((np.abs(rr) > k * sg).sum())))
    sv = pd.DataFrame(sv)
    sv.to_csv(SALIDA / "tracking_yeast6_1_atipicos_ventana.csv", index=False)

    # ---- 4. trayectoria 3D y D por MSD
    dfil, msd_t = [], []
    for fuente, (pr, _, plat) in fuentes.items():
        if not np.isfinite(tray[fuente, BETA_REF][0]).any():
            continue
        xy, xy_lat = pr * px, plat * px
        for b in BETAS:
            z, at, _ = tray[fuente, b]
            for rech in (True, False):
                zz = np.where(at, np.nan, z) if rech else z
                r3 = np.where(np.isfinite(zz)[:, None], np.c_[xy, zz], np.nan)   # 3D y z: cuadros con z valida
                res_b = d_por_msd(r3, rng, ejes={"3d": [0, 1, 2], "z": [2]})
                if b == BETA_REF and rech:                                     # lateral: todos los cuadros de la pista
                    res_b.update(d_por_msd(np.c_[xy_lat, np.zeros(T)], rng, ejes={"lateral": [0, 1], "x": [0], "y": [1]}))
                for e_, val in res_b.items():
                    dfil.append(dict(fuente=fuente, beta=b, atipicos_rechazados=rech, eje=e_, D_um2_s=val["D"], lo=val["lo"],
                                     hi=val["hi"], ordenada_nm2=val["ordenada_nm2"], pares_lag1=int(val["pares"][0])))
                    if b == BETA_REF:
                        for k in range(N_LAGS):
                            msd_t.append(dict(fuente=fuente, atipicos_rechazados=rech, eje=e_, lag=k + 1,
                                              msd_nm2=val["msd"][k], pares=int(val["pares"][k])))
    td = pd.DataFrame(dfil)
    td.to_csv(SALIDA / "tracking_yeast6_1_D.csv", index=False)
    tm = pd.DataFrame(msd_t)
    tm.to_csv(SALIDA / "tracking_yeast6_1_msd.csv", index=False)
    pc = pd.DataFrame(pc)
    pc.to_csv(SALIDA / "tracking_yeast6_1_por_cuadro.csv", index=False)
    tf = pd.DataFrame(filas)
    tf.to_csv(SALIDA / "tracking_yeast6_1_resumen.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 400,
                           "display.max_colwidth", 120):
        f1 = lambda v: f"{v:.4g}"  # noqa: E731
        print(tf.to_string(index=False))
        for t_ in (comp, comp_refl, pd.concat([r["iter"] for r in res.values()]), pd.concat([r["sens"] for r in res.values()]),
                   pd.concat([r["pistas"] for r in res.values()]), pd.concat([r["pares"] for r in res.values()]),
                   zt[zt.beta.isin([40, 42.5, 45, 47.5, 50])], sv,
                   td[td.beta.isin([40, 45, 50]) | td.eje.isin(["lateral", "x", "y"])]):
            print(t_.to_string(index=False, float_format=f1))
            print()
    figuras(pc, zt, td, tm, tray, geo, res, cl)


# ----------------------------------------------------------------------------
# figuras
# ----------------------------------------------------------------------------

def figuras(pc, zt, td, tm, tray, geo, res, cl):
    from espejo import coordenadas_espejo
    plt = _estilo()
    px, phi, q, T = geo["px"], geo["phi_deg"], np.array(geo["q_px"]), geo["T"]
    x0, y0 = geo["ventana_x0"], geo["ventana_y0"]
    st = lambda x, y: coordenadas_espejo(x, y, px, phi, q)  # noqa: E731
    fr = np.arange(T)
    grid = dict(color="#e6e5e0", lw=0.6)
    ad = res["adaptada"]["o"]["tray"]

    # -- 1-2: deteccion, pistas y emparejamiento
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1.4, 1]))
    ax = axs[0]
    med = np.load(CACHE / "media_ventana.npy")
    ax.imshow(med, cmap="gray", vmin=np.percentile(med, 1), vmax=np.percentile(med, 99.7),
              extent=(x0 - 0.5, x0 + VENTANA - 0.5, y0 + VENTANA - 0.5, y0 - 0.5))
    det = pd.read_csv(CACHE / "detecciones_red.csv")
    d2 = det[det.conf >= UMBRAL_MIN]
    ax.scatter(d2.x_px, d2.y_px, s=3, color="#c9c8c2", lw=0, label="red, conf >= 0.2 (todos los cuadros)")
    sup = res["adaptada"]["det"]
    ax.scatter(sup.x_px, sup.y_px, s=4, color=COLOR["red"], lw=0, label="tras suprimir anillos")
    man = res["adaptada"]["cand"]
    ax.scatter(man.x_px, man.y_px, s=4, color=COLOR["reflejo"], lw=0, label="manchas candidatas a reflejo")
    ax.plot(cl.x_real, cl.y_real, "-", color="white", lw=0.8)
    ax.plot(cl.x_refl, cl.y_refl, "-", color="white", lw=0.8, label="par clasico")
    ok = ad.x_refl_px.notna()
    ax.scatter(ad.x_tri_px[ok], ad.y_tri_px[ok], s=7, marker="o", facecolor="none", edgecolor=COLOR["tri"], lw=0.7,
               label="pipeline: cuadros triangulados")
    ax.scatter(ad.x_refl_px[ok], ad.y_refl_px[ok], s=7, marker="o", facecolor="none", edgecolor=COLOR["tri"], lw=0.7)
    ax.set_xlim(x0 - 0.5, x0 + VENTANA - 0.5); ax.set_ylim(y0 + VENTANA - 0.5, y0 - 0.5)
    ax.set_xlabel("x [px, canal derecho]"); ax.set_ylabel("y [px]")
    ax.set_title("ventana de la red (media de los cuadros 0-9)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    ax = axs[1]
    sRc, tRc = st(cl.x_real, cl.y_real)
    for col, nombre, mk in (("tal_cual", "red, umbral 0.4 (tal cual)", "x"), ("adaptada", "red, anillos suprimidos", "o")):
        trr = res[col]["o"]["tray"]
        s_, t_ = st(trr.x_px, trr.y_px)
        ax.plot(fr, s_ - sRc, mk, ms=3, mfc="none", color=COLOR["red"] if col == "adaptada" else COLOR["gris"], lw=0,
                label=f"{nombre}: s - s clasico")
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_xlabel("cuadro (10 Hz)"); ax.set_ylabel("real de la pista - real clasico, a lo largo de m [nm]")
    ax.set_ylim(-1600, 1600)
    ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7)
    ax.set_title("posicion del real en la pista de la particula (los saltos de ~600-1500 nm son anillos)", fontsize=8.5)
    ax = axs[2]
    sFc, tFc = st(cl.x_refl, cl.y_refl)
    from espejo import triangular_z
    z45 = triangular_z(sRc, sFc, BETA_REF)
    dtc = tRc - tFc
    ax.scatter(z45, dtc, s=9, color=COLOR["clasico"], label="par clasico")
    p_ = np.polyfit(z45, dtc, 1)
    xx = np.array([z45.min(), z45.max()])
    ax.plot(xx, np.polyval(p_, xx), "-", color=COLOR["clasico"], lw=1.2,
            label=f"pendiente {p_[0]:+.2f}, r = {np.corrcoef(z45, dtc)[0, 1]:+.2f}")
    sRa, tRa = st(ad.x_tri_px, ad.y_tri_px)
    sFa, tFa = st(ad.x_refl_px, ad.y_refl_px)
    ax.scatter(triangular_z(sRa, sFa, BETA_REF), tRa - tFa, s=12, marker="o", facecolor="none", edgecolor=COLOR["tri"],
               label="pipeline (adaptada)")
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_xlabel("z triangulada, beta 45 [nm]"); ax.set_ylabel("dt = t real - t reflejo [nm]")
    ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7)
    ax.set_title("el modelo pide dt = 0 (t' = t): aqui depende de z", fontsize=8.5)
    fig.suptitle("Yeast6_1, canal derecho: red s1 frente a la deteccion clasica y emparejamiento real <-> reflejo", fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_yeast6_1_deteccion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -- 3: z(t) con atipicos
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[2.4, 1]))
    ax = axs[0]
    zc = {b: tray["clasico", b][0] for b in (40.0, 50.0)}
    ax.fill_between(fr, zc[40.0], zc[50.0], color=COLOR["clasico"], alpha=0.12, lw=0, label="par clasico, beta 40-50")
    zcl, atc, _ = tray["clasico", BETA_REF]
    ax.plot(fr, zcl, "-", color=COLOR["clasico"], lw=1, label="par clasico, beta 45")
    ax.plot(fr[atc], zcl[atc], "x", color=COLOR["clasico"], ms=5, mew=1, label=f"atipicos del clasico (k = {K_ATIP:g})")
    zw, atw, _ = tray["adaptada_ancha", BETA_REF]
    ax.plot(fr, zw, "s", ms=5, mfc="none", mec="#7fcfae", mew=0.8, label="pipeline, sigma_dt x 2, beta 45")
    ax.plot(fr[atw], zw[atw], "s", ms=8, mfc="none", mec=COLOR["atipico"], mew=0.8, alpha=0.6)
    za, ata, meda = tray["adaptada", BETA_REF]
    ax.plot(fr, za, "o", ms=3.5, color=COLOR["tri"], label="pipeline (adaptada), beta 45")
    ax.plot(fr, meda, "-", color=COLOR["tri"], lw=0.9, alpha=0.7, label=f"mediana movil ({VENT_MED} cuadros)")
    ax.plot(fr[ata], za[ata], "o", ms=8, mfc="none", mec=COLOR["atipico"], mew=1.3, label=f"atipicos del pipeline (k = {K_ATIP:g})")
    ax.set_xlabel("cuadro (10 Hz)"); ax.set_ylabel("z del real [nm] (+z hacia el objetivo; linea en el borde de la franja)")
    ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = axs[1]
    for fuente, c, mk in (("clasico", COLOR["clasico"], "s"), ("adaptada", COLOR["tri"], "o"), ("adaptada_ancha", "#7fcfae", "s"),
                          ("tal_cual", COLOR["gris"], "x")):
        z_ = zt[zt.fuente == fuente]
        if z_.cuadros.max() == 0:
            continue
        ax.plot(z_.beta, 100 * z_[f"atipicos_k{K_ATIP:g}"] / z_.cuadros, "-" + mk, color=c, ms=3.5, lw=1.2, mfc="none" if mk == "o" else c,
                label=f"{NOMBRE_VAR[fuente]} ({int(z_.cuadros.iloc[0])} cuadros)")
    ax.set_xlabel("beta [grados]"); ax.set_ylabel(f"cuadros rechazados (k = {K_ATIP:g}) [%]")
    ax.set_ylim(bottom=0); ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Yeast6_1: z triangulada por cuadro y rechazo de atipicos por consistencia temporal (no por la z de la red)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_yeast6_1_z.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -- 4: trayectoria 3D (variante con mas cuadros triangulados)
    aw = res["adaptada_ancha"]["o"]["tray"]
    zw_r = np.where(atw, np.nan, zw) / 1000
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1.2, 0.04], wspace=0.5)
    xc, yc = cl.x_real.mean(), cl.y_real.mean()
    zcl_r = np.where(atc, np.nan, zcl) / 1000
    Xc, Yc = (cl.x_real - xc) * px / 1000, (cl.y_real - yc) * px / 1000
    Xa, Ya = (aw.x_tri_px - xc) * px / 1000, (aw.y_tri_px - yc) * px / 1000
    et = "pipeline (adaptada, sigma_dt x 2)"
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

    # -- 5: MSD por eje y D_z(beta)
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.4))
    dims = {"lateral": 2, "z": 1, "3d": 3}
    cols = {"lateral": COLOR["lat"], "z": COLOR["z"], "3d": COLOR["3d"]}
    tau = DT * np.arange(1, N_LAGS + 1)
    for ax, fuente in zip(axs[:3], ("adaptada", "adaptada_ancha", "clasico")):
        for e_, d_ in dims.items():
            m_ = tm[(tm.fuente == fuente) & tm.atipicos_rechazados & (tm.eje == e_)].sort_values("lag")
            f_ = td[(td.fuente == fuente) & (td.beta == BETA_REF) & td.atipicos_rechazados & (td.eje == e_)].iloc[0]
            ax.plot(tau, m_.msd_nm2 / d_ / 1e3, "o", color=cols[e_], ms=5)
            ax.plot(tau, (2 * f_.D_um2_s * 1e6 * tau + f_.ordenada_nm2 / d_) / 1e3, "-", color=cols[e_], lw=1.5,
                    label=f"{e_}: D = {f_.D_um2_s:.3f} [{f_.lo:.3f}, {f_.hi:.3f}] um^2/s")
        ax.set_xlabel("lag [s]"); ax.set_ylabel("MSD por eje [10^3 nm^2]")
        ax.set_xlim(0, tau[-1] * 1.08); ax.set_ylim(bottom=0)
        ax.grid(True, **grid); ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_title(f"{NOMBRE_VAR[fuente]}\n(beta 45, sin atipicos; IC 95 % bootstrap de bloques)", fontsize=8)
    ax = axs[3]
    for fuente, c, mk in (("clasico", COLOR["clasico"], "s"), ("adaptada", COLOR["tri"], "o"), ("adaptada_ancha", "#7fcfae", "^")):
        f_ = td[(td.fuente == fuente) & td.atipicos_rechazados & (td.eje == "z")].sort_values("beta")
        ax.fill_between(f_.beta, f_.lo, f_.hi, color=c, alpha=0.10, lw=0)
        ax.plot(f_.beta, f_.D_um2_s, "-" + mk, color=c, ms=3.5, lw=1.2, label=f"D_z, {NOMBRE_VAR[fuente]}")
        fl = td[(td.fuente == fuente) & (td.eje == "lateral")].iloc[0]
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
