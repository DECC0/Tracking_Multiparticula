"""
Cuanto se gana combinando frames, con y sin el reflejo del espejo (sin
entrenar nada). Version oraculo: trayectorias del GT de simulador_nuevo
(videos de 50 frames con espejo, 64 x 96 px, alpha y phi por defecto, la
linea del espejo en el borde izquierdo del FOV, dist_max_espejo_um = 4), enlazadas por particle_id, mas ruido de localizacion
gaussiano con su covarianza conocida.

Modelo (suavizador de Kalman / RTS por trayectoria, todo en nm):
    estado      p_k (posicion 3D del real)
    dinamica    p_k+1 = p_k + w,  w ~ N(0, 2 D dt I)   (D por particula; sin paredes)
    medidas     real:     y_k  = p_k + e,             e  ~ N(0, diag(s_lat, s_lat, s_z)^2)
                reflejo:  y'_k = p0 + R (p_k - p0) + e', e' ~ N(0, diag(s'_lat, s'_lat, s'_z)^2)
                (solo si el real / el reflejo estan en el FOV y se "detectan")
Como R es una involucion, el reflejo equivale a una medida de p, R (y' - c)
con c = p0 - R p0, de covarianza R Sigma' R: cada frame suma la informacion
del real y la del reflejo.

Escenarios de ruido:
    crlb   real y reflejo con su CRLB (crlb_3d, emisor solo: fotones, z, fondo
           medio del video), detectados siempre que esten en el FOV.
    red    real con el RMSE lateral y z y el recall de la red con espejo de esa
           variante (localizacion/corridas/hypatia_espejo_b45_<variante>_*), por rango de
           fotones y de distancia al espejo; reflejo con su CRLB escalada por
           la ineficiencia de la red respecto de la CRLB del real a iguales
           fotones (kappa = RMSE red / CRLB por rango de fotones), detectado
           con el recall de la red a 0.8 N fotones. Es un localizador de
           reflejos hipotetico: la red no localiza reflejos.

Estimadores (se evaluan en los frames con el real en el FOV y detectado):
    frame_solo   la localizacion del real
    frame_conj   real + reflejo del mismo frame (suma de informacion)
    suav_solo    RTS con las localizaciones del real, D verdadera
    suav_conj    RTS con real + reflejo, D verdadera
    suav_*_Dml   D por maxima verosimilitud (verosimilitud de las innovaciones
                 del filtro sobre una rejilla de D, con refinamiento parabolico)
    suav_*_Dmsd  D por MSD lateral del real (lags 1-3, tolera huecos) restando
                 el ruido de localizacion conocido; sin pares suficientes, la
                 mediana de las demas trayectorias
Ademas: RMSE del suavizado en los frames con el real en el FOV pero no detectado.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/temporal_espejo.py [--videos 500] [--variantes lejos cerca] [--seed 0]
    $PY figuras/temporal_espejo.py --celula     # emisores en una celula de 5 um (geometria de configs/celula.yaml),
                                                # confinados vs libres desde las mismas posiciones (escenario crlb)
Convencion: +z hacia el objetivo, espejo a 45 grados del eje optico; variantes
de la PSF lejos / cerca (anillos_hacia_objetivo False / True). Sin la corrida
y el test de una variante (hypatia_espejo_b45_<variante>, test_espejo_b45_<variante>)
solo se corre el escenario crlb. Los resultados de resultados_temporal_espejo.md
son de la convencion vieja (alpha = 34 desde el plano focal, signo_z).
Escribe figuras/fig_temporal_espejo_b45.png/.pdf, figuras/temporal_espejo_b45_rmse.csv
(RMSE e IC por rango, estimador, escenario y variante) y
figuras/temporal_espejo_b45_D.csv (D estimada por trayectoria).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "simulador_nuevo"))
sys.path.insert(0, str(RAIZ / "figuras"))
import simulador as sim  # noqa: E402
import crlb_espejo as ce  # noqa: E402  (Fisher de un emisor: derivadas, var_parametros)

VARIANTES = {"lejos": False, "cerca": True}      # -> anillos_hacia_objetivo


def evaluacion_red(variante):
    """Carpeta de evaluacion de la corrida con espejo mas reciente de la variante, o None."""
    c = sorted((RAIZ / "localizacion/corridas").glob(f"hypatia_espejo_b45_{variante}_*/evaluacion_test_espejo_b45_{variante}"))
    return c[-1] if c else None
Z_TABLA = np.arange(-8000.0, 8000.1, 100.0)
N_TABLA = np.geomspace(300.0, 30000.0, 15)
B_TABLA = np.geomspace(4.0, 80.0, 7)
BORDES = {"D": np.array([0.01, 0.05, 0.1, 0.2, 0.3]),                  # um^2/s
          "fotones": np.geomspace(1000.0, 30000.0, 5),
          "dist": np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0, np.inf])}   # nm al plano del espejo
BORDES_RED_N = np.geomspace(1000.0, 30000.0, 9)                        # tabla de la red: 8 rangos de fotones
BORDES_RED_D = np.array([0.0, 1000.0, 2000.0, 3000.0, np.inf])
D_REJILLA = np.geomspace(1e-3, 1.0, 31)                                 # um^2/s, para la maxima verosimilitud
P0 = 1e10                                                               # nm^2: prior difuso
DT = sim.Config().dt_s                                                  # s entre frames (10 Hz)
METODOS = ["frame_solo", "frame_conj", "suav_solo", "suav_conj",
           "suav_solo_Dml", "suav_solo_Dmsd", "suav_conj_Dml", "suav_conj_Dmsd"]


# ----------------------------------------------------------------------------
# Trayectorias (las del GT de generar_video, sin renderizar imagenes)
# ----------------------------------------------------------------------------

DIAMETRO_CELULA_UM = 5.0


def config(variante, n_videos, seed, celula=False):
    if celula:
        # como configs/celula.yaml: 96 px, celula de 5 um con el centro en el foco y el contacto
        # celula-espejo en el centro del FOV (beta = 45, phi = 0, z_ref = 0)
        base = sim.Config()
        b, R, px = np.radians(base.angulo_espejo_eje_deg), DIAMETRO_CELULA_UM * 500.0, base.pixel_nm
        Q = np.array([47.5, 47.5]) * px
        z_q = 0.0 - R * np.sin(b)                                       # altura del contacto
        p0 = Q - (0.0 - z_q) * np.tan(b) * np.array([1.0, 0.0])
        c = Q + R * np.cos(b) * np.array([1.0, 0.0])
        return sim.Config(usar_espejo=True, usar_celula=True, n_frames=50, n_videos=n_videos, H=96, W=96,
                          z_rango_nm=(-2500.0, 2500.0), diametro_celula_um=DIAMETRO_CELULA_UM,
                          espejo_punto_px=tuple(p0 / px), celula_centro_px=tuple(c / px),
                          anillos_hacia_objetivo=VARIANTES[variante], seed=seed)
    # linea del espejo en el borde izquierdo del FOV (como espejo_linea_rango_fov al entrenar) y W = 96 px,
    # para que haya reales en el FOV a todas las distancias de 0 a 4 um
    return sim.Config(usar_espejo=True, n_frames=50, n_videos=n_videos, dist_max_espejo_um=4.0,
                      H=64, W=96, espejo_punto_px=(-0.5, 32.0), anillos_hacia_objetivo=VARIANTES[variante], seed=seed)


def trayectorias_libres(t, cfg, rng):
    """Las mismas posiciones iniciales y D que t, con browniano libre (sin la
    celula ni el espejo como paredes): la linea de base del confinamiento."""
    e, px = sim.Espejo(cfg), cfg.pixel_nm
    n, T = t["d"].shape
    pasos = rng.normal(size=(n, T, 3)) * (np.sqrt(2 * t["D"] * cfg.dt_s) * 1000.0)[:, None, None]
    pasos[:, 0] = 0.0
    p = t["p"][:, :1] + np.cumsum(pasos, axis=1)
    pr = e.reflejar_nm(p)
    u = dict(t, p=p, pr=pr, d=e.distancia_nm(p))
    u["fov"] = sim.en_fov(cfg, p[..., 0] / px, p[..., 1] / px)
    u["fovr"] = sim.en_fov(cfg, pr[..., 0] / px, pr[..., 1] / px)
    return u


def trayectorias(cfg, videos):
    """Mismos sorteos que generar_video (un flujo por componente): posiciones
    del real y del reflejo [nm] (n, T, 3), FOV, fotones, D, fondo medio, video."""
    e, px = sim.Espejo(cfg), cfg.pixel_nm
    out = {k: [] for k in ("p", "pr", "fov", "fovr", "N", "D", "b", "video", "pid")}
    for v in videos:
        r = sim.rngs_video(cfg.seed, v)
        n = int(r["muestra"].integers(cfg.n_emisores_rango[0], cfg.n_emisores_rango[1] + 1))
        tray, _, D = sim.generar_trayectorias(cfg, n, r["trayectorias"])
        N = sim.muestrear_fotones(cfg, n, r["fotones"])
        b = float(sim.generar_fondo(cfg, r["fondo"]).mean())
        trr = e.reflejar(tray)
        out["p"].append(e.a_nm(tray)); out["pr"].append(e.a_nm(trr))
        out["fov"].append(sim.en_fov(cfg, tray[..., 0], tray[..., 1]))
        out["fovr"].append(sim.en_fov(cfg, trr[..., 0], trr[..., 1]))
        out["N"].append(N); out["D"].append(D); out["b"].append(np.full(n, b))
        out["video"].append(np.full(n, v)); out["pid"].append(np.arange(n))
    t = {k: np.concatenate(v) for k, v in out.items()}
    t["d"] = e.distancia_nm(t["p"])                                     # (n, T) distancia al plano del espejo
    t["R"], t["c"] = e.R, e.p0 - e.R @ e.p0                             # p' = c + R p
    return t


def verificar_trayectorias(cfg, t):
    """Las trayectorias coinciden con el GT de generar_video (videos 0 y 1)."""
    banco = sim.BancoPSF(cfg)
    px = cfg.pixel_nm
    for v in (0, 1):
        _, gt = sim.generar_video(cfg, banco, v)
        sel = t["video"] == v
        p = t["p"][sel].transpose(1, 0, 2).reshape(-1, 3)                # orden del GT: frame, particula
        pr = t["pr"][sel].transpose(1, 0, 2).reshape(-1, 3)
        assert np.allclose(p[:, :2] / px, gt[["x_px", "y_px"]], atol=1e-9) and np.allclose(p[:, 2], gt.z_nm, atol=1e-9)
        assert np.allclose(pr[:, :2] / px, gt[["x_reflejo_px", "y_reflejo_px"]], atol=1e-9)
        assert np.allclose(t["N"][sel][gt.particle_id], gt.photons) and np.array_equal(t["fov"][sel].T.ravel(), gt.en_fov)
    print("verificacion: trayectorias == GT de generar_video (videos 0 y 1)")


# ----------------------------------------------------------------------------
# Precision de localizacion
# ----------------------------------------------------------------------------

def tabla_crlb(banco):
    """crlb(N, z, b) -> (sigma lateral por eje, sigma z) [nm] de un emisor solo
    (crlb_3d), interpolada en (log N, z, log b)."""
    v, px = sim._var_lectura_e2(banco.cfg), banco.cfg.pixel_nm
    L = 2 * ce.MARGEN_PX + 1
    lat = np.empty((len(N_TABLA), len(Z_TABLA), len(B_TABLA)))
    zz = np.empty_like(lat)
    for j, z in enumerate(Z_TABLA):
        g, G = ce.derivadas(banco, ce.MARGEN_PX, ce.MARGEN_PX, z, (L, L))
        for i, N in enumerate(N_TABLA):
            for k, b in enumerate(B_TABLA):
                vv = ce.var_parametros(g, G, N, b, v)
                lat[i, j, k], zz[i, j, k] = np.sqrt(0.5 * (vv[0] + vv[1])) * px, np.sqrt(vv[2])
    ejes = (np.log(N_TABLA), Z_TABLA, np.log(B_TABLA))
    fl, fz = RegularGridInterpolator(ejes, np.log(lat)), RegularGridInterpolator(ejes, np.log(zz))

    def crlb(N, z, b):
        N, z, b = np.broadcast_arrays(N, z, b)
        pts = np.c_[np.log(np.clip(N, N_TABLA[0], N_TABLA[-1])).ravel(), np.clip(z, Z_TABLA[0], Z_TABLA[-1]).ravel(),
                    np.log(np.clip(b, B_TABLA[0], B_TABLA[-1])).ravel()]
        return np.exp(fl(pts)).reshape(N.shape), np.exp(fz(pts)).reshape(N.shape)
    return crlb


def indice(valores, bordes):
    return np.clip(np.digitize(valores, bordes[1:-1]), 0, len(bordes) - 2)


def tabla_red(variante, crlb):
    """De la evaluacion de la red con espejo sobre su test: RMSE lateral (por
    eje) y z y recall por (fotones, distancia), y kappa_lat, kappa_z por
    fotones (RMSE de la red / RMS de la CRLB de los mismos emisores)."""
    g = pd.read_csv(evaluacion_red(variante) / "gt_eval.csv")
    m = pd.read_csv(RAIZ / f"localizacion/datos/test_espejo_b45_{variante}/muestras.csv")[["video_id", "fondo_medio"]]
    g = g[g.en_fov].merge(m, on="video_id", how="left")
    i_n, i_d = indice(g.photons, BORDES_RED_N), indice(g.d_espejo_nm, BORDES_RED_D)
    nn, nd = len(BORDES_RED_N) - 1, len(BORDES_RED_D) - 1
    lat, z, rec = (np.full((nn, nd), np.nan) for _ in range(3))
    for a in range(nn):
        for c in range(nd):
            s = g[(i_n == a) & (i_d == c)]
            p = s[s.detectado]
            rec[a, c] = len(p) / len(s)
            lat[a, c] = np.sqrt(np.mean((p.dx_nm ** 2 + p.dy_nm ** 2) / 2))
            z[a, c] = np.sqrt(np.mean(p.dz_nm ** 2))
    det = g[g.detectado]
    cl, cz = crlb(det.photons.to_numpy(), det.z_nm.to_numpy(), det.fondo_medio.to_numpy())
    i_det = indice(det.photons, BORDES_RED_N)
    k_lat = np.array([np.sqrt(np.mean((det.dx_nm[i_det == a] ** 2 + det.dy_nm[i_det == a] ** 2) / 2))
                      / np.sqrt(np.mean(cl[i_det == a] ** 2)) for a in range(nn)])
    k_z = np.array([np.sqrt(np.mean(det.dz_nm[i_det == a] ** 2)) / np.sqrt(np.mean(cz[i_det == a] ** 2)) for a in range(nn)])
    rec_n = np.array([g.detectado[i_n == a].mean() for a in range(nn)])
    return dict(lat=lat, z=z, rec=rec, rec_n=rec_n, k_lat=k_lat, k_z=k_z)


def precisiones(t, escenario, crlb, red):
    """sigma lateral y z del real y del reflejo (n, T) y probabilidad de detectarlos."""
    n, T = t["d"].shape
    N, b = t["N"][:, None] * np.ones((1, T)), t["b"][:, None] * np.ones((1, T))
    rl, rz = crlb(0.8 * N, t["pr"][..., 2], b)                          # CRLB del reflejo
    if escenario == "crlb":
        sl, sz = crlb(N, t["p"][..., 2], b)
        return sl, sz, rl, rz, np.ones((n, T)), np.ones((n, T))
    a, c = indice(N, BORDES_RED_N), indice(t["d"], BORDES_RED_D)
    ar = indice(0.8 * N, BORDES_RED_N)
    return (red["lat"][a, c], red["z"][a, c], red["k_lat"][ar] * rl, red["k_z"][ar] * rz,
            red["rec"][a, c], red["rec_n"][ar])


def medir(t, sig, rng):
    """Localizaciones con ruido: informacion J (n, T, 3, 3) y vector b = J y del
    real solo y de real + reflejo, y la medida del real (y, detectado)."""
    sl, sz, rl, rz, pd_, pdr = sig
    n, T = sl.shape
    det = t["fov"] & (rng.random((n, T)) < pd_)
    detr = t["fovr"] & (rng.random((n, T)) < pdr)
    s_real = np.stack([sl, sl, sz], -1)
    s_ref = np.stack([rl, rl, rz], -1)
    y = t["p"] + rng.normal(size=(n, T, 3)) * s_real
    yr = t["pr"] + rng.normal(size=(n, T, 3)) * s_ref
    R, c = t["R"], t["c"]
    J_real = np.zeros((n, T, 3, 3))
    J_real[..., [0, 1, 2], [0, 1, 2]] = np.where(det[..., None], 1.0 / s_real ** 2, 0.0)
    b_real = np.einsum("ntij,ntj->nti", J_real, y)
    Jd = np.zeros((n, T, 3, 3))
    Jd[..., [0, 1, 2], [0, 1, 2]] = np.where(detr[..., None], 1.0 / s_ref ** 2, 0.0)
    J_ref = R @ Jd @ R                                                  # informacion de p: R Sigma'^-1 R
    p_ref = (yr - c) @ R.T                                              # R (y' - c)
    b_ref = np.einsum("ntij,ntj->nti", J_ref, p_ref)
    return dict(solo=(J_real, b_real), conj=(J_real + J_ref, b_real + b_ref), y=y, det=det, detr=detr, s_lat=sl)


# ----------------------------------------------------------------------------
# Kalman / RTS (todas las trayectorias a la vez)
# ----------------------------------------------------------------------------

def kalman(J, b, D, suavizar=True):
    """Filtro de informacion + RTS para p_k+1 = p_k + w, w ~ N(0, 2 D dt I).
    J, b (n, T, 3, 3), (n, T, 3); D (n,) [um^2/s]. Devuelve (estimacion
    suavizada (n, T, 3) o None, log-verosimilitud de las innovaciones (n,))."""
    n, T = J.shape[:2]
    q = 2.0 * np.asarray(D) * 1e6 * DT                                  # nm^2 por eje y por frame
    I = np.eye(3)
    x, P = np.zeros((n, 3)), np.broadcast_to(I * P0, (n, 3, 3)).copy()
    hay = np.abs(J).reshape(n, T, 9).sum(-1) > 0
    ll = np.zeros(n)
    xp, Pp, xf, Pf = (np.empty((n, T, 3)), np.empty((n, T, 3, 3)), np.empty((n, T, 3)), np.empty((n, T, 3, 3)))
    for k in range(T):
        if k > 0:
            P = P + q[:, None, None] * I
        xp[:, k], Pp[:, k] = x, P
        Pinv = np.linalg.inv(P)
        m = hay[:, k]
        if m.any():                                                     # innovacion de la medida combinada z = J^-1 b
            Jm = J[m, k]
            Rm = np.linalg.inv(Jm)
            nu = np.einsum("nij,nj->ni", Rm, b[m, k]) - x[m]
            S = P[m] + Rm
            ll[m] -= 0.5 * (np.einsum("ni,ni->n", nu, np.linalg.solve(S, nu[..., None])[..., 0])
                            + np.linalg.slogdet(S)[1])
        P = np.linalg.inv(Pinv + J[:, k])
        x = np.einsum("nij,nj->ni", P, np.einsum("nij,nj->ni", Pinv, x) + b[:, k])
        xf[:, k], Pf[:, k] = x, P
    if not suavizar:
        return None, ll
    xs = xf.copy()
    for k in range(T - 2, -1, -1):
        C = Pf[:, k] @ np.linalg.inv(Pp[:, k + 1])
        xs[:, k] = xf[:, k] + np.einsum("nij,nj->ni", C, xs[:, k + 1] - xp[:, k + 1])
    return xs, ll


def d_maxima_verosimilitud(J, b):
    ll = np.stack([kalman(J, b, np.full(J.shape[0], Dg), suavizar=False)[1] for Dg in D_REJILLA], 1)
    i = np.clip(ll.argmax(1), 1, len(D_REJILLA) - 2)                    # refinamiento parabolico en log D
    r = np.arange(len(i))
    y0, y1, y2 = ll[r, i - 1], ll[r, i], ll[r, i + 1]
    den = y0 - 2 * y1 + y2
    paso = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
    h = np.log(D_REJILLA[1] / D_REJILLA[0])
    return np.exp(np.log(D_REJILLA[i]) + np.clip(paso, -1, 1) * h)


LAGS_MSD = (1, 2, 3)


def d_msd(y, det, s_lat):
    """D por MSD lateral del real con el ruido de localizacion conocido restado,
    tolerando huecos: pares de frames detectados a lag L = 1, 2, 3, con
    E[dx^2 + dy^2] = 4 D L dt + 2 (s_a^2 + s_b^2). NaN si hay menos de 5 pares."""
    num, den, n = 0.0, 0.0, 0
    for L in LAGS_MSD:
        ok = det[:, L:] & det[:, :-L]
        inc2 = ((y[:, L:, :2] - y[:, :-L, :2]) ** 2).sum(-1)
        ruido = 2 * (s_lat[:, L:] ** 2 + s_lat[:, :-L] ** 2)
        num = num + np.where(ok, inc2 - ruido, 0).sum(1)
        den = den + ok.sum(1) * 4 * L * DT
        n = n + ok.sum(1)
    D = num / np.maximum(den, 1e-12) / 1e6
    return np.where(n >= 5, np.clip(D, 1e-3, 2.0), np.nan)


# ----------------------------------------------------------------------------
# Evaluacion
# ----------------------------------------------------------------------------

def estimaciones(t, med):
    """Estimacion (n, T, 3) de cada metodo (NaN donde no hay) y D estimadas."""
    Js, bs = med["solo"]
    Jc, bc = med["conj"]
    est = {"frame_solo": np.where(med["det"][..., None], med["y"], np.nan)}
    hay = np.abs(Jc).reshape(*Jc.shape[:2], 9).sum(-1) > 0
    Jc_seg = np.where(hay[..., None, None], Jc, np.eye(3))
    est["frame_conj"] = np.where(hay[..., None], np.linalg.solve(Jc_seg, bc[..., None])[..., 0], np.nan)
    est["suav_solo"] = kalman(Js, bs, t["D"])[0]
    est["suav_conj"] = kalman(Jc, bc, t["D"])[0]
    Dest = {"ml_solo": d_maxima_verosimilitud(Js, bs), "ml_conj": d_maxima_verosimilitud(Jc, bc),
            "msd": d_msd(med["y"], med["det"], med["s_lat"])}
    respaldo = np.nanmedian(Dest["msd"])
    Dest["msd_usada"] = np.where(np.isnan(Dest["msd"]), respaldo, Dest["msd"])
    est["suav_solo_Dml"] = kalman(Js, bs, Dest["ml_solo"])[0]
    est["suav_conj_Dml"] = kalman(Jc, bc, Dest["ml_conj"])[0]
    est["suav_solo_Dmsd"] = kalman(Js, bs, Dest["msd_usada"])[0]
    est["suav_conj_Dmsd"] = kalman(Jc, bc, Dest["msd_usada"])[0]
    return est, Dest


def rmse_ic(e2, mascara, rango, n_rangos, tray, W):
    """RMSE por rango con IC 95% por bootstrap de trayectorias."""
    n_tr = W.shape[1]
    idx = (tray[mascara] * n_rangos + rango[mascara])
    S = np.bincount(idx, weights=e2[mascara], minlength=n_tr * n_rangos).reshape(n_tr, n_rangos)
    C = np.bincount(idx, minlength=n_tr * n_rangos).reshape(n_tr, n_rangos).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = np.sqrt(S.sum(0) / C.sum(0))
        boot = np.sqrt((W @ S) / (W @ C))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return est, lo, hi, C.sum(0)


def evaluar(t, med, est, W, variante, escenario):
    n, T = t["d"].shape
    tray = np.repeat(np.arange(n)[:, None], T, 1)
    rangos = {"D": np.repeat(indice(t["D"], BORDES["D"])[:, None], T, 1),
              "fotones": np.repeat(indice(t["N"], BORDES["fotones"])[:, None], T, 1),
              "dist": indice(t["d"], BORDES["dist"]), "global": np.zeros((n, T), int)}
    conjuntos = {"detectados": t["fov"] & med["det"], "no_detectados": t["fov"] & ~med["det"]}
    filas = []
    for metodo, x in est.items():
        e_lat2 = ((x[..., :2] - t["p"][..., :2]) ** 2).sum(-1)
        e_z2 = (x[..., 2] - t["p"][..., 2]) ** 2
        for conj, masc in conjuntos.items():
            m = masc & np.isfinite(e_z2)
            if conj == "no_detectados" and not metodo.startswith("suav") and metodo != "frame_conj":
                continue
            for var, r in rangos.items():
                nr = 1 if var == "global" else len(BORDES[var]) - 1
                ez = rmse_ic(np.nan_to_num(e_z2), m, r, nr, tray, W)
                el = rmse_ic(np.nan_to_num(e_lat2), m, r, nr, tray, W)
                for k in range(nr):
                    lo_b, hi_b = (BORDES[var][k], BORDES[var][k + 1]) if var != "global" else (np.nan, np.nan)
                    filas.append(dict(variante=variante, escenario=escenario, metodo=metodo, frames=conj, variable=var,
                                      desde=lo_b, hasta=hi_b, n=int(ez[3][k]),
                                      cobertura=float(ez[3][k] / max(1, (masc & (r == k)).sum())),
                                      rmse_z=ez[0][k], rmse_z_lo=ez[1][k], rmse_z_hi=ez[2][k],
                                      rmse_lat=el[0][k], rmse_lat_lo=el[1][k], rmse_lat_hi=el[2][k]))
    return filas


# ----------------------------------------------------------------------------
# Figura
# ----------------------------------------------------------------------------

ESTILOS = {"frame_solo": ("#52514e", "o", "--", "frame a frame, real"),
           "frame_conj": ("#eb6834", "s", "--", "frame a frame, real + reflejo"),
           "suav_solo": ("#2a78d6", "o", "-", "suavizado, real"),
           "suav_conj": ("#1baf7a", "s", "-", "suavizado, real + reflejo")}
ETIQ = {"D": "D [µm²/s]", "fotones": "fotones del real", "dist": "distancia al espejo [µm]"}


def x_rango(var):
    b = BORDES[var].astype(float)
    if var == "dist":
        b = np.append(b[:-1], 5000.0) / 1000
        return (b[:-1] + b[1:]) / 2
    return np.sqrt(b[:-1] * b[1:]) if var == "fotones" else (b[:-1] + b[1:]) / 2


def figura(res, Dtab, ruta, escenario="red"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 11))
    r = res[(res.escenario == escenario) & (res.frames == "detectados")]

    def curvas(ax, variante, var, col, metodos=ESTILOS):
        for k, mt in enumerate(metodos):
            c, mk, ls, lab = ESTILOS[mt]
            s = r[(r.variante == variante) & (r.metodo == mt) & (r.variable == var)]
            xc = x_rango(var)
            x = xc * np.exp((k - 1.5) * 0.04) if var == "fotones" else xc + (k - 1.5) * 0.012 * xc[-1]   # desfase para que no se tapen
            ax.errorbar(x, s[col], yerr=[s[col] - s[col + "_lo"], s[col + "_hi"] - s[col]], color=c, marker=mk, ls=ls,
                        ms=5, lw=1.5, capsize=2, label=lab)
        if var == "fotones":
            from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        if var == "dist":
            ax.set_xticks(x_rango(var), ["0–1", "1–2", "2–3", "3–4", "> 4"])
        ax.set_xlabel(ETIQ[var])
        ax.set_ylim(bottom=0)
        ax.grid(True, color="#e6e5e0", lw=0.6)

    for j, var in enumerate(("D", "fotones", "dist")):
        curvas(axs[0, j], "lejos", var, "rmse_z")
        curvas(axs[1, j], "lejos", var, "rmse_lat")
        axs[0, j].set_title(f"RMSE z, PSF lejos, por {ETIQ[var].split(' [')[0]}")
        axs[1, j].set_title(f"RMSE lateral, PSF lejos, por {ETIQ[var].split(' [')[0]}")
    axs[0, 0].set_ylabel("RMSE z [nm]")
    axs[1, 0].set_ylabel("RMSE lateral [nm]")
    axs[0, 0].legend(frameon=False, fontsize=8)
    curvas(axs[2, 0], "cerca", "dist", "rmse_z")
    axs[2, 0].set_title("RMSE z, PSF cerca, por distancia al espejo")
    axs[2, 0].set_ylabel("RMSE z [nm]")

    # estimacion de D
    ax = axs[2, 1]
    d = Dtab[(Dtab.escenario == escenario) & (Dtab.variante == "lejos")]
    k = indice(d.D, BORDES["D"])
    xc = x_rango("D")
    for j, (col, c, lab) in enumerate((("ml_conj", "#1baf7a", "máx. verosimilitud, real + reflejo"),
                                       ("ml_solo", "#2a78d6", "máx. verosimilitud, real"),
                                       ("msd", "#52514e", "MSD (ruido restado), real"))):
        q = np.array([np.nanpercentile(d[col][k == i] / d.D[k == i], [25, 50, 75]) for i in range(len(xc))])
        ax.errorbar(xc + (j - 1) * 0.008, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], color=c, marker="o",
                    ms=5, capsize=2, lw=1.5, label=lab)
    ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_xlabel(ETIQ["D"]); ax.set_ylabel("D estimada / D verdadera (mediana, IQR)")
    ax.set_title("estimación de D por trayectoria (50 frames), PSF lejos")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(True, color="#e6e5e0", lw=0.6)

    # cuanto empeora el suavizado con D estimada
    ax = axs[2, 2]
    for mt, base, c, ls, lab in (("suav_conj_Dml", "suav_conj", "#1baf7a", "-", "real + reflejo, D máx. verosimilitud"),
                                 ("suav_conj_Dmsd", "suav_conj", "#1baf7a", "--", "real + reflejo, D por MSD"),
                                 ("suav_solo_Dml", "suav_solo", "#2a78d6", "-", "real, D máx. verosimilitud"),
                                 ("suav_solo_Dmsd", "suav_solo", "#2a78d6", "--", "real, D por MSD")):
        a = r[(r.variante == "lejos") & (r.metodo == mt) & (r.variable == "D")].rmse_z.to_numpy()
        b = r[(r.variante == "lejos") & (r.metodo == base) & (r.variable == "D")].rmse_z.to_numpy()
        ax.plot(xc, a / b, color=c, ls=ls, marker="o", ms=5, lw=1.5, label=lab)
    ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_xlabel(ETIQ["D"]); ax.set_ylabel("RMSE z con D estimada / con D verdadera")
    ax.set_title("costo de no conocer D, PSF lejos")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    origen = ("la precision de la red con espejo (escenario 'red')" if escenario == "red"
              else "la precision de la CRLB (escenario 'crlb')")
    fig.suptitle(f"Combinar frames (Kalman/RTS) con y sin el reflejo, espejo a 45 grados del eje optico: oraculo con "
                 f"trayectorias del GT y {origen}; IC 95% por bootstrap de trayectorias", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ruta.with_suffix(".png"), dpi=170)
    fig.savefig(ruta.with_suffix(".pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--videos", type=int, default=500)
    p.add_argument("--variantes", nargs="+", default=list(VARIANTES), choices=list(VARIANTES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--salida", default=str(Path(__file__).resolve().parent))
    p.add_argument("--celula", action="store_true", help="emisores en una celula (confinados vs libres)")
    a = p.parse_args()
    salida = Path(a.salida)
    if a.celula:
        return main_celula(a, salida)
    t0 = time.time()
    filas, Dfilas = [], []
    rng = np.random.default_rng(a.seed)
    for variante in a.variantes:
        crlb = tabla_crlb(sim.BancoPSF(sim.Config(z_banco_nm=(-8000.0, 8000.0),
                                                  anillos_hacia_objetivo=VARIANTES[variante])))
        print(f"\n{variante}: tabla de la CRLB ({time.time() - t0:.0f} s)")
        cfg = config(variante, a.videos, a.seed)
        t = trayectorias(cfg, range(a.videos))
        if variante == a.variantes[0]:
            verificar_trayectorias(cfg, t)
        n_tr = len(t["D"])
        W = rng.multinomial(n_tr, np.full(n_tr, 1.0 / n_tr), size=a.n_boot).astype(float)
        print(f"  {n_tr} trayectorias de {cfg.n_frames} frames; "
              f"real en el FOV {t['fov'].mean():.0%}, reflejo en el FOV {t['fovr'].mean():.0%} de los frames")
        escenarios = ["crlb"]
        red = None
        if evaluacion_red(variante) is not None and (RAIZ / f"localizacion/datos/test_espejo_b45_{variante}").exists():
            red = tabla_red(variante, crlb)
            escenarios.append("red")
            print("  kappa_lat (red / CRLB) por fotones:", np.round(red["k_lat"], 2))
            print("  kappa_z   (red / CRLB) por fotones:", np.round(red["k_z"], 2))
        else:
            print(f"  sin la corrida o el test de la variante {variante}: solo el escenario crlb")
        for escenario in escenarios:
            t1 = time.time()
            sig = precisiones(t, escenario, crlb, red)
            med = medir(t, sig, rng)
            est, Dest = estimaciones(t, med)
            filas += evaluar(t, med, est, W, variante, escenario)
            Dfilas.append(pd.DataFrame(dict(variante=variante, escenario=escenario, video=t["video"], particle_id=t["pid"],
                                            D=t["D"], fotones=t["N"], ml_solo=Dest["ml_solo"], ml_conj=Dest["ml_conj"],
                                            msd=Dest["msd"])))
            print(f"  escenario {escenario}: {time.time() - t1:.0f} s; detectados real {med['det'][t['fov']].mean():.0%}, "
                  f"reflejo {med['detr'][t['fovr']].mean():.0%}")
    res = pd.DataFrame(filas)
    Dtab = pd.concat(Dfilas, ignore_index=True)
    res.to_csv(salida / "temporal_espejo_b45_rmse.csv", index=False)
    Dtab.to_csv(salida / "temporal_espejo_b45_D.csv", index=False)
    if set(a.variantes) == set(VARIANTES):
        escenario = "red" if (res.escenario == "red").any() and set(res[res.escenario == "red"].variante) == set(VARIANTES) \
            else "crlb"
        figura(res, Dtab, salida / "fig_temporal_espejo_b45", escenario)
    g = res[(res.variable == "global") & (res.frames == "detectados")]
    print("\nRMSE global (frames con el real detectado) [nm]:")
    print(g.pivot_table(index=["variante", "escenario"], columns="metodo", values="rmse_z")[METODOS].round(1).to_string())
    print(g.pivot_table(index=["variante", "escenario"], columns="metodo", values="rmse_lat")[METODOS].round(1).to_string())
    print(f"\ntotal {time.time() - t0:.0f} s -> {salida}")


def figura_celula(res, Dtab, tray, R_um, ruta, variante="lejos"):
    """Efecto del confinamiento: D estimada, RMSE z y MSD, confinado vs libre."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    xc = x_rango("D")
    col = {"confinado": "#2a78d6", "libre": "#52514e"}
    ax = axs[0]
    for mov in ("confinado", "libre"):
        d = Dtab[Dtab.variante == f"{variante}_{mov}"]
        k = indice(d.D, BORDES["D"])
        for c, ls, lab in (("ml_conj", "-", "máx. verosimilitud"), ("msd", "--", "MSD")):
            q = np.array([np.nanpercentile(d[c][k == i] / d.D[k == i], [25, 50, 75]) for i in range(len(xc))])
            ax.errorbar(xc, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], color=col[mov], ls=ls, marker="o",
                        ms=4, capsize=2, label=f"{mov}, {lab}")
    ax.axhline(1, color="#999999", lw=0.8)
    ax.set_xlabel(ETIQ["D"]); ax.set_ylabel("D estimada / D (mediana, IQR)")
    ax.set_title("estimación de D (50 frames)"); ax.legend(frameon=False, fontsize=7.5)
    ax = axs[1]
    r = res[(res.frames == "detectados") & (res.variable == "D")]
    for mov in ("confinado", "libre"):
        for mt, ls in (("frame_conj", ":"), ("suav_conj", "-"), ("suav_conj_Dml", "--")):
            s = r[(r.variante == f"{variante}_{mov}") & (r.metodo == mt)]
            ax.plot(xc, s.rmse_z, ls, marker="o", ms=4, color=col[mov], label=f"{mov}, {mt}")
    ax.set_xlabel(ETIQ["D"]); ax.set_ylabel("RMSE z [nm]"); ax.set_ylim(bottom=0)
    ax.set_title("real + reflejo: frame a frame y suavizado"); ax.legend(frameon=False, fontsize=7)
    ax = axs[2]
    p, D = tray
    lags = np.arange(1, 50)
    for (lo, hi), c in zip(((0.01, 0.05), (0.1, 0.2), (0.2, 0.3)), ("#9ec5f0", "#4a90d9", "#1b4f8f")):
        sel = (D >= lo) & (D < hi)
        msd = [np.mean(np.sum((p[sel][:, L:] - p[sel][:, :-L]) ** 2, -1)) / 1e6 / 3 for L in lags]
        ax.plot(lags * 0.1, msd, color=c, lw=2, label=f"D {lo:g}–{hi:g}")
        ax.plot(lags * 0.1, 2 * np.mean(D[sel]) * lags * 0.1, color=c, lw=0.8, ls=":")
    ax.axhline(2 * R_um ** 2 / 5, color="#e34948", ls="--", lw=1, label="meseta 2R²/5 por eje")
    ax.set_xlabel("retardo [s]"); ax.set_ylabel("MSD por eje [µm²]"); ax.set_title(f"MSD confinado (R = {R_um} µm)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Difusión confinada en una célula de {2 * R_um:g} µm vs libre desde las mismas posiciones "
                 f"(PSF {variante}, escenario crlb, espejo a 45°)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ruta.with_suffix(".png"), dpi=170)
    fig.savefig(ruta.with_suffix(".pdf"))
    plt.close(fig)


def main_celula(a, salida):
    t0 = time.time()
    filas, Dfilas = [], []
    rng = np.random.default_rng(a.seed)
    tray_msd = None
    for variante in a.variantes:
        crlb = tabla_crlb(sim.BancoPSF(sim.Config(z_banco_nm=(-8000.0, 8000.0),
                                                  anillos_hacia_objetivo=VARIANTES[variante])))
        cfg = config(variante, a.videos, a.seed, celula=True)
        t = trayectorias(cfg, range(a.videos))
        if variante == a.variantes[0]:
            verificar_trayectorias(cfg, t)
            cel = sim.Celula(cfg)
            assert (cel.profundidad_nm(t["p"]) >= -1e-6).all() and (t["d"] >= -1e-6).all()
            tray_msd = (t["p"], t["D"])
        n_tr = len(t["D"])
        W = rng.multinomial(n_tr, np.full(n_tr, 1.0 / n_tr), size=a.n_boot).astype(float)
        for mov, tt in (("confinado", t), ("libre", trayectorias_libres(t, cfg, rng))):
            sig = precisiones(tt, "crlb", crlb, None)
            med = medir(tt, sig, rng)
            est, Dest = estimaciones(tt, med)
            etiq = f"{variante}_{mov}"
            filas += evaluar(tt, med, est, W, etiq, "crlb")
            Dfilas.append(pd.DataFrame(dict(variante=etiq, escenario="crlb", D=tt["D"], ml_solo=Dest["ml_solo"],
                                            ml_conj=Dest["ml_conj"], msd=Dest["msd"])))
            print(f"{etiq}: {n_tr} trayectorias, real en el FOV {tt['fov'].mean():.0%}, reflejo en el FOV "
                  f"{tt['fovr'].mean():.0%} ({time.time() - t0:.0f} s)")
    res = pd.DataFrame(filas)
    Dtab = pd.concat(Dfilas, ignore_index=True)
    res.to_csv(salida / "temporal_espejo_celula_rmse.csv", index=False)
    Dtab.to_csv(salida / "temporal_espejo_celula_D.csv", index=False)
    figura_celula(res, Dtab, tray_msd, DIAMETRO_CELULA_UM / 2, salida / "fig_temporal_espejo_celula", a.variantes[0])
    g = res[(res.variable == "global") & (res.frames == "detectados")]
    print("\nRMSE z global [nm]:")
    print(g.pivot_table(index="variante", columns="metodo", values="rmse_z")[METODOS].round(1).to_string())
    print("RMSE lateral global [nm]:")
    print(g.pivot_table(index="variante", columns="metodo", values="rmse_lat")[METODOS].round(1).to_string())
    print("\nD estimada / D, mediana por rango de D:")
    k = indice(Dtab.D, BORDES["D"])
    print(Dtab.assign(r=k).groupby(["variante", "r"])[["ml_conj", "msd"]].apply(
        lambda x: (x.div(Dtab.loc[x.index, "D"], axis=0)).median()).round(3).unstack().to_string())
    print(f"\ntotal {time.time() - t0:.0f} s -> {salida}")


if __name__ == "__main__":
    main()
