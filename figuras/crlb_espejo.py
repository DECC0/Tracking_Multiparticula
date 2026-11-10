"""
Cuanto puede aportar el reflejo del espejo a la z de un emisor, y si las
redes lo usan. Convencion: +z hacia el objetivo, espejo a beta =
angulo_espejo_eje_deg por defecto (45 grados desde el eje optico). Por cada
variante de la PSF (lejos / cerca: anillos_hacia_objetivo False / True), con
ratio_reflejo = 0.8 y la camara y la PSF por defecto de simulador_nuevo:

1. CRLB en z de un emisor solo (el mismo calculo que crlb_3d de simulador_nuevo,
   verificado contra ella) y de emisor + reflejo con un modelo conjunto:
       mu = b + N g(p) + r N g(p'),   p' = p0 + R (p - p0),   r = ratio_reflejo,
   con parametros (x, y, z, N, b). La PSF del reflejo se mueve con p a traves de
   R, comparte fondo y ruido de camara, y cerca del espejo las dos PSF se
   superponen en la misma ROI (caja de las dos posiciones + MARGEN_PX).
2. Triangulacion oraculo: z por minimos cuadrados ponderados a partir de las
   posiciones laterales del GT del test con espejo, con la geometria exacta de
   cada muestra (geometria_espejo.py): real + N(0, 14 nm), reflejo + N(0, su
   CRLB lateral para sus fotones, su z' y el fondo medio de la muestra). Solo
   emisores con el reflejo dentro del FOV.
3. Una figura por variante: sqrt(CRLB_z) solo y con reflejo, RMSE de la
   triangulacion y RMSE z de las redes evaluadas sobre ese test
   (por_dist_espejo_fotones.csv), por rango de distancia al espejo.
Sin el test de la variante (localizacion/datos/test_espejo_b45_<variante>,
generar_espejo.py) solo se calcula la CRLB, con rangos de d y de fotones y
fondos por defecto (los del test anterior).

Las CRLB se promedian como varianzas (RMS) sobre z en [-800, 800], sobre los
quintiles del fondo medio del test y sobre las distancias de cada rango, que
es lo comparable con un RMSE. No modela el recorte de pupila del reflejo
(a 45 grados parte del cono de coleccion del reflejo no llega por el espejo), ni
el FOV, ni otros emisores, ni el prior z en [-800, 800] que la red si aprende.
Los resultados anteriores (resultados_crlb_espejo.md, fig_crlb_espejo_sz+-1)
son de la convencion vieja: alpha = 34 desde el plano focal y signo_z.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/crlb_espejo.py              # redes: corridas/*/evaluacion_test_espejo_b45_<variante>/ (salvo humo_*)
    $PY figuras/crlb_espejo.py --red espejo localizacion/corridas/<corrida>/evaluacion_test_espejo_b45_lejos [--red ...]
    $PY figuras/crlb_espejo.py --variantes cerca --sufijo _x
Escribe en figuras/: fig_crlb_espejo_b45_<variante>.png/.pdf, crlb_espejo_b45_<variante>.csv
(rejilla completa de la CRLB) y crlb_espejo_rangos_b45_<variante>.csv (por rango, con las redes).
"""
import argparse
import json
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import RegularGridInterpolator

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "simulador_nuevo"))
import simulador as sim  # noqa: E402
from geometria_espejo import GeometriaEspejo  # noqa: E402  (raiz al final del sys.path, via simulador)

BETA_DEG = sim.Config().angulo_espejo_eje_deg   # 45: desde el eje optico (el montaje)
VARIANTES = {"lejos": False, "cerca": True}      # -> anillos_hacia_objetivo
FONDOS_POR_DEFECTO = np.array([16.0, 27.0, 35.0, 43.0, 54.0])   # quintiles del fondo medio del test anterior
BORDES_D_POR_DEFECTO = np.linspace(0.0, 4000.0, 5)
BORDES_N_POR_DEFECTO = np.geomspace(1000.0, 30000.0, 5)
RATIO = 0.8
SIGMA_REAL_NM = 14.0                    # precision lateral de los reales en la triangulacion
MARGEN_PX = 40                          # ROI: caja de real y reflejo + 40 px (>= 81 px alrededor de cada uno)
Z_BANCO_NM = (-8000.0, 8000.0)          # cubre z' con d <= 4 um y cualquier beta (y z en [-800, 800])
D_NM = np.r_[0, 25, 50, 100, 150, 200, 300:4001:100].astype(float)
Z_NM = np.linspace(-800.0, 800.0, 9)
N_FOT = np.geomspace(1000.0, 30000.0, 13)
ZP_TABLA = np.arange(-8000.0, 8000.1, 100.0)                # tabla de la CRLB lateral del reflejo
NP_TABLA = np.geomspace(500.0, 30000.0, 13)
B_TABLA = np.geomspace(4.0, 80.0, 7)
N_REPETICIONES = 50                     # sorteos de ruido por emisor en la triangulacion
N_RANGOS = 4                            # como N_RANGOS_2D de evaluate.py
COLOR = {"solo": "#52514e", "conjunto": "#2a78d6", "tri": "#eb6834"}
COLORES_REDES = ["#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MARCAS_REDES = ["D", "^", "v", "P"]


# ----------------------------------------------------------------------------
# Modelo y Fisher
# ----------------------------------------------------------------------------

def unitaria(banco, x, y, z, forma):
    img = np.zeros(forma)
    banco.colocar(img, x, y, z, 1.0)
    return img.ravel()


def derivadas(banco, x, y, z, forma):
    """PSF unitaria g (P,) y sus derivadas G (3, P) respecto de (x [px], y [px],
    z [nm]), con los mismos pasos que simulador._fisher_roi."""
    h = 1e-3
    zg = banco.z_grid
    hz = np.diff(zg).min() / 2
    z1, z2 = max(z - hz, zg[0]), min(z + hz, zg[-1])
    u = lambda xx, yy, zz: unitaria(banco, xx, yy, zz, forma)   # noqa: E731
    G = np.stack([(u(x + h, y, z) - u(x - h, y, z)) / (2 * h),
                  (u(x, y + h, z) - u(x, y - h, z)) / (2 * h),
                  (u(x, y, z2) - u(x, y, z1)) / (z2 - z1)])
    return u(x, y, z), G


def var_parametros(s, Ju, N, b, v):
    """Diagonal de la inversa de Fisher para (x, y, z, N, b) con mu = b + N s,
    d mu / d(x, y, z) = N Ju, ruido Poisson + lectura (var = mu + v). s (P,),
    Ju (3, P); N, b escalares."""
    J = np.vstack([N * Ju, s, np.ones_like(s)])
    F = (J / (b + N * s + v)) @ J.T
    return np.diag(np.linalg.inv(F))


def transformacion_px(geo, px):
    """T = d p' / d p en unidades (x_px, y_px, z_nm)."""
    S = np.diag([px, px, 1.0])
    return np.linalg.inv(S) @ geo.R @ S


def geometria_canonica():
    return GeometriaEspejo(BETA_DEG, 0.0, [0.0, 0.0, 0.0], -800.0, 800.0)


def real_y_reflejo(geo, d, z):
    """Real a distancia d [nm] del plano y altura z [nm] (sobre la recta p0 + s m), y su reflejo, en nm."""
    s_m = (d - np.sin(geo.beta) * z) / np.cos(geo.beta)
    p = s_m * geo.m + np.array([0.0, 0.0, z])
    assert abs(geo.distancia_nm(p) - d) < 1e-6
    return p, geo.reflejar_nm(p)


def roi_conjunta(p, pr, px):
    """Forma de la ROI y posiciones [px] del real (en un pixel entero, como crlb_3d) y del reflejo."""
    dx, dy = (pr[:2] - p[:2]) / px
    x0, y0 = int(np.floor(min(0.0, dx))) - MARGEN_PX, int(np.floor(min(0.0, dy))) - MARGEN_PX
    x1, y1 = int(np.ceil(max(0.0, dx))) + MARGEN_PX, int(np.ceil(max(0.0, dy))) + MARGEN_PX
    cx, cy = -x0, -y0
    return (y1 - y0 + 1, x1 - x0 + 1), (cx, cy), (cx + dx, cy + dy)


def crlb_rejilla(banco, variante, fondos):
    """DataFrame con sqrt(CRLB) en z [nm] solo y con reflejo para cada (d, z, N, b)."""
    cfg, px = banco.cfg, banco.cfg.pixel_nm
    v = sim._var_lectura_e2(cfg)
    geo = geometria_canonica()
    T = transformacion_px(geo, px)
    filas = []
    for d in D_NM:
        for z in Z_NM:
            p, pr = real_y_reflejo(geo, d, z)
            forma, (cx, cy), (rx, ry) = roi_conjunta(p, pr, px)
            g1, G1 = derivadas(banco, cx, cy, z, forma)
            g2, G2 = derivadas(banco, rx, ry, pr[2], forma)
            s_conj = g1 + RATIO * g2
            Ju_conj = G1 + RATIO * (T.T @ G2)        # fila j: sum_i T[i, j] G2[i]
            for N in N_FOT:
                for b in fondos:
                    v_solo = var_parametros(g1, G1, N, b, v)
                    v_conj = var_parametros(s_conj, Ju_conj, N, b, v)
                    filas.append(dict(variante=variante, d_nm=d, z_nm=z, z_reflejo_nm=pr[2],
                                      separacion_px=np.hypot(rx - cx, ry - cy), fotones=N, fondo=b,
                                      crlb_z_solo_nm=np.sqrt(v_solo[2]), crlb_z_conjunto_nm=np.sqrt(v_conj[2]),
                                      crlb_x_solo_nm=np.sqrt(v_solo[0]) * px,
                                      crlb_x_conjunto_nm=np.sqrt(v_conj[0]) * px))
    return pd.DataFrame(filas)


def tabla_lateral_reflejo(banco):
    """Interpolador de log(CRLB lateral [nm]) de un emisor solo (z desconocido,
    como crlb_3d) en (log fotones, z [nm], log fondo)."""
    cfg, px = banco.cfg, banco.cfg.pixel_nm
    v = sim._var_lectura_e2(cfg)
    L = 2 * MARGEN_PX + 1
    tabla = np.empty((len(NP_TABLA), len(ZP_TABLA), len(B_TABLA)))
    for j, z in enumerate(ZP_TABLA):
        g, G = derivadas(banco, MARGEN_PX, MARGEN_PX, z, (L, L))
        for i, N in enumerate(NP_TABLA):
            for k, b in enumerate(B_TABLA):
                vv = var_parametros(g, G, N, b, v)
                tabla[i, j, k] = np.sqrt(0.5 * (vv[0] + vv[1])) * px
    return RegularGridInterpolator((np.log(NP_TABLA), ZP_TABLA, np.log(B_TABLA)), np.log(tabla))


def verificar(banco):
    """Aserciones: Fisher propia == crlb_3d de simulador_nuevo, y el jacobiano
    conjunto (regla de la cadena con R) == diferencias finitas del modelo conjunto."""
    cfg, px = banco.cfg, banco.cfg.pixel_nm
    v = sim._var_lectura_e2(cfg)
    for z in (-600.0, 0.0, 500.0):
        g, G = derivadas(banco, 7, 7, z, (15, 15))
        ref = sim.crlb_3d(banco, 3000.0, 30.0, z, tam_roi=15)
        mio = np.sqrt(var_parametros(g, G, 3000.0, 30.0, v))
        assert np.allclose(mio, [ref[k] for k in ("x", "y", "z", "N", "b")], rtol=1e-9), (mio, ref)
    for beta in (BETA_DEG, -30.0):
        geo = GeometriaEspejo(beta, 0.0, [0.0, 0.0, 0.0], -800.0, 800.0)
        T = transformacion_px(geo, px)
        p, pr = real_y_reflejo(geo, 800.0, 300.0)
        forma, (cx, cy), (rx, ry) = roi_conjunta(p, pr, px)
        _, G1 = derivadas(banco, cx, cy, p[2], forma)
        _, G2 = derivadas(banco, rx, ry, pr[2], forma)
        Ju = G1 + RATIO * (T.T @ G2)

        def modelo(dp):                                  # dp en (px, px, nm), real movido y reflejo recalculado
            q = p + dp * np.array([px, px, 1.0])
            qr = geo.reflejar_nm(q)
            xr, yr = cx + (qr[0] - p[0]) / px, cy + (qr[1] - p[1]) / px
            return (unitaria(banco, cx + dp[0], cy + dp[1], q[2], forma)
                    + RATIO * unitaria(banco, xr, yr, qr[2], forma))
        for j in (0, 1):                                 # x, y: el shift es exacto, pero el reflejo cambia de z'
            e = np.zeros(3)
            e[j] = 1e-3
            fd = (modelo(e) - modelo(-e)) / 2e-3
            err = np.abs(fd - Ju[j]).max() / np.abs(Ju[j]).max()
            assert err < 0.05, (j, err)
        # z: el banco interpola linealmente; con pasos pequenos dentro de un intervalo coincide
        e = np.array([0.0, 0.0, 1e-3])
        fd = (modelo(e) - modelo(-e)) / 2e-3
        err = np.abs(fd - Ju[2]).max() / np.abs(Ju[2]).max()
        assert err < 0.05, err
    print("verificacion: Fisher == crlb_3d y jacobiano conjunto == diferencias finitas")


# ----------------------------------------------------------------------------
# Test, triangulacion y redes
# ----------------------------------------------------------------------------

def bordes(valores, escala, n=N_RANGOS):
    """Como evaluate.bordes: n + 1 bordes entre min y max (geometricos con escala log)."""
    lo, hi = float(np.min(valores)), float(np.max(valores))
    return np.geomspace(lo, hi, n + 1) if escala == "log" else np.linspace(lo, hi, n + 1)


def cargar_test(carpeta):
    """GT del test con la geometria de cada muestra y d al espejo (como
    metricas.distancia_al_espejo_nm). Devuelve (cfg, gt, muestras): cfg es la
    simulador.Config del test si es de la convencion actual de z, o None si es
    de la anterior (sin convencion_z en config_simulador.json): entonces solo
    sirven su GT, d y sus rangos (z en la convencion vieja), no la geometria."""
    carpeta = Path(carpeta)
    meta = json.loads((carpeta / "config_simulador.json").read_text())
    cfg = sim.cargar_config(carpeta / "config_simulador.json") if meta.get("convencion_z") == sim.CONVENCION_Z else None
    gt = pd.read_csv(carpeta / "ground_truth.csv")
    m = pd.read_csv(carpeta / "muestras.csv")
    cols = [c for c in ("fondo_medio", "espejo_angulo_eje_deg", "espejo_angulo_deg", "espejo_orientacion_deg",
                        "espejo_punto_x_px", "espejo_punto_y_px") if c in m]
    gt = gt.merge(m[["video_id"] + cols], on="video_id", how="left")
    px = meta["config"].get("pixel_camara_um", 6.5) * 1000.0 / meta["config"].get("magnificacion", 66.0)
    dif = np.c_[(gt.x_reflejo_px - gt.x_px) * px, (gt.y_reflejo_px - gt.y_px) * px, gt.z_reflejo_nm - gt.z_nm]
    gt["d_espejo_nm"] = 0.5 * np.linalg.norm(dif, axis=1)
    return cfg, gt, m


def matrices_geometria(cfg, gt):
    """R (n, 3, 3) y p0 (n, 3) [nm] de la geometria de la muestra de cada fila."""
    px = cfg.pixel_nm
    R = np.empty((len(gt), 3, 3))
    p0 = np.empty((len(gt), 3))
    claves = ["espejo_angulo_eje_deg", "espejo_orientacion_deg", "espejo_punto_x_px", "espejo_punto_y_px"]
    for (b, f, x0, y0), idx in gt.groupby(claves).indices.items():
        geo = GeometriaEspejo(b, f, [x0 * px, y0 * px, cfg.z_ref_espejo_nm], *cfg.z_rango_nm)
        R[idx], p0[idx] = geo.R, geo.p0
    return R, p0


def triangular(R, p0, lat_real, lat_reflejo, s_real, s_reflejo):
    """z [nm] por minimos cuadrados ponderados: incognita p (3,), medidas p_xy
    (precision s_real) y p'_xy = [p0 + R (p - p0)]_xy (precision s_reflejo).
    Todo en nm; R (n, 3, 3), p0 (n, 3), lat_* (n, 2), s_* (n,)."""
    A = np.concatenate([np.broadcast_to(np.eye(3)[:2], (len(R), 2, 3)), R[:, :2, :]], axis=1)   # (n, 4, 3)
    c = p0 - np.einsum("nij,nj->ni", R, p0)                                                       # p' = c + R p
    y = np.concatenate([lat_real, lat_reflejo - c[:, :2]], axis=1)
    w = np.concatenate([np.repeat(1 / s_real[:, None] ** 2, 2, 1), np.repeat(1 / s_reflejo[:, None] ** 2, 2, 1)], 1)
    AtW = np.transpose(A, (0, 2, 1)) * w[:, None, :]
    return np.linalg.solve(AtW @ A, np.einsum("nij,nj->ni", AtW, y)[..., None])[:, 2, 0]


def triangulacion(cfg, gt, interp, rng):
    """Error de z de la triangulacion oraculo (N_REPETICIONES sorteos por fila) de
    los reales en el FOV con el reflejo en el FOV. Devuelve (filas usadas, errores (n, rep))."""
    px = cfg.pixel_nm
    g = gt[gt.en_fov & gt.reflejo_en_fov].reset_index(drop=True)
    R, p0 = matrices_geometria(cfg, g)
    p = np.c_[g.x_px * px, g.y_px * px, g.z_nm]
    pr = np.c_[g.x_reflejo_px * px, g.y_reflejo_px * px, g.z_reflejo_nm]
    error_geo = np.abs(p0 + np.einsum("nij,nj->ni", R, p - p0) - pr).max()
    assert error_geo < 1e-6, f"la geometria de muestras.csv no reproduce los reflejos del GT ({error_geo} nm)"
    sin_ruido = triangular(R, p0, p[:, :2], pr[:, :2], np.full(len(g), SIGMA_REAL_NM), np.ones(len(g)))
    assert np.abs(sin_ruido - g.z_nm).max() < 1e-6
    pts = np.c_[np.log(np.clip(g.photons_reflejo, NP_TABLA[0], NP_TABLA[-1])),
                np.clip(g.z_reflejo_nm, ZP_TABLA[0], ZP_TABLA[-1]),
                np.log(np.clip(g.fondo_medio, B_TABLA[0], B_TABLA[-1]))]
    s_ref = np.exp(interp(pts))
    g["sigma_lat_reflejo_nm"] = s_ref
    s_real = np.full(len(g), SIGMA_REAL_NM)
    err = np.empty((len(g), N_REPETICIONES))
    for k in range(N_REPETICIONES):
        lr = p[:, :2] + rng.normal(0, 1, (len(g), 2)) * s_real[:, None]
        lf = pr[:, :2] + rng.normal(0, 1, (len(g), 2)) * s_ref[:, None]
        err[:, k] = triangular(R, p0, lr, lf, s_real, s_ref) - g.z_nm
    return g, err


def redes_por_defecto(variante):
    etiq = f"evaluacion_test_espejo_b45_{variante}"
    return [(None, c) for c in sorted((RAIZ / "localizacion" / "corridas").glob(f"*/{etiq}"))
            if not c.parent.name.startswith("humo")]


def con_espejo(carpeta):
    """True si la corrida de la carpeta de evaluacion se entreno con espejo (None si no se sabe)."""
    try:
        cfg = yaml.safe_load(open(Path(carpeta).parent / "config.yaml"))
        return bool(cfg["datos"]["simulador"].get("usar_espejo", False))
    except (OSError, KeyError, TypeError):
        return None


def etiqueta_red(carpeta):
    nombre = re.sub(r"^hypatia_|_\d{8}-\d{6}$", "", Path(carpeta).parent.name)
    return nombre + (" (sin espejo)" if con_espejo(carpeta) is False else "")


def variante_de_evaluacion(carpeta):
    r = yaml.safe_load(open(Path(carpeta) / "resumen.yaml"))
    entrada = Path(r["entradas"][0])
    entrada = entrada if entrada.is_absolute() else RAIZ / entrada
    an = json.load(open(entrada / "config_simulador.json"))["config"].get("anillos_hacia_objetivo")
    return {False: "lejos", True: "cerca"}.get(an)


# ----------------------------------------------------------------------------
# Tablas por rango y figura
# ----------------------------------------------------------------------------

def rms(a):
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


CLAVES = ["video_id", "frame", "particle_id"]


def bootstrap_redes(redes, bd, bf, n_boot, rng):
    """RMSE z por rango de cada red con IC 95% por bootstrap de imagenes
    (video_id: los emisores de una imagen no son independientes), y diferencias
    contra la red de referencia (la primera entrenada sin espejo):
        dif_propia   RMSE de cada red sobre sus propios detectados (lo de evaluate.py)
        dif_pareada  RMSE sobre los emisores que detectan las dos redes
    redes: [(etiqueta, gt_eval, con_espejo)], todas sobre el mismo GT.
    Devuelve {columna: (n_rangos,)} con estimacion e IC (lo, hi) de cada cantidad."""
    base = redes[0][1]
    gts = []
    for etiq, g, _ in redes:
        g = g.sort_values(CLAVES).reset_index(drop=True)
        assert len(g) == len(base) and np.allclose(g[["x_px", "y_px", "z_nm", "photons"]].to_numpy(float),
                                                   base.sort_values(CLAVES)[["x_px", "y_px", "z_nm", "photons"]]
                                                   .to_numpy(float)), f"{etiq}: otro GT"
        gts.append(g[g.en_fov].reset_index(drop=True))
    g0 = gts[0]
    nr = (len(bd) - 1) * (len(bf) - 1)
    rango = np.digitize(g0.d_espejo_nm, bd[1:-1]) * (len(bf) - 1) + np.digitize(g0.photons, bf[1:-1])
    vid, n_vid = pd.factorize(g0.video_id)[0], g0.video_id.nunique()
    celda = vid * nr + rango

    def sumas(peso):                                    # (n_vid, nr) de sum(peso)
        return np.bincount(celda, weights=peso, minlength=n_vid * nr).reshape(n_vid, nr)

    W = rng.multinomial(n_vid, np.full(n_vid, 1.0 / n_vid), size=n_boot).astype(float)   # (n_boot, n_vid)
    uno = np.ones((1, n_vid))

    def rmse(det, dz):                                  # estimacion (nr,) y bootstrap (n_boot, nr)
        sq, n = sumas(np.where(det, dz, 0.0) ** 2), sumas(det.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt((uno @ sq) / (uno @ n))[0], np.sqrt((W @ sq) / (W @ n))

    def ic(est, boot, nombre):
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        return {nombre: est, f"{nombre}_ic_lo": lo, f"{nombre}_ic_hi": hi}

    out = {}
    est = []
    for (etiq, _, _), g in zip(redes, gts):
        det, dz = g.detectado.to_numpy(bool), g.dz_nm.fillna(0.0).to_numpy(float)
        e, b = rmse(det, dz)
        est.append((det, dz, e, b))
        out.update(ic(e, b, f"rmse_z_red[{etiq}]"))
        n_tot = sumas(np.ones(len(g)))
        r_e, r_b = (uno @ sumas(det.astype(float)))[0] / (uno @ n_tot)[0], (W @ sumas(det.astype(float))) / (W @ n_tot)
        out.update(ic(r_e, r_b, f"recall_red[{etiq}]"))
    ref = next((k for k, r in enumerate(redes) if r[2] is False), None)
    if ref is None:
        return out
    det_r, dz_r, e_r, b_r = est[ref]
    for k, (etiq, _, _) in enumerate(redes):
        if k == ref:
            continue
        det, dz, e, b = est[k]
        out.update(ic(e - e_r, b - b_r, f"dif_propia[{etiq}]"))
        comun = det & det_r
        e1, b1 = rmse(comun, dz)
        e2, b2 = rmse(comun, dz_r)
        out.update(ic(e1 - e2, b1 - b2, f"dif_pareada[{etiq}]"))
        out[f"rmse_pareado[{etiq}]"], out[f"rmse_pareado_ref[{etiq}]"] = e1, e2
        out[f"n_comun[{etiq}]"] = (uno @ sumas(comun.astype(float)))[0]
    return out


def tabla_rangos(crlb, g_tri, err, gt, bd, bf, redes_boot):
    """Por (rango de d, rango de fotones): CRLB solo / conjunta (RMS sobre d, z,
    fondo y fotones del rango, en la rejilla), triangulacion y las columnas de
    bootstrap_redes (en el mismo orden de rangos: d por fuera, fotones por dentro)."""
    g_fov = gt[gt.en_fov]
    kd_t, kf_t = np.digitize(g_tri.d_espejo_nm, bd[1:-1]), np.digitize(g_tri.photons, bf[1:-1])
    kd_g, kf_g = np.digitize(g_fov.d_espejo_nm, bd[1:-1]), np.digitize(g_fov.photons, bf[1:-1])
    filas = []
    for i in range(len(bd) - 1):
        for j in range(len(bf) - 1):
            c = crlb[(crlb.d_nm >= bd[i]) & (crlb.d_nm <= bd[i + 1])
                     & (crlb.fotones >= bf[j]) & (crlb.fotones <= bf[j + 1])]
            sel = (kd_t == i) & (kf_t == j)
            fila = dict(d_desde_nm=bd[i], d_hasta_nm=bd[i + 1], fotones_desde=bf[j], fotones_hasta=bf[j + 1],
                        n_fov=int(((kd_g == i) & (kf_g == j)).sum()), n_con_reflejo_en_fov=int(sel.sum()),
                        crlb_z_solo_nm=rms(c.crlb_z_solo_nm), crlb_z_conjunto_nm=rms(c.crlb_z_conjunto_nm),
                        rmse_z_triangulacion_nm=rms(err[sel]),
                        sigma_lat_reflejo_mediana_nm=float(np.median(g_tri.sigma_lat_reflejo_nm[sel]))
                        if sel.any() else float("nan"))
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    for col, v in redes_boot.items():
        tabla[col] = v
    return tabla


def curva_por_fotones(crlb, d0, d1, col):
    c = crlb[(crlb.d_nm >= d0) & (crlb.d_nm <= d1)]
    return c.groupby("fotones")[col].apply(rms)


def formato_log(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_xscale("log")
    ax.set_yscale("log")
    for eje in (ax.xaxis, ax.yaxis):
        eje.set_major_locator(LogLocator(subs=(1, 2, 5)))
        eje.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        eje.set_minor_formatter(NullFormatter())


def figura(variante, crlb, tabla, redes, bd, bf, fondos, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28,
                          left=0.06, right=0.985, top=0.89, bottom=0.2)
    ymin, ymax = 2.0, 600.0
    for i in range(len(bd) - 1):
        ax = fig.add_subplot(gs[0, i])
        d0, d1 = bd[i], bd[i + 1]
        cs = curva_por_fotones(crlb, d0, d1, "crlb_z_solo_nm")
        cc = curva_por_fotones(crlb, d0, d1, "crlb_z_conjunto_nm")
        ax.plot(cs.index, cs.values, "-", color=COLOR["solo"], lw=2, label="CRLB, emisor solo")
        ax.plot(cc.index, cc.values, "-", color=COLOR["conjunto"], lw=2, label="CRLB, emisor + reflejo")
        t = tabla[np.isclose(tabla.d_desde_nm, d0)]
        xc = np.sqrt(t.fotones_desde * t.fotones_hasta).to_numpy()
        if t.rmse_z_triangulacion_nm.notna().any():
            ax.plot(xc, t.rmse_z_triangulacion_nm, "s--", color=COLOR["tri"], lw=1.5, ms=7,
                    mec="white", mew=1, label="triangulacion oraculo")
        desfase = np.linspace(-1, 1, len(redes)) * 0.07 if len(redes) > 1 else [0.0]
        for k, etiq in enumerate(redes):
            col = f"rmse_z_red[{etiq}]"
            y = t[col].to_numpy()
            yerr = [y - t[f"{col}_ic_lo"].to_numpy(), t[f"{col}_ic_hi"].to_numpy() - y]
            ax.errorbar(xc * np.exp(desfase[k]), y, yerr=yerr, fmt=MARCAS_REDES[k % 4], color=COLORES_REDES[k % 4],
                        ms=7, mec="white", mew=1, elinewidth=1.5, capsize=2, ls="none",
                        label=f"red {etiq}")
        formato_log(ax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(900, 33000)
        ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
        ax.set_title(f"d al espejo {d0 / 1000:.1f}-{d1 / 1000:.1f} um", fontsize=10)
        ax.set_xlabel("fotones del real")
        if i == 0:
            ax.set_ylabel("error en z [nm]")
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 0.975), fontsize=8.5)
    fig.suptitle(f"Informacion del reflejo sobre z: espejo a {BETA_DEG:g} grados del eje optico, PSF '{variante}' "
                 f"(anillos_hacia_objetivo = {VARIANTES[variante]}), ratio_reflejo = {RATIO}, +z hacia el objetivo",
                 y=0.998, fontsize=11)

    # abajo: razon de ganancia vs d, CRLB vs z, z' vs d
    ax = fig.add_subplot(gs[1, 0:2])
    estilos = [(1000.0, ":"), (5000.0, "--"), (30000.0, "-")]
    for N, ls in estilos:
        Nn = N_FOT[np.argmin(np.abs(N_FOT - N))]
        c = crlb[np.isclose(crlb.fotones, Nn)]
        r = (c.groupby("d_nm").crlb_z_solo_nm.apply(rms) / c.groupby("d_nm").crlb_z_conjunto_nm.apply(rms))
        ax.plot(r.index / 1000, r.values, ls, color=COLOR["conjunto"], lw=2, label=f"N = {Nn:.0f}")
    ax.axhline(1, color=COLOR["solo"], lw=1)
    ax.annotate("d -> 0: real y reflejo coinciden y un paso a lo largo de n\nlos mueve en sentidos opuestos (la CRLB no ve el prior d >= 0)",
                xy=(0.0, 0.3), xytext=(0.35, 0.25), fontsize=7.5, color="#52514e",
                arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.8))
    ax.set_xlabel("distancia al espejo d [um]")
    ax.set_ylabel("ganancia CRLB_z: solo / con reflejo")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, title="RMS sobre z y fondo", title_fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    Nn = N_FOT[np.argmin(np.abs(N_FOT - 5000.0))]
    for d, ls in ((200.0, ":"), (1000.0, "--"), (3000.0, "-")):
        c = crlb[np.isclose(crlb.fotones, Nn) & np.isclose(crlb.d_nm, d)]
        ax.plot(Z_NM, c.groupby("z_nm").crlb_z_solo_nm.apply(rms).values, ls, color=COLOR["solo"], lw=1.5)
        ax.plot(Z_NM, c.groupby("z_nm").crlb_z_conjunto_nm.apply(rms).values, ls, color=COLOR["conjunto"],
                lw=2, label=f"d = {d / 1000:g} um")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("z del real [nm]")
    ax.set_ylabel(f"sqrt(CRLB_z) [nm], N = {Nn:.0f}")
    ax.set_title("gris: solo; azul: con reflejo", fontsize=9)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 3])
    c = crlb[np.isclose(crlb.fotones, Nn)].drop_duplicates(["d_nm", "z_nm"])
    for z, ls in ((-800.0, ":"), (0.0, "-"), (800.0, "--")):
        cz = c[np.isclose(c.z_nm, z)]
        ax.plot(cz.d_nm / 1000, cz.z_reflejo_nm / 1000, ls, color=COLOR["tri"], lw=2, label=f"z = {z:g} nm")
    ax.set_xlabel("distancia al espejo d [um]")
    ax.set_ylabel("z' del reflejo [um]")
    ax.set_title("desenfoque del reflejo", fontsize=9)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=8)

    nota = [f"CRLB: Poisson + lectura (Orca, 1.5 e-), parametros (x, y, z, N, b), fondo uniforme en los quintiles del test "
            f"({', '.join(f'{b:.0f}' for b in fondos)} fotones/px), RMS sobre z en [-800, 800] nm y sobre d y N de cada rango; "
            f"ROI = caja real-reflejo + {MARGEN_PX} px (sin recorte por el FOV ni otros emisores; la red ademas aprende el "
            "prior z en [-800, 800], que la CRLB no incluye).",
            f"Triangulacion oraculo (si hay test): GT del test (geometria de cada muestra), real + {SIGMA_REAL_NM:g} nm, "
            "reflejo + su CRLB lateral (fotones, z', fondo medio); solo reales con el reflejo en el FOV, sin superposicion de "
            "PSF (por eso puede quedar bajo la CRLB conjunta, en la que la precision lateral del real a pocos fotones es "
            "peor que 14 nm). Redes: RMSE z de los detectados (sesgo de supervivencia a pocos fotones), IC 95% por "
            "bootstrap de imagenes del test.",
            f"Limitaciones: el reflejo es una fuente puntual vista con la pupila completa. A {BETA_DEG:g} grados del eje "
            "optico parte del cono de coleccion del reflejo no llega por el espejo (recorte de pupila, no modelado; ~6% de su "
            "luz segun el asesor), lo que le quita fotones, puede volver asimetrica su PSF y desplazar su centroide con z' "
            "(sesgo de la triangulacion si no se modela). Espejo plano ideal e infinito, ratio_reflejo constante."]
    nota = "\n".join(textwrap.fill(t, 225) for t in nota)
    fig.text(0.06, 0.012, nota, fontsize=7.2, color="#52514e", va="bottom", ha="left")
    fig.savefig(ruta.with_suffix(".png"), dpi=200)
    fig.savefig(ruta.with_suffix(".pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--variantes", nargs="+", default=list(VARIANTES), choices=list(VARIANTES))
    p.add_argument("--red", nargs=2, action="append", metavar=("ETIQUETA", "CARPETA_EVALUACION"),
                   help="carpeta evaluacion_test_espejo_b45_<variante> de evaluate.py; sin --red se buscan en corridas/")
    p.add_argument("--datos", default=str(RAIZ / "localizacion" / "datos"),
                   help="carpeta con test_espejo_b45_lejos y test_espejo_b45_cerca (generar_espejo.py)")
    p.add_argument("--salida", default=str(Path(__file__).resolve().parent))
    p.add_argument("--sufijo", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000, help="remuestreos de imagenes para los IC de las redes")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    salida = Path(a.salida)

    for variante in a.variantes:
        t0 = time.time()
        banco = sim.BancoPSF(sim.Config(z_banco_nm=Z_BANCO_NM, anillos_hacia_objetivo=VARIANTES[variante]))
        print(f"\n{variante}: banco de {len(banco.z_grid)} planos en [{banco.z_grid[0]:.0f}, {banco.z_grid[-1]:.0f}] nm, "
              f"K = {banco.K}")
        verificar(banco)
        interp = tabla_lateral_reflejo(banco)
        print(f"tabla de la CRLB lateral del reflejo: {time.time() - t0:.0f} s")
        carpeta_test = Path(a.datos) / f"test_espejo_b45_{variante}"
        cfg_t = None
        if carpeta_test.exists():
            cfg_t, gt, muestras = cargar_test(carpeta_test)
            if cfg_t is None or cfg_t.anillos_hacia_objetivo != VARIANTES[variante]:
                raise ValueError(f"{carpeta_test}: no es un test de la convencion actual con la PSF '{variante}'")
            assert np.isclose(cfg_t.ratio_reflejo, RATIO)
            fondos = np.quantile(muestras.fondo_medio, [0.1, 0.3, 0.5, 0.7, 0.9])
            g_fov = gt[gt.en_fov]
            bd, bf = bordes(g_fov.d_espejo_nm, "lineal"), bordes(g_fov.photons, "log")
        else:
            print(f"  sin {carpeta_test}: solo la CRLB, con rangos y fondos por defecto (sin triangulacion ni redes)")
            gt = pd.DataFrame(columns=["en_fov", "d_espejo_nm", "photons"]).astype({"en_fov": bool})
            fondos, bd, bf = FONDOS_POR_DEFECTO, BORDES_D_POR_DEFECTO, BORDES_N_POR_DEFECTO

        entradas = ([(e, Path(c)) for e, c in a.red] if a.red else redes_por_defecto(variante)) if cfg_t else []
        redes, tablas_eval = [], []                     # gt_eval de cada red, y su por_dist_espejo_fotones.csv
        for etiq, carpeta in entradas:
            if a.red and variante_de_evaluacion(carpeta) != variante:
                continue
            t = pd.read_csv(carpeta / "por_dist_espejo_fotones.csv")
            if not (np.allclose(np.unique(t.d_desde_nm), bd[:-1]) and np.allclose(np.unique(t.fotones_desde), bf[:-1])):
                print(f"  aviso: los rangos de {carpeta} no son los de {carpeta_test}; se omite")
                continue
            etiq = etiq or etiqueta_red(carpeta)
            redes.append((etiq, pd.read_csv(carpeta / "gt_eval.csv"), con_espejo(carpeta)))
            tablas_eval.append((etiq, t))
        redes.sort(key=lambda r: r[2] is not False)     # la referencia (sin espejo) primero
        boot = bootstrap_redes(redes, bd, bf, a.n_boot, rng) if redes else {}
        for etiq, t in tablas_eval:                     # la estimacion reproduce por_dist_espejo_fotones.csv
            assert np.allclose(boot[f"rmse_z_red[{etiq}]"], t.rmse_z_nm, equal_nan=True), etiq

        t1 = time.time()
        crlb = crlb_rejilla(banco, variante, fondos)
        if cfg_t is not None:
            g_tri, err = triangulacion(cfg_t, gt, interp, rng)
        else:
            g_tri = pd.DataFrame(columns=["d_espejo_nm", "photons", "sigma_lat_reflejo_nm"], dtype=float)
            err = np.zeros((0, N_REPETICIONES))
        tabla = tabla_rangos(crlb, g_tri, err, gt, bd, bf, boot)
        nombre = f"b45_{variante}{a.sufijo}"
        crlb.to_csv(salida / f"crlb_espejo_{nombre}.csv", index=False)
        tabla.to_csv(salida / f"crlb_espejo_rangos_{nombre}.csv", index=False)
        figura(variante, crlb, tabla, [r[0] for r in redes], bd, bf, fondos, salida / f"fig_crlb_espejo_{nombre}")
        print(f"\n{variante}  ({time.time() - t1:.0f} s; redes: {[r[0] for r in redes] or 'ninguna'})")
        if cfg_t is not None:
            print(f"  reales en el FOV con el reflejo en el FOV: {len(g_tri)} de {len(g_fov)}")
        cols = ["d_desde_nm", "d_hasta_nm", "fotones_desde", "fotones_hasta", "crlb_z_solo_nm", "crlb_z_conjunto_nm",
                "rmse_z_triangulacion_nm", "sigma_lat_reflejo_mediana_nm"]
        with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.1f}".format):
            print(tabla[cols].to_string(index=False))
            redes_cols = [c for c in tabla if c.startswith(("rmse_z_red", "dif_", "recall_red", "n_comun"))]
            if redes_cols:
                print(tabla[["d_desde_nm", "fotones_desde"] + redes_cols].T.to_string())
        print(f"  -> {salida / f'fig_crlb_espejo_{nombre}.png'}")


if __name__ == "__main__":
    main()
