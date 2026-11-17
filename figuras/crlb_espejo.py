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

4. Con --celula, en lugar de 1-3: CRLB por emisor del test con celula
   (localizacion/datos/test_celula_b45_<variante>, sin rejilla canonica):
   cada emisor evaluable con la geometria, ratio_reflejo y el mapa de fondo
   exactos de su muestra (regenerado con su flujo del simulador), la imagen
   entera de 96 x 96 como ROI (corte del FOV incluido), real y reflejo con el
   recorte de pupila (PSFRecortada, z exacta, sin banco) y parametros (x, y,
   z, N, b) con b un desplazamiento del fondo; solo, conjunto (r conocido) y
   conjunto con r desconocido. Por rango de z, fotones y distancia al espejo,
   junto al RMSE z de las redes (evaluacion_test_celula_b45_<variante>), con
   IC 95% por bootstrap de imagenes, tambien solo de los emisores aislados
   (> 1 um lateral de otro real y de otro reflejo: sin otros emisores, como la
   CRLB). Sin el prior de la celula ni los demas emisores.

Las CRLB se promedian como varianzas (RMS) sobre z en [-800, 800], sobre los
quintiles del fondo medio del test y sobre las distancias de cada rango, que
es lo comparable con un RMSE. No modela el recorte de pupila del reflejo
(a 45 grados parte del cono de coleccion del reflejo no llega por el espejo;
su efecto en la CRLB esta en figuras/recorte_pupila.py: <= ~12%), ni
el FOV, ni otros emisores, ni el prior z en [-800, 800] que la red si aprende.
Los resultados anteriores (resultados_crlb_espejo.md, fig_crlb_espejo_sz+-1)
son de la convencion vieja: alpha = 34 desde el plano focal y signo_z.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/crlb_espejo.py              # redes: corridas/*/evaluacion_test_espejo_b45_<variante>/ (salvo humo_*)
    $PY figuras/crlb_espejo.py --red espejo localizacion/corridas/<corrida>/evaluacion_test_espejo_b45_lejos [--red ...]
    $PY figuras/crlb_espejo.py --variantes cerca --sufijo _x
    $PY figuras/crlb_espejo.py --celula [--variantes cerca lejos] [--procesos 14]   # ~2 min por variante
Escribe en figuras/crlb_espejo/: fig_crlb_espejo_b45_<variante>.png/.pdf, crlb_espejo_b45_<variante>.csv
(rejilla completa de la CRLB) y crlb_espejo_rangos_b45_<variante>.csv (por rango, con las redes);
con --celula: fig_crlb_celula_b45_<variante>.png/.pdf, crlb_celula_b45_<variante>.csv (por emisor) y
crlb_celula_rangos_b45_<variante>.csv (por rango de z, fotones y distancia al espejo, columna tabla).
"""
import argparse
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import replace
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


def derivadas(banco, x, y, z, forma, hz=None):
    """PSF unitaria g (P,) y sus derivadas G (3, P) respecto de (x [px], y [px],
    z [nm]), con los mismos pasos que simulador._fisher_roi. banco es un
    BancoPSF (z con medio paso de su rejilla) o, con hz [nm], cualquier objeto
    con colocar, p. ej. PSFRecortada (PSF exacta en cada z: diferencia centrada
    con paso hz)."""
    h = 1e-3
    if hz is None:
        zg = banco.z_grid
        hz = np.diff(zg).min() / 2
        z1, z2 = max(z - hz, zg[0]), min(z + hz, zg[-1])
    else:
        z1, z2 = z - hz, z + hz
    u = lambda xx, yy, zz: unitaria(banco, xx, yy, zz, forma)   # noqa: E731
    G = np.stack([(u(x + h, y, z) - u(x - h, y, z)) / (2 * h),
                  (u(x, y + h, z) - u(x, y - h, z)) / (2 * h),
                  (u(x, y, z2) - u(x, y, z1)) / (z2 - z1)])
    return u(x, y, z), G


def var_parametros(s, Ju, N, b, v, extra=()):
    """Diagonal de la inversa de Fisher para (x, y, z, N, b[, extra]) con mu =
    b + N s, d mu / d(x, y, z) = N Ju, ruido Poisson + lectura (var = mu + v).
    s (P,), Ju (3, P); N escalar; b escalar o mapa (P,) de fondo conocido en
    forma (el parametro es entonces un desplazamiento uniforme sobre el);
    extra: derivadas (P,) de parametros de molestia adicionales."""
    J = np.vstack([N * Ju, s, np.ones_like(s), *extra])
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


# ----------------------------------------------------------------------------
# Celula: CRLB por emisor del test (--celula)
# ----------------------------------------------------------------------------

HZ_CELULA_NM = 10.0                     # paso de la derivada en z con PSFRecortada (PSF exacta en cada z)
BORDES_CELULA = {"z": ("z_nm", np.linspace(-2500.0, 2500.0, 11), "z del emisor [nm]", False),
                 "fotones": ("photons", np.geomspace(1000.0, 30000.0, 9), "fotones del real", True),
                 "dist_espejo": ("d_espejo_nm", np.array([0, 500, 1000, 1500, 2000, 3000, 4000, 6000.0]),
                                 "distancia al espejo [nm]", False)}
AISLADO_NM = 1000.0                     # sin otro real ni otro reflejo a menos de esto (lateral): comparable con la CRLB


def config_muestra_celula(cfg, fila):
    """simulador.Config de una muestra del test con celula (geometria, ratio y fondo de muestras.csv)."""
    return replace(cfg, angulo_espejo_eje_deg=float(fila.espejo_angulo_eje_deg),
                   orientacion_espejo_deg=float(fila.espejo_orientacion_deg),
                   espejo_punto_px=(float(fila.espejo_punto_x_px), float(fila.espejo_punto_y_px)),
                   ratio_reflejo=float(fila.ratio_reflejo), fondo_no_uniforme=bool(fila.fondo_no_uniforme))


def _crlb_muestra(tarea):
    """sqrt(CRLB) en z y lateral de los emisores evaluables de una muestra: en la
    imagen entera (H x W, con el corte del FOV), con su fondo exacto (mapa
    conocido en forma, desplazamiento desconocido), real y reflejo con el
    recorte de pupila (PSFRecortada) y parametros (x, y, z, N, b):
        solo         mu = b + N g(p)                   (real recortado, sin reflejo)
        conjunto     mu = b + N g(p) + r N g'(p')      (r conocido)
        conjunto_r   lo mismo con r desconocido (parametro de molestia)"""
    cfg, K, idx, filas = tarea
    psf, esp, px = sim.PSFRecortada(cfg, K), sim.Espejo(cfg), cfg.pixel_nm
    assert not psf.completa
    fondo = sim.generar_fondo(cfg, sim.rngs_video(cfg.seed, idx)["fondo"])
    v, r, forma = sim._var_lectura_e2(cfg), cfg.ratio_reflejo, (cfg.H, cfg.W)
    T = transformacion_px(esp, px)
    b = fondo.ravel()
    out = []
    for pid, x, y, z, N, xr_gt, yr_gt, zr_gt in filas:
        xr, yr, zr = esp.reflejar(np.array([x, y, z]))
        assert abs(xr - xr_gt) < 1e-6 and abs(yr - yr_gt) < 1e-6 and abs(zr - zr_gt) < 1e-3
        g1, G1 = derivadas(psf, x, y, z, forma, HZ_CELULA_NM)
        g2, G2 = derivadas(psf, xr, yr, zr, forma, HZ_CELULA_NM)
        v_solo = var_parametros(g1, G1, N, b, v)
        s, Ju = g1 + r * g2, G1 + r * (T.T @ G2)
        v_conj = var_parametros(s, Ju, N, b, v)
        v_conj_r = var_parametros(s, Ju, N, b, v, extra=[N * g2]) if g2.sum() > 1e-4 else v_conj
        out.append(dict(video_id=idx, particle_id=pid, fondo_local=float(fondo[int(np.clip(round(y), 0, cfg.H - 1)),
                                                                               int(np.clip(round(x), 0, cfg.W - 1))]),
                        energia_real_en_fov=float(g1.sum()), energia_reflejo_en_fov=float(g2.sum()),
                        crlb_z_solo_nm=np.sqrt(v_solo[2]), crlb_z_conjunto_nm=np.sqrt(v_conj[2]),
                        crlb_z_conjunto_r_nm=np.sqrt(v_conj_r[2]),
                        crlb_lat_solo_nm=np.sqrt(v_solo[0] + v_solo[1]) * px,
                        crlb_lat_conjunto_nm=np.sqrt(v_conj[0] + v_conj[1]) * px))
    return out, float(fondo.mean())


def crlb_test_celula(carpeta, procesos):
    """DataFrame por emisor evaluable del test con celula: GT + sqrt(CRLB) de _crlb_muestra."""
    from concurrent.futures import ProcessPoolExecutor
    carpeta = Path(carpeta)
    meta = json.loads((carpeta / "config_simulador.json").read_text())
    cfg = sim.cargar_config(carpeta / "config_simulador.json")
    assert cfg.usar_celula and cfg.recorte_pupila_real and cfg.recorte_pupila_reflejo
    K = int(meta["K_psf_usado"])
    gt = pd.read_csv(carpeta / "ground_truth.csv")
    m = pd.read_csv(carpeta / "muestras.csv").set_index("video_id")
    assert (m.reintentos == 0).all()
    ev = gt[gt.en_fov & gt.en_rango_z]
    cols = ["particle_id", "x_px", "y_px", "z_nm", "photons", "x_reflejo_px", "y_reflejo_px", "z_reflejo_nm"]
    tareas = [(config_muestra_celula(cfg, m.loc[v]), K, int(v), g[cols].to_numpy(float).tolist())
              for v, g in ev.groupby("video_id")]
    filas = []
    with ProcessPoolExecutor(procesos) as ex:
        for (out, fondo_medio), t in zip(ex.map(_crlb_muestra, tareas, chunksize=4), tareas):
            assert np.isclose(fondo_medio, m.loc[t[2]].fondo_medio), "el fondo regenerado no es el del test"
            filas += out
    c = pd.DataFrame(filas)
    c["particle_id"] = c.particle_id.astype(int)
    px = cfg.pixel_nm
    g = ev.merge(c, on=["video_id", "particle_id"], validate="1:1")
    dif = np.c_[(g.x_reflejo_px - g.x_px) * px, (g.y_reflejo_px - g.y_px) * px, g.z_reflejo_nm - g.z_nm]
    g["d_espejo_nm"] = 0.5 * np.linalg.norm(dif, axis=1)
    g["ratio_reflejo"] = g.video_id.map(m.ratio_reflejo)
    return cfg, g


def vecinos_laterales(g, px):
    """Distancia lateral [nm] al otro real y al reflejo de otro emisor mas cercanos del frame (todo el GT)."""
    d_real, d_refl = np.full(len(g), np.inf), np.full(len(g), np.inf)
    xy = g[["x_px", "y_px"]].to_numpy(float) * px
    xr = g[["x_reflejo_px", "y_reflejo_px"]].to_numpy(float) * px
    for _, idx in g.groupby("video_id").indices.items():
        if len(idx) < 2:
            continue
        for a, out in ((xy, d_real), (xr, d_refl)):
            d = np.hypot(*(xy[idx][:, None] - a[idx][None]).transpose(2, 0, 1))
            np.fill_diagonal(d, np.inf)
            out[idx] = d.min(1)
    return d_real, d_refl


def tablas_celula(crlb, redes, n_boot, rng):
    """{variable: DataFrame} por rango: RMS de sqrt(CRLB) (de todos los evaluables y de
    los detectados por cada red) y RMSE z / recall de cada red, con IC 95% por bootstrap de
    imagenes; tambien el RMSE de los emisores aislados (AISLADO_NM)."""
    vids = np.unique(crlb.video_id)
    W = rng.multinomial(len(vids), np.full(len(vids), 1.0 / len(vids)), size=n_boot).astype(float)
    iv = np.searchsorted(vids, crlb.video_id)
    aislado = ((crlb.d_vecino_nm > AISLADO_NM) & (crlb.d_reflejo_vecino_nm > AISLADO_NM)).to_numpy()
    out = {}
    for nombre, (col, b, _, _) in BORDES_CELULA.items():
        nb = len(b) - 1
        k = np.clip(np.digitize(crlb[col], b[1:-1]), 0, nb - 1)

        def S(peso):
            M = np.bincount(iv * nb + k, weights=np.asarray(peso, float), minlength=len(vids) * nb).reshape(len(vids), nb)
            return M.sum(0), W @ M

        def rms_ic(peso_sel, valor):
            (n0, nB), (s0, sB) = S(peso_sel), S(np.where(peso_sel, valor, 0.0) ** 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                e, bs = np.sqrt(s0 / n0), np.sqrt(sB / nB)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
            return e, lo, hi

        t = pd.DataFrame({"desde": b[:-1], "hasta": b[1:], "n": S(np.ones(len(crlb)))[0]})
        todos = np.ones(len(crlb), bool)
        for c in ("crlb_z_solo_nm", "crlb_z_conjunto_nm", "crlb_z_conjunto_r_nm"):
            t[c] = rms_ic(todos, crlb[c].to_numpy())[0]
        for etiq, det, dz in redes:
            t[f"crlb_z_conjunto_detectados[{etiq}]"] = rms_ic(det, crlb.crlb_z_conjunto_nm.to_numpy())[0]
            t[f"crlb_z_solo_detectados[{etiq}]"] = rms_ic(det, crlb.crlb_z_solo_nm.to_numpy())[0]
            for sufijo, sel in (("", det), ("_aislados", det & aislado)):
                e, lo, hi = rms_ic(sel, dz)
                t[f"rmse_z{sufijo}[{etiq}]"], t[f"rmse_z{sufijo}[{etiq}]_lo"], t[f"rmse_z{sufijo}[{etiq}]_hi"] = e, lo, hi
            (n0, nB), (d0, dB) = S(todos), S(det)
            t[f"recall[{etiq}]"] = d0 / n0
            t[f"n_aislados_detectados[{etiq}]"] = S(det & aislado)[0]
        out[nombre] = t
    return out


def eje_x_log(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def figura_celula(variante, tablas, redes, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), height_ratios=[1.4, 1])
    for j, (nombre, (_, b, etiq_x, log)) in enumerate(BORDES_CELULA.items()):
        t = tablas[nombre]
        x = np.sqrt(t.desde * t.hasta) if log else (t.desde + t.hasta) / 2
        ax = axes[0, j]
        ax.plot(x, t.crlb_z_solo_nm, "-", color=COLOR["solo"], lw=2, label="CRLB, real solo (recortado)")
        ax.plot(x, t.crlb_z_conjunto_nm, "-", color=COLOR["conjunto"], lw=2, label="CRLB, real + reflejo (r conocido)")
        ax.plot(x, t.crlb_z_conjunto_r_nm, "--", color=COLOR["conjunto"], lw=1.5, label="CRLB, real + reflejo (r desconocido)")
        for k, etiq in enumerate(redes):
            c = f"rmse_z[{etiq}]"
            ax.errorbar(x, t[c], yerr=[t[c] - t[c + "_lo"], t[c + "_hi"] - t[c]], fmt=MARCAS_REDES[k % 4],
                        color=COLORES_REDES[k % 4], ms=6, mec="white", mew=1, capsize=2, ls="none",
                        label=f"red {etiq}: RMSE z de los detectados")
            c = f"rmse_z_aislados[{etiq}]"
            ax.plot(x, t[c], MARCAS_REDES[k % 4], mfc="none", mec=COLORES_REDES[k % 4], ms=7,
                    label=f"red {etiq}: solo aislados (> {AISLADO_NM / 1000:g} um de otro real y reflejo)")
            ax.plot(x, t[f"crlb_z_conjunto_detectados[{etiq}]"], ":", color=COLOR["conjunto"], lw=1.2,
                    label="CRLB conjunta de los detectados por la red")
        ax.set_yscale("log")
        if log:
            eje_x_log(ax)
        ax.set_ylim(5, 400)
        ax.set_ylabel("error en z [nm]")
        ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax2 = axes[1, j]
        for k, etiq in enumerate(redes):
            ax2.plot(x, t[f"rmse_z[{etiq}]"] / t[f"crlb_z_conjunto_detectados[{etiq}]"], "o-", color=COLORES_REDES[k % 4],
                     lw=1.8, ms=4, label="RMSE red / CRLB conjunta (detectados)")
            ax2.plot(x, t[f"rmse_z_aislados[{etiq}]"] / t[f"crlb_z_conjunto_detectados[{etiq}]"], "o--",
                     color=COLORES_REDES[k % 4], mfc="none", lw=1.2, ms=4, label="idem, aislados")
        ax2.plot(x, t.crlb_z_solo_nm / t.crlb_z_conjunto_nm, "-", color=COLOR["conjunto"], lw=1.8,
                 label="ganancia del reflejo: CRLB solo / conjunta")
        ax2.axhline(1, color=COLOR["solo"], lw=0.8)
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel(etiq_x)
        ax2.grid(True, color="#e6e5e0", lw=0.6)
        if log:
            eje_x_log(ax2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"CRLB por emisor del test con celula ({variante}): geometria, ratio y fondo de cada muestra, FOV 96 px, "
                 "recorte de pupila del real y del reflejo; sin otros emisores ni el prior de la celula", fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(ruta.with_suffix(".png"), dpi=200)
    fig.savefig(ruta.with_suffix(".pdf"))
    plt.close(fig)


def modo_celula(a, rng, salida):
    for variante in a.variantes:
        t0 = time.time()
        carpeta = Path(a.datos) / f"test_celula_b45_{variante}"
        cfg, crlb = crlb_test_celula(carpeta, a.procesos)
        if cfg.anillos_hacia_objetivo != VARIANTES[variante]:
            raise ValueError(f"{carpeta}: no es un test con la PSF '{variante}'")
        gt_todo = pd.read_csv(carpeta / "ground_truth.csv")
        dv, dr = vecinos_laterales(gt_todo, cfg.pixel_nm)
        gt_todo = gt_todo.assign(d_vecino_nm=dv, d_reflejo_vecino_nm=dr)
        crlb = crlb.merge(
            gt_todo[CLAVES + ["d_vecino_nm", "d_reflejo_vecino_nm"]], on=CLAVES, validate="1:1")
        print(f"\n{variante}: CRLB de {len(crlb)} emisores evaluables de {carpeta.name} ({time.time() - t0:.0f} s)")
        entradas = [(e, Path(c)) for e, c in a.red] if a.red else [
            (None, c) for c in sorted((RAIZ / "localizacion" / "corridas").glob(f"*/evaluacion_test_celula_b45_{variante}"))]
        redes = []
        for etiq, carpeta_eval in entradas:
            if variante_de_evaluacion(carpeta_eval) != variante:
                continue
            ge = pd.read_csv(carpeta_eval / "gt_eval.csv")
            ge = crlb[CLAVES].merge(ge[CLAVES + ["detectado", "dz_nm"]], on=CLAVES, how="left", validate="1:1")
            assert ge.detectado.notna().all(), f"{carpeta_eval}: otro GT"
            redes.append((etiq or etiqueta_red(carpeta_eval), ge.detectado.to_numpy(bool), ge.dz_nm.fillna(0.0).to_numpy()))
        tablas = tablas_celula(crlb, redes, a.n_boot, rng)
        nombre = f"b45_{variante}{a.sufijo}"
        cols = CLAVES + ["en_rango_z", "d_espejo_nm", "photons", "ratio_reflejo", "fondo_local", "energia_real_en_fov",
                         "energia_reflejo_en_fov", "crlb_z_solo_nm", "crlb_z_conjunto_nm", "crlb_z_conjunto_r_nm",
                         "crlb_lat_solo_nm", "crlb_lat_conjunto_nm", "d_vecino_nm", "d_reflejo_vecino_nm"]
        crlb[cols].to_csv(salida / f"crlb_celula_{nombre}.csv", index=False, float_format="%.5g")
        pd.concat([t.assign(tabla=k) for k, t in tablas.items()], ignore_index=True).to_csv(
            salida / f"crlb_celula_rangos_{nombre}.csv", index=False)
        figura_celula(variante, tablas, [r[0] for r in redes], salida / f"fig_crlb_celula_{nombre}")
        ais = (crlb.d_vecino_nm > AISLADO_NM) & (crlb.d_reflejo_vecino_nm > AISLADO_NM)
        print(f"  RMS sqrt(CRLB_z): solo {rms(crlb.crlb_z_solo_nm):.1f}, conjunto {rms(crlb.crlb_z_conjunto_nm):.1f}, "
              f"conjunto con r desconocido {rms(crlb.crlb_z_conjunto_r_nm):.1f} nm; lateral solo "
              f"{rms(crlb.crlb_lat_solo_nm):.1f}, conjunto {rms(crlb.crlb_lat_conjunto_nm):.1f} nm; aislados {ais.mean():.2f}")
        for etiq, det, dz in redes:
            print(f"  red {etiq}: RMSE z {rms(dz[det]):.1f} nm (aislados {rms(dz[det & ais.to_numpy()]):.1f}); "
                  f"CRLB conjunta de sus detectados {rms(crlb.crlb_z_conjunto_nm[det]):.1f}, solo "
                  f"{rms(crlb.crlb_z_solo_nm[det]):.1f} nm -> RMSE / CRLB conjunta {rms(dz[det]) / rms(crlb.crlb_z_conjunto_nm[det]):.2f}")
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3g}".format):
            for k, t in tablas.items():
                print(f"\n  por {k}:\n" + t[[c for c in t if not c.endswith(("_lo", "_hi"))]].to_string(index=False))
        print(f"  -> {salida / f'fig_crlb_celula_{nombre}.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--variantes", nargs="+", default=list(VARIANTES), choices=list(VARIANTES))
    p.add_argument("--red", nargs=2, action="append", metavar=("ETIQUETA", "CARPETA_EVALUACION"),
                   help="carpeta evaluacion_test_espejo_b45_<variante> de evaluate.py; sin --red se buscan en corridas/")
    p.add_argument("--datos", default=str(RAIZ / "localizacion" / "datos"),
                   help="carpeta con test_espejo_b45_lejos y test_espejo_b45_cerca (generar_espejo.py)")
    p.add_argument("--salida", default=str(Path(__file__).resolve().parent / "crlb_espejo"))
    p.add_argument("--sufijo", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000, help="remuestreos de imagenes para los IC de las redes")
    p.add_argument("--celula", action="store_true",
                   help="CRLB por emisor de test_celula_b45_<variante> contra evaluacion_test_celula_b45_<variante>")
    p.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="--celula: procesos")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    if a.celula:
        modo_celula(a, rng, salida)
        return

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
