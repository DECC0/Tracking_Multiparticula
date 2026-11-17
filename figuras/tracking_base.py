"""
Primera version del tracking: red de localizacion v2 (s1) + enlace LAP
clasico, validado en simulacion. Ver resultados/resultados_tracking_base.md.

Partes:
    datos      200 videos de 50 cuadros a 10 Hz con la config de la corrida s1
               (configs/celula_v2.yaml, cerca): 50 muestras base (celula,
               PSF, geometria, fotones, fondo y posiciones iniciales de la
               muestra idx de la semilla SEMILLA) x 4 valores de D (DS), con
               n = 1 + idx % 12 emisores por celula. Solo cambia la config del
               simulador (n_frames, D_rango_um2_s, n_emisores_rango); el GT ya
               trae la identidad (particle_id). -> localizacion/datos/tracking_celula_v2_cerca/
    localizar  la red s1 cuadro a cuadro (GPU si hay), con la geometria de
               cada video para el canal de distancia -> detecciones.csv en la
               misma carpeta
    analisis   error de la red (emparejamiento por cuadro, 150 / 300 nm),
               enlace (tracking/enlace.py) de tres entradas: las detecciones de
               la red, el GT con ruido gaussiano del tamano del error de la red
               y el GT exacto; metricas ISBI, identidades y D por MSD
               (tracking/metricas.py) por D y por densidad, con IC por
               bootstrap de videos -> figuras/tracking_base/

Enlace: z reescalada por sigma_lat / sigma_z (los RMSE de la red en estos
videos), radio = R_CHI sqrt(2 D dt + 2 sigma_lat^2) con el D de la condicion
(el radio de busqueda se elige por el D esperado, como se hace en la
practica), cierre de huecos de hasta 2 cuadros. Evaluacion: distancia en 3D
con z * 150/300 y eps = 150 nm (el elipsoide de 150 nm lateral y 300 nm axial).

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/tracking_base.py datos [--procesos 8]                       # ~3 min
    LD_PRELOAD=$HOME/miniforge3/envs/tesis/lib/libstdc++.so.6 $PY figuras/tracking_base.py localizar   # ~2 min
    $PY figuras/tracking_base.py analisis                                   # ~1 min
"""
import argparse
import glob
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "localizacion"))
sys.path.insert(0, str(RAIZ / "tracking"))
from config import cargar_config  # noqa: E402
from data.simulacion import sim, generar_muestra, evaluable, COLUMNAS_GEOMETRIA  # noqa: E402
from enlace import enlazar, coordenadas  # noqa: E402
from metricas import a_arreglo, isbi, de_sumas, identidades, d_msd, particula_mayoritaria  # noqa: E402

CORRIDA = Path(glob.glob(str(RAIZ / "localizacion" / "corridas" / "hypatia_celula_v2_b45_cerca_s1_*"))[0])
DATOS = RAIZ / "localizacion" / "datos" / "tracking_celula_v2_cerca"
SALIDA = Path(__file__).resolve().parent / "tracking_base"
SEMILLA = 5                       # distinta de seed_train 1, seed_val 2 y la del test (3)
N_BASE, T = 50, 50
DS = (0.001, 0.02, 0.2, 1.5)      # um^2/s
N_MAX = 12
DENSIDAD = {"1-4": (1, 4), "5-8": (5, 8), "9-12": (9, 12)}
R_XY, R_Z = 150.0, 300.0          # tolerancia de evaluacion [nm]
EPS = 150.0                       # ISBI: distancia en (x, y, z * R_XY / R_Z) [nm]
R_CHI = 4.59                      # sqrt(chi2_3(0.9999)): radio de busqueda en sigmas del paso; con el
                                  # cuantil 99% (3.37) el 39% de las pistas de 50 cuadros se cortaba solo por el ruido
MAX_HUECO = 2
N_BOOT = 1000
ENTRADAS = ("red", "gt_ruido", "gt")
NOMBRE = {"red": "detecciones de la red", "gt_ruido": "GT + ruido gaussiano (error de la red)", "gt": "GT exacto"}
COLOR = {"red": "#2a78d6", "gt_ruido": "#eb6834", "gt": "#52514e"}


def _config():
    cfg = cargar_config(CORRIDA / "config.yaml")
    return cfg


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------

_G = {}


def _video(args):
    k, idx = args
    cfg_datos, cfg_sim, banco = _G["cfg_datos"], _G["cfg_sim"], _G["banco"]
    n = 1 + idx % N_MAX
    cv = replace(cfg_sim, n_frames=T, D_rango_um2_s=(DS[k], DS[k]), n_emisores_rango=(n, n))
    video, gt, meta = generar_muestra(cfg_datos, cv, banco, idx)
    vid = k * N_BASE + idx
    gt["video_id"] = vid
    meta.update(video_id=vid, muestra=idx, D_um2_s=DS[k], n_set=n, n_emisores_evaluables=int(evaluable(gt[gt.frame == 0]).sum()))
    return vid, video, gt, meta


def datos(procesos):
    from multiprocessing import get_context
    from data.dataset import DatasetSimulado
    cfg = _config().datos
    ds = DatasetSimulado(cfg, 1, SEMILLA)
    _G.update(cfg_datos=cfg, cfg_sim=ds.cfg_sim, banco=ds.banco)
    DATOS.mkdir(parents=True, exist_ok=True)
    n_v = len(DS) * N_BASE
    imgs = np.lib.format.open_memmap(DATOS / "imagenes.npy", mode="w+", dtype=np.uint16,
                                     shape=(n_v, T, ds.cfg_sim.H, ds.cfg_sim.W))
    tablas, metas = [], []
    t0 = time.time()
    trabajos = [(k, i) for k in range(len(DS)) for i in range(N_BASE)]
    with get_context("fork").Pool(procesos) as pool:
        for j, (vid, video, gt, meta) in enumerate(pool.imap_unordered(_video, trabajos)):
            imgs[vid] = video
            tablas.append(gt)
            metas.append(meta)
            if (j + 1) % 20 == 0:
                print(f"  {j + 1}/{n_v} videos ({time.time() - t0:.0f} s)")
    imgs.flush()
    gt = pd.concat(tablas, ignore_index=True).sort_values(["video_id", "frame", "particle_id"])
    gt.to_csv(DATOS / "ground_truth.csv", index=False)
    pd.DataFrame(metas).sort_values("video_id").to_csv(DATOS / "muestras.csv", index=False)
    sim.guardar_config(replace(ds.cfg_sim, n_frames=T), DATOS / "config_simulador.json", ds.banco)
    print(f"{n_v} videos ({T} cuadros, dt {ds.cfg_sim.dt_s} s) -> {DATOS}  [{time.time() - t0:.0f} s; "
          f"{gt.groupby('video_id').particle_id.nunique().sum()} trayectorias]")


# ----------------------------------------------------------------------------
# Localizacion
# ----------------------------------------------------------------------------

def localizar_videos():
    from utils import cargar_modelo, elegir_dispositivo
    from infer import localizar
    from data.simulacion import config_simulador
    disp = elegir_dispositivo("auto")
    modelo, cfg, _ = cargar_modelo(CORRIDA / "mejor.pt", disp)
    cs = config_simulador(cfg.datos, cfg.datos.seed_train)
    imgs = np.load(DATOS / "imagenes.npy", mmap_mode="r")
    m = pd.read_csv(DATOS / "muestras.csv").set_index("video_id")
    partes = []
    t0 = time.time()
    for v in range(len(imgs)):
        geo = pd.DataFrame([m.loc[v, COLUMNAS_GEOMETRIA].to_numpy()] * T, columns=COLUMNAS_GEOMETRIA)
        d = localizar(modelo, np.ascontiguousarray(imgs[v]), cfg, replace(cs, H=imgs.shape[2], W=imgs.shape[3]), disp,
                      geometria=geo)
        partes.append(d.rename(columns={"t": "frame"}).assign(video_id=v))
    det = pd.concat(partes, ignore_index=True)[["video_id", "frame", "x_px", "y_px", "z_nm", "conf"]]
    det.to_csv(DATOS / "detecciones.csv", index=False)
    print(f"{len(det)} detecciones en {len(imgs) * T} cuadros ({time.time() - t0:.0f} s) -> {DATOS / 'detecciones.csv'}")


# ----------------------------------------------------------------------------
# Analisis
# ----------------------------------------------------------------------------

def emparejar_cuadro(g, d, px):
    """Pares (i, j) GT / deteccion de un cuadro con tolerancia cilindrica R_XY / R_Z (hungaro)."""
    from scipy.optimize import linear_sum_assignment
    if len(g) == 0 or len(d) == 0:
        return np.zeros((0, 2), int)
    dl = np.hypot((g.x_px.to_numpy()[:, None] - d.x_px.to_numpy()[None]) * px,
                  (g.y_px.to_numpy()[:, None] - d.y_px.to_numpy()[None]) * px)
    dz = np.abs(g.z_nm.to_numpy()[:, None] - d.z_nm.to_numpy()[None])
    ok = (dl <= R_XY) & (dz <= R_Z)
    c = np.where(ok, (dl / R_XY) ** 2 + (dz / R_Z) ** 2, 1e9)
    f, k = linear_sum_assignment(c)
    s = ok[f, k]
    return np.c_[f[s], k[s]]


def error_red(gt, det, px):
    """Emparejamiento por cuadro de las detecciones con el GT evaluable: errores por par y conteos por video."""
    filas, cuentas = [], []
    dg = {k: g for k, g in gt.groupby(["video_id", "frame"])}
    dd = {k: g for k, g in det.groupby(["video_id", "frame"])}
    for v in sorted(gt.video_id.unique()):
        tp = fp = fn = 0
        for t in range(T):
            g = dg.get((v, t), gt.iloc[:0])
            d = dd.get((v, t), det.iloc[:0])
            p = emparejar_cuadro(g, d, px)
            tp += len(p); fn += len(g) - len(p); fp += len(d) - len(p)
            if len(p):
                a, b = g.iloc[p[:, 0]], d.iloc[p[:, 1]]
                filas.append(pd.DataFrame(dict(video_id=v, dx_nm=(b.x_px.to_numpy() - a.x_px.to_numpy()) * px,
                                               dy_nm=(b.y_px.to_numpy() - a.y_px.to_numpy()) * px,
                                               dz_nm=b.z_nm.to_numpy() - a.z_nm.to_numpy())))
        cuentas.append(dict(video_id=v, tp=tp, fp=fp, fn=fn))
    return pd.concat(filas, ignore_index=True), pd.DataFrame(cuentas)


def vecino_mas_cercano(gt, px):
    """Mediana por video de la distancia 3D y lateral [nm] de cada emisor a su vecino mas cercano (todos los cuadros)."""
    out = {}
    for v, g in gt.groupby("video_id"):
        d3, dl = [], []
        for _, f in g.groupby("frame"):
            if len(f) < 2:
                continue
            p = np.c_[f.x_px * px, f.y_px * px, f.z_nm]
            for lista, q in ((d3, p), (dl, p[:, :2])):
                d = np.sqrt(((q[:, None] - q[None]) ** 2).sum(-1))
                np.fill_diagonal(d, np.inf)
                lista.append(d.min(1))
        out[v] = ((float(np.median(np.concatenate(d3))), float(np.median(np.concatenate(dl)))) if d3 else (np.nan, np.nan))
    return pd.DataFrame(out, index=["vecino_nm", "vecino_lateral_nm"]).T


def analizar_video(v, entrada, pts, gt_v, D, sigma_lat, escala_z, px):
    """Enlaza pts (DataFrame frame, x_px, y_px, z_nm) y devuelve las sumas ISBI, identidades y D por MSD."""
    radio = R_CHI * np.sqrt(2 * D * 1e6 * 0.1 + 2 * sigma_lat ** 2)
    fr = pts.frame.to_numpy()
    pista = enlazar(fr, coordenadas(pts, px, escala_z), radio, MAX_HUECO)
    ev = lambda df: np.c_[df.x_px.to_numpy() * px, df.y_px.to_numpy() * px, df.z_nm.to_numpy() * R_XY / R_Z]  # noqa: E731
    X, _ = a_arreglo(gt_v.particle_id.to_numpy(), gt_v.frame.to_numpy(), ev(gt_v), T)
    Y, _ = a_arreglo(pista, fr, ev(pts), T) if len(pts) else (np.zeros((0, T, 3)), None)
    s = isbi(X, Y, EPS)
    idsw, frag, n_ids, emp = identidades(X, Y, EPS, emparejamiento=True)
    mayor, comparte = particula_mayoritaria(emp, len(Y))
    fila = dict(video_id=v, entrada=entrada, radio_nm=radio, **s, idsw=int(idsw.sum()), frag=int(frag.sum()),
                n_ids=int(n_ids.sum()), pistas_gt_largas=int((np.isfinite(X[..., 0]).sum(1) >= 10).sum()))
    # D por MSD de cada pista estimada con >= 10 puntos (en nm, z sin reescalar), y la referencia: el
    # mismo estimador sobre la pista GT completa de su particula mayoritaria
    R, _ = a_arreglo(pista, fr, np.c_[pts.x_px * px, pts.y_px * px, pts.z_nm], T) if len(pts) else (np.zeros((0, T, 3)), None)
    G, pids = a_arreglo(gt_v.particle_id.to_numpy(), gt_v.frame.to_numpy(),
                        np.c_[gt_v.x_px * px, gt_v.y_px * px, gt_v.z_nm], T)
    ref3 = np.array([d_msd(g, 0.1) for g in G]) / 1e6
    refz = np.array([d_msd(g[:, 2:], 0.1) for g in G]) / 1e6
    msd = []
    for j, r in enumerate(R):
        k = mayor[j]
        msd.append(dict(video_id=v, entrada=entrada, D_um2_s=D, puntos=int(np.isfinite(r[:, 0]).sum()),
                        D3_um2_s=d_msd(r, 0.1) / 1e6, Dz_um2_s=d_msd(r[:, 2:], 0.1) / 1e6,
                        particle_id=int(pids[k]) if k >= 0 else -1, cuadros_compartidos=int(comparte[j]),
                        D3_ref_um2_s=ref3[k] if k >= 0 else np.nan, Dz_ref_um2_s=refz[k] if k >= 0 else np.nan))
    return fila, [m for m in msd if np.isfinite(m["D3_um2_s"])]


def bootstrap_grupo(t, rng, n_boot=N_BOOT):
    """alpha, beta, JSC, JSC_theta e identidades por pista de un grupo de videos, con IC por bootstrap de videos."""
    cols = ["d_xy", "d_x0", "d_ybar0", "tp_det", "fn_det", "fp_det", "tp_pistas", "fn_pistas", "fp_pistas",
            "idsw", "frag", "n_ids", "n_pistas_gt"]
    S = t[cols].to_numpy(float)

    def est(s):
        d = dict(zip(cols, s))
        m = de_sumas(d)
        m.update(idsw_por_pista=d["idsw"] / d["n_pistas_gt"], frag_por_pista=d["frag"] / d["n_pistas_gt"],
                 ids_por_pista=d["n_ids"] / d["n_pistas_gt"])
        return m
    e = est(S.sum(0))
    w = rng.multinomial(len(S), np.full(len(S), 1 / len(S)), size=n_boot)
    bs = pd.DataFrame([est(wi @ S) for wi in w])
    out = dict(videos=len(t), pistas_gt=int(t.n_pistas_gt.sum()), pistas_est=int(t.n_pistas_est.sum()))
    for k, v in e.items():
        out[k] = v
        out[k + "_lo"], out[k + "_hi"] = np.nanpercentile(bs[k], [2.5, 97.5])
    return out


def analisis():
    rng = np.random.default_rng(0)
    SALIDA.mkdir(parents=True, exist_ok=True)
    cfg_sim = sim.cargar_config(DATOS / "config_simulador.json")
    px = cfg_sim.pixel_nm
    gt_todo = pd.read_csv(DATOS / "ground_truth.csv")
    gt = gt_todo[evaluable(gt_todo)].copy()
    det = pd.read_csv(DATOS / "detecciones.csv")
    m = pd.read_csv(DATOS / "muestras.csv").set_index("video_id")
    m["densidad"] = pd.cut(m.n_set, [0, 4, 8, 12], labels=list(DENSIDAD))
    m = m.join(vecino_mas_cercano(gt, px))

    # error de la red en estos videos
    err, cuentas = error_red(gt, det, px)
    sx, sy, sz = (float(np.sqrt(np.mean(err[c] ** 2))) for c in ("dx_nm", "dy_nm", "dz_nm"))
    sigma_lat = float(np.sqrt((sx ** 2 + sy ** 2) / 2))
    escala_z = sigma_lat / sz
    c = cuentas.merge(m[["D_um2_s", "densidad"]], left_on="video_id", right_index=True)
    e = err.merge(m[["D_um2_s"]], left_on="video_id", right_index=True)
    filas_red = []
    for D, s in c.groupby("D_um2_s"):
        ee = e[e.D_um2_s == D]
        filas_red.append(dict(D_um2_s=D, recall=s.tp.sum() / (s.tp.sum() + s.fn.sum()),
                              precision=s.tp.sum() / (s.tp.sum() + s.fp.sum()),
                              fp_por_cuadro=s.fp.sum() / (len(s) * T),
                              rmse_x_nm=np.sqrt(np.mean(ee.dx_nm ** 2)), rmse_y_nm=np.sqrt(np.mean(ee.dy_nm ** 2)),
                              rmse_z_nm=np.sqrt(np.mean(ee.dz_nm ** 2))))
    filas_red.append(dict(D_um2_s=np.nan, recall=c.tp.sum() / (c.tp.sum() + c.fn.sum()),
                          precision=c.tp.sum() / (c.tp.sum() + c.fp.sum()), fp_por_cuadro=c.fp.sum() / (len(c) * T),
                          rmse_x_nm=sx, rmse_y_nm=sy, rmse_z_nm=sz))
    t_red = pd.DataFrame(filas_red)
    t_red.to_csv(SALIDA / "tracking_base_red.csv", index=False)
    print(f"error de la red: RMSE x {sx:.1f}, y {sy:.1f}, z {sz:.1f} nm -> sigma_lat {sigma_lat:.1f} nm, "
          f"escala_z {escala_z:.3f}")
    print(t_red.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    dens = m.groupby("densidad", observed=True).agg(videos=("n_set", "size"), n_medio=("n_set", "mean"),
                                                    vecino_mediana_nm=("vecino_nm", "median"),
                                                    vecino_lateral_mediana_nm=("vecino_lateral_nm", "median"),
                                                    diametro_celula_um=("celula_diametro_um", "mean"))
    dens.to_csv(SALIDA / "tracking_base_densidad.csv")
    print(f"\nvecino mas cercano (3D, mediana por video) por densidad:\n{dens.to_string(float_format=lambda v: f'{v:.0f}')}")

    # entradas: red, GT + ruido, GT exacto
    ruido = gt.copy()
    ruido["x_px"] += rng.normal(0, sx, len(ruido)) / px
    ruido["y_px"] += rng.normal(0, sy, len(ruido)) / px
    ruido["z_nm"] += rng.normal(0, sz, len(ruido))
    fuentes = {"red": det, "gt_ruido": ruido, "gt": gt}
    filas, msd = [], []
    gv = {v: g for v, g in gt.groupby("video_id")}
    t0 = time.time()
    for entrada, df in fuentes.items():
        por_v = {v: g for v, g in df.groupby("video_id")}
        for v in m.index:
            pts = por_v.get(v, df.iloc[:0])
            f, ms = analizar_video(v, entrada, pts, gv.get(v, gt.iloc[:0]), m.loc[v, "D_um2_s"], sigma_lat, escala_z, px)
            filas.append(f)
            msd += ms
    print(f"\nenlace y metricas: {time.time() - t0:.0f} s")
    # sensibilidad: z sin reescalar (escala_z = 1, radio con el error medio por eje) a D alto
    sens = []
    sigma_eje = float(np.sqrt((sx ** 2 + sy ** 2 + sz ** 2) / 3))
    for entrada, df in fuentes.items():
        por_v = {v: g for v, g in df.groupby("video_id")}
        for v in m.index[m.D_um2_s >= 0.2]:
            for esc, sig in ((escala_z, sigma_lat), (1.0, sigma_eje)):
                f, _ = analizar_video(v, entrada, por_v.get(v, df.iloc[:0]), gv.get(v, gt.iloc[:0]), m.loc[v, "D_um2_s"],
                                      sig, esc, px)
                sens.append(dict(f, escala_z=esc))
    ts = pd.DataFrame(sens).merge(m[["D_um2_s"]], left_on="video_id", right_index=True)
    filas_s = []
    for (entrada, D, esc), s in ts.groupby(["entrada", "D_um2_s", "escala_z"]):
        filas_s.append(dict(entrada=entrada, D_um2_s=D, escala_z=esc, **bootstrap_grupo(s, rng)))
    tsens = pd.DataFrame(filas_s)
    tsens.to_csv(SALIDA / "tracking_base_sensibilidad_z.csv", index=False)
    with pd.option_context("display.width", 250):
        print("\nsensibilidad a la escala de z en el enlace:")
        print(tsens[["entrada", "D_um2_s", "escala_z", "alpha", "beta", "jsc", "jsc_pistas", "idsw_por_pista",
                     "frag_por_pista"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # MSD de las pistas GT reales (sin enlace): referencia del estimador (confinamiento)
    for v, g in gv.items():
        R, _ = a_arreglo(g.particle_id.to_numpy(), g.frame.to_numpy(), np.c_[g.x_px * px, g.y_px * px, g.z_nm], T)
        for r in R:
            d3 = d_msd(r, 0.1)
            if np.isfinite(d3):
                dz = d_msd(r[:, 2:], 0.1) / 1e6
                msd.append(dict(video_id=v, entrada="pistas_gt", D_um2_s=m.loc[v, "D_um2_s"], puntos=int(np.isfinite(r[:, 0]).sum()),
                                D3_um2_s=d3 / 1e6, Dz_um2_s=dz, D3_ref_um2_s=d3 / 1e6, Dz_ref_um2_s=dz))
    tv = pd.DataFrame(filas).merge(m[["D_um2_s", "densidad", "n_set"]], left_on="video_id", right_index=True)
    tv.to_csv(SALIDA / "tracking_base_videos.csv", index=False)
    tm = pd.DataFrame(msd).merge(m[["densidad"]], left_on="video_id", right_index=True)

    # agregados por D, por densidad y por D x densidad
    agg = []
    for entrada in ENTRADAS:
        s = tv[tv.entrada == entrada]
        for D, sd in s.groupby("D_um2_s"):
            agg.append(dict(entrada=entrada, D_um2_s=D, densidad="todas", **bootstrap_grupo(sd, rng)))
            for de, sdd in sd.groupby("densidad", observed=True):
                agg.append(dict(entrada=entrada, D_um2_s=D, densidad=de, **bootstrap_grupo(sdd, rng)))
        for de, sdd in s.groupby("densidad", observed=True):
            agg.append(dict(entrada=entrada, D_um2_s=np.nan, densidad=de, **bootstrap_grupo(sdd, rng)))
    ta = pd.DataFrame(agg)
    ta.to_csv(SALIDA / "tracking_base_metricas.csv", index=False)
    # D por MSD: cada pista contra el mismo estimador sobre la pista GT de su particula (referencia
    # principal: incluye el confinamiento y los lags) y, como referencia, contra el D nominal
    tm["r3_ref"] = tm.D3_um2_s / tm.D3_ref_um2_s
    tm["rz_ref"] = tm.Dz_um2_s / tm.Dz_ref_um2_s
    tm["r3_nom"] = tm.D3_um2_s / tm.D_um2_s
    tm["rz_nom"] = tm.Dz_um2_s / tm.D_um2_s
    tm.to_csv(SALIDA / "tracking_base_msd_pistas.csv", index=False)
    dm = []
    for (entrada, D), s in tm.groupby(["entrada", "D_um2_s"]):
        for de, sd in [("todas", s)] + list(s.groupby("densidad", observed=True)):
            con = sd[sd.D3_ref_um2_s.notna()]
            fila = dict(entrada=entrada, D_um2_s=D, densidad=de, pistas=len(sd), pistas_sin_particula=int(len(sd) - len(con)),
                        puntos_medio=sd.puntos.mean(), D3_mediana=sd.D3_um2_s.median(), D3_ref_mediana=con.D3_ref_um2_s.median(),
                        Dz_mediana=sd.Dz_um2_s.median(), Dz_ref_mediana=con.Dz_ref_um2_s.median(),
                        frac_negativos_3d=(sd.D3_um2_s <= 0).mean(), frac_negativos_z=(sd.Dz_um2_s <= 0).mean())
            for k, col, base in (("3_ref", "r3_ref", con), ("z_ref", "rz_ref", con), ("3_nom", "r3_nom", sd), ("z_nom", "rz_nom", sd)):
                r = base[col]
                fila.update({f"ratio{k}_mediana": r.median(), f"ratio{k}_p25": r.quantile(.25), f"ratio{k}_p75": r.quantile(.75),
                             f"err{k}_rel_mediano": (r - 1).abs().median()})
            dm.append(fila)
    td = pd.DataFrame(dm)
    td.to_csv(SALIDA / "tracking_base_msd.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_columns", 40):
        cols = ["entrada", "D_um2_s", "densidad", "videos", "pistas_gt", "pistas_est", "alpha", "alpha_lo", "alpha_hi",
                "beta", "beta_lo", "beta_hi", "jsc", "jsc_lo", "jsc_hi", "jsc_pistas", "jsc_pistas_lo", "jsc_pistas_hi",
                "idsw_por_pista", "frag_por_pista", "ids_por_pista"]
        print(ta[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(td.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    figuras(ta, td, tm, m, gt, fuentes, sigma_lat, escala_z, px)


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------

def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def figuras(ta, td, tm, m, gt, fuentes, sigma_lat, escala_z, px):
    plt = _estilo()
    metricas = [("alpha", "alpha"), ("beta", "beta"), ("jsc", "JSC detecciones"), ("jsc_pistas", "JSC pistas"),
                ("idsw_por_pista", "cambios de identidad / pista"), ("frag_por_pista", "interrupciones / pista")]
    filas = ["todas"] + list(DENSIDAD)
    fig, axs = plt.subplots(len(filas), len(metricas), figsize=(16, 10), sharex=True)
    for i, de in enumerate(filas):
        for j, (k, et) in enumerate(metricas):
            ax = axs[i, j]
            for entrada in ENTRADAS:
                s = ta[(ta.entrada == entrada) & (ta.densidad == de) & ta.D_um2_s.notna()].sort_values("D_um2_s")
                ax.plot(s.D_um2_s, s[k], "o-", color=COLOR[entrada], lw=1.4, ms=3.5, label=NOMBRE[entrada])
                if k + "_lo" in s:
                    ax.fill_between(s.D_um2_s, s[k + "_lo"], s[k + "_hi"], color=COLOR[entrada], alpha=0.15, lw=0)
            ax.set_xscale("log")
            if k in ("alpha", "beta", "jsc", "jsc_pistas"):
                ax.set_ylim(0, 1.02)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            if i == 0:
                ax.set_title(et, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"emisores por celula: {de}")
            if i == len(filas) - 1:
                ax.set_xlabel("D [um^2/s]")
    axs[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.suptitle(f"Tracking base (LAP, huecos <= {MAX_HUECO}): metricas ISBI e identidades por D y densidad "
                 f"(50 videos por D; bandas: IC 95% bootstrap de videos; eps = {EPS:.0f} nm con z x {R_XY / R_Z:g})",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_base_metricas.png", dpi=300)
    plt.close(fig)

    # D por MSD
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.4))
    orden = ["pistas_gt"] + list(ENTRADAS)
    col = dict(COLOR, pistas_gt="#1baf7a")
    nom = dict(NOMBRE, pistas_gt="pistas GT (sin enlace ni ruido)")
    paneles = (("r3_ref", "3D, contra el estimador sobre la pista GT"), ("rz_ref", "solo z, contra el estimador sobre la pista GT"),
               ("r3_nom", "3D, contra el D nominal"), ("rz_nom", "solo z, contra el D nominal"))
    for ax, (c, et) in zip(axs.flat, paneles):
        for k, D in enumerate(DS):
            for j, e in enumerate(orden):
                s = tm[(tm.entrada == e) & (tm.D_um2_s == D)][c].dropna()
                pos = k * (len(orden) + 1) + j
                if len(s):
                    bp = ax.boxplot(np.clip(s, 1e-3, 1e4), positions=[pos], widths=0.7, showfliers=False, patch_artist=True)
                    for b in bp["boxes"]:
                        b.set_facecolor(col[e]); b.set_alpha(0.6)
                    ax.text(pos, 3e3, f"{len(s)}", ha="center", fontsize=5.5, rotation=90)
        ax.set_yscale("log")
        ax.axhline(1, color="#8a8984", lw=0.8)
        ax.set_xticks([k * (len(orden) + 1) + 1.5 for k in range(len(DS))], [f"D = {D:g}" for D in DS])
        ax.set_ylabel("D estimado / referencia")
        ax.set_ylim(1e-3, 1e4)
        ax.set_title(f"{et} (numero: pistas)", fontsize=8.5)
        ax.grid(True, color="#e6e5e0", lw=0.6, axis="y")
    for e in orden:
        axs[0, 0].plot([], [], "s", color=col[e], alpha=0.6, label=nom[e])
    axs[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("D por MSD: ajuste MSD = 2 n D tau + 2 n sigma^2 en tau = 1-4 cuadros, pistas con >= 10 puntos; "
                 "valores <= 0 recortados a 1e-3", fontsize=9)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_base_msd.png", dpi=300)
    plt.close(fig)

    # ejemplo: un video denso con D = 0.2 y otro con D = 1.5 (misma muestra base), x(t) y z(t)
    idx = int(m[(m.n_set == 9) & (m.D_um2_s == DS[0])].muestra.iloc[0])
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True)
    for j, D in enumerate((DS[2], DS[3])):
        v = int(m[(m.muestra == idx) & (m.D_um2_s == D)].index[0])
        g = gt[gt.video_id == v]
        for i, (entrada, c, et) in enumerate((("gt", "x_px", "x [px]"), ("gt", "z_nm", "z [nm]"))):
            ax = axs[i, 2 * j]
            for pid, s in g.groupby("particle_id"):
                ax.plot(s.frame, s[c], "-", lw=1.2)
            ax.set_title(f"GT, D = {D:g} um^2/s, {g.particle_id.nunique()} emisores", fontsize=8.5)
            ax.set_ylabel(et)
            ax = axs[i, 2 * j + 1]
            d = fuentes["red"][fuentes["red"].video_id == v]
            radio = R_CHI * np.sqrt(2 * D * 1e6 * 0.1 + 2 * sigma_lat ** 2)
            p = enlazar(d.frame.to_numpy(), coordenadas(d, px, escala_z), radio, MAX_HUECO)
            for k in np.unique(p):
                s = d[p == k]
                ax.plot(s.frame, s[c], "-o", lw=0.9, ms=1.5)
            ax.set_title(f"red + LAP: {len(np.unique(p))} pistas", fontsize=8.5)
    for ax in axs[1]:
        ax.set_xlabel("cuadro (10 Hz)")
    for ax in axs.flat:
        ax.grid(True, color="#e6e5e0", lw=0.6)
    fig.suptitle(f"Ejemplo: muestra base {idx} (mismas condiciones iniciales) con D = {DS[2]:g} y {DS[3]:g} um^2/s; "
                 "cada color es una pista", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_base_ejemplo.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("partes", nargs="+", choices=("datos", "localizar", "analisis"))
    p.add_argument("--procesos", type=int, default=8)
    a = p.parse_args()
    if "datos" in a.partes:
        datos(a.procesos)
    if "localizar" in a.partes:
        localizar_videos()
    if "analisis" in a.partes:
        analisis()


if __name__ == "__main__":
    main()
