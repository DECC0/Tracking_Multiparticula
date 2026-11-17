"""
Linea base clasica de deteccion en simulacion: el detector de manchas de
tracking/reflejos.py (diferencia de gaussianas, maximos locales sobre k sigmas
de ruido y ajuste gaussiano 2D) aplicado a las particulas reales, frente a la
red s1 (hypatia_celula_v2_b45_cerca_s1), sin reentrenar. Ver
resultados/resultados_linea_base_clasica.md.

Datos:
    test v2      localizacion/datos/test_celula_v2_b45_cerca (2000 imagenes de 96 px);
                 la red: predicciones.csv de su evaluacion local (umbral 0.4)
    tracking     los 200 videos de tracking_base.py (50 cuadros, 4 D x 50 muestras
                 base); la red: detecciones_u02.csv de tracking_3d.py con conf >= 0.4
Mitades: el umbral k y el radio r del detector se eligen en la de seleccion
(test v2: imagenes 0-999; tracking: muestras base 0-24) y todo lo demas se
evalua en la de prueba (imagenes 1000-1999; muestras 25-49).

Partes:
    detectar   el detector con k = 3 (el umbral mayor es un subconjunto: se filtra
               por snr) y r = 1, 2, 3 en los dos conjuntos -> CRUDOS (no versionado)
    analisis   ajuste de (r, k), deteccion emisor a emisor (recall, precision y RMSE
               lateral por fotones, fondo, vecino, autofluorescencia y z; reflejos
               detectados como particulas aparte) y tracking con el LAP mejorado de
               tracking_3d (JSC, cambios de identidad, error de D lateral, por D y
               densidad) -> figuras/linea_base_clasica/

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/linea_base_clasica.py detectar [--procesos 12]   # ~10 min
    $PY figuras/linea_base_clasica.py analisis [--procesos 12]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
import tracking_base as tb  # noqa: E402  (inserta localizacion/ y tracking/ en sys.path)
import tracking_3d as t3  # noqa: E402

TEST = RAIZ / "localizacion" / "datos" / "test_celula_v2_b45_cerca"
PRED_TEST = tb.CORRIDA / "evaluacion_local_test_celula_v2_b45_cerca" / "predicciones.csv"
TRACK = tb.DATOS
CRUDOS = RAIZ / "localizacion" / "datos" / "linea_base_clasica"
SALIDA = AQUI / "linea_base_clasica"
CONJUNTOS = ("test", "tracking")
NOMBRE_CONJ = {"test": "test v2 (2000 imagenes)", "tracking": "videos de tracking"}
RADIOS = (1, 2, 3)
K_MIN = 3.0
KS = np.round(np.arange(3.0, 14.01, 0.5), 2)
UMBRAL_RED = 0.4
TOL_NM = tb.R_XY                  # 150 nm lateral (y 300 en z para la red con el criterio 3D)
T, DT = tb.T, 0.1
N_BOOT = 1000
METODOS = ("clasico", "red")
NOMBRE = {"clasico": "detector clasico", "red": "red s1", "red_3d": "red s1, enlace 3D", "red_2d": "red s1, enlace 2D",
          "clasico_2d": "detector clasico, enlace 2D", "clasico_2d_oraculo": "clasico sin reflejos (oraculo), enlace 2D"}
COLOR = {"clasico": "#eb6834", "red": "#2a78d6", "red_3d": "#2a78d6", "red_2d": "#7fb0ea", "clasico_2d": "#eb6834",
         "clasico_2d_oraculo": "#f2a477",
         "gris": "#8a8984"}
BINS = {"photons": np.array([1000, 2000, 5000, 10000, 20000, 30000, 50000]),
        "fondo_medio": np.array([0, 3, 6, 12, 25, 50, 200]),
        "vecino_nm": np.array([0, 300, 500, 1000, 2000, 1e9]),
        "contraste_celula": np.array([0, 0.25, 0.5, 1, 2, 4]),
        "z_nm": np.arange(-3000, 3001, 1000)}
ETIQ = {"photons": "fotones", "fondo_medio": "fondo [e-/px]", "vecino_nm": "vecino mas cercano [nm]",
        "contraste_celula": "contraste de la autofluorescencia", "z_nm": "z [nm]"}


def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


# ----------------------------------------------------------------------------
# datos
# ----------------------------------------------------------------------------

def imagenes(conj):
    d = TEST if conj == "test" else TRACK
    return np.load(d / "imagenes.npy", mmap_mode="r"), tb.sim.cargar_config(d / "config_simulador.json")


def cargar(conj):
    """GT (todo), muestras (con contraste, mitad, densidad) y detecciones de la red (conf >= 0.4)."""
    d = TEST if conj == "test" else TRACK
    gt = pd.read_csv(d / "ground_truth.csv")
    m = pd.read_csv(d / "muestras.csv").set_index("video_id")
    m["contraste_celula"] = m.fondo_celula / np.maximum(m.fondo_medio, 1e-9)
    if conj == "test":
        m["mitad"] = np.where(m.index < 1000, "seleccion", "prueba")
        red = pd.read_csv(PRED_TEST)
    else:
        m["mitad"] = np.where(m.muestra < 25, "seleccion", "prueba")
        m["densidad"] = pd.cut(m.n_set, [0, 4, 8, 12], labels=list(tb.DENSIDAD))
        red = pd.read_csv(TRACK / "detecciones_u02.csv")
    red = red[red.conf >= UMBRAL_RED].reset_index(drop=True)
    gt["evaluable"] = tb.evaluable(gt)
    return gt, m, red


# ----------------------------------------------------------------------------
# detectar
# ----------------------------------------------------------------------------

_G = {}


def _trabajo_detectar(v):
    from reflejos import detectar_manchas
    img = _G["imgs"][v]
    partes = []
    for f in range(img.shape[0]):
        e = (img[f].astype(float) - _G["offset"]) * _G["gain"]
        partes.append(pd.DataFrame(detectar_manchas(e, k=K_MIN, r=_G["r"])).assign(video_id=v, frame=f))
    return pd.concat(partes, ignore_index=True)


def detectar(procesos):
    from multiprocessing import get_context
    CRUDOS.mkdir(parents=True, exist_ok=True)
    for conj in CONJUNTOS:
        imgs, cs = imagenes(conj)
        for r in RADIOS:
            t0 = time.time()
            _G.update(imgs=imgs, gain=cs.gain, offset=cs.offset, r=r)
            with get_context("fork").Pool(procesos) as pool:
                partes = pool.map(_trabajo_detectar, range(len(imgs)), chunksize=4)
            c = pd.concat(partes, ignore_index=True)[["video_id", "frame", "x_px", "y_px", "amplitud", "sigma_px", "fondo",
                                                       "fotones", "snr"]]
            c.to_csv(CRUDOS / f"manchas_{conj}_r{r}.csv", index=False)
            print(f"{conj}, r = {r}: {len(c)} manchas (DoG > {K_MIN} sigma) en {imgs.shape[0] * imgs.shape[1]} cuadros "
                  f"({time.time() - t0:.0f} s)", flush=True)


# ----------------------------------------------------------------------------
# emparejamiento por cuadro
# ----------------------------------------------------------------------------

def por_cuadro(df, cols):
    """dict (video_id, frame) -> arreglo de las columnas cols."""
    return {k: g[cols].to_numpy(float) for k, g in df.groupby(["video_id", "frame"])}


def emparejar(ge, go, d, refl, tol_px, z=None):
    """Emparejamiento hungaro de un cuadro.
    ge: (n, 2) GT evaluable; go: (m, 2) GT no evaluable (se puede emparejar, no cuenta); d: (p, 2) detecciones;
    refl: (q, 2) reflejos GT en el FOV; z: opcional (z_ge, z_go, z_d) para el criterio cilindrico 3D de la red.
    Devuelve (indices de ge emparejados, sus detecciones, mascara de FP, mascara de FP sobre un reflejo)."""
    from scipy.optimize import linear_sum_assignment
    g = np.r_[ge, go] if len(go) else ge
    n, p = len(ge), len(d)
    fp = np.ones(p, bool)
    ie, je = np.zeros(0, int), np.zeros(0, int)
    if len(g) and p:
        dl = np.hypot(g[:, 0][:, None] - d[:, 0][None], g[:, 1][:, None] - d[:, 1][None])
        ok = dl <= tol_px
        c = (dl / tol_px) ** 2
        if z is not None:
            zg = np.r_[z[0], z[1]] if len(go) else z[0]
            dz = np.abs(zg[:, None] - z[2][None])
            ok &= dz <= tb.R_Z
            c = c + (dz / tb.R_Z) ** 2
        f, k = linear_sum_assignment(np.where(ok, c, 1e9))
        s = ok[f, k]
        f, k = f[s], k[s]
        fp[k] = False
        e = f < n
        ie, je = f[e], k[e]
    fpr = np.zeros(p, bool)
    if fp.any() and len(refl):
        dr = np.hypot(d[fp, 0][:, None] - refl[:, 0][None], d[fp, 1][:, None] - refl[:, 1][None])
        fpr[np.flatnonzero(fp)] = dr.min(1) <= tol_px
    return ie, je, fp, fpr


def estructura(gt, px):
    """Por cuadro: GT evaluable (x, y, z, indice), no evaluable (x, y, z), reflejos en el FOV (x, y)."""
    ge = gt[gt.evaluable]
    go = gt[~gt.evaluable]
    gr = gt[gt.reflejo_en_fov]
    E = {k: (g[["x_px", "y_px"]].to_numpy(float), g.z_nm.to_numpy(float), g.index.to_numpy())
         for k, g in ge.groupby(["video_id", "frame"])}
    O = {k: (g[["x_px", "y_px"]].to_numpy(float), g.z_nm.to_numpy(float)) for k, g in go.groupby(["video_id", "frame"])}
    Rf = por_cuadro(gr, ["x_reflejo_px", "y_reflejo_px"])
    claves = sorted(set(gt.groupby(["video_id", "frame"]).groups))
    return E, O, Rf, claves


def evaluar_detecciones(gt, det, px, con_z=False):
    """Por emisor GT evaluable: detectado, dx, dy [nm]; por cuadro: n_det, tp, fp, fp_reflejo."""
    E, O, Rf, claves = estructura(gt, px)
    D = {k: g for k, g in det.groupby(["video_id", "frame"])}
    vac2, vac1 = np.zeros((0, 2)), np.zeros(0)
    det_e = pd.Series(False, index=gt.index[gt.evaluable])
    dx = pd.Series(np.nan, index=det_e.index)
    dy = pd.Series(np.nan, index=det_e.index)
    filas = []
    tol = TOL_NM / px
    for k in claves:
        ge, zge, idx = E.get(k, (vac2, vac1, np.zeros(0, int)))
        go, zgo = O.get(k, (vac2, vac1))
        dd = D.get(k)
        d = dd[["x_px", "y_px"]].to_numpy(float) if dd is not None else vac2
        z = (zge, zgo, dd.z_nm.to_numpy(float) if dd is not None else vac1) if con_z else None
        ie, je, fp, fpr = emparejar(ge, go, d, Rf.get(k, vac2), tol, z)
        if len(ie):
            det_e.loc[idx[ie]] = True
            dx.loc[idx[ie]] = (d[je, 0] - ge[ie, 0]) * px
            dy.loc[idx[ie]] = (d[je, 1] - ge[ie, 1]) * px
        filas.append((k[0], k[1], len(d), len(ie), int(fp.sum()), int(fpr.sum()), len(ge)))
    cu = pd.DataFrame(filas, columns=["video_id", "frame", "n_det", "tp", "fp", "fp_reflejo", "n_gt"])
    return pd.DataFrame(dict(detectado=det_e, dx_nm=dx, dy_nm=dy)), cu


def sumas_k(gt, man, px, ks):
    """TP, FP, FP sobre reflejo y FN por video para cada umbral k (manchas con snr >= k), sin pandas por k."""
    E, O, Rf, claves = estructura(gt, px)
    M = {k: g[["x_px", "y_px", "snr"]].to_numpy(float) for k, g in man.groupby(["video_id", "frame"])}
    vac2, vac3, vac1 = np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0)
    vids = np.array(sorted({k[0] for k in claves}))
    pos = {v: i for i, v in enumerate(vids)}
    S = np.zeros((len(ks), len(vids), 4), int)
    tol = TOL_NM / px
    for k in claves:
        ge, _, _ = E.get(k, (vac2, vac1, None))
        go, _ = O.get(k, (vac2, vac1))
        m = M.get(k, vac3)
        rf = Rf.get(k, vac2)
        i = pos[k[0]]
        for j, kk in enumerate(ks):
            d = m[m[:, 2] >= kk, :2]
            ie, _, fp, fpr = emparejar(ge, go, d, rf, tol)
            S[j, i] += (len(ie), int(fp.sum()), int(fpr.sum()), len(ge) - len(ie))
    return vids, S


# ----------------------------------------------------------------------------
# tracking
# ----------------------------------------------------------------------------

def metricas_2d(pista, pts, gt_v, D, px):
    """ISBI e identidades en (x, y) con eps = 150 nm, y D lateral por MSD de cada pista contra el mismo estimador
    sobre la pista GT de su particula mayoritaria."""
    from metricas import a_arreglo, isbi, identidades, d_msd, particula_mayoritaria
    fr = pts.frame.to_numpy()
    ev = lambda df: np.c_[df.x_px.to_numpy() * px, df.y_px.to_numpy() * px]  # noqa: E731
    X, pids = a_arreglo(gt_v.particle_id.to_numpy(), gt_v.frame.to_numpy(), ev(gt_v), T)
    Y, _ = a_arreglo(pista, fr, ev(pts), T) if len(pts) else (np.zeros((0, T, 2)), None)
    s = isbi(X, Y, tb.EPS)
    idsw, frag, n_ids, emp = identidades(X, Y, tb.EPS, emparejamiento=True)
    mayor, _ = particula_mayoritaria(emp, len(Y))
    ref = np.array([d_msd(g, DT) for g in X]) / 1e6
    msd = []
    for j, r in enumerate(Y):
        k = mayor[j]
        dl = d_msd(r, DT) / 1e6
        if np.isfinite(dl):
            msd.append(dict(D_lat=dl, D_lat_ref=ref[k] if k >= 0 else np.nan))
    return dict(**s, idsw=int(idsw.sum()), frag=int(frag.sum()), n_ids=int(n_ids.sum())), msd


def _trabajo_tracking(v):
    from enlace import CostoNormalizado, enlazar, filtrar_cortas
    m_v, px, P = _G["m"].loc[v], _G["px"], _G["P"]
    D = float(m_v.D_um2_s)
    gt_v = _G["gt"].get(v, _G["gt0"])
    filas, msd = [], []
    for entrada, (pts_v, sig, dims) in _G["entradas"].items():
        pts = pts_v.get(v, _G["vacio"]).reset_index(drop=True)
        if len(pts):
            c = CostoNormalizado(sig[:dims], D * 1e6, DT, tb.R_CHI, c_loc=2.0)
            pos = np.c_[pts.x_px * px, pts.y_px * px, pts.z_nm][:, :dims] if dims == 3 else np.c_[pts.x_px * px, pts.y_px * px]
            pista = enlazar(pts.frame.to_numpy(), pos, max_hueco=P["hueco"], costo=c)
            keep = filtrar_cortas(pista, P["largo"])
            pista, pts = pista[keep], pts[keep]
        else:
            pista = np.zeros(0, int)
        f, ms = metricas_2d(pista, pts, gt_v, D, px)
        filas.append(dict(video_id=v, entrada=entrada, **f))
        msd += [dict(video_id=v, entrada=entrada, **x) for x in ms]
    return filas, msd


def bootstrap_unidades(S, est, rng, n_boot=N_BOOT):
    """Estadistico est(sumas) con IC por bootstrap de unidades (filas de S)."""
    e = est(S.sum(0))
    w = rng.multinomial(len(S), np.full(len(S), 1 / len(S)), size=n_boot)
    bs = np.array([est(wi @ S) for wi in w])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return e, lo, hi


# ----------------------------------------------------------------------------
# analisis
# ----------------------------------------------------------------------------

def vecinos(gt, px):
    """Distancia lateral [nm] de cada emisor GT al vecino real mas cercano del mismo cuadro."""
    out = pd.Series(np.inf, index=gt.index)
    for _, g in gt.groupby(["video_id", "frame"]):
        if len(g) < 2:
            continue
        p = g[["x_px", "y_px"]].to_numpy(float) * px
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        out.loc[g.index] = d.min(1)
    return out


def analisis(procesos):
    from multiprocessing import get_context
    rng = np.random.default_rng(0)
    SALIDA.mkdir(parents=True, exist_ok=True)
    datos = {c: cargar(c) for c in CONJUNTOS}
    px = imagenes("test")[1].pixel_nm
    assert abs(px - imagenes("tracking")[1].pixel_nm) < 1e-9

    # ---- 1. ajuste de (r, k) en la mitad de seleccion
    t0 = time.time()
    aj = []
    for conj in CONJUNTOS:
        gt, m, _ = datos[conj]
        sel = m.index[m.mitad == "seleccion"]
        gs = gt[gt.video_id.isin(sel)]
        for r in RADIOS:
            man = pd.read_csv(CRUDOS / f"manchas_{conj}_r{r}.csv")
            vids, S = sumas_k(gs, man[man.video_id.isin(sel)], px, KS)
            for j, k in enumerate(KS):
                tp, fp, fpr, fn = S[j].sum(0)
                aj.append(dict(conjunto=conj, r=r, k=k, tp=tp, fp=fp, fp_reflejo=fpr, fn=fn))
    aj = pd.DataFrame(aj)
    tot = aj.groupby(["r", "k"])[["tp", "fp", "fp_reflejo", "fn"]].sum().reset_index().assign(conjunto="ambos")
    aj = pd.concat([aj, tot], ignore_index=True)
    aj["jaccard"] = aj.tp / (aj.tp + aj.fp + aj.fn)
    aj["recall"] = aj.tp / (aj.tp + aj.fn)
    aj["precision"] = aj.tp / (aj.tp + aj.fp)
    aj["precision_sin_reflejos"] = aj.tp / (aj.tp + aj.fp - aj.fp_reflejo)
    aj.to_csv(SALIDA / "linea_base_clasica_ajuste.csv", index=False)
    mejor = aj[aj.conjunto == "ambos"].sort_values("jaccard", ascending=False).iloc[0]
    R_SEL, K_SEL = int(mejor.r), float(mejor.k)
    por_conj = aj[aj.conjunto != "ambos"].sort_values("jaccard", ascending=False).groupby("conjunto").head(1)
    print(f"ajuste ({time.time() - t0:.0f} s): r = {R_SEL}, k = {K_SEL:g} (Jaccard {mejor.jaccard:.3f} en las dos mitades de "
          f"seleccion); el mejor por conjunto:\n{por_conj[['conjunto', 'r', 'k', 'jaccard', 'recall', 'precision']].to_string(index=False)}")

    # ---- 2. deteccion emisor a emisor en la mitad de prueba
    emis, cuad, sig_lat = [], [], {}
    for conj in CONJUNTOS:
        gt, m, red = datos[conj]
        gt = gt.copy()
        gt["vecino_nm"] = vecinos(gt, px)
        man = pd.read_csv(CRUDOS / f"manchas_{conj}_r{R_SEL}.csv")
        cla = man[man.snr >= K_SEL].reset_index(drop=True)
        cla_por_mitad = {}
        for mitad in ("seleccion", "prueba"):
            vs = m.index[m.mitad == mitad]
            g_ = gt[gt.video_id.isin(vs)]
            for met, dets, con_z in (("clasico", cla, False), ("red", red, False), ("red_3d", red, True)):
                d_ = dets[dets.video_id.isin(vs)]
                e, c = evaluar_detecciones(g_, d_, px, con_z)
                if mitad == "seleccion":       # sigma lateral por eje para el costo del enlace (sin mirar la prueba)
                    if met in ("clasico", "red") and conj == "tracking":
                        sig_lat[met] = float(np.sqrt(np.nanmean(np.r_[e.dx_nm, e.dy_nm] ** 2)))
                    continue
                cols = ["video_id", "frame", "particle_id", "photons", "z_nm", "vecino_nm", "snr"]
                e = e.join(g_[cols]).assign(conjunto=conj, metodo=met)
                emis.append(e)
                cuad.append(c.assign(conjunto=conj, metodo=met))
            cla_por_mitad[mitad] = cla
        datos[conj] = (gt, m, red, cla)
    emis = pd.concat(emis, ignore_index=True)
    cuad = pd.concat(cuad, ignore_index=True)
    meta_cols = ["fondo_medio", "contraste_celula"]
    mm = pd.concat([datos[c][1][meta_cols].assign(conjunto=c).reset_index() for c in CONJUNTOS], ignore_index=True)
    emis = emis.merge(mm, on=["conjunto", "video_id"])
    cuad = cuad.merge(mm, on=["conjunto", "video_id"])
    print(f"sigma lateral (seleccion, tracking): {sig_lat}")

    # globales con IC (bootstrap de imagenes / videos, pareado entre metodos)
    glob = []
    for conj in CONJUNTOS:
        for met in ("clasico", "red", "red_3d"):
            cc_ = cuad[(cuad.conjunto == conj) & (cuad.metodo == met)].assign(cuadros=1)
            c = cc_.groupby("video_id")[["tp", "fp", "fp_reflejo", "n_gt", "n_det", "cuadros"]].sum()
            e = emis[(emis.conjunto == conj) & (emis.metodo == met)]
            err = e.groupby("video_id").apply(lambda g: pd.Series(dict(se2=np.nansum(g.dx_nm ** 2 + g.dy_nm ** 2),
                                                                       nd=g.detectado.sum())), include_groups=False)
            S = np.c_[c.to_numpy(float), err.reindex(c.index).to_numpy(float)]

            def est(s):
                tp, fp, fpr, ngt, ndet, nc, se2, nd = s
                return np.array([tp / ngt, tp / (tp + fp), tp / (tp + fp - fpr), tp / (ngt + fp), fp / nc,
                                 fpr / nc, np.sqrt(se2 / nd)])
            e_, lo, hi = bootstrap_unidades(S, est, rng)
            nombres = ["recall", "precision", "precision_sin_reflejos", "jaccard", "fp_por_cuadro", "fp_reflejo_por_cuadro",
                       "rmse_lat_nm"]
            fila = dict(conjunto=conj, metodo=met, unidades=len(c), n_gt=int(S[:, 3].sum()))
            for n_, a, b, cc in zip(nombres, e_, lo, hi):
                fila.update({n_: a, f"{n_}_lo": b, f"{n_}_hi": cc})
            glob.append(fila)
    glob = pd.DataFrame(glob)
    glob.to_csv(SALIDA / "linea_base_clasica_global.csv", index=False)

    # por rangos: recall y RMSE lateral por emisor; precision y FP por imagen para fondo y autofluorescencia
    rang = []
    for conj in CONJUNTOS:
        for var, bins in BINS.items():
            for i in range(len(bins) - 1):
                for met in ("clasico", "red"):
                    e = emis[(emis.conjunto == conj) & (emis.metodo == met)]
                    sel = (e[var] >= bins[i]) & (e[var] < bins[i + 1])
                    es = e[sel]
                    S = es.groupby("video_id").agg(det=("detectado", "sum"), n=("detectado", "size")).to_numpy(float)
                    if len(S) == 0:
                        continue
                    r_, lo, hi = bootstrap_unidades(S, lambda s: np.array([s[0] / s[1]]), rng, 300)
                    fila = dict(conjunto=conj, variable=var, lo_bin=bins[i], hi_bin=bins[i + 1], metodo=met, n=len(es),
                                recall=r_[0], recall_lo=lo[0], recall_hi=hi[0],
                                rmse_lat_nm=float(np.sqrt(np.nanmean(es.dx_nm ** 2 + es.dy_nm ** 2))) if es.detectado.any() else np.nan)
                    if var in ("fondo_medio", "contraste_celula"):
                        c = cuad[(cuad.conjunto == conj) & (cuad.metodo == met)]
                        c = c[(c[var] >= bins[i]) & (c[var] < bins[i + 1])]
                        fila.update(precision=c.tp.sum() / max(c.tp.sum() + c.fp.sum(), 1),
                                    fp_por_cuadro=c.fp.sum() / max(len(c), 1), fp_reflejo_por_cuadro=c.fp_reflejo.sum() / max(len(c), 1))
                    rang.append(fila)
    rang = pd.DataFrame(rang)
    rang.to_csv(SALIDA / "linea_base_clasica_rangos.csv", index=False)

    # regimen: diferencia de recall red - clasico por fotones x vecino (los dos conjuntos juntos, pareado por emisor)
    reg = []
    piv = emis[emis.metodo.isin(METODOS)].pivot_table(index=["conjunto", "video_id", "frame", "particle_id"], columns="metodo",
                                                       values="detectado", aggfunc="first").reset_index()
    piv = piv.merge(emis[emis.metodo == "red"][["conjunto", "video_id", "frame", "particle_id", "photons", "vecino_nm"]],
                    on=["conjunto", "video_id", "frame", "particle_id"])
    fb, vb = BINS["photons"], np.array([0, 500, 1000, 2000, 1e9])
    for i in range(len(fb) - 1):
        for j in range(len(vb) - 1):
            s = piv[(piv.photons >= fb[i]) & (piv.photons < fb[i + 1]) & (piv.vecino_nm >= vb[j]) & (piv.vecino_nm < vb[j + 1])]
            if len(s) < 20:
                continue
            u = s.assign(u=s.conjunto + "_" + s.video_id.astype(str)).groupby("u").agg(
                r=("red", "sum"), c=("clasico", "sum"), n=("red", "size")).to_numpy(float)
            d_, lo, hi = bootstrap_unidades(u, lambda q: np.array([q[0] / q[2], q[1] / q[2], (q[0] - q[1]) / q[2]]), rng, 500)
            reg.append(dict(fotones_lo=fb[i], fotones_hi=fb[i + 1], vecino_lo=vb[j], vecino_hi=vb[j + 1], n=len(s),
                            recall_red=d_[0], recall_clasico=d_[1], diferencia=d_[2], diferencia_lo=lo[2], diferencia_hi=hi[2],
                            solo_red=int((s.red & ~s.clasico).sum()), solo_clasico=int((s.clasico & ~s.red).sum())))
    reg = pd.DataFrame(reg)
    reg.to_csv(SALIDA / "linea_base_clasica_regimen.csv", index=False)

    # curvas precision - recall en la mitad de prueba: clasico (k) y red (conf; en tracking desde 0.2)
    pr = []
    for conj in CONJUNTOS:
        gt, m, red, cla = datos[conj]
        vs = m.index[m.mitad == "prueba"]
        g_ = gt[gt.video_id.isin(vs)]
        man = pd.read_csv(CRUDOS / f"manchas_{conj}_r{R_SEL}.csv")
        rr = pd.read_csv(TRACK / "detecciones_u02.csv") if conj == "tracking" else red
        for met, d_, ks in (("clasico", man[man.video_id.isin(vs)], KS),
                            ("red", rr[rr.video_id.isin(vs)].rename(columns={"conf": "snr"}),
                             np.round(np.arange(0.2 if conj == "tracking" else UMBRAL_RED, 1.501, 0.1), 2))):
            _, S = sumas_k(g_, d_, px, ks)
            for j, k in enumerate(ks):
                tp, fp, fpr, fn = S[j].sum(0)
                pr.append(dict(conjunto=conj, metodo=met, umbral=k, recall=tp / (tp + fn), precision=tp / (tp + fp),
                               precision_sin_reflejos=tp / (tp + fp - fpr)))
    pr = pd.DataFrame(pr)
    pr.to_csv(SALIDA / "linea_base_clasica_precision_recall.csv", index=False)

    # ---- 3. tracking con el LAP mejorado (tracking_3d), mitad de prueba
    gt, m, red, cla = datos["tracking"]
    mejor_enl = pd.read_csv(t3.SALIDA / "tracking_3d_enlace_mejor.csv").iloc[0]
    sig3 = pd.read_csv(t3.SALIDA / "tracking_3d_enlace_sigmas.csv").set_index("umbral").loc[UMBRAL_RED,
                                                                                           ["sigma_x", "sigma_y", "sigma_z"]].to_numpy(float)
    P = dict(hueco=int(mejor_enl.hueco), largo=int(mejor_enl.largo))
    prueba = m.index[m.mitad == "prueba"]
    ge = gt[gt.evaluable]
    sc, sr = sig_lat["clasico"] / np.sqrt(2), sig_lat["red"] / np.sqrt(2)    # por eje
    # oraculo: el clasico sin las detecciones que caen sobre un reflejo GT (a <= 150 nm) y lejos de toda particula real
    keep = np.ones(len(cla), bool)
    R_ = {k: g[["x_reflejo_px", "y_reflejo_px"]].to_numpy(float) for k, g in gt[gt.reflejo_en_fov].groupby(["video_id", "frame"])}
    G_ = {k: g[["x_px", "y_px"]].to_numpy(float) for k, g in gt.groupby(["video_id", "frame"])}
    tol = TOL_NM / px
    for k, g in cla.groupby(["video_id", "frame"]):
        d = g[["x_px", "y_px"]].to_numpy(float)
        rf, gg = R_.get(k), G_.get(k)
        if rf is None:
            continue
        cerca_r = np.hypot(d[:, 0][:, None] - rf[:, 0][None], d[:, 1][:, None] - rf[:, 1][None]).min(1) <= tol
        cerca_g = (np.hypot(d[:, 0][:, None] - gg[:, 0][None], d[:, 1][:, None] - gg[:, 1][None]).min(1) <= tol) if gg is not None \
            else np.zeros(len(d), bool)
        keep[g.index.to_numpy()] = ~(cerca_r & ~cerca_g)
    cla_or = cla[keep]
    entradas = {"red_3d": ({v: g for v, g in red.groupby("video_id")}, sig3, 3),
                "red_2d": ({v: g for v, g in red.groupby("video_id")}, np.array([sr, sr]), 2),
                "clasico_2d": ({v: g for v, g in cla.assign(z_nm=0.0).groupby("video_id")}, np.array([sc, sc]), 2),
                "clasico_2d_oraculo": ({v: g for v, g in cla_or.assign(z_nm=0.0).groupby("video_id")}, np.array([sc, sc]), 2)}
    _G.update(m=m, px=px, P=P, gt={v: g for v, g in ge.groupby("video_id")}, gt0=ge.iloc[:0], entradas=entradas,
              vacio=red.iloc[:0])
    t0 = time.time()
    filas, msd = [], []
    with get_context("fork").Pool(procesos) as pool:
        for f_, m_ in pool.imap_unordered(_trabajo_tracking, list(prueba), chunksize=2):
            filas += f_
            msd += m_
    print(f"tracking: {time.time() - t0:.0f} s; sigma por eje del costo: clasico {sc:.1f}, red {sr:.1f} nm, red 3D {sig3.round(1)}")
    tv = pd.DataFrame(filas).merge(m[["D_um2_s", "densidad", "n_set"]], left_on="video_id", right_index=True)
    tmsd = pd.DataFrame(msd).merge(m[["D_um2_s", "densidad"]], left_on="video_id", right_index=True)
    tv.to_csv(CRUDOS / "tracking_por_video.csv", index=False)
    tmsd.to_csv(CRUDOS / "tracking_msd.csv", index=False)
    trk = []
    for (ent, eje, val), t in [((e, "D", d), tv[(tv.entrada == e) & (tv.D_um2_s == d)]) for e in entradas for d in tb.DS] + \
                              [((e, "densidad", d), tv[(tv.entrada == e) & (tv.densidad == d)]) for e in entradas for d in tb.DENSIDAD] + \
                              [((e, "todos", "-"), tv[tv.entrada == e]) for e in entradas]:
        cols = ["d_xy", "d_x0", "d_ybar0", "tp_det", "fn_det", "fp_det", "tp_pistas", "fn_pistas", "fp_pistas", "idsw", "frag", "n_ids",
                "n_pistas_gt", "n_pistas_est"]
        S = t[cols].to_numpy(float)

        def est(s):
            d = dict(zip(cols, s))
            from metricas import de_sumas
            q = de_sumas(d)
            return np.array([q["jsc"], q["alpha"], q["beta"], d["idsw"] / d["n_pistas_gt"], d["frag"] / d["n_pistas_gt"],
                             d["n_pistas_est"] / d["n_pistas_gt"]])
        e_, lo, hi = bootstrap_unidades(S, est, rng)
        fila = dict(entrada=ent, por=eje, valor=val, videos=len(t))
        for n_, a, b, c in zip(["jsc", "alpha", "beta", "idsw_por_pista", "frag_por_pista", "pistas_est_por_gt"], e_, lo, hi):
            fila.update({n_: a, f"{n_}_lo": b, f"{n_}_hi": c})
        ms = tmsd[(tmsd.entrada == ent) & ((tmsd.D_um2_s == val) if eje == "D" else (tmsd.densidad == val) if eje == "densidad"
                                           else np.ones(len(tmsd), bool))]
        ms = ms[ms.D_lat_ref.notna() & (ms.D_lat_ref > 0)]
        if len(ms):
            err = (ms.D_lat / ms.D_lat_ref - 1).abs().to_numpy()
            bs = [np.median(rng.choice(err, len(err))) for _ in range(500)]
            fila.update(err_D_lat_rel=float(np.median(err)), err_D_lat_rel_lo=np.percentile(bs, 2.5),
                        err_D_lat_rel_hi=np.percentile(bs, 97.5), ratio_D_lat=float((ms.D_lat / ms.D_lat_ref).median()),
                        pistas_msd=len(ms))
        trk.append(fila)
    trk = pd.DataFrame(trk)
    trk.to_csv(SALIDA / "linea_base_clasica_tracking.csv", index=False)

    pd.DataFrame([dict(r=R_SEL, k=K_SEL, jaccard_seleccion=mejor.jaccard, sigma_lat_clasico_nm=sig_lat["clasico"],
                       sigma_lat_red_nm=sig_lat["red"], hueco=P["hueco"], largo=P["largo"])]).to_csv(
        SALIDA / "linea_base_clasica_parametros.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 500):
        f1 = lambda v: f"{v:.4g}"  # noqa: E731
        print(glob.to_string(index=False, float_format=f1))
        print(rang.drop(columns=[c for c in rang if c.endswith("_lo") or c.endswith("_hi")]).to_string(index=False, float_format=f1))
        print(reg.to_string(index=False, float_format=f1))
        print(trk.to_string(index=False, float_format=f1))
    figuras(aj, R_SEL, K_SEL, glob, rang, reg, trk, pr)


# ----------------------------------------------------------------------------
# figuras
# ----------------------------------------------------------------------------

def figuras(aj, R_SEL, K_SEL, glob, rang, reg, trk, pr):
    plt = _estilo()
    grid = dict(color="#e6e5e0", lw=0.6)

    # -- ajuste del detector
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.2))
    ax = axs[3]
    for conj, mk in zip(CONJUNTOS, ("o", "s")):
        for met in METODOS:
            q = pr[(pr.conjunto == conj) & (pr.metodo == met)]
            ax.plot(q.recall, q.precision, "-" + mk, color=COLOR[met], ms=3, lw=1.1, label=f"{NOMBRE[met]}, {NOMBRE_CONJ[conj]}")
            ax.plot(q.recall, q.precision_sin_reflejos, ":" + mk, color=COLOR[met], ms=2.5, lw=0.9, mfc="none")
    ax.set_xlabel("recall"); ax.set_ylabel("precision (punteada: sin contar reflejos)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, **grid); ax.legend(frameon=False, fontsize=6)
    ax.set_title("mitad de prueba: barrido de k (clasico) y de conf (red)", fontsize=8.5)
    for ax, conj in zip(axs, ("ambos",) + CONJUNTOS):
        for r, ls in zip(RADIOS, (":", "-", "--")):
            a = aj[(aj.conjunto == conj) & (aj.r == r)]
            ax.plot(a.k, a.jaccard, ls, color=COLOR["clasico"], lw=1.4, label=f"Jaccard, r = {r}")
            if r == R_SEL:
                ax.plot(a.k, a.recall, ls, color="#52514e", lw=1, label=f"recall, r = {r}")
                ax.plot(a.k, a.precision, ls, color=COLOR["gris"], lw=1, label=f"precision, r = {r}")
        ax.axvline(K_SEL, color="#8a8984", lw=0.8)
        ax.set_xlabel("umbral k [sigmas de ruido de la DoG]"); ax.set_ylim(0, 1)
        ax.grid(True, **grid); ax.legend(frameon=False, fontsize=6.5)
        ax.set_title(f"mitad de seleccion: {'los dos conjuntos' if conj == 'ambos' else NOMBRE_CONJ[conj]}", fontsize=8.5)
    fig.suptitle(f"Detector clasico: ajuste del umbral k y del radio r de los maximos locales (elegidos: r = {R_SEL}, k = {K_SEL:g})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_linea_base_clasica_ajuste.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -- recall y RMSE por rangos
    vars_ = list(BINS)
    fig, axs = plt.subplots(3, len(vars_), figsize=(20, 10.5))
    for fila, (conj, magn) in enumerate((("test", "recall"), ("tracking", "recall"), ("test", "rmse_lat_nm"))):
        for col, var in enumerate(vars_):
            ax = axs[fila, col]
            for met in METODOS:
                r = rang[(rang.conjunto == conj) & (rang.variable == var) & (rang.metodo == met)]
                xc = np.arange(len(r))
                if magn == "recall":
                    ax.errorbar(xc, r.recall, yerr=[r.recall - r.recall_lo, r.recall_hi - r.recall], fmt="o-", color=COLOR[met],
                                ms=4, lw=1.2, capsize=2, label=NOMBRE[met])
                    if var in ("fondo_medio", "contraste_celula"):
                        ax.plot(xc, r.precision, "s--", color=COLOR[met], ms=3, lw=0.9, mfc="none", label=f"precision, {NOMBRE[met]}")
                    ax.set_ylim(0, 1.02)
                else:
                    ax.plot(xc, r.rmse_lat_nm, "o-", color=COLOR[met], ms=4, lw=1.2, label=NOMBRE[met])
                    ax.set_ylim(bottom=0)
                lab = [f"{a:g}-{b:g}" if b < 1e8 else f">{a:g}" for a, b in zip(r.lo_bin, r.hi_bin)]
                ax.set_xticks(xc)
                ax.set_xticklabels(lab, rotation=35, fontsize=6.5)
            ax.set_xlabel(ETIQ[var]); ax.grid(True, **grid)
            if col == 0:
                ax.set_ylabel(f"{NOMBRE_CONJ[conj]}: {'recall (precision a trazos)' if magn == 'recall' else 'RMSE lateral [nm]'}")
            if fila == 0 and col == 0:
                ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Deteccion emisor a emisor en la mitad de prueba (tolerancia lateral 150 nm; IC 95 % por bootstrap de imagenes / videos)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_linea_base_clasica_deteccion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -- regimen fotones x vecino
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    fb = sorted(reg.fotones_lo.unique())
    vb = sorted(reg.vecino_lo.unique())
    for ax, (col, tit, cmap, lim) in zip(axs, (("recall_clasico", "recall, detector clasico", "Greys", (0, 1)),
                                               ("recall_red", "recall, red s1", "Blues", (0, 1)),
                                               ("diferencia", "recall red - clasico", "RdBu", (-0.6, 0.6)))):
        M = np.full((len(vb), len(fb)), np.nan)
        for _, q in reg.iterrows():
            M[vb.index(q.vecino_lo), fb.index(q.fotones_lo)] = q[col]
        im = ax.imshow(M, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
        for (i, j), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}" if col == "diferencia" else f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (abs(v) > 0.45 if col == "diferencia" else v > 0.6) else "#1f1f1d")
        ax.set_xticks(range(len(fb)))
        ax.set_xticklabels([f"{a / 1000:g}k" for a in fb], fontsize=7)
        ax.set_yticks(range(len(vb)))
        ax.set_yticklabels([f">{a:g}" if a >= 2000 else f"{a:g}-{b:g}" for a, b in zip(vb, [500, 1000, 2000, 1e9])], fontsize=7)
        ax.set_xlabel("fotones (limite inferior del bin)"); ax.set_ylabel("vecino mas cercano [nm]")
        ax.set_title(tit, fontsize=8.5)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Regimen: recall por fotones y distancia al vecino (mitad de prueba de los dos conjuntos)", fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_linea_base_clasica_regimen.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -- tracking
    ents = ("red_3d", "red_2d", "clasico_2d", "clasico_2d_oraculo")
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for fila, (por, vals, xl) in enumerate((("D", list(tb.DS), "D [um^2/s]"), ("densidad", list(tb.DENSIDAD), "emisores por celula"))):
        for col, (mag, yl) in enumerate((("jsc", "JSC (lateral)"), ("idsw_por_pista", "cambios de identidad por pista GT"),
                                         ("err_D_lat_rel", "error relativo mediano de D lateral"))):
            ax = axs[fila, col]
            for k, e in enumerate(ents):
                t = trk[(trk.entrada == e) & (trk.por == por)].set_index("valor").loc[[str(v) if por == "densidad" else v for v in vals]] \
                    if por == "densidad" else trk[(trk.entrada == e) & (trk.por == por)].set_index("valor").loc[vals]
                xc = np.arange(len(vals)) + (k - 1.5) * 0.1
                ax.errorbar(xc, t[mag], yerr=[t[mag] - t[f"{mag}_lo"], t[f"{mag}_hi"] - t[mag]], fmt="o-", color=COLOR[e],
                            ms=4, lw=1.2, capsize=2, label=NOMBRE[e], mfc="none" if e in ("red_2d", "clasico_2d_oraculo") else COLOR[e],
                            ls="--" if e == "clasico_2d_oraculo" else "-")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([f"{v:g}" if por == "D" else v for v in vals])
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, **grid)
            if mag != "jsc":
                ax.set_yscale("log") if mag == "idsw_por_pista" else None
            if fila == 0 and col == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Tracking con el LAP mejorado (tracking_3d) en la mitad de prueba: red s1 frente al detector clasico "
                 "(metricas en x, y; IC 95 % por bootstrap de videos)", fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_linea_base_clasica_tracking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("partes", nargs="+", choices=("detectar", "analisis"))
    p.add_argument("--procesos", type=int, default=12)
    a = p.parse_args()
    if "detectar" in a.partes:
        detectar(a.procesos)
    if "analisis" in a.partes:
        analisis(a.procesos)


if __name__ == "__main__":
    main()
