"""
Tracking 3D en la celula con el espejo: mejoras del enlace sin reentrenar y z
por triangulacion real <-> reflejo, sobre los 200 videos de tracking_base.py
(localizacion/datos/tracking_celula_v2_cerca/). Ver
resultados/resultados_tracking_3d.md.

Partes (en orden; cada una lee lo que escribio la anterior):
    localizar  la red s1 cuadro a cuadro con postproc.umbral = 0.2 (superconjunto
               de las detecciones con umbral 0.3 y 0.4: los maximos locales y su
               conf no dependen del umbral) -> detecciones_u02.csv en DATOS
    enlace     B1: rejilla costo (z reescalada / normalizado por eje) x huecos
               (2, 5, 10) x umbral de conf (0.2, 0.3, 0.4) x largo minimo de
               pista, contra la linea base de tracking_base.py
               -> figuras/tracking_3d/tracking_3d_enlace_*.csv (agregados; por video
               y por pista en DATOS/tracking_3d/)
    reflejos   B3: deteccion clasica de los reflejos (maximos locales + ajuste
               gaussiano 2D) en todos los cuadros, sin las detecciones de la
               red y del lado del espejo de algun real -> reflejos.csv en DATOS
    analisis   B4-B7: enlace de los reflejos, emparejamiento de pistas real <->
               reflejo, z triangulada + Kalman/RTS (beta verdadero y 45 fijo) y
               metricas por D, densidad y distancia al espejo
               -> figuras/tracking_3d/ (agregados y figuras) y DATOS/tracking_3d/
               (por video, pista, par y punto)

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    LD_PRELOAD=$HOME/miniforge3/envs/tesis/lib/libstdc++.so.6 $PY figuras/tracking_3d.py localizar   # ~30 min GPU
    $PY figuras/tracking_3d.py enlace [--procesos 12]
    $PY figuras/tracking_3d.py reflejos [--procesos 12]
    $PY figuras/tracking_3d.py analisis [--procesos 12]
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
import tracking_base as tb  # noqa: E402  (inserta localizacion/ y tracking/ en sys.path)
import enlace as tb_enlace  # noqa: E402  (tracking/enlace.py)

DATOS = tb.DATOS
SALIDA = AQUI / "tracking_3d"
CRUDOS = DATOS / "tracking_3d"     # tablas por video / pista / punto (grandes, no se versionan)
T, DT = tb.T, 0.1
UMBRAL_MIN = 0.2


# ----------------------------------------------------------------------------
# Localizacion con umbral bajo
# ----------------------------------------------------------------------------

def localizar_videos():
    from utils import cargar_modelo, elegir_dispositivo
    from infer import localizar
    from data.simulacion import config_simulador, COLUMNAS_GEOMETRIA
    disp = elegir_dispositivo("auto")
    modelo, cfg, _ = cargar_modelo(tb.CORRIDA / "mejor.pt", disp)
    cfg = replace(cfg, postproc=replace(cfg.postproc, umbral=UMBRAL_MIN))
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
        if (v + 1) % 20 == 0:
            print(f"  {v + 1}/{len(imgs)} videos ({time.time() - t0:.0f} s)", flush=True)
    det = pd.concat(partes, ignore_index=True)[["video_id", "frame", "x_px", "y_px", "z_nm", "conf"]]
    det.to_csv(DATOS / "detecciones_u02.csv", index=False)
    print(f"{len(det)} detecciones (conf >= {UMBRAL_MIN}) en {len(imgs) * T} cuadros ({time.time() - t0:.0f} s)")
    # control: con conf >= 0.4 deben salir las de tracking_base (salvo el ultimo digito de fp16)
    base = pd.read_csv(DATOS / "detecciones.csv")
    d4 = det[det.conf >= 0.4]
    print(f"control: {len(d4)} con conf >= 0.4 frente a {len(base)} de detecciones.csv")


# ----------------------------------------------------------------------------
# Comun
# ----------------------------------------------------------------------------

R_XY, R_Z, EPS, R_CHI = tb.R_XY, tb.R_Z, tb.EPS, tb.R_CHI
DS = tb.DS
DENSIDAD = tb.DENSIDAD
COSTOS = ("zesc", "norm2", "norm1")
NOMBRE_COSTO = {"zesc": "z reescalada (base)", "norm2": "normalizado, 2 sigma^2 + 2 D dt",
                "norm1": "normalizado, sigma^2 + 2 D dt"}
HUECOS = (2, 5, 10)
UMBRALES = (0.4, 0.3, 0.2)
LARGOS = (1, 3, 5, 8)
BASE = dict(costo="zesc", hueco=2, umbral=0.4, largo=1)
N_BOOT = 1000


def cargar():
    """GT completo, GT evaluable, muestras (con densidad y half de seleccion) y px."""
    cfg_sim = tb.sim.cargar_config(DATOS / "config_simulador.json")
    px = cfg_sim.pixel_nm
    gt_todo = pd.read_csv(DATOS / "ground_truth.csv")
    gt = gt_todo[tb.evaluable(gt_todo)].copy()
    m = pd.read_csv(DATOS / "muestras.csv").set_index("video_id")
    m["densidad"] = pd.cut(m.n_set, [0, 4, 8, 12], labels=list(DENSIDAD))
    m["mitad"] = np.where(m.muestra < 25, "seleccion", "prueba")
    return gt_todo, gt, m, px


def detecciones(umbral):
    det = pd.read_csv(DATOS / "detecciones_u02.csv")
    return det[det.conf >= umbral].reset_index(drop=True)


def errores_emparejados(gt, det, px):
    """Errores por par GT / deteccion (emparejamiento por cuadro de tracking_base), con particle_id."""
    filas = []
    dg = {k: g for k, g in gt.groupby(["video_id", "frame"])}
    dd = {k: g for k, g in det.groupby(["video_id", "frame"])}
    for (v, t), g in dg.items():
        d = dd.get((v, t))
        if d is None:
            continue
        p = tb.emparejar_cuadro(g, d, px)
        if len(p):
            a, b = g.iloc[p[:, 0]], d.iloc[p[:, 1]]
            filas.append(pd.DataFrame(dict(video_id=v, frame=t, particle_id=a.particle_id.to_numpy(),
                                           dx=(b.x_px.to_numpy() - a.x_px.to_numpy()) * px,
                                           dy=(b.y_px.to_numpy() - a.y_px.to_numpy()) * px,
                                           dz=b.z_nm.to_numpy() - a.z_nm.to_numpy(), indice=b.index.to_numpy())))
    return pd.concat(filas, ignore_index=True)


def sigma_y_cloc(err):
    """RMSE por eje y c_loc efectivo por eje: var(e_{t+1} - e_t) / sigma^2 para la misma particula."""
    sig = np.array([np.sqrt(np.mean(err[c] ** 2)) for c in ("dx", "dy", "dz")])
    a = err.set_index(["video_id", "particle_id", "frame"])
    b = a.copy()
    b.index = b.index.set_levels(b.index.levels[2] - 1, level=2)
    j = a.join(b, rsuffix="_1", how="inner")
    cloc = np.array([np.var(j[c + "_1"] - j[c]) / s ** 2 for c, s in zip(("dx", "dy", "dz"), sig)])
    return sig, cloc


def enlazar_cfg(pts, costo, hueco, D_um2, sig, px):
    """Pistas de pts (frame, x_px, y_px, z_nm) con el costo y el hueco dados."""
    fr = pts.frame.to_numpy()
    if costo == "zesc":
        s_lat = float(np.sqrt((sig[0] ** 2 + sig[1] ** 2) / 2))
        radio = R_CHI * np.sqrt(2 * D_um2 * 1e6 * DT + 2 * s_lat ** 2)
        return tb.enlazar(fr, tb.coordenadas(pts, px, s_lat / sig[2]), radio, hueco)
    c = tb_enlace.CostoNormalizado(sig, D_um2 * 1e6, DT, R_CHI, c_loc=2.0 if costo == "norm2" else 1.0)
    return tb.enlazar(fr, np.c_[pts.x_px.to_numpy() * px, pts.y_px.to_numpy() * px, pts.z_nm.to_numpy()],
                      max_hueco=hueco, costo=c)


def metricas_video(pista, pts, gt_v, D, px, por_pista=False):
    """Sumas ISBI e identidades del video y D por MSD de cada pista estimada (como tracking_base)."""
    from metricas import a_arreglo, isbi, identidades, d_msd, particula_mayoritaria
    fr = pts.frame.to_numpy()
    ev = lambda df: np.c_[df.x_px.to_numpy() * px, df.y_px.to_numpy() * px, df.z_nm.to_numpy() * R_XY / R_Z]  # noqa: E731
    X, pids = a_arreglo(gt_v.particle_id.to_numpy(), gt_v.frame.to_numpy(), ev(gt_v), T)
    Y, ids_y = a_arreglo(pista, fr, ev(pts), T) if len(pts) else (np.zeros((0, T, 3)), np.zeros(0, int))
    r = isbi(X, Y, EPS, por_pista=por_pista)
    s, pp = r if por_pista else (r, None)
    idsw, frag, n_ids, emp = identidades(X, Y, EPS, emparejamiento=True)
    mayor, comparte = particula_mayoritaria(emp, len(Y))
    fila = dict(**s, idsw=int(idsw.sum()), frag=int(frag.sum()), n_ids=int(n_ids.sum()))
    R, _ = a_arreglo(pista, fr, np.c_[pts.x_px * px, pts.y_px * px, pts.z_nm], T) if len(pts) else (np.zeros((0, T, 3)), None)
    G, _ = a_arreglo(gt_v.particle_id.to_numpy(), gt_v.frame.to_numpy(), np.c_[gt_v.x_px * px, gt_v.y_px * px, gt_v.z_nm], T)
    ref3 = np.array([d_msd(g, DT) for g in G]) / 1e6
    refz = np.array([d_msd(g[:, 2:], DT) for g in G]) / 1e6
    msd = []
    for j, rr in enumerate(R):
        k = mayor[j]
        d3 = d_msd(rr, DT) / 1e6
        if np.isfinite(d3):
            msd.append(dict(pista=int(ids_y[j]), D3=d3, Dz=d_msd(rr[:, 2:], DT) / 1e6, particle_id=int(pids[k]) if k >= 0 else -1,
                            D3_ref=ref3[k] if k >= 0 else np.nan, Dz_ref=refz[k] if k >= 0 else np.nan))
    if por_pista:
        pp = dict(pp, particle_id=pids, idsw=idsw, frag=frag)
        return fila, msd, pp
    return fila, msd


COLS_SUMA = ["d_xy", "d_x0", "d_ybar0", "tp_det", "fn_det", "fp_det", "tp_pistas", "fn_pistas", "fp_pistas",
             "idsw", "frag", "n_ids", "n_pistas_gt", "n_pistas_est"]


def resumir(t):
    """Metricas de un grupo de filas por video (sin IC)."""
    from metricas import de_sumas
    s = t[COLS_SUMA].sum()
    out = de_sumas(s)
    out.update(idsw_por_pista=s.idsw / s.n_pistas_gt, frag_por_pista=s.frag / s.n_pistas_gt,
               ids_por_pista=s.n_ids / s.n_pistas_gt, pistas_est=int(s.n_pistas_est), pistas_gt=int(s.n_pistas_gt))
    return out


def resumir_msd(t):
    """Error relativo mediano de D (3D y z) contra el estimador sobre la pista GT, y fraccion D <= 0."""
    c = t[t.D3_ref.notna()]
    return dict(pistas_msd=len(t), err3_rel=float((c.D3 / c.D3_ref - 1).abs().median()) if len(c) else np.nan,
                errz_rel=float((c.Dz / c.Dz_ref - 1).abs().median()) if len(c) else np.nan,
                ratio3=float((c.D3 / c.D3_ref).median()) if len(c) else np.nan,
                ratioz=float((c.Dz / c.Dz_ref).median()) if len(c) else np.nan,
                neg3=float((t.D3 <= 0).mean()) if len(t) else np.nan, negz=float((t.Dz <= 0).mean()) if len(t) else np.nan)


# ----------------------------------------------------------------------------
# B1: enlace
# ----------------------------------------------------------------------------

_G = {}


def _trabajo_enlace(v):
    gt_v, m_v, px = _G["gt"].get(v), _G["m"].loc[v], _G["px"]
    gt_v = gt_v if gt_v is not None else _G["gt0"]
    D = float(m_v.D_um2_s)
    filas, msd = [], []
    for entrada, umbral, sig, pts_por_v in _G["entradas"]:
        pts = pts_por_v.get(v, _G["vacio"]).reset_index(drop=True)
        costos = COSTOS
        for costo in costos:
            for hueco in HUECOS:
                pista = enlazar_cfg(pts, costo, hueco, D, sig, px) if len(pts) else np.zeros(0, int)
                largos = LARGOS if entrada == "red" else (1,)
                for largo in largos:
                    keep = tb_enlace.filtrar_cortas(pista, largo) if len(pista) else np.zeros(0, bool)
                    f, ms = metricas_video(pista[keep], pts[keep], gt_v, D, px)
                    clave = dict(video_id=v, entrada=entrada, costo=costo, hueco=hueco, umbral=umbral, largo=largo)
                    filas.append(dict(clave, **f))
                    msd += [dict(clave, **x) for x in ms]
    return filas, msd


def enlace(procesos):
    from multiprocessing import get_context
    SALIDA.mkdir(parents=True, exist_ok=True)
    gt_todo, gt, m, px = cargar()
    sig_u, cloc_u = {}, {}
    dets = {}
    for u in UMBRALES:
        dets[u] = detecciones(u)
        err = errores_emparejados(gt, dets[u], px)
        sig_u[u], cloc_u[u] = sigma_y_cloc(err)
        n_gt = len(gt)
        print(f"umbral {u}: {len(dets[u])} detecciones, recall {len(err) / n_gt:.3f}, precision {len(err) / len(dets[u]):.3f}, "
              f"RMSE x/y/z {sig_u[u].round(1)} nm, c_loc efectivo x/y/z {cloc_u[u].round(2)}")
    base = pd.read_csv(DATOS / "detecciones.csv")
    print(f"control: detecciones.csv (tracking_base) {len(base)}, conf >= 0.4 de esta inferencia {len(dets[0.4])}")
    # GT + ruido con el mismo generador que tracking_base (misma semilla y orden: identico si sigma coincide)
    rng = np.random.default_rng(0)
    s4 = sig_u[0.4]
    ruido = gt.copy()
    ruido["x_px"] += rng.normal(0, s4[0], len(ruido)) / px
    ruido["y_px"] += rng.normal(0, s4[1], len(ruido)) / px
    ruido["z_nm"] += rng.normal(0, s4[2], len(ruido))
    entradas = [("red", u, sig_u[u], {v: g for v, g in dets[u].groupby("video_id")}) for u in UMBRALES]
    entradas += [(e, np.nan, s4, {v: g for v, g in df.groupby("video_id")}) for e, df in (("gt_ruido", ruido), ("gt", gt))]
    _G.update(gt={v: g for v, g in gt.groupby("video_id")}, gt0=gt.iloc[:0], m=m, px=px, entradas=entradas,
              vacio=dets[0.4].iloc[:0])
    t0 = time.time()
    filas, msd = [], []
    with get_context("fork").Pool(procesos) as pool:
        for f, ms in pool.imap_unordered(_trabajo_enlace, list(m.index), chunksize=2):
            filas += f
            msd += ms
    print(f"enlace de la rejilla: {time.time() - t0:.0f} s")
    tv = pd.DataFrame(filas).merge(m[["D_um2_s", "densidad", "mitad"]], left_on="video_id", right_index=True)
    tm = pd.DataFrame(msd).merge(m[["D_um2_s", "densidad", "mitad"]], left_on="video_id", right_index=True)
    CRUDOS.mkdir(parents=True, exist_ok=True)
    tv.to_csv(CRUDOS / "enlace_videos.csv", index=False)
    tm.to_csv(CRUDOS / "enlace_msd_pistas.csv", index=False)
    pd.DataFrame([dict(umbral=u, sigma_x=sig_u[u][0], sigma_y=sig_u[u][1], sigma_z=sig_u[u][2], cloc_x=cloc_u[u][0],
                       cloc_y=cloc_u[u][1], cloc_z=cloc_u[u][2]) for u in UMBRALES]).to_csv(
        SALIDA / "tracking_3d_enlace_sigmas.csv", index=False)
    resumen_enlace(tv, tm)


CLAVES = ["entrada", "costo", "hueco", "umbral", "largo"]


def resumen_enlace(tv, tm):
    """Rejilla agregada por D (todas y por mitad) y seleccion de la config en la mitad de seleccion."""
    filas = []
    tv = tv.fillna({"umbral": -1})
    tm = tm.fillna({"umbral": -1})
    gm = {k: g for k, g in tm.groupby(CLAVES + ["D_um2_s", "mitad"])}
    for k, g in tv.groupby(CLAVES + ["D_um2_s", "mitad"]):
        filas.append(dict(zip(CLAVES + ["D_um2_s", "mitad"], k), **resumir(g), **resumir_msd(gm.get(k, tm.iloc[:0]))))
    gm2 = {k: g for k, g in tm.groupby(CLAVES + ["D_um2_s"])}
    for k, g in tv.groupby(CLAVES + ["D_um2_s"]):
        filas.append(dict(zip(CLAVES + ["D_um2_s"], k), mitad="todas", **resumir(g), **resumir_msd(gm2.get(k, tm.iloc[:0]))))
    ta = pd.DataFrame(filas)
    ta.to_csv(SALIDA / "tracking_3d_enlace_rejilla.csv", index=False)
    # seleccion: mayor JSC medio sobre los 4 D en la mitad de seleccion (muestras base 0-24), entrada red
    r = ta[(ta.entrada == "red") & (ta.mitad == "seleccion")]
    puntaje = r.groupby(CLAVES).jsc.mean().sort_values(ascending=False)
    print("\nmejores configs (JSC medio en D, mitad de seleccion):")
    print(puntaje.head(12).to_string(float_format=lambda v: f"{v:.4f}"))
    mejor = dict(zip(CLAVES, puntaje.index[0]))
    pd.DataFrame([mejor]).to_csv(SALIDA / "tracking_3d_enlace_mejor.csv", index=False)
    print(f"elegida: {mejor}")
    comparar_enlace(tv, tm, ta, puntaje, mejor)
    return ta, mejor


def configs_comparadas(puntaje, mejor):
    """Linea base, cada cambio por separado (umbral con su mejor largo en la mitad de seleccion) y la elegida."""
    def mejor_largo(**k):
        p = puntaje.reset_index()
        for c, x in k.items():
            p = p[p[c] == x]
        return int(p.sort_values("jsc", ascending=False).largo.iloc[0])
    cs = [("base", BASE),
          ("costo normalizado (2 sigma^2)", dict(BASE, costo="norm2")),
          ("costo normalizado (sigma^2)", dict(BASE, costo="norm1")),
          ("huecos <= 5", dict(BASE, hueco=5)),
          ("huecos <= 10", dict(BASE, hueco=10))]
    for u in (0.3, 0.2):
        L = mejor_largo(costo="zesc", hueco=2, umbral=u)
        cs.append((f"umbral {u}, largo >= {L}", dict(BASE, umbral=u, largo=L)))
    cs.append(("elegida", dict(mejor)))
    return cs


def _sel(t, cfg):
    s = t[t.entrada == "red"]
    for k, x in cfg.items():
        s = s[s[k] == x]
    return s


def comparar_enlace(tv, tm, ta, puntaje, mejor):
    rng = np.random.default_rng(1)
    filas = []
    for nombre, cfg in configs_comparadas(puntaje, mejor):
        sv, sm = _sel(tv, cfg), _sel(tm, cfg)
        for D in DS:
            for mitad in ("todas", "prueba"):
                a = sv[sv.D_um2_s == D] if mitad == "todas" else sv[(sv.D_um2_s == D) & (sv.mitad == "prueba")]
                b = sm[sm.D_um2_s == D] if mitad == "todas" else sm[(sm.D_um2_s == D) & (sm.mitad == "prueba")]
                f = dict(config=nombre, **cfg, D_um2_s=D, mitad=mitad, **resumir_msd(b))
                f.update(tb.bootstrap_grupo(a, rng, n_boot=N_BOOT if mitad == "todas" else 200))
                filas.append(f)
    tc = pd.DataFrame(filas)
    tc.to_csv(SALIDA / "tracking_3d_enlace_comparacion.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(tc[tc.mitad == "todas"][["config", "D_um2_s", "alpha", "beta", "jsc", "jsc_lo", "jsc_hi", "jsc_pistas",
                                        "idsw_por_pista", "frag_por_pista", "pistas_est", "err3_rel", "errz_rel"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nmitad de prueba (muestras base 25-49):")
        print(tc[tc.mitad == "prueba"][["config", "D_um2_s", "alpha", "jsc", "jsc_lo", "jsc_hi", "idsw_por_pista",
                                         "frag_por_pista", "err3_rel", "errz_rel"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # controles: costo con GT + ruido y GT exacto
    c = ta[(ta.entrada != "red") & (ta.mitad == "todas") & (ta.largo == 1)]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print("\ncontroles (largo 1):")
        print(c[["entrada", "costo", "hueco", "D_um2_s", "alpha", "jsc", "idsw_por_pista", "frag_por_pista", "err3_rel",
                 "errz_rel"]].sort_values(["entrada", "D_um2_s", "costo", "hueco"])
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    figura_enlace(tc)


def figura_enlace(tc):
    plt = tb._estilo()
    t = tc[tc.mitad == "todas"]
    noms = list(dict.fromkeys(t.config))
    colores = ["#52514e", "#2a78d6", "#7fb0ea", "#eb6834", "#f2a477", "#1baf7a", "#86d6b3", "#c0392b"]
    paneles = [("jsc", "JSC detecciones"), ("alpha", "alpha"), ("jsc_pistas", "JSC pistas"),
               ("idsw_por_pista", "cambios de identidad / pista"), ("frag_por_pista", "interrupciones / pista"),
               ("err3_rel", "error relativo de D (3D), mediana"), ("errz_rel", "error relativo de D (z), mediana"),
               ("pistas_est", "pistas estimadas")]
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, (k, et) in zip(axs.flat, paneles):
        for n, col in zip(noms, colores):
            s = t[t.config == n].sort_values("D_um2_s")
            lw = 2.2 if n in ("base", "elegida") else 1.1
            ax.plot(s.D_um2_s, s[k], "o-", color=col, lw=lw, ms=3, label=n)
            if k + "_lo" in s and n in ("base", "elegida"):
                ax.fill_between(s.D_um2_s, s[k + "_lo"], s[k + "_hi"], color=col, alpha=0.15, lw=0)
        ax.set_xscale("log")
        if k in ("err3_rel", "errz_rel"):
            ax.set_yscale("log")
        ax.set_title(et, fontsize=9)
        ax.set_xlabel("D [um^2/s]")
        ax.grid(True, color="#e6e5e0", lw=0.6)
    axs[0, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("B1: mejoras del enlace sobre la red v2 s1 (200 videos; bandas: IC 95% bootstrap de videos para base y elegida)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_3d_enlace.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# B3: deteccion clasica de manchas (todos los candidatos; los filtros van en analisis)
# ----------------------------------------------------------------------------

K_REFL = 5.0                     # umbral del DoG en sigmas de ruido


def _trabajo_reflejos(v):
    from reflejos import detectar_manchas
    img = _G["imgs"][v]
    partes = []
    for f in range(T):
        e = (img[f].astype(float) - _G["offset"]) * _G["gain"]
        c = detectar_manchas(e, k=K_REFL)
        partes.append(pd.DataFrame(c).assign(video_id=v, frame=f))
    return pd.concat(partes, ignore_index=True)


def reflejos(procesos):
    from multiprocessing import get_context
    cfg_sim = tb.sim.cargar_config(DATOS / "config_simulador.json")
    _G.update(imgs=np.load(DATOS / "imagenes.npy", mmap_mode="r"), gain=cfg_sim.gain, offset=cfg_sim.offset)
    t0 = time.time()
    with get_context("fork").Pool(procesos) as pool:
        partes = pool.map(_trabajo_reflejos, range(len(_G["imgs"])), chunksize=2)
    c = pd.concat(partes, ignore_index=True)
    c = c[["video_id", "frame", "x_px", "y_px", "amplitud", "sigma_px", "fondo", "fotones", "snr"]]
    c.to_csv(DATOS / "manchas.csv", index=False)
    print(f"{len(c)} manchas (DoG > {K_REFL} sigma) en {len(_G['imgs']) * T} cuadros ({time.time() - t0:.0f} s) "
          f"-> {DATOS / 'manchas.csv'}")


# ----------------------------------------------------------------------------
# B4-B7: reflejos, emparejamiento, triangulacion y metricas
# ----------------------------------------------------------------------------

R_EXCL_PX = 2.0                  # manchas a <= 2 px de una deteccion de la red: son los reales
TOL_T_NM = 600.0                 # deteccion: |t - t_real| para "del lado del espejo de algun real"
MARGEN_S_NM = 150.0              # s_mancha < s_real + margen
R_REFL_NM = 200.0                # tolerancia lateral para emparejar manchas con reflejos GT (evaluacion)
Z_POSIBLE = (-4000.0, 4000.0)    # z triangulada (mediana del par) fuera de esto: par imposible
BETAS = ("verdadero", "45")
VARIANTES_TRI = ("verdadero", "45", "ctrl")     # ctrl: beta verdadero y control por la z de la red
K_CTRL = 3.0
DIST_BINS = [0, 1, 2, 3, 4, 10]  # distancia del real al plano del espejo [um]
DIST_ETIQ = ["0-1", "1-2", "2-3", "3-4", ">4"]


def geometria(m_v):
    return float(m_v.espejo_orientacion_deg), np.array([m_v.espejo_punto_x_px, m_v.espejo_punto_y_px])


def filtrar_manchas(man, det, phi, p0, px):
    """Manchas candidatas a reflejo de un video: lejos (> R_EXCL_PX) de toda deteccion de la red
    del cuadro y del lado del espejo de alguna (s < s_real + margen, |t - t_real| < TOL_T_NM)."""
    from espejo import coordenadas_espejo
    keep = np.zeros(len(man), bool)
    dd = {f: g for f, g in det.groupby("frame")}
    for f, g in man.groupby("frame"):
        d = dd.get(f)
        if d is None:
            continue
        dist = np.hypot(g.x_px.to_numpy()[:, None] - d.x_px.to_numpy()[None], g.y_px.to_numpy()[:, None] - d.y_px.to_numpy()[None])
        sc, tc = coordenadas_espejo(g.x_px, g.y_px, px, phi, p0)
        sr, tr = coordenadas_espejo(d.x_px, d.y_px, px, phi, p0)
        lado = (sc[:, None] < sr[None] + MARGEN_S_NM) & (np.abs(tc[:, None] - tr[None]) < TOL_T_NM)
        keep[g.index.to_numpy() - man.index[0]] = (dist.min(1) > R_EXCL_PX) & lado.any(1)
    return keep


def emparejar_lateral(ax, ay, bx, by, tol_nm, px):
    from scipy.optimize import linear_sum_assignment
    if len(ax) == 0 or len(bx) == 0:
        return np.zeros((0, 2), int)
    d = np.hypot(np.asarray(ax)[:, None] - np.asarray(bx)[None], np.asarray(ay)[:, None] - np.asarray(by)[None]) * px
    f, c = linear_sum_assignment(np.where(d < tol_nm, d, 1e9))
    s = d[f, c] < tol_nm
    return np.c_[f[s], c[s]]


def asociar_gt(pts, gt_v, xcol, ycol, tol_nm, px):
    """(n,) particle_id GT de cada punto de pts (emparejamiento lateral por cuadro; -1 = ninguno)."""
    pid = -np.ones(len(pts), int)
    gg = {f: g for f, g in gt_v.groupby("frame")}
    for f, d in pts.groupby("frame"):
        g = gg.get(f)
        if g is None:
            continue
        p = emparejar_lateral(g[xcol], g[ycol], d.x_px, d.y_px, tol_nm, px)
        if len(p):
            pid[d.index.to_numpy()[p[:, 1]]] = g.particle_id.to_numpy()[p[:, 0]]
    return pid


def mayoria(pista, pid):
    """dict pista -> particle_id mayoritario entre sus puntos asociados (-1 si ninguno)."""
    out = {}
    for k in np.unique(pista):
        p = pid[pista == k]
        p = p[p >= 0]
        out[int(k)] = int(np.bincount(p).argmax()) if len(p) else -1
    return out


def _trabajo_analisis(v):
    from espejo import coordenadas_espejo, triangular_z, sigma_z_triangulada, kalman_rts, emparejar_pistas
    from metricas import a_arreglo, isbi
    P = _G["P"]
    px, m_v = _G["px"], _G["m"].loc[v]
    D = float(m_v.D_um2_s)
    phi, p0 = geometria(m_v)
    beta_v = float(m_v.espejo_angulo_eje_deg)
    gt_v = _G["gt"].get(v, _G["gt0"])
    gt_all = _G["gt_todo"].get(v)
    det = _G["det"].get(v, _G["det0"]).reset_index(drop=True)
    man = _G["man"].get(v, _G["man0"]).reset_index(drop=True)
    out = dict(filas=[], msd=[], pista_gt=[], puntos=[], pares=[], refl=[], refl_isbi=[], pistas=[])

    # pistas de reales (config elegida en B1)
    pista = enlazar_cfg(det, P["costo"], P["hueco"], D, P["sig"], px) if len(det) else np.zeros(0, int)
    keep = tb_enlace.filtrar_cortas(pista, P["largo"]) if len(pista) else np.zeros(0, bool)
    det, pista = det[keep].reset_index(drop=True), tb_enlace._renumerar(pista[keep]) if keep.any() else np.zeros(0, int)

    # manchas candidatas a reflejo y sus pistas (2D, mismo LAP con sigma del reflejo)
    km = filtrar_manchas(man, _G["det"].get(v, _G["det0"]), phi, p0, px) if len(man) else np.zeros(0, bool)
    cand = man[km].reset_index(drop=True)
    if len(cand):
        c = tb_enlace.CostoNormalizado(P["sig_refl"], D * 1e6, DT, R_CHI, c_loc=2.0)
        pf = tb.enlazar(cand.frame.to_numpy(), np.c_[cand.x_px * px, cand.y_px * px], max_hueco=P["hueco"], costo=c)
        kf = tb_enlace.filtrar_cortas(pf, P["largo_refl"])
        cand, pf = cand[kf].reset_index(drop=True), tb_enlace._renumerar(pf[kf]) if kf.any() else np.zeros(0, int)
    else:
        pf = np.zeros(0, int)

    # evaluacion de manchas y pistas de reflejo contra los reflejos GT (en el FOV)
    gr = gt_all[gt_all.reflejo_en_fov].copy() if gt_all is not None else _G["gt0"]
    pid_c = asociar_gt(cand, gr, "x_reflejo_px", "y_reflejo_px", R_REFL_NM, px) if len(cand) else np.zeros(0, int)
    out["refl"].append(dict(video_id=v, manchas=len(man), candidatas=int(km.sum()), en_pistas=len(cand),
                            tp=int((pid_c >= 0).sum()), reflejos_gt=len(gr)))
    ev2 = lambda df, xc, yc: np.c_[df[xc].to_numpy() * px, df[yc].to_numpy() * px]  # noqa: E731
    if len(gr):
        Xr, _ = a_arreglo(gr.particle_id.to_numpy(), gr.frame.to_numpy(), ev2(gr, "x_reflejo_px", "y_reflejo_px"), T)
        Yr, _ = a_arreglo(pf, cand.frame.to_numpy(), ev2(cand, "x_px", "y_px"), T) if len(cand) else (np.zeros((0, T, 2)), None)
        out["refl_isbi"].append(dict(video_id=v, **isbi(Xr, Yr, EPS)))

    # asociacion con el GT (lateral) para saber que particula sigue cada pista
    pid_r = asociar_gt(det, gt_v, "x_px", "y_px", R_XY, px) if len(det) else np.zeros(0, int)
    may_r, may_f = mayoria(pista, pid_r), mayoria(pf, pid_c)

    # (s, t) de las pistas
    sR, tR = coordenadas_espejo(det.x_px, det.y_px, px, phi, p0)
    sF, tF = coordenadas_espejo(cand.x_px, cand.y_px, px, phi, p0) if len(cand) else (np.zeros(0), np.zeros(0))
    R, ids_r = a_arreglo(pista, det.frame.to_numpy(), np.c_[sR, tR], T) if len(det) else (np.zeros((0, T, 2)), np.zeros(0, int))
    F, ids_f = a_arreglo(pf, cand.frame.to_numpy(), np.c_[sF, tF], T) if len(cand) else (np.zeros((0, T, 2)), np.zeros(0, int))
    # particula GT del reflejo usado en cada cuadro de cada pista de reflejo (-1: ninguna / no hay)
    Fpid, _ = a_arreglo(pf, cand.frame.to_numpy(), pid_c[:, None].astype(float), T) if len(cand) else (np.zeros((0, T, 1)), None)
    Fpid = np.nan_to_num(Fpid[..., 0], nan=-1).astype(int)
    Rpid, _ = a_arreglo(pista, det.frame.to_numpy(), pid_r[:, None].astype(float), T) if len(det) else (np.zeros((0, T, 1)), None)
    Rpid = np.nan_to_num(Rpid[..., 0], nan=-1).astype(int)
    sigma_dt = float(np.hypot(P["sig_lat"], P["sig_refl_lat"]))
    q = 2 * D * 1e6 * DT

    zs = {"red": det.z_nm.to_numpy().copy()}
    for nb in BETAS:
        beta = beta_v if nb == "verdadero" else 45.0

        def z_par(i, j, ok, beta=beta):
            z = triangular_z(R[i, ok, 0], F[j, ok, 0], beta)
            return Z_POSIBLE[0] <= np.median(z) <= Z_POSIBLE[1]
        pares, _ = emparejar_pistas(R, F, sigma_dt, z_par=z_par)
        var_obs = sigma_z_triangulada(P["sig_lat"], P["sig_refl_lat"], beta) ** 2
        z_tri = np.full(len(det), np.nan)
        z_suav = np.full(len(det), np.nan)
        ztri_t = np.full((len(R), T), np.nan)
        pid_refl_t = -np.ones((len(R), T), int)
        for p in pares:
            ok = p["cuadros"]
            ztri_t[p["i"], ok] = triangular_z(R[p["i"], ok, 0], F[p["j"], ok, 0], beta)
            pid_refl_t[p["i"], ok] = Fpid[p["j"], ok]
            kr, kf_ = ids_r[p["i"]], ids_f[p["j"]]
            out["pares"].append(dict(video_id=v, beta=nb, pista=int(kr), pista_refl=int(kf_), n=p["n_cons"], comunes=p["n"],
                                     ronda=p["ronda"], puntaje=p["puntaje"], corr=p["corr"], lado=p["lado"],
                                     dt_medio=p["dt_medio"], mov_t=p["mov_t"],
                                     cuadros_correctos=int(((Fpid[p["j"], ok] == Rpid[p["i"], ok]) & (Rpid[p["i"], ok] >= 0)).sum()),
                                     particula=may_r.get(int(kr), -1), particula_refl=may_f.get(int(kf_), -1)))
        fr = det.frame.to_numpy()
        fila_de = np.searchsorted(ids_r, pista)
        variantes = [(nb, ztri_t)]
        if nb == "verdadero":
            # control por la red: se descartan los cuadros con |z_tri - z_red| > K_CTRL sigma combinado
            Zred, _ = a_arreglo(pista, fr, det.z_nm.to_numpy()[:, None], T) if len(det) else (np.zeros((0, T, 1)), None)
            s_c = np.sqrt(P["sig"][2] ** 2 + var_obs)
            variantes.append(("ctrl", np.where(np.abs(ztri_t - Zred[..., 0]) <= K_CTRL * s_c, ztri_t, np.nan)))
        for nv, zt in variantes:
            z_tri = np.full(len(det), np.nan)
            z_suav = np.full(len(det), np.nan)
            for i in np.unique(fila_de):
                if np.isfinite(zt[i]).any():
                    mu, _ = kalman_rts(zt[i], var_obs, q)
                    sel = fila_de == i
                    z_tri[sel] = zt[i, fr[sel]]
                    z_suav[sel] = mu[fr[sel]]
            for i, k in enumerate(ids_r):
                out["pistas"].append(dict(video_id=v, beta=nv, pista=int(k), particula=may_r.get(int(k), -1),
                                          largo=int(np.isfinite(R[i, :, 0]).sum()), cuadros_tri=int(np.isfinite(zt[i]).sum())))
            zs[f"tri_{nv}"] = np.where(np.isfinite(z_tri), z_tri, det.z_nm.to_numpy())
            zs[f"suav_{nv}"] = np.where(np.isfinite(z_suav), z_suav, det.z_nm.to_numpy())
            zs[f"_tri_{nv}"], zs[f"_suav_{nv}"] = z_tri, z_suav
            zs[f"_pidrefl_{nv}"] = np.where(np.isfinite(zt), pid_refl_t, -1)[fila_de, fr] if len(det) else np.zeros(0, int)

    # puntos (deteccion asociada a GT): errores de z de cada fuente sobre los mismos puntos
    if len(det):
        g = gt_v.set_index(["frame", "particle_id"])
        a = pid_r >= 0
        idx = pd.MultiIndex.from_arrays([det.frame.to_numpy()[a], pid_r[a]])
        zg = g.loc[idx, "z_nm"].to_numpy()
        gr_ = gt_all.set_index(["frame", "particle_id"]).loc[idx]
        pts = pd.DataFrame(dict(video_id=v, frame=det.frame.to_numpy()[a], particle_id=pid_r[a], pista=pista[a],
                                z_gt=zg, z_refl_gt=gr_.z_reflejo_nm.to_numpy(), e_red=zs["red"][a] - zg))
        for nb in VARIANTES_TRI:
            pts[f"e_tri_{nb}"] = zs[f"_tri_{nb}"][a] - zg
            pts[f"e_suav_{nb}"] = zs[f"_suav_{nb}"][a] - zg
        for nb in VARIANTES_TRI:
            # correcto si la mancha usada en ese cuadro esta sobre el reflejo GT de la misma particula
            pts[f"par_correcto_{nb}"] = np.where(np.isfinite(pts[f"e_tri_{nb}"]),
                                                 (zs[f"_pidrefl_{nb}"][a] == pid_r[a]).astype(float), np.nan)
        out["puntos"].append(pts)

    # metricas de tracking con cada fuente de z
    for nz in Z_VARIANTES:
        d2 = det.assign(z_nm=zs[nz])
        f, ms, pp = metricas_video(pista, d2, gt_v, D, px, por_pista=True) if len(det) else (
            *metricas_video(pista, d2, gt_v, D, px), None)
        out["filas"].append(dict(video_id=v, z=nz, **f))
        out["msd"] += [dict(video_id=v, z=nz, **x) for x in ms]
        if pp is not None:
            out["pista_gt"].append(pd.DataFrame(dict(video_id=v, z=nz, **pp)))
    return out


def analisis(procesos):
    from multiprocessing import get_context
    SALIDA.mkdir(parents=True, exist_ok=True)
    gt_todo, gt, m, px = cargar()
    mejor = pd.read_csv(SALIDA / "tracking_3d_enlace_mejor.csv").iloc[0].to_dict()
    sig_t = pd.read_csv(SALIDA / "tracking_3d_enlace_sigmas.csv").set_index("umbral")
    u = float(mejor["umbral"])
    sig = sig_t.loc[u, ["sigma_x", "sigma_y", "sigma_z"]].to_numpy(float)
    det = detecciones(u)
    man = pd.read_csv(DATOS / "manchas.csv")
    # error lateral de las manchas que caen sobre reflejos GT (calibra el costo del enlace y sigma_t)
    e = []
    for v, g in man.groupby("video_id"):
        gr = gt_todo[(gt_todo.video_id == v) & gt_todo.reflejo_en_fov]
        for f, c in g.groupby("frame"):
            r = gr[gr.frame == f]
            p = emparejar_lateral(r.x_reflejo_px, r.y_reflejo_px, c.x_px, c.y_px, R_REFL_NM, px)
            if len(p):
                e.append(np.c_[(c.x_px.to_numpy()[p[:, 1]] - r.x_reflejo_px.to_numpy()[p[:, 0]]) * px,
                               (c.y_px.to_numpy()[p[:, 1]] - r.y_reflejo_px.to_numpy()[p[:, 0]]) * px])
    e = np.concatenate(e)
    sig_refl = np.sqrt((e ** 2).mean(0))
    P = dict(costo=mejor["costo"], hueco=int(mejor["hueco"]), largo=int(mejor["largo"]), largo_refl=int(mejor["largo"]),
             sig=sig, sig_lat=float(np.sqrt((sig[0] ** 2 + sig[1] ** 2) / 2)), sig_refl=sig_refl,
             sig_refl_lat=float(np.sqrt((sig_refl ** 2).mean())))
    print(f"config de enlace: {mejor}; sigma red {sig.round(1)} nm; sigma manchas sobre reflejos {sig_refl.round(1)} nm "
          f"({len(e)} pares, tolerancia {R_REFL_NM:.0f} nm)")
    _G.update(P=P, px=px, m=m, gt={v: g for v, g in gt.groupby("video_id")}, gt0=gt.iloc[:0],
              gt_todo={v: g for v, g in gt_todo.groupby("video_id")},
              det={v: g for v, g in det.groupby("video_id")}, det0=det.iloc[:0],
              man={v: g for v, g in man.groupby("video_id")}, man0=man.iloc[:0])
    t0 = time.time()
    res = dict(filas=[], msd=[], pista_gt=[], puntos=[], pares=[], refl=[], refl_isbi=[], pistas=[])
    with get_context("fork").Pool(procesos) as pool:
        for o in pool.imap_unordered(_trabajo_analisis, list(m.index), chunksize=2):
            for k in res:
                res[k] += o[k]
    print(f"analisis por video: {time.time() - t0:.0f} s")
    mc = m[["D_um2_s", "densidad", "n_set", "espejo_angulo_eje_deg"]]
    t = {k: (pd.concat(res[k], ignore_index=True) if k in ("pista_gt", "puntos") else pd.DataFrame(res[k]))
         for k in res}
    for k in t:
        t[k] = t[k].merge(mc, left_on="video_id", right_index=True)
    # distancia media de cada particula al plano del espejo (GT, geometria verdadera)
    t["dist"] = distancias_gt(gt, m, px)
    for k in ("pista_gt", "puntos", "msd"):
        t[k] = t[k].merge(t["dist"], on=["video_id", "particle_id"], how="left")
    t["pistas"] = t["pistas"].merge(t["dist"].rename(columns={"particle_id": "particula"}), on=["video_id", "particula"],
                                    how="left")
    CRUDOS.mkdir(parents=True, exist_ok=True)
    for k, df in t.items():
        df.to_csv(CRUDOS / f"{k}.csv", index=False)
    pd.DataFrame([dict(P, sig=list(P["sig"]), sig_refl=list(P["sig_refl"]))]).to_csv(SALIDA / "tracking_3d_parametros.csv", index=False)
    resumen_analisis(t, P)


def distancias_gt(gt, m, px):
    filas = []
    for v, g in gt.groupby("video_id"):
        mv = m.loc[v]
        phi, p0 = geometria(mv)
        from espejo import coordenadas_espejo
        s, _ = coordenadas_espejo(g.x_px, g.y_px, px, phi, p0)
        b = np.radians(mv.espejo_angulo_eje_deg)
        d = s * np.cos(b) + g.z_nm.to_numpy() * np.sin(b)
        filas.append(pd.DataFrame(dict(video_id=v, particle_id=g.particle_id.to_numpy(), d_nm=d)))
    d = pd.concat(filas).groupby(["video_id", "particle_id"]).d_nm.mean().reset_index()
    d["dist_um"] = pd.cut(d.d_nm / 1000, DIST_BINS, labels=DIST_ETIQ, right=False)
    return d


Z_VARIANTES = ("red", "tri_verdadero", "suav_verdadero", "tri_45", "suav_45", "tri_ctrl", "suav_ctrl")
NOMBRE_Z = {"red": "z de la red", "tri_verdadero": "triangulada, beta verdadero", "suav_verdadero": "suavizada (RTS), beta verdadero",
            "tri_45": "triangulada, beta = 45", "suav_45": "suavizada (RTS), beta = 45",
            "tri_ctrl": "triangulada + control red, beta verdadero", "suav_ctrl": "suavizada + control red, beta verdadero"}
COLOR_Z = {"red": "#52514e", "tri_verdadero": "#7fb0ea", "suav_verdadero": "#2a78d6", "tri_45": "#f2a477", "suav_45": "#eb6834",
           "tri_ctrl": "#86d6b3", "suav_ctrl": "#1baf7a"}


def _rmse(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan


def errores_z(p, nb):
    """RMSE z de la red, triangulada y suavizada sobre los mismos puntos (con z triangulada), mediana de |e|
    y fraccion |e| > 300 nm."""
    q = p[np.isfinite(p[f"e_tri_{nb}"])]
    out = dict(puntos=len(q), puntos_total=len(p), cobertura=len(q) / max(len(p), 1),
               par_correcto=float(q[f"par_correcto_{nb}"].mean()) if len(q) else np.nan)
    for f, c in (("red", "e_red"), ("tri", f"e_tri_{nb}"), ("suav", f"e_suav_{nb}")):
        e = q[c].to_numpy()
        out[f"rmse_{f}"] = _rmse(e)
        out[f"mae_med_{f}"] = float(np.median(np.abs(e))) if len(e) else np.nan
        out[f"frac300_{f}"] = float(np.mean(np.abs(e) > 300)) if len(e) else np.nan
        out[f"sesgo_{f}"] = float(np.mean(e)) if len(e) else np.nan
    c = q[q[f"par_correcto_{nb}"] == 1]
    out["rmse_tri_correctos"] = _rmse(c[f"e_tri_{nb}"])
    out["rmse_suav_correctos"] = _rmse(c[f"e_suav_{nb}"])
    out["rmse_red_correctos"] = _rmse(c["e_red"])
    return out


def resumen_analisis(t, P):
    rng = np.random.default_rng(2)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_columns", 60)
    # B3-B4: manchas y pistas de reflejos
    r = t["refl"].groupby("D_um2_s")[["manchas", "candidatas", "en_pistas", "tp", "reflejos_gt"]].sum()
    r["recall"] = r.tp / r.reflejos_gt
    r["precision"] = r.tp / r.en_pistas
    from metricas import de_sumas
    ri = t["refl_isbi"].groupby("D_um2_s")[COLS_SUMA[:9] + ["n_pistas_gt", "n_pistas_est"]].sum()
    for D, s in ri.iterrows():
        for k, x in de_sumas(s).items():
            r.loc[D, f"{k}_reflejos"] = x
    r["pistas_refl_gt"], r["pistas_refl_est"] = ri.n_pistas_gt, ri.n_pistas_est
    r.to_csv(SALIDA / "tracking_3d_reflejos_resumen.csv")
    print("\nB3-B4 manchas -> candidatas a reflejo -> en pistas; contra los reflejos GT en el FOV (lateral < "
          f"{R_REFL_NM:.0f} nm):\n{r.to_string(float_format=fmt)}")

    # B5: emparejamiento
    pr = t["pares"]
    pr["correcto"] = (pr.particula >= 0) & (pr.particula == pr.particula_refl)
    pi = t["pistas"]
    pi = pi[pi.largo >= 5].copy()
    pi["emparejada"] = pi.cuadros_tri > 0
    pts = t["puntos"]
    fc = pd.concat([pts.groupby(["video_id", "pista"])[f"par_correcto_{nb}"].mean().rename("frac_correctos").reset_index()
                    .assign(beta=nb) for nb in VARIANTES_TRI])
    pi = pi.merge(fc, on=["video_id", "beta", "pista"], how="left")
    pi["emparejada_bien"] = pi.emparejada & (pi.frac_correctos.fillna(0) >= 0.9)
    pi["frac_tri"] = pi.cuadros_tri / pi.largo
    filas = []
    for (nb, de), g in [((nb, de), g) for (nb, de), g in pi.groupby(["beta", "dist_um"], observed=True)] + \
            [((nb, "todas"), g) for nb, g in pi.groupby("beta")]:
        filas.append(dict(beta=nb, dist_um=de, pistas=len(g), frac_emparejadas=g.emparejada.mean(),
                          frac_emparejadas_bien=g.emparejada_bien.mean(), frac_cuadros_tri=g.frac_tri.mean()))
    te = pd.DataFrame(filas)
    filas = []
    for (nb, D), g in pi.groupby(["beta", "D_um2_s"]):
        filas.append(dict(beta=nb, D_um2_s=D, pistas=len(g), frac_emparejadas=g.emparejada.mean(),
                          frac_emparejadas_bien=g.emparejada_bien.mean(), frac_cuadros_tri=g.frac_tri.mean()))
    teD = pd.DataFrame(filas)
    tp_ = pr.groupby(["beta", "D_um2_s"]).agg(pares=("n", "size"), cuadros=("n", "sum"), cuadros_correctos=("cuadros_correctos", "sum"),
                                              pares_mayoria_correcta=("correcto", "mean"),
                                              rondas_extra=("ronda", lambda x: (x > 0).mean()))
    tp_["frac_cuadros_correctos"] = tp_.cuadros_correctos / tp_.cuadros
    te.to_csv(SALIDA / "tracking_3d_emparejamiento_distancia.csv", index=False)
    teD.merge(tp_.reset_index(), on=["beta", "D_um2_s"]).to_csv(SALIDA / "tracking_3d_emparejamiento_D.csv", index=False)
    print(f"\nB5 emparejamiento (pistas reales con >= 5 cuadros) por distancia de su particula al espejo:\n{te.to_string(index=False, float_format=fmt)}")
    print(f"\npor D:\n{teD.to_string(index=False, float_format=fmt)}\n{tp_.to_string(float_format=fmt)}")

    # B6: errores de z sobre los mismos puntos
    filas = []
    for nb in VARIANTES_TRI:
        filas.append(dict(beta=nb, grupo="todos", **errores_z(pts, nb)))
        for D, g in pts.groupby("D_um2_s"):
            filas.append(dict(beta=nb, grupo=f"D={D:g}", **errores_z(g, nb)))
        for de, g in pts.groupby("densidad", observed=True):
            filas.append(dict(beta=nb, grupo=f"densidad {de}", **errores_z(g, nb)))
        for de, g in pts.groupby("dist_um", observed=True):
            filas.append(dict(beta=nb, grupo=f"distancia {de} um", **errores_z(g, nb)))
    tz = pd.DataFrame(filas)
    tz.to_csv(SALIDA / "tracking_3d_z.csv", index=False)
    # sesgo del reflejo: error de la z triangulada (beta verdadero, cuadros bien emparejados) por z' y beta
    c = pts[pts.par_correcto_verdadero == 1].copy()
    c["z_refl_um"] = pd.cut(c.z_refl_gt / 1000, [-8, -6, -5, -4, -3, -2, 0])
    c["beta_deg"] = pd.cut(c.espejo_angulo_eje_deg, [40, 42.5, 45, 47.5, 50])
    sz = c.groupby(["z_refl_um", "beta_deg"], observed=True).e_tri_verdadero.agg(
        puntos="size", sesgo="mean", mediana="median", sd="std").reset_index()
    sz.to_csv(SALIDA / "tracking_3d_sesgo_reflejo.csv", index=False)
    print("\nsesgo de la z triangulada (beta verdadero, bien emparejados) por z' del reflejo y beta:")
    print(c.groupby("z_refl_um", observed=True).e_tri_verdadero.agg(["size", "mean", "median", "std"]).to_string(float_format=fmt))
    print(sz.pivot(index="z_refl_um", columns="beta_deg", values="mediana").to_string(float_format=lambda v: f"{v:.0f}"))
    print(f"\nB6 errores de z sobre los mismos puntos:\n{tz.to_string(index=False, float_format=fmt)}")

    # B7: tracking por D y densidad (IC bootstrap de videos)
    fv = t["filas"]
    agg = []
    for nz in Z_VARIANTES:
        s = fv[fv.z == nz]
        for D, sd in s.groupby("D_um2_s"):
            agg.append(dict(z=nz, D_um2_s=D, densidad="todas", **tb.bootstrap_grupo(sd, rng, N_BOOT)))
            for de, sdd in sd.groupby("densidad", observed=True):
                agg.append(dict(z=nz, D_um2_s=D, densidad=de, **tb.bootstrap_grupo(sdd, rng, 200)))
    ta = pd.DataFrame(agg)
    ta.to_csv(SALIDA / "tracking_3d_metricas.csv", index=False)
    print("\nB7 ISBI e identidades por fuente de z, D (todas las densidades):")
    print(ta[ta.densidad == "todas"][["z", "D_um2_s", "alpha", "alpha_lo", "alpha_hi", "beta", "jsc", "jsc_lo", "jsc_hi",
                                       "jsc_pistas", "idsw_por_pista", "frag_por_pista"]].to_string(index=False, float_format=fmt))
    print("\npor densidad (z red / suav_verdadero):")
    print(ta[(ta.densidad != "todas") & ta.z.isin(["red", "suav_verdadero"])][
        ["z", "D_um2_s", "densidad", "alpha", "beta", "jsc", "idsw_por_pista", "frag_por_pista"]].to_string(index=False, float_format=fmt))

    # B7 por distancia al espejo (por pista GT: alpha, JSC sin espurias, identidades)
    pg = t["pista_gt"]
    filas = []
    for (nz, D, de), g in pg.groupby(["z", "D_um2_s", "dist_um"], observed=True):
        filas.append(dict(z=nz, D_um2_s=D, dist_um=de, pistas_gt=len(g), alpha=1 - g.d.sum() / g.d0.sum(),
                          jsc_sin_espurias=g.tp.sum() / (g.largo.sum() + g.fp.sum()), idsw_por_pista=g.idsw.mean(),
                          frag_por_pista=g.frag.mean()))
    for (nz, de), g in pg.groupby(["z", "dist_um"], observed=True):
        filas.append(dict(z=nz, D_um2_s=np.nan, dist_um=de, pistas_gt=len(g), alpha=1 - g.d.sum() / g.d0.sum(),
                          jsc_sin_espurias=g.tp.sum() / (g.largo.sum() + g.fp.sum()), idsw_por_pista=g.idsw.mean(),
                          frag_por_pista=g.frag.mean()))
    td = pd.DataFrame(filas)
    td.to_csv(SALIDA / "tracking_3d_metricas_distancia.csv", index=False)
    print("\npor distancia al espejo (todas las D):")
    print(td[td.D_um2_s.isna()].to_string(index=False, float_format=fmt))

    # B7 D por MSD
    ms = t["msd"].merge(pi[pi.beta == "verdadero"][["video_id", "pista", "frac_tri"]], on=["video_id", "pista"], how="left")
    ms["emparejada"] = ms.frac_tri.fillna(0) >= 0.5
    filas = []
    for (nz, D), g in ms.groupby(["z", "D_um2_s"]):
        filas.append(dict(z=nz, D_um2_s=D, grupo="todas", **resumir_msd(g)))
        filas.append(dict(z=nz, D_um2_s=D, grupo="emparejadas (>= 50% de cuadros)", **resumir_msd(g[g.emparejada])))
    for (nz, de), g in ms.groupby(["z", "dist_um"], observed=True):
        filas.append(dict(z=nz, D_um2_s=np.nan, grupo=f"distancia {de} um", **resumir_msd(g)))
    tdm = pd.DataFrame(filas)
    tdm.to_csv(SALIDA / "tracking_3d_msd.csv", index=False)
    print("\nB7 D por MSD (error relativo mediano contra el estimador sobre la pista GT):")
    print(tdm.to_string(index=False, float_format=fmt))
    figuras_analisis(t, te, tz, ta, td, tdm, pi)


def figuras_analisis(t, te, tz, ta, td, tdm, pi):
    plt = tb._estilo()
    # emparejamiento y error de z por distancia
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.4))
    ax = axs[0]
    for nb, c, et in (("verdadero", "#2a78d6", "beta verdadero (igual con 45)"), ("ctrl", "#1baf7a", "con control por la red")):
        s = te[(te.beta == nb) & (te.dist_um != "todas")]
        ax.plot(s.dist_um.astype(str), s.frac_emparejadas, "o-", color=c, label=f"emparejadas, {et}")
        ax.plot(s.dist_um.astype(str), s.frac_emparejadas_bien, "s--", color=c, label=f"bien emparejadas (>= 90% de cuadros), {et}")
        ax.plot(s.dist_um.astype(str), s.frac_cuadros_tri, ":", color=c, label=f"fraccion de cuadros con z triangulada, {et}")
    ax.set_xlabel("distancia de la particula al plano del espejo [um]")
    ax.set_ylabel("fraccion de pistas reales (>= 5 cuadros)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("B5: emparejamiento real <-> reflejo", fontsize=9)
    for ax, nb in zip(axs[1:], VARIANTES_TRI):
        s = tz[(tz.beta == nb) & tz.grupo.str.startswith("distancia")]
        x = s.grupo.str.replace("distancia ", "").str.replace(" um", "")
        for f, c, et in (("red", "#52514e", "red"), ("tri", "#7fb0ea", "triangulada"), ("suav", "#2a78d6", "suavizada (RTS)")):
            ax.plot(x, s[f"rmse_{f}"], "o-", color=c, label=f"RMSE {et}")
            ax.plot(x, s[f"mae_med_{f}"], "o:", color=c, label=f"mediana |e| {et}")
        ax.set_yscale("log")
        ax.set_xlabel("distancia al espejo [um]")
        ax.set_ylabel("error de z [nm] (mismos puntos)")
        ax.set_title(f"B6: error de z, {'beta verdadero + control por la red' if nb == 'ctrl' else 'beta ' + nb}", fontsize=9)
        ax.legend(frameon=False, fontsize=6.5)
    for ax in axs:
        ax.grid(True, color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_3d_espejo.png", dpi=300)
    plt.close(fig)

    # metricas por D para cada fuente de z
    s0 = ta[ta.densidad == "todas"]
    paneles = [("alpha", "alpha"), ("beta", "beta"), ("jsc", "JSC detecciones"), ("idsw_por_pista", "cambios de identidad / pista")]
    fig, axs = plt.subplots(1, 6, figsize=(21, 3.8))
    for ax, (k, et) in zip(axs, paneles):
        for nz in Z_VARIANTES:
            s = s0[s0.z == nz].sort_values("D_um2_s")
            ax.plot(s.D_um2_s, s[k], "o-", color=COLOR_Z[nz], lw=1.3, ms=3, label=NOMBRE_Z[nz])
            ax.fill_between(s.D_um2_s, s[k + "_lo"], s[k + "_hi"], color=COLOR_Z[nz], alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_title(et, fontsize=9)
    for ax, (k, et) in zip(axs[4:], (("err3_rel", "error relativo de D, 3D"), ("errz_rel", "error relativo de D, solo z"))):
        for nz in Z_VARIANTES:
            s = tdm[(tdm.z == nz) & (tdm.grupo == "todas")].sort_values("D_um2_s")
            ax.plot(s.D_um2_s, s[k], "o-", color=COLOR_Z[nz], lw=1.3, ms=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(et + " (mediana)", fontsize=9)
    for ax in axs:
        ax.set_xlabel("D [um^2/s]")
        ax.grid(True, color="#e6e5e0", lw=0.6)
    axs[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("B7: tracking 3D por fuente de z (mismas pistas; eps = 150 nm con z x 0.5; bandas: IC 95% bootstrap de videos)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_3d_metricas.png", dpi=300)
    plt.close(fig)

    # ejemplo: z(t) de algunas pistas emparejadas
    pts = t["puntos"]
    q = pts[np.isfinite(pts.e_tri_verdadero) & (pts.par_correcto_verdadero == 1)]
    cand = q.groupby(["video_id", "pista"]).size().sort_values(ascending=False)
    elegidos = []
    for D in (DS[1], DS[2], DS[3]):
        c = cand[[pts.loc[(pts.video_id == v), "D_um2_s"].iloc[0] == D for v, _ in cand.index]]
        if len(c):
            elegidos.append(c.index[0])
    fig, axs = plt.subplots(1, len(elegidos), figsize=(5.2 * len(elegidos), 3.8), squeeze=False)
    for ax, (v, k) in zip(axs[0], elegidos):
        g = pts[(pts.video_id == v) & (pts.pista == k)].sort_values("frame")
        ax.plot(g.frame, g.z_gt, "-", color="#1baf7a", lw=2, label="GT")
        ax.plot(g.frame, g.z_gt + g.e_red, "o", color="#52514e", ms=3, label="red")
        ax.plot(g.frame, g.z_gt + g.e_tri_verdadero, "o", color="#7fb0ea", ms=3, label="triangulada")
        ax.plot(g.frame, g.z_gt + g.e_suav_verdadero, "-", color="#2a78d6", lw=1.3, label="RTS")
        ax.plot(g.frame, g.z_gt + g.e_tri_45, "x", color="#eb6834", ms=3, label="triangulada, beta = 45")
        ax.set_title(f"video {v} (D = {g.D_um2_s.iloc[0]:g} um^2/s, beta = {g.espejo_angulo_eje_deg.iloc[0]:.1f}), pista {k}",
                     fontsize=8.5)
        ax.set_xlabel("cuadro (10 Hz)")
        ax.set_ylabel("z [nm]")
        ax.grid(True, color="#e6e5e0", lw=0.6)
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_tracking_3d_ejemplo.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("partes", nargs="+", choices=("localizar", "enlace", "reflejos", "analisis"))
    p.add_argument("--procesos", type=int, default=12)
    a = p.parse_args()
    if "localizar" in a.partes:
        localizar_videos()
    if "enlace" in a.partes:
        enlace(a.procesos)
    if "reflejos" in a.partes:
        reflejos(a.procesos)
    if "analisis" in a.partes:
        analisis(a.procesos)


if __name__ == "__main__":
    main()
