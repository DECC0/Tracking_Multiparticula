"""
Diagnostico de las corridas con celula (hypatia_celula_b45_cerca y _lejos)
sobre sus tests (localizacion/datos/test_celula_b45_<variante>) y control sin
reflejo de la variante cerca. Solo lee lo que escribe evaluate.py
(gt_eval.csv, predicciones.csv) y los tests; no corre la red.

1. Recall, RMSE y sesgo en z por z (rangos de 250 nm, y |z| <= 800 / 800-1600
   / > 1600 nm), por fotones, por distancia al espejo, por (z, fotones), por
   distancia al vecino real mas cercano y al reflejo mas cercano. IC 95% por
   bootstrap de imagenes (los emisores de una imagen no son independientes).
2. Falsos positivos: z predicha, distancia (con signo) a la superficie de la
   celula y clase, la primera que cumple, lateral <= r_xy (150 nm) del
       duplicado      emisor evaluable ya detectado por otra deteccion
       z_fuera        emisor evaluable no detectado (la deteccion es el, con |dz| > r_z)
       fuera_rango_z  emisor dentro del FOV lateral con z fuera del rango de la red (en_rango_z False)
       borde_fov      emisor fuera del FOV lateral
       reflejo        reflejo de cualquier emisor del frame (fp_sobre_reflejos de metricas.py)
       vecindad       emisor real a <= 3 r_xy (localizacion lateral mala)
       espurio        nada cerca
3. Control sin reflejo (--control): el mismo test regenerado con ratio_reflejo
   = 0 (mismas posiciones, fotones, fondo y geometria; solo cambia la luz del
   reflejo y, por eso, la realizacion del ruido de Poisson), evaluado con la
   misma red. Diferencias pareadas (con - sin) de recall y de RMSE z, sobre los
   detectados de cada uno y sobre los emisores detectados en los dos, por rango
   de z, fotones, distancia al espejo y ratio_reflejo, con IC 95% por bootstrap
   pareado de imagenes. Las dos evaluaciones deben correr en la misma maquina
   (evaluate.py --salida evaluacion_local_...).

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/diagnostico_celula.py [--variantes cerca lejos] [--eval VARIANTE CARPETA ...]
        [--control CON SIN] [--n-boot 2000] [--salida figuras]
Escribe en --salida: diag_celula_rangos.csv (todas las tablas por rango, columnas variante y
tabla), diag_celula_fp.csv (los FP con su clase), fig_diag_celula.png/.pdf,
fig_diag_celula_fp.png/.pdf y, con --control, diag_celula_control.csv y
fig_diag_celula_control.png/.pdf. Imprime todas las cifras de resultados_celula.md.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "simulador_nuevo"))
import simulador as sim  # noqa: E402

CORRIDAS = RAIZ / "localizacion" / "corridas"
DATOS = RAIZ / "localizacion" / "datos"
CORRIDA = {"cerca": "hypatia_celula_b45_cerca_20261110-234627", "lejos": "hypatia_celula_b45_lejos_20261110-234617"}
COLOR = {"cerca": "#2a78d6", "lejos": "#eb6834", "sin": "#52514e", "dif": "#1baf7a"}
R_XY, R_Z = 150.0, 300.0
CLASES_FP = ["duplicado", "z_fuera", "fuera_rango_z", "borde_fov", "reflejo", "vecindad", "espurio"]
Z_FINO = np.arange(-2500.0, 2500.1, 250.0)
Z_GRUESO = np.array([0.0, 800.0, 1600.0, 2500.0])          # en |z|
CLAVES = ["video_id", "frame", "particle_id"]


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------

def config_test(carpeta):
    meta = yaml.safe_load((carpeta / "datos.yaml").read_text())
    return sim.cargar_config(carpeta / "config_simulador.json"), meta


def centros_celula(cfg, muestras):
    """(n_muestras, 3) centro de la celula [nm] y (n_muestras,) radio [nm] de cada muestra."""
    c, R = np.empty((len(muestras), 3)), np.empty(len(muestras))
    for i, f in enumerate(muestras.itertuples()):
        cm = replace(cfg, angulo_espejo_eje_deg=f.espejo_angulo_eje_deg, orientacion_espejo_deg=f.espejo_orientacion_deg,
                     espejo_punto_px=(f.espejo_punto_x_px, f.espejo_punto_y_px), diametro_celula_um=f.celula_diametro_um,
                     altura_celula_nm=f.celula_altura_nm, celula_centro_px=(f.celula_centro_x_px, f.celula_centro_y_px))
        cel = sim.Celula(cm)
        c[i], R[i] = cel.c, cel.R
    return c, R


def vecinos(gt, px):
    """Distancia lateral [nm] de cada emisor al real y al reflejo (de otro emisor) mas cercanos del frame."""
    d_real = np.full(len(gt), np.inf)
    d_refl = np.full(len(gt), np.inf)
    xy = gt[["x_px", "y_px"]].to_numpy(float) * px
    xr = gt[["x_reflejo_px", "y_reflejo_px"]].to_numpy(float) * px
    for _, idx in gt.groupby(["video_id", "frame"]).indices.items():
        if len(idx) < 2:
            continue
        a = xy[idx]
        d = np.hypot(*(a[:, None] - a[None]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        d_real[idx] = d.min(1)
        d = np.hypot(*(a[:, None] - xr[idx][None]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        d_refl[idx] = d.min(1)
    return d_real, d_refl


def cargar(eval_dir, test_dir):
    """(gt, pred, cfg, muestras): gt_eval.csv con en_fov lateral original, celula,
    vecinos y ratio; predicciones.csv con la profundidad en la celula."""
    eval_dir, test_dir = Path(eval_dir), Path(test_dir)
    cfg, _ = config_test(test_dir)
    px = cfg.pixel_nm
    g = pd.read_csv(eval_dir / "gt_eval.csv")
    g0 = pd.read_csv(test_dir / "ground_truth.csv")
    m = pd.read_csv(test_dir / "muestras.csv")
    g = g.drop(columns=[c for c in ("en_fov_lat",) if c in g]).merge(
        g0[CLAVES + ["en_fov"]].rename(columns={"en_fov": "en_fov_lat"}), on=CLAVES, validate="1:1")
    assert (g.en_fov == (g.en_fov_lat & g.en_rango_z)).all()
    c, R = centros_celula(cfg, m)
    pos = {v: i for i, v in enumerate(m.video_id)}
    k = g.video_id.map(pos).to_numpy()
    p = np.c_[g.x_px * px, g.y_px * px, g.z_nm]
    g["prof_celula_nm"] = np.linalg.norm(p - c[k], axis=1) - R[k]      # <= 0: dentro
    assert g.prof_celula_nm.max() < 1e-6, "emisores fuera de su celula"
    g["d_vecino_nm"], g["d_reflejo_vecino_nm"] = vecinos(g, px)
    g["ratio_reflejo"] = g.video_id.map(m.set_index("video_id").ratio_reflejo)
    g["fondo_medio"] = g.video_id.map(m.set_index("video_id").fondo_medio)
    pr = pd.read_csv(eval_dir / "predicciones.csv")
    kp = pr.video_id.map(pos).to_numpy()
    q = np.c_[pr.x_px * px, pr.y_px * px, pr.z_nm]
    pr["prof_celula_nm"] = np.linalg.norm(q - c[kp], axis=1) - R[kp]
    return g, pr, cfg, m


# ----------------------------------------------------------------------------
# Bootstrap por imagenes
# ----------------------------------------------------------------------------

class Bootstrap:
    """Pesos multinomiales por imagen (video_id) compartidos por todas las tablas: diferencias pareadas."""

    def __init__(self, video_ids, n_boot, rng):
        self.vids = np.unique(video_ids)
        n = len(self.vids)
        self.W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)

    def sumas(self, vid, k, nb, peso):
        """(1 + n_boot, nb): estimacion (todos los pesos 1) y remuestreos de sum(peso) por rango."""
        iv = np.searchsorted(self.vids, vid)
        S = np.bincount(iv * nb + k, weights=peso, minlength=len(self.vids) * nb).reshape(len(self.vids), nb)
        return np.vstack([S.sum(0, keepdims=True), self.W @ S])


def estadisticos(bs, g, k, nb, comun=None):
    """{nombre: (1 + n_boot, nb)} de recall, rmse_z, sesgo_z, rmse_lat y n de los
    emisores evaluables g (rango k). comun (bool) restringe el RMSE a esos emisores."""
    vid = g.video_id.to_numpy()
    det = g.detectado.to_numpy(bool)
    sel = det if comun is None else comun
    dz = np.where(sel, g.dz_nm.fillna(0.0), 0.0)
    dl2 = np.where(sel, g.dx_nm.fillna(0.0) ** 2 + g.dy_nm.fillna(0.0) ** 2, 0.0)
    n, nd, ns = (bs.sumas(vid, k, nb, a.astype(float)) for a in (np.ones(len(g)), det, sel))
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(n=n, recall=nd / n, rmse_z=np.sqrt(bs.sumas(vid, k, nb, dz ** 2) / ns),
                    sesgo_z=bs.sumas(vid, k, nb, dz) / ns, rmse_lat=np.sqrt(bs.sumas(vid, k, nb, dl2) / ns))


def a_tabla(est, bordes, prefijo=""):
    """DataFrame (rango, estimacion, IC 95%) de un dict de estadisticos."""
    t = pd.DataFrame({"desde": bordes[:-1], "hasta": bordes[1:]})
    for nombre, a in est.items():
        t[prefijo + nombre] = a[0]
        if nombre != "n":
            lo, hi = np.nanpercentile(a[1:], [2.5, 97.5], axis=0)
            t[prefijo + nombre + "_lo"], t[prefijo + nombre + "_hi"] = lo, hi
    return t


def rangos(valores, bordes):
    """Indice de rango (lo de fuera va al primero o al ultimo)."""
    return np.clip(np.digitize(valores, bordes[1:-1]), 0, len(bordes) - 2)


def variables(g, cfg):
    """{nombre: (columna, bordes)} de las tablas por rango."""
    ev = g[g.en_fov]
    return {"z": (ev.z_nm, Z_FINO),
            "z_abs": (ev.z_nm.abs(), Z_GRUESO),
            "fotones": (ev.photons, np.geomspace(1000.0, 30000.0, 9)),
            "dist_espejo": (ev.d_espejo_nm, np.array([0, 500, 1000, 1500, 2000, 3000, 4000, 6000.0])),
            "vecino": (ev.d_vecino_nm, np.array([0, 300, 500, 750, 1000, 1500, 2000, 3000, np.inf])),
            "reflejo_vecino": (ev.d_reflejo_vecino_nm, np.array([0, 300, 500, 750, 1000, 1500, 2000, 3000, np.inf])),
            "ratio_reflejo": (ev.ratio_reflejo, np.linspace(0.45, 1.0, 5)),
            "prof_celula": (-ev.prof_celula_nm, np.array([0, 100, 250, 500, 1000, 3000.0]))}


def tablas(bs, g, cfg, comun=None, prefijo=""):
    ev = g[g.en_fov].reset_index(drop=True)
    c = None if comun is None else comun[g.en_fov.to_numpy()]
    out = {}
    for nombre, (col, b) in variables(g, cfg).items():
        k = rangos(col.to_numpy(float), b)
        out[nombre] = a_tabla(estadisticos(bs, ev, k, len(b) - 1, c), b, prefijo)
    # (z grueso) x (fotones en terciles fijos)
    bf = np.array([1000.0, 3000.0, 9000.0, 30000.0])
    kz, kf = rangos(ev.z_nm.abs().to_numpy(), Z_GRUESO), rangos(ev.photons.to_numpy(), bf)
    est = estadisticos(bs, ev, kz * 3 + kf, 9, c)
    t = a_tabla(est, np.arange(10.0), prefijo).drop(columns=["desde", "hasta"])
    t.insert(0, "abs_z_desde", np.repeat(Z_GRUESO[:-1], 3))
    t.insert(1, "fotones_desde", np.tile(bf[:-1], 3))
    out["z_abs_fotones"] = t
    return out


def largo(tablas):
    """Las tablas de un dict {nombre: DataFrame} en un solo DataFrame con la columna tabla."""
    return pd.concat([t.assign(tabla=n) for n, t in tablas.items()], ignore_index=True)


# ----------------------------------------------------------------------------
# Falsos positivos
# ----------------------------------------------------------------------------


def distancia_fp_reflejo(g, pr, px):
    """Distancia lateral [nm] de cada FP de pr al reflejo mas cercano de su imagen."""
    f = pr[~pr.tp & ~pr.fuera_fov]
    gi = g.groupby("video_id").indices
    xr = g[["x_reflejo_px", "y_reflejo_px"]].to_numpy(float)
    return np.array([np.hypot(*(xr[gi[v]] - [x, y]).T).min() * px for v, x, y in zip(f.video_id, f.x_px, f.y_px)])

def clasificar_fp(g, pr, px):
    """pr + clase (solo FP: ni tp ni fuera_fov) y dz al emisor evaluable mas cercano lateral."""
    pr = pr.copy()
    fp = (~pr.tp & ~pr.fuera_fov).to_numpy()
    clase = np.full(len(pr), "", dtype=object)
    dz_cerca = np.full(len(pr), np.nan)
    grupos = g.groupby(["video_id", "frame"]).indices
    pxy = pr[["x_px", "y_px"]].to_numpy(float) * px
    gxy = g[["x_px", "y_px"]].to_numpy(float) * px
    rxy = g[["x_reflejo_px", "y_reflejo_px"]].to_numpy(float) * px
    ev, det = g.en_fov.to_numpy(bool), g.detectado.to_numpy(bool)
    lat, rz = g.en_fov_lat.to_numpy(bool), g.en_rango_z.to_numpy(bool)
    gz = g.z_nm.to_numpy(float)
    for i in np.flatnonzero(fp):
        ig = grupos.get((pr.video_id.iat[i], pr.frame.iat[i]), np.zeros(0, int))
        d = np.hypot(*(gxy[ig] - pxy[i]).T)
        dr = np.hypot(*(rxy[ig] - pxy[i]).T)
        cerca = d <= R_XY
        if (cerca & ev[ig]).any():
            j = ig[cerca & ev[ig]][np.argmin(np.abs(gz[ig[cerca & ev[ig]]] - pr.z_nm.iat[i]))]
            dz_cerca[i] = pr.z_nm.iat[i] - gz[j]
        if (cerca & ev[ig] & det[ig]).any():
            clase[i] = "duplicado"
        elif (cerca & ev[ig]).any():
            clase[i] = "z_fuera"
        elif (cerca & lat[ig] & ~rz[ig]).any():
            clase[i] = "fuera_rango_z"
        elif (cerca & ~lat[ig]).any():
            clase[i] = "borde_fov"
        elif (dr <= R_XY).any():
            clase[i] = "reflejo"
        elif (d <= 3 * R_XY).any():
            clase[i] = "vecindad"
        else:
            clase[i] = "espurio"
    pr["clase_fp"] = np.where(fp, clase, "")
    pr["dz_emisor_nm"] = dz_cerca
    return pr


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------

def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _eje_x_log(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _x(t, log=False):
    hasta = np.where(np.isfinite(t.hasta), t.hasta, t.desde * 1.5 if not log else t.desde * 2)
    return np.sqrt(t.desde * hasta) if log else (t.desde + hasta) / 2


def _banda(ax, x, t, col, color, label, marca="o-"):
    ax.plot(x, t[col], marca, color=color, lw=1.8, ms=4, label=label)
    ax.fill_between(x, t[col + "_lo"], t[col + "_hi"], color=color, alpha=0.18, lw=0)


def figura_diag(tabs, ruta):
    plt = _estilo()
    filas = [("z", "z del emisor [nm]", False), ("fotones", "fotones del emisor", True),
             ("dist_espejo", "distancia al espejo [nm]", False), ("vecino", "real vecino mas cercano [nm]", False)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    for j, (nombre, etiq, log) in enumerate(filas):
        for var, t in tabs.items():
            tt = t[nombre]
            x = _x(tt, log)
            if nombre == "vecino":
                x = np.where(np.isfinite(tt.hasta), x, 3500.0)
            _banda(axes[0, j], x, tt, "recall", COLOR[var], var)
            _banda(axes[1, j], x, tt, "rmse_z", COLOR[var], var)
        for ax in axes[:, j]:
            ax.set_xlabel(etiq)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            if log:
                _eje_x_log(ax)
            if nombre == "z":
                for v in (-800, 800):
                    ax.axvline(v, color="#8a8984", lw=0.8, ls=":")
        axes[0, j].set_ylim(0, 1.02)
        axes[1, j].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("recall")
    axes[1, 0].set_ylabel("RMSE z de los detectados [nm]")
    axes[0, 0].legend(frameon=False, title="red y test")
    fig.suptitle("Celula (96 px, z +-2500 nm): recall y RMSE z por rango, IC 95% por bootstrap de imagenes", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta.with_suffix(".png"), dpi=200)
    fig.savefig(ruta.with_suffix(".pdf"))
    plt.close(fig)


def figura_fp(fps, gts, ruta):
    plt = _estilo()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    ancho = 0.38
    for k, (var, pr) in enumerate(fps.items()):
        f = pr[pr.clase_fp != ""]
        cuenta = f.clase_fp.value_counts().reindex(CLASES_FP, fill_value=0)
        axes[0].bar(np.arange(len(CLASES_FP)) + (k - 0.5) * ancho, cuenta.values, ancho, color=COLOR[var], label=var)
        tp = pr[pr.tp]
        bz = np.arange(-2900, 2901, 200)
        axes[1].hist(f.z_nm, bz, histtype="step", lw=1.8, color=COLOR[var], density=True, label=f"FP {var}")
        axes[1].hist(tp.z_nm, bz, histtype="step", lw=1, ls="--", color=COLOR[var], density=True, label=f"TP {var}")
        bp = np.arange(-3000, 1001, 100)
        axes[2].hist(f.prof_celula_nm, bp, histtype="step", lw=1.8, color=COLOR[var], density=True)
        axes[2].hist(tp.prof_celula_nm, bp, histtype="step", lw=1, ls="--", color=COLOR[var], density=True)
        bc = np.linspace(0, max(3.0, pr.conf.quantile(0.99)), 41)
        axes[3].hist(f.conf, bc, histtype="step", lw=1.8, color=COLOR[var], density=True)
        axes[3].hist(tp.conf, bc, histtype="step", lw=1, ls="--", color=COLOR[var], density=True)
    axes[0].set_xticks(range(len(CLASES_FP)))
    axes[0].set_xticklabels(CLASES_FP, rotation=35, ha="right")
    axes[0].set_ylabel("falsos positivos")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("z predicha [nm]")
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].set_xlabel("distancia a la superficie de la celula [nm] (< 0 dentro)")
    axes[2].axvline(0, color="#8a8984", lw=0.8)
    axes[3].set_xlabel("confianza (masa)")
    for ax in axes[1:]:
        ax.set_ylabel("densidad")
    for ax in axes:
        ax.grid(True, color="#e6e5e0", lw=0.6)
    fig.suptitle("Falsos positivos (trazo grueso) frente a verdaderos positivos (discontinuo)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta.with_suffix(".png"), dpi=200)
    fig.savefig(ruta.with_suffix(".pdf"))
    plt.close(fig)


def figura_control(t_con, t_sin, t_dif, ruta):
    plt = _estilo()
    filas = [("z", "z del emisor [nm]", False), ("fotones", "fotones del emisor", True),
             ("dist_espejo", "distancia al espejo [nm]", False), ("ratio_reflejo", "ratio_reflejo de la muestra", False)]
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.6), sharex="col")
    for j, (nombre, etiq, log) in enumerate(filas):
        a, b, d = t_con[nombre], t_sin[nombre], t_dif[nombre]
        x = _x(a, log)
        _banda(axes[0, j], x, a, "recall", COLOR["cerca"], "con reflejo")
        _banda(axes[0, j], x, b, "recall", COLOR["sin"], "ratio_reflejo = 0")
        _banda(axes[1, j], x, a, "rmse_z", COLOR["cerca"], "con reflejo")
        _banda(axes[1, j], x, b, "rmse_z", COLOR["sin"], "ratio_reflejo = 0")
        _banda(axes[2, j], x, d, "dif_rmse_z_comun", COLOR["dif"], "RMSE z, detectados en los dos")
        _banda(axes[2, j], x, d, "dif_recall", COLOR["lejos"], "recall")
        axes[2, j].axhline(0, color="#52514e", lw=0.8)
        axes[2, j].set_xlabel(etiq)
        for ax in axes[:, j]:
            ax.grid(True, color="#e6e5e0", lw=0.6)
            if log:
                _eje_x_log(ax)
        axes[0, j].set_ylim(0, 1.02)
        axes[1, j].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("recall")
    axes[1, 0].set_ylabel("RMSE z [nm]")
    axes[2, 0].set_ylabel("con - sin  (recall x 100; RMSE z en nm)")
    axes[0, 0].legend(frameon=False)
    axes[2, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Control sin reflejo: red cerca sobre el mismo test con ratio_reflejo = 0 (IC 95% bootstrap pareado)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta.with_suffix(".png"), dpi=200)
    fig.savefig(ruta.with_suffix(".pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------

def imprimir(titulo, t, cols):
    print(f"\n{titulo}")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(t[[c for c in cols if c in t]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def control(con_dir, sin_dir, test_dir, n_boot, rng, salida):
    g1, p1, cfg, _ = cargar(con_dir, test_dir)
    g0, p0, _, _ = cargar(sin_dir, Path(str(test_dir) + "_sin_reflejo"))
    g1, g0 = g1.sort_values(CLAVES).reset_index(drop=True), g0.sort_values(CLAVES).reset_index(drop=True)
    assert np.allclose(g1[["x_px", "y_px", "z_nm", "photons"]], g0[["x_px", "y_px", "z_nm", "photons"]])
    assert (g0.photons_reflejo == 0).all() and (g1.en_fov == g0.en_fov).all()
    g0["ratio_reflejo"] = g1.ratio_reflejo          # la de la muestra con reflejo, para los rangos
    bs = Bootstrap(g1.video_id, n_boot, rng)
    comun = (g1.detectado & g0.detectado).to_numpy()
    t1, t0 = tablas(bs, g1, cfg), tablas(bs, g0, cfg)
    c1, c0 = tablas(bs, g1, cfg, comun), tablas(bs, g0, cfg, comun)
    # diferencias pareadas con los mismos pesos
    ev = g1.en_fov.to_numpy()
    dif = {}
    for nombre, (col, b) in variables(g1, cfg).items():
        k = rangos(col.to_numpy(float), b)
        e1 = estadisticos(bs, g1[ev].reset_index(drop=True), k, len(b) - 1)
        e0 = estadisticos(bs, g0[ev].reset_index(drop=True), k, len(b) - 1)
        f1 = estadisticos(bs, g1[ev].reset_index(drop=True), k, len(b) - 1, comun[ev])
        f0 = estadisticos(bs, g0[ev].reset_index(drop=True), k, len(b) - 1, comun[ev])
        est = dict(n=e1["n"], dif_recall=100 * (e1["recall"] - e0["recall"]), dif_rmse_z=e1["rmse_z"] - e0["rmse_z"],
                   dif_rmse_z_comun=f1["rmse_z"] - f0["rmse_z"],
                   n_comun=bs.sumas(g1.video_id[ev].to_numpy(), k, len(b) - 1, comun[ev].astype(float)),
                   rmse_z_comun_con=f1["rmse_z"], rmse_z_comun_sin=f0["rmse_z"],
                   dif_sesgo_z_comun=f1["sesgo_z"] - f0["sesgo_z"])
        dif[nombre] = a_tabla(est, b)
    # global
    k0 = np.zeros(int(ev.sum()), int)
    e1 = estadisticos(bs, g1[ev].reset_index(drop=True), k0, 1)
    e0 = estadisticos(bs, g0[ev].reset_index(drop=True), k0, 1)
    f1 = estadisticos(bs, g1[ev].reset_index(drop=True), k0, 1, comun[ev])
    f0 = estadisticos(bs, g0[ev].reset_index(drop=True), k0, 1, comun[ev])
    glob = a_tabla(dict(recall_con=e1["recall"], recall_sin=e0["recall"], dif_recall=e1["recall"] - e0["recall"],
                        rmse_z_con=e1["rmse_z"], rmse_z_sin=e0["rmse_z"], dif_rmse_z=e1["rmse_z"] - e0["rmse_z"],
                        rmse_z_comun_con=f1["rmse_z"], rmse_z_comun_sin=f0["rmse_z"],
                        dif_rmse_z_comun=f1["rmse_z"] - f0["rmse_z"],
                        dif_rmse_lat_comun=f1["rmse_lat"] - f0["rmse_lat"]), np.array([0.0, 1.0]))
    fp1, fp0 = int((~p1.tp & ~p1.fuera_fov).sum()), int((~p0.tp & ~p0.fuera_fov).sum())
    print(f"\n[control sin reflejo] {con_dir.name} vs {sin_dir.name}: {int(ev.sum())} emisores evaluables, "
          f"detectados en los dos {int(comun[ev].sum())}; FP con {fp1}, sin {fp0}")
    for nombre, pr in (("con", p1), ("sin", p0)):
        d = distancia_fp_reflejo(g1, pr, cfg.pixel_nm)
        print(f"  FP {nombre} reflejo a < 150 / 300 / 500 / 1000 nm (lateral) de algun reflejo: "
              f"{[int((d < r).sum()) for r in (150, 300, 500, 1000)]}")
    imprimir("global (con - sin; IC 95% pareado)", glob.T.reset_index(), list(glob.T.reset_index().columns))
    for nombre in ("z", "z_abs", "fotones", "dist_espejo", "ratio_reflejo", "reflejo_vecino", "z_abs_fotones"):
        if nombre in dif:
            imprimir(f"por {nombre}: diferencias con - sin", dif[nombre],
                     ["desde", "hasta", "n", "dif_recall", "dif_recall_lo", "dif_recall_hi", "rmse_z_comun_con",
                      "rmse_z_comun_sin", "dif_rmse_z_comun", "dif_rmse_z_comun_lo", "dif_rmse_z_comun_hi",
                      "dif_sesgo_z_comun"])
    largo({**dif, "global": glob}).to_csv(salida / "diag_celula_control.csv", index=False)
    figura_control(t1, t0, dif, salida / "fig_diag_celula_control")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--variantes", nargs="+", default=["cerca", "lejos"], choices=list(CORRIDA))
    p.add_argument("--eval", nargs=2, action="append", metavar=("VARIANTE", "CARPETA"),
                   help="carpeta de evaluate.py de la variante (por defecto la de Hypatia de su corrida)")
    p.add_argument("--control", nargs=2, metavar=("CON", "SIN"), default=None,
                   help="evaluaciones de la red cerca (misma maquina) sobre test_celula_b45_cerca y su _sin_reflejo")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--salida", default=str(Path(__file__).resolve().parent))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    salida = Path(a.salida)
    evals = dict(a.eval or [])

    tabs, fps, gts, bs = {}, {}, {}, None
    for var in a.variantes:
        e = Path(evals.get(var, CORRIDAS / CORRIDA[var] / f"evaluacion_test_celula_b45_{var}"))
        test = DATOS / f"test_celula_b45_{var}"
        g, pr, cfg, m = cargar(e, test)
        bs = bs or Bootstrap(g.video_id, a.n_boot, rng)      # mismos pesos: los dos tests tienen los mismos emisores
        r = yaml.safe_load(open(e / "resumen.yaml"))["metricas"]
        print(f"\n===== {var}: {e.relative_to(RAIZ) if e.is_relative_to(RAIZ) else e}  "
              f"(recall {r['recall']:.3f}, precision {r['precision']:.3f}, RMSE z {r['rmse_z_nm']:.1f} nm)")
        tabs[var] = tablas(bs, g, cfg)
        for nombre in ("z", "z_abs", "fotones", "dist_espejo", "vecino", "reflejo_vecino", "prof_celula", "z_abs_fotones"):
            imprimir(f"{var}, por {nombre}", tabs[var][nombre],
                     ["abs_z_desde", "fotones_desde", "desde", "hasta", "n", "recall", "recall_lo", "recall_hi",
                      "rmse_z", "rmse_z_lo", "rmse_z_hi", "sesgo_z", "rmse_lat"])
        ev = g[g.en_fov]
        print(f"  fuera del rango z (en el FOV lateral): {int((g.en_fov_lat & ~g.en_rango_z).sum())}; "
              f"FN {int((~ev.detectado).sum())}, de ellos con < 3000 fotones {int(((~ev.detectado) & (ev.photons < 3000)).sum())}, "
              f"con un real a < 500 nm {int(((~ev.detectado) & (ev.d_vecino_nm < 500)).sum())}")
        pr = clasificar_fp(g, pr, cfg.pixel_nm)
        fps[var], gts[var] = pr, g
        f = pr[pr.clase_fp != ""]
        cuenta = f.clase_fp.value_counts().reindex(CLASES_FP, fill_value=0)
        print(f"  FP: {len(f)}; por clase: {cuenta.to_dict()}")
        zf = f[f.clase_fp == "z_fuera"].dz_emisor_nm
        print(f"    z_fuera: |dz| mediana {zf.abs().median():.0f} nm, fraccion con dz > 0 {(zf > 0).mean():.2f}; "
              f"duplicados: dz mediana {f[f.clase_fp == 'duplicado'].dz_emisor_nm.abs().median():.0f} nm")
        tp = pr[pr.tp]
        for nombre, s in (("FP", f), ("TP", tp)):
            print(f"    {nombre}: z predicha |z| > 2000 nm {(s.z_nm.abs() > 2000).mean():.2f}, |z| > 1600 {(s.z_nm.abs() > 1600).mean():.2f}; "
                  f"fuera de la celula (> 0 nm) {(s.prof_celula_nm > 0).mean():.2f}, a > 300 nm fuera "
                  f"{(s.prof_celula_nm > 300).mean():.2f}, a < 250 nm de la superficie {(s.prof_celula_nm.abs() < 250).mean():.2f}; "
                  f"conf mediana {s.conf.median():.2f}")
        n_ev = int(g.en_fov.sum())
        print("    umbral de confianza (aprox.: quitar detecciones, sin re-emparejar): " + "; ".join(
            f"conf >= {u}: FP {int((f.conf >= u).sum())}, recall {(tp.conf >= u).sum() / n_ev:.3f}"
            for u in (0.0, 0.6, 0.7, 0.8, 0.9)))
        fz = pd.crosstab(pd.cut(f.z_nm, [-3000, -1600, -800, 800, 1600, 3000]), f.clase_fp)
        print("    FP por z predicha y clase:\n" + fz.to_string())
    pd.concat([largo(t).assign(variante=v) for v, t in tabs.items()]).to_csv(salida / "diag_celula_rangos.csv", index=False)
    pd.concat([pr[pr.clase_fp != ""].assign(variante=v) for v, pr in fps.items()]).to_csv(
        salida / "diag_celula_fp.csv", index=False)
    figura_diag(tabs, salida / "fig_diag_celula")
    figura_fp(fps, gts, salida / "fig_diag_celula_fp")
    if a.control:
        control(Path(a.control[0]), Path(a.control[1]), DATOS / "test_celula_b45_cerca", a.n_boot, rng, salida)


if __name__ == "__main__":
    main()
