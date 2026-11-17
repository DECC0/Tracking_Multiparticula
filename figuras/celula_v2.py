"""
Evaluacion de las corridas v2 (configs/celula_v2.yaml, variante cerca, dos
semillas de entrenamiento: hypatia_celula_v2_b45_cerca_s0 y _s1) frente a la
red cerca actual (hypatia_celula_b45_cerca_20261110-234627). Ver
resultados/resultados_celula_v2.md.

Partes:
    convergencia  log.csv de las dos semillas: perdidas y metricas de val por
                  epoca, epoca del minimo de val (mejor.pt), lr y tendencia de
                  las ultimas 20 epocas (pendiente con el mayor de los ee OLS y
                  Newey-West, como figuras/convergencia.py)
                  -> celula_v2_convergencia.csv, fig_celula_v2_convergencia.png
    evaluacion    solo lee las salidas de evaluate.py (evaluacion_local_<test>
                  de cada semilla; las de la red actual son las de
                  resultados/resultados_simulador_v2.md) sobre el test v2, el control con
                  la PSF fija y el test v1: metricas globales con IC 95% por
                  bootstrap de imagenes, diferencia pareada s0 - s1 (mismos
                  emisores), desglose por amp_esferica, astigmatismo, coma,
                  contraste de la autofluorescencia, fotones y z, agujeros en z
                  (recall por 100 nm de z real y densidad de z predicha frente a
                  la de z real) y acuerdo emisor a emisor entre semillas
                  -> celula_v2_global.csv, celula_v2_rangos.csv, celula_v2_z.csv,
                  fig_celula_v2_evaluacion.png, fig_celula_v2_z.png
    reales        corre las redes (GPU si hay) sobre Yeast6_1, canal derecho,
                  con el procedimiento de `red_celula.py reales` (ventana de
                  96 px en el punto medio del par, linea en el borde de la
                  franja, z de la deteccion a <= 3 px del real) y la z
                  triangulada con beta 40, 42.5, 45, 47.5 y 50 grados (la red no
                  depende de beta: el canal de distancia solo usa la linea).
                  Criterios fijados antes de ver los resultados:
                    - sin saltos entre modos: ningun |dz| > SALTO_NM (1000 nm)
                      entre cuadros consecutivos con el real detectado;
                    - correlacion con la z triangulada > 0.7;
                    - pendiente (OLS, z red ~ z tri) en [0.7, 1.3].
                  Pasa o no por la estimacion puntual; IC 95% por bootstrap de
                  bloques de 10 cuadros (red_celula.ajuste)
                  -> celula_v2_reales.csv (por cuadro), celula_v2_reales_resumen.csv,
                  fig_celula_v2_reales.png

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/celula_v2.py convergencia evaluacion
    LD_PRELOAD=$HOME/miniforge3/envs/tesis/lib/libstdc++.so.6 $PY figuras/celula_v2.py reales [--tif RUTA]
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG))
CORRIDAS = RAIZ / "localizacion" / "corridas"
DATOS = RAIZ / "localizacion" / "datos"
REDES = {"s0": "hypatia_celula_v2_b45_cerca_s0_20261114-090158",
         "s1": "hypatia_celula_v2_b45_cerca_s1_20261114-090158",
         "actual": "hypatia_celula_b45_cerca_20261110-234627"}
TESTS = {"v2": "test_celula_v2_b45_cerca", "control": "test_celula_v2_psf_fija_b45_cerca", "v1": "test_celula_b45_cerca"}
# carpeta de evaluate.py de cada (red, test): las semillas, evaluadas localmente; la red actual, las de
# resultados/resultados_simulador_v2.md (v1 de Hypatia, v2 y control locales)
EVAL = {(r, t): f"evaluacion_local_{TESTS[t]}" for r in ("s0", "s1") for t in TESTS}
EVAL.update({("actual", t): f"evaluacion_{TESTS[t]}" for t in TESTS})
COLOR = {"s0": "#2a78d6", "s1": "#1baf7a", "actual": "#eb6834"}
NOMBRE = {"s0": "v2 semilla 0", "s1": "v2 semilla 1", "actual": "red cerca actual (v1)"}
NOMBRE_TEST = {"v2": "test v2", "control": "control v2 (PSF fija)", "v1": "test v1"}
BINS = {"amp_esferica": np.linspace(4.0, 6.5, 6), "amp_astig": np.linspace(0, 1.3, 6), "amp_coma": np.linspace(0, 1.2, 7),
        "contraste_celula": np.array([0, 0.25, 0.5, 1, 2, 4]), "photons": np.array([1000, 2000, 5000, 10000, 20000, 30000, 50000]),
        "z_nm": np.arange(-3000, 3001, 250), "fondo_medio": np.array([0, 3, 6, 12, 25, 50, 200])}
N_BOOT = 2000
SALTO_NM = 1000.0
BETAS = (40.0, 42.5, 45.0, 47.5, 50.0)


def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


# ----------------------------------------------------------------------------
# 1. Convergencia
# ----------------------------------------------------------------------------

def convergencia(salida, ultimas=20):
    from convergencia import pendiente, lr_por_epoca
    filas, logs = [], {}
    for s in ("s0", "s1"):
        c = CORRIDAS / REDES[s]
        log = pd.read_csv(c / "log.csv")
        lr = lr_por_epoca(pd.read_csv(c / "log_iter.csv"))
        assert np.allclose(lr.values, log.lr.values), "log.csv debe tener la lr usada (corridas >= 2026-11-12)"
        logs[s] = log
        k = int(log.val_total.idxmin())
        sel = log[log.epoca >= log.epoca.max() - ultimas + 1]
        cambios = log.epoca[log.lr.diff().fillna(0) != 0].tolist()
        base = dict(semilla=s, epocas=len(log), epoca_mejor=int(log.epoca[k]), val_total_mejor=log.val_total[k],
                    mejor_ultimas_10=log.epoca[log.mejor & (log.epoca >= log.epoca.max() - 9)].tolist(),
                    epoca_lr_min=int(log.epoca[log.lr <= 1e-5 + 1e-12].min()), caidas_lr=cambios,
                    **{f"{m}_mejor": log[m][k] for m in ("val_recall", "val_precision", "val_jaccard", "val_rmse_lat_nm",
                                                         "val_rmse_z_nm")})
        for col in ("val_total", "val_mse", "val_dice", "train_total", "val_recall", "val_precision", "val_jaccard",
                    "val_rmse_lat_nm", "val_rmse_z_nm"):
            b, ee, ee_nw, _ = pendiente(sel.epoca, sel[col])
            e = max(ee, ee_nw)
            filas.append(dict(base, serie=col, ventana=f"{sel.epoca.min()}-{sel.epoca.max()}", media=sel[col].mean(),
                              pendiente_10_epocas=10 * b, ic95_lo=10 * (b - 1.96 * e), ic95_hi=10 * (b + 1.96 * e),
                              relativo_10_epocas=10 * b / sel[col].mean(), plana=(b - 1.96 * e) <= 0 <= (b + 1.96 * e),
                              std_epocas=float(sel[col].std())))
    t = pd.DataFrame(filas)
    t.to_csv(salida / "celula_v2_convergencia.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for s, log in logs.items():
            r = t[t.semilla == s].iloc[0]
            print(f"{s}: {r.epocas} epocas; minimo de val_total {r.val_total_mejor:.4f} en la epoca {r.epoca_mejor}; "
                  f"mejor.pt en las ultimas 10: {r.mejor_ultimas_10}; caidas de lr en {r.caidas_lr}, lr_min desde la "
                  f"{r.epoca_lr_min}; en la mejor: recall {r.val_recall_mejor:.3f}, precision {r.val_precision_mejor:.3f}, "
                  f"RMSE z {r.val_rmse_z_nm_mejor:.1f} nm")
        print(t[["semilla", "serie", "ventana", "media", "pendiente_10_epocas", "ic95_lo", "ic95_hi", "relativo_10_epocas",
                 "plana", "std_epocas"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        # metricas de val en ventanas de 10 epocas (cuando dejan de cambiar)
        for s, log in logs.items():
            v = log.groupby(log.epoca // 10 * 10)[["val_total", "val_recall", "val_precision", "val_rmse_z_nm"]].mean()
            print(f"\n{s}, medias por ventana de 10 epocas:\n{v.T.to_string(float_format=lambda x: f'{x:.4g}')}")

    plt = _estilo()
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.6))
    paneles = [("val_total", "perdida total (MSE + Dice)"), ("val_recall", "recall de val"),
               ("val_precision", "precision de val"), ("lr", "tasa de aprendizaje"),
               ("val_rmse_z_nm", "RMSE z de val [nm]"), ("val_rmse_lat_nm", "RMSE lateral de val [nm]")]
    for ax, (col, et) in zip(axs.flat, paneles):
        for s, log in logs.items():
            c = COLOR[s]
            if col == "val_total":
                ax.plot(log.epoca, log.train_total, "-", color=c, lw=0.9, alpha=0.6, label=f"{NOMBRE[s]}: train")
            ax.plot(log.epoca, log[col], "-" if col == "lr" else "--" if col == "val_total" else "-", color=c, lw=1.4,
                    label=f"{NOMBRE[s]}" + (": val" if col == "val_total" else ""))
            k = log.val_total.idxmin()
            if col != "lr":
                ax.plot(log.epoca[k], log[col][k], "*", color=c, ms=10, mec="white", mew=0.8)
        if col in ("val_total", "lr"):
            ax.set_yscale("log")
        if col == "val_total":
            ax.set_ylim(0.48, 0.75)
        if col == "val_rmse_z_nm":
            ax.set_ylim(90, 135)
        ax.set_title(et, fontsize=9)
        ax.set_xlabel("epoca (0 = la primera)")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Corridas v2 (celula_v2.yaml, cerca): log.csv por epoca; estrella = minimo de la perdida de val "
                 "(mejor.pt); val = 1000 muestras fijas", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(salida / "fig_celula_v2_convergencia.png", dpi=300)
    plt.close(fig)
    return t


# ----------------------------------------------------------------------------
# 2. Evaluacion
# ----------------------------------------------------------------------------

def cargar(red, test):
    e = CORRIDAS / REDES[red] / EVAL[(red, test)]
    gt = pd.read_csv(e / "gt_eval.csv")
    pred = pd.read_csv(e / "predicciones.csv")
    m = pd.read_csv(DATOS / TESTS[test] / "muestras.csv")
    if "fondo_celula" in m:
        m["contraste_celula"] = m.fondo_celula / np.maximum(m.fondo_medio, 1e-9)
    if test == "control":           # el control comparte los video_id: la PSF del v2 sirve para agrupar
        m2 = pd.read_csv(DATOS / TESTS["v2"] / "muestras.csv")
        m = m.drop(columns=[c for c in ("amp_esferica", "amp_astig", "amp_coma") if c in m]).merge(
            m2[["video_id", "amp_esferica", "amp_astig", "amp_coma"]], on="video_id")
    ev = gt[gt.en_fov & gt.en_rango_z].merge(m, on="video_id", how="left", suffixes=("", "_m"))
    fp = pred[~pred.tp & ~pred.fuera_fov]
    return ev, pred, fp, m


class Bootstrap:
    """Remuestreo de imagenes: pesos multinomiales por video_id, comunes a todas las tablas (pareado)."""

    def __init__(self, n_videos, n_boot, rng):
        self.w = rng.multinomial(n_videos, np.full(n_videos, 1 / n_videos), size=n_boot).astype(np.float64)
        self.n = n_videos

    def sumas(self, vid, valores):
        return self.w @ np.bincount(vid, weights=valores, minlength=self.n)


def metricas_boot(bs, ev, fp, n_img):
    """(estimacion, (n_boot,) muestras) de recall, precision, FP por imagen, RMSE lat/z, sesgo z."""
    v = ev.video_id.to_numpy()
    d = ev.detectado.to_numpy().astype(float)
    dz2 = np.where(d > 0, ev.dz_nm.fillna(0).to_numpy() ** 2, 0)
    dl2 = np.where(d > 0, (ev.dx_nm.fillna(0) ** 2 + ev.dy_nm.fillna(0) ** 2).to_numpy(), 0)
    dzs = np.where(d > 0, ev.dz_nm.fillna(0).to_numpy(), 0)
    nfp = np.bincount(fp.video_id.to_numpy(), minlength=n_img).astype(float)
    S = lambda x: (np.bincount(v, weights=x, minlength=n_img).sum(), bs.sumas(v, x))  # noqa: E731
    (n, nb), (tp, tpb), (z2, z2b), (l2, l2b), (zs, zsb) = (S(np.ones_like(d)), S(d), S(dz2), S(dl2), S(dzs))
    f, fb = nfp.sum(), bs.w @ nfp
    return {"recall": (tp / n, tpb / nb), "precision": (tp / (tp + f), tpb / (tpb + fb)),
            "fp_por_imagen": (f / n_img, fb / bs.w.sum(1)), "rmse_lat_nm": (np.sqrt(l2 / tp), np.sqrt(l2b / tpb)),
            "rmse_z_nm": (np.sqrt(z2 / tp), np.sqrt(z2b / tpb)), "sesgo_z_nm": (zs / tp, zsb / tpb)}


def simples(g):
    d = g[g.detectado]
    return dict(n=len(g), recall=len(d) / max(len(g), 1),
                rmse_z_nm=float(np.sqrt(np.mean(d.dz_nm ** 2))) if len(d) >= 5 else np.nan,
                sesgo_z_nm=float(d.dz_nm.mean()) if len(d) >= 5 else np.nan,
                rmse_lat_nm=float(np.sqrt(np.mean(d.dx_nm ** 2 + d.dy_nm ** 2))) if len(d) >= 5 else np.nan)


def agujeros_z(ev, pred, paso=100.0, lim=3000.0):
    """Por bin de z: recall (z real) y densidad de z predicha (todas las detecciones en el FOV) / densidad de z real."""
    b = np.arange(-lim, lim + paso, paso)
    hz, _ = np.histogram(ev.z_nm, b)
    hp, _ = np.histogram(pred.z_nm[~pred.fuera_fov], b)
    filas = []
    for i, (lo, hi) in enumerate(zip(b[:-1], b[1:])):
        s = ev[(ev.z_nm >= lo) & (ev.z_nm < hi)]
        filas.append(dict(desde=lo, hasta=hi, n_gt=int(hz[i]), n_pred=int(hp[i]),
                          razon_pred_gt=hp[i] / hz[i] if hz[i] else np.nan, **{k: v for k, v in simples(s).items() if k != "n"}))
    return pd.DataFrame(filas)


def evaluacion(salida, rng):
    nets = list(REDES)
    datos = {(r, t): cargar(r, t) for r in nets for t in TESTS}
    filas_g, filas_r, filas_z = [], [], []
    for t in TESTS:
        n_img = int(datos[("s0", t)][3].video_id.max()) + 1
        bs = Bootstrap(n_img, N_BOOT, rng)
        boot = {}
        for r in nets:
            ev, pred, fp, m = datos[(r, t)]
            mb = metricas_boot(bs, ev, fp, n_img)
            boot[r] = mb
            fila = dict(red=r, test=t, n_gt=len(ev), tp=int(ev.detectado.sum()), fp=len(fp))
            for k, (e, b) in mb.items():
                fila[k], fila[k + "_lo"], fila[k + "_hi"] = e, *np.percentile(b, [2.5, 97.5])
            g25 = ev[ev.z_nm.abs() <= 2500]
            fila["recall_z_le_2500"] = g25.detectado.mean()
            fila["recall_z_gt_2500"] = ev[ev.z_nm.abs() > 2500].detectado.mean() if (ev.z_nm.abs() > 2500).any() else np.nan
            p = pred[~pred.fuera_fov]
            fila["z_pred_min"], fila["z_pred_max"] = p.z_nm.min(), p.z_nm.max()
            filas_g.append(fila)
        for a, b_ in (("s0", "s1"), ("s0", "actual"), ("s1", "actual")):
            fila = dict(red=f"{a} - {b_}", test=t)
            for k in boot[a]:
                dif = boot[a][k][1] - boot[b_][k][1]
                fila[k] = boot[a][k][0] - boot[b_][k][0]
                fila[k + "_lo"], fila[k + "_hi"] = np.percentile(dif, [2.5, 97.5])
            filas_g.append(fila)
        # acuerdo emisor a emisor entre semillas
        cl = ["video_id", "particle_id"]
        a0 = datos[("s0", t)][0][cl + ["detectado"]].merge(datos[("s1", t)][0][cl + ["detectado"]], on=cl, suffixes=("_s0", "_s1"))
        filas_g.append(dict(red="acuerdo s0/s1", test=t, n_gt=len(a0), ambos=np.mean(a0.detectado_s0 & a0.detectado_s1),
                            solo_s0=np.mean(a0.detectado_s0 & ~a0.detectado_s1),
                            solo_s1=np.mean(~a0.detectado_s0 & a0.detectado_s1),
                            ninguno=np.mean(~a0.detectado_s0 & ~a0.detectado_s1)))
        # rangos
        for r in nets:
            ev, pred, fp, m = datos[(r, t)]
            nfp = fp.groupby("video_id").size()
            mm = m.assign(fp=m.video_id.map(nfp).fillna(0))
            for f, b in BINS.items():
                if f not in ev:
                    continue
                if f in ("amp_esferica", "amp_astig", "amp_coma") and t != "v2":
                    continue
                for lo, hi in zip(b[:-1], b[1:]):
                    s = ev[(ev[f] >= lo) & (ev[f] < hi)]
                    fila = dict(red=r, test=t, factor=f, desde=lo, hasta=hi, **simples(s))
                    if f in mm:
                        si = mm[(mm[f] >= lo) & (mm[f] < hi)]
                        fila.update(imagenes=len(si), fp_por_imagen=si.fp.mean() if len(si) else np.nan)
                    filas_r.append(fila)
            filas_z.append(agujeros_z(ev, pred).assign(red=r, test=t))
    g = pd.DataFrame(filas_g)
    rg = pd.DataFrame(filas_r)
    zz = pd.concat(filas_z, ignore_index=True)
    g.to_csv(salida / "celula_v2_global.csv", index=False)
    rg.to_csv(salida / "celula_v2_rangos.csv", index=False)
    zz.to_csv(salida / "celula_v2_z.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_rows", 900, "display.max_columns", 40):
        print(g.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        for f in BINS:
            s = rg[rg.factor == f]
            if not len(s):
                continue
            cols = ["recall", "rmse_z_nm", "sesgo_z_nm", "rmse_lat_nm"] + (["fp_por_imagen"] if "fp_por_imagen" in s and
                                                                          s.fp_por_imagen.notna().any() else [])
            pv = s.pivot_table(index=["test", "desde", "hasta"], columns="red", values=cols + ["n"], sort=False)
            print(f"\n== {f}\n{pv.to_string(float_format=lambda v: f'{v:.3g}')}")
        sel = zz[(zz.test != "v1") & (zz.desde >= 0) & (zz.desde < 1000)]
        print("\n== z en [0, 1000) cada 100 nm (agujero de +400)")
        print(sel.pivot_table(index=["test", "desde"], columns="red", values=["recall", "razon_pred_gt", "rmse_z_nm"],
                              sort=False).to_string(float_format=lambda v: f"{v:.3g}"))
        sel = zz[(zz.test == "v2") & (zz.desde.abs() >= 2000)]
        print("\n== test v2, |z| >= 2000")
        print(sel.pivot_table(index=["desde"], columns="red", values=["recall", "razon_pred_gt", "rmse_z_nm", "sesgo_z_nm"],
                              sort=False).to_string(float_format=lambda v: f"{v:.3g}"))
    figura_evaluacion(rg, salida / "fig_celula_v2_evaluacion.png")
    figura_z(zz, salida / "fig_celula_v2_z.png")
    return g, rg, zz


def figura_evaluacion(rg, ruta):
    plt = _estilo()
    fig, axs = plt.subplots(3, 4, figsize=(16, 10.5))
    paneles = [("v2", "amp_esferica", "amp_esferica [rad]"), ("v2", "amp_astig", "astigmatismo [rad]"),
               ("v2", "amp_coma", "coma [rad]"), ("v2", "photons", "fotones"),
               ("v2", "contraste_celula", "contraste de la autofluorescencia"), ("v2", "z_nm", "z [nm]"),
               ("control", "z_nm", "z [nm]"), ("v1", "z_nm", "z [nm]")]
    for k, (t, f, et) in enumerate(paneles):
        ax = axs.flat[k] if k < 4 else axs.flat[k]
        for r in REDES:
            s = rg[(rg.red == r) & (rg.test == t) & (rg.factor == f) & (rg.n >= 20)]
            x = 0.5 * (s.desde + s.hasta)
            ax.plot(x, s.recall, "o-", color=COLOR[r], lw=1.5, ms=3.5, label=f"{NOMBRE[r]}: recall")
            ax.plot(x, s.rmse_z_nm / 1000, "s--", color=COLOR[r], lw=1.1, ms=3, label=f"{NOMBRE[r]}: RMSE z [um]")
        if f == "photons":
            ax.set_xscale("log")
            ax.set_xticks([1e3, 2e3, 5e3, 1e4, 2e4, 5e4], ["1k", "2k", "5k", "10k", "20k", "50k"])
            ax.minorticks_off()
        if f == "z_nm":
            for v in (-2500, 2500):
                ax.axvline(v, color="#8a8984", lw=0.8, ls=":")
            ax.axvspan(250, 500, color="#e87ba4", alpha=0.12, lw=0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(et)
        ax.set_title(f"{NOMBRE_TEST[t]}: por {et}", fontsize=8.5)
        ax.grid(True, color="#e6e5e0", lw=0.6)
    axs.flat[0].legend(frameon=False, fontsize=6, loc="lower left")
    # FP por imagen
    for k, (t, f, et) in enumerate([("v2", "amp_esferica", "amp_esferica [rad]"),
                                    ("v2", "contraste_celula", "contraste de la autofluorescencia"),
                                    ("control", "contraste_celula", "contraste de la autofluorescencia"),
                                    ("v2", "fondo_medio", "fondo medio [e-/px]")]):
        ax = axs.flat[8 + k]
        for r in REDES:
            s = rg[(rg.red == r) & (rg.test == t) & (rg.factor == f)]
            if f == "fondo_medio":
                s = s[s.n >= 20]
                ax.plot(0.5 * (s.desde + s.hasta), s.recall, "o-", color=COLOR[r], lw=1.5, ms=3.5, label=NOMBRE[r])
                ax.set_ylabel("recall"); ax.set_xscale("log"); ax.set_ylim(0, 1.02)
            else:
                ax.plot(0.5 * (s.desde + s.hasta), s.fp_por_imagen, "o-", color=COLOR[r], lw=1.5, ms=3.5, label=NOMBRE[r])
                ax.set_ylabel("FP por imagen")
        ax.set_xlabel(et)
        ax.set_title(f"{NOMBRE_TEST[t]}: " + ("recall por fondo" if f == "fondo_medio" else f"FP por imagen por {et}"),
                     fontsize=8.5)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Redes v2 (dos semillas) frente a la red cerca actual; tolerancia 150 / 300 nm; franja rosa: z de 250 a 500 nm",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def figura_z(zz, ruta):
    plt = _estilo()
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for j, t in enumerate(TESTS):
        for r in REDES:
            s = zz[(zz.red == r) & (zz.test == t) & (zz.n_gt >= 10)]
            x = 0.5 * (s.desde + s.hasta)
            axs[0, j].plot(x, s.recall, "-", color=COLOR[r], lw=1.3, label=NOMBRE[r])
            axs[1, j].plot(x, s.razon_pred_gt, "-", color=COLOR[r], lw=1.3, label=NOMBRE[r])
        for ax in axs[:, j]:
            for v in (-2500, 2500):
                ax.axvline(v, color="#8a8984", lw=0.8, ls=":")
            ax.axvspan(250, 500, color="#e87ba4", alpha=0.12, lw=0)
            ax.grid(True, color="#e6e5e0", lw=0.6)
        axs[0, j].set_title(f"{NOMBRE_TEST[t]}: recall por z real (bins de 100 nm)", fontsize=8.5)
        axs[1, j].set_title(f"{NOMBRE_TEST[t]}: detecciones con z predicha en el bin / emisores con z real en el bin",
                            fontsize=8)
        axs[0, j].set_ylim(0, 1.02)
        axs[1, j].set_ylim(0, 2)
        axs[1, j].axhline(1, color="#8a8984", lw=0.8)
        axs[1, j].set_xlabel("z [nm]")
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Agujeros en z: un hueco en la densidad de z predicha (abajo) es una z que la red no asigna", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. Datos reales
# ----------------------------------------------------------------------------

def saltos(z, rng, bloque=10, n_boot=2000):
    """Saltos |dz| > SALTO_NM entre cuadros consecutivos con deteccion; IC de la fraccion por bootstrap de bloques."""
    zz = z[np.isfinite(z)]
    d = np.abs(np.diff(zz)) > SALTO_NM
    n = len(d)
    nb = int(np.ceil(n / bloque))
    bs = []
    for _ in range(n_boot):
        ini = rng.integers(0, n - bloque + 1, nb)
        bs.append(d[(ini[:, None] + np.arange(bloque)).ravel()[:n]].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n_pasos=n, n_saltos=int(d.sum()), frac_saltos=d.mean(), frac_saltos_lo=lo, frac_saltos_hi=hi,
                max_paso_nm=float(np.abs(np.diff(zz)).max()))


def reales(a, dispositivo, rng, salida):
    import analisis_datos_reales as adr
    import red_celula as rc
    from infer import localizar
    rc.verificar_triangulacion()
    px = adr.PIXEL_NM
    video, _ = adr.cargar(a.tif)
    T = len(video)
    c = adr.buscar_corte(video.mean(0))
    corte = c["corte"]
    vid = video[:, :, corte:].astype(np.float32)
    crudo = video[:, :, corte:]
    del video
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
    z_tri = {b: rc.z_triangulada(real, refl, q, m, np.radians(b), px) for b in BETAS}
    print(f"Yeast6_1 canal derecho: {T} cuadros, phi {phi:+.1f} deg, linea a {fr['v_borde']:+.1f} px del punto medio")
    for b, z in z_tri.items():
        print(f"  z triangulada beta {b:g}: media {np.nanmean(z):+.0f}, std {np.nanstd(z):.0f}, "
              f"max |paso| {np.nanmax(np.abs(np.diff(z))):.0f} nm")
    medio = np.nanmean((real + refl) / 2, 0)
    ventanas = [("medio", medio, 0.0), ("medio_linea-10px", medio, -10.0), ("medio_linea+10px", medio, 10.0),
                ("real", np.nanmean(real, 0), 0.0)]
    filas, resumen, porcuadro = [], [], []
    for red in ("s0", "s1"):
        modelo, cfg, cs = rc.cargar_red("cerca", dispositivo, CORRIDAS / REDES[red] / "mejor.pt")
        for nombre, cen, dlin in ventanas:
            x0, y0 = int(round(cen[0])) - rc.W // 2, int(round(cen[1])) - rc.H // 2
            assert ap[y0:y0 + rc.H, x0:x0 + rc.W].all(), "la ventana se sale de la apertura"
            ims = np.ascontiguousarray(crudo[:, y0:y0 + rc.H, x0:x0 + rc.W])
            qv = q + dlin * m - np.array([x0, y0])
            geo = rc.geometria(45.0, phi, qv[0], qv[1], T)
            det = localizar(modelo, ims, cfg, replace(cs, H=rc.H, W=rc.W), dispositivo, geometria=geo)
            rv, fv = real - np.array([x0, y0]), refl - np.array([x0, y0])
            z_red, conf = np.full(T, np.nan), np.full(T, np.nan)
            n_det, n_refl = np.zeros(T, int), np.zeros(T, int)
            for t, d in det.groupby("t"):
                dl = np.hypot(d.x_px - rv[t, 0], d.y_px - rv[t, 1]).to_numpy()
                n_det[t] = len(d)
                n_refl[t] = int((np.hypot(d.x_px - fv[t, 0], d.y_px - fv[t, 1]) <= rc.R_ASOC_REAL_PX).sum())
                if (dl <= rc.R_ASOC_REAL_PX).any():
                    j = np.argmin(dl)
                    z_red[t], conf[t] = d.z_nm.iat[j], d.conf.iat[j]
            sal = saltos(z_red, rng)
            ok = np.isfinite(z_red)
            signo = np.sign(np.nanmedian(z_red))
            base = dict(red=red, ventana=nombre, detectado=ok.mean(), otras_det_por_cuadro=(n_det - ok).mean(),
                        det_sobre_reflejo=n_refl.mean(), conf_mediana=np.nanmedian(conf), z_red_media=np.nanmean(z_red),
                        z_red_std=np.nanstd(z_red), frac_signo_dominante=np.mean(np.sign(z_red[ok]) == signo), **sal)
            for b in BETAS:
                r = rc.ajuste(z_tri[b], z_red, rng)
                fila = dict(base, beta_deg=b, z_tri_media=np.nanmean(z_tri[b]), **r)
                fila["pasa_saltos"] = sal["n_saltos"] == 0
                fila["pasa_correlacion"] = r["correlacion"] > 0.7
                fila["pasa_pendiente"] = 0.7 <= r["pendiente"] <= 1.3
                fila["ic_correlacion"] = ("dentro" if r["correlacion_lo"] > 0.7 else "fuera" if r["correlacion_hi"] <= 0.7
                                          else "cruza")
                fila["ic_pendiente"] = ("dentro" if 0.7 <= r["pendiente_lo"] and r["pendiente_hi"] <= 1.3 else
                                        "fuera" if r["pendiente_hi"] < 0.7 or r["pendiente_lo"] > 1.3 else "cruza")
                resumen.append(fila)
            r45 = [f for f in resumen if f["red"] == red and f["ventana"] == nombre and f["beta_deg"] == 45.0][0]
            print(f"  red {red}, ventana {nombre}: detectado {ok.mean():.2f}, otras det/cuadro {(n_det - ok).mean():.2f} "
                  f"(sobre el reflejo {n_refl.mean():.2f}), z red {np.nanmean(z_red):+.0f} +- {np.nanstd(z_red):.0f}; "
                  f"saltos {sal['n_saltos']}/{sal['n_pasos']} (max paso {sal['max_paso_nm']:.0f} nm); beta 45: corr "
                  f"{r45['correlacion']:+.3f} [{r45['correlacion_lo']:+.2f}, {r45['correlacion_hi']:+.2f}], pendiente "
                  f"{r45['pendiente']:+.2f} [{r45['pendiente_lo']:+.2f}, {r45['pendiente_hi']:+.2f}], desfase "
                  f"{r45['desfase_nm']:+.0f} nm")
            porcuadro.append(pd.DataFrame(dict(red=red, ventana=nombre, frame=np.arange(T), z_red_nm=z_red, conf=conf,
                                               n_det=n_det, **{f"z_tri_b{b:g}_nm": z_tri[b] for b in BETAS})))
            if nombre == "medio":
                filas.append((red, ims, det, rv, fv, qv))
    res = pd.DataFrame(resumen)
    res.to_csv(salida / "celula_v2_reales_resumen.csv", index=False)
    pc = pd.concat(porcuadro, ignore_index=True)
    pc.to_csv(salida / "celula_v2_reales.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        cols = ["red", "ventana", "beta_deg", "detectado", "otras_det_por_cuadro", "n_saltos", "correlacion", "correlacion_lo",
                "correlacion_hi", "pendiente", "pendiente_lo", "pendiente_hi", "desfase_nm", "pasa_saltos",
                "pasa_correlacion", "ic_correlacion", "pasa_pendiente", "ic_pendiente"]
        print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    figura_reales(pc, filas, phi, salida / "fig_celula_v2_reales.png")
    return res


def figura_reales(pc, imagenes, phi, ruta):
    plt = _estilo()
    fig = plt.figure(figsize=(15, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.7, 1, 1], wspace=0.35)
    red0, ims, det, rv, fv, qv = imagenes[0]
    ax = fig.add_subplot(gs[0])
    med = ims[:10].mean(0)
    ax.imshow(med, cmap="gray", vmin=np.percentile(med, 1), vmax=np.percentile(med, 99.7))
    ax.plot(rv[:, 0], rv[:, 1], "-", color="#1baf7a", lw=0.8)
    ax.plot(fv[:, 0], fv[:, 1], "-", color="#eda100", lw=0.8)
    for red, _, d, *_r in imagenes:
        d0 = d[d.t < 10]
        ax.scatter(d0.x_px, d0.y_px, marker="+", s=28, color=COLOR[red], lw=1, label=f"{NOMBRE[red]} (cuadros 0-9)")
    ax.set_xlim(-0.5, 95.5); ax.set_ylim(95.5, -0.5)
    ax.set_title("ventana de 96 px (media de los cuadros 0-9)\nverde: real; amarillo: reflejo", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.1))
    ax = fig.add_subplot(gs[1])
    s = pc[(pc.ventana == "medio") & (pc.red == red0)]
    ax.fill_between(s.frame, s["z_tri_b40_nm"], s["z_tri_b50_nm"], color="#1baf7a", alpha=0.18, lw=0,
                    label="triangulada, beta 40-50")
    ax.plot(s.frame, s["z_tri_b45_nm"], "-", color="#1baf7a", lw=1.8, label="triangulada, beta 45")
    for red in ("s0", "s1"):
        s = pc[(pc.ventana == "medio") & (pc.red == red)]
        ax.plot(s.frame, s.z_red_nm, "o-", color=COLOR[red], lw=0.9, ms=2.5, label=NOMBRE[red])
    ax.set_xlabel("cuadro (10 Hz)"); ax.set_ylabel("z del real [nm]")
    ax.grid(True, color="#e6e5e0", lw=0.6); ax.legend(frameon=False, fontsize=7)
    for k, red in enumerate(("s0", "s1")):
        ax = fig.add_subplot(gs[2 + k])
        s = pc[(pc.ventana == "medio") & (pc.red == red)]
        ok = s.z_red_nm.notna()
        x, y = s["z_tri_b45_nm"][ok], s.z_red_nm[ok]
        ax.scatter(x, y, s=9, color=COLOR[red])
        p = np.polyfit(x, y, 1)
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, np.polyval(p, xx), "-", color=COLOR[red], lw=1.5,
                label=f"pendiente {p[0]:+.2f}, r = {np.corrcoef(x, y)[0, 1]:+.2f}")
        ax.plot(xx, xx + np.mean(y - x), ":", color="#8a8984", lw=1, label="pendiente 1")
        ax.set_xlabel("z triangulada, beta 45 [nm]"); ax.set_ylabel(f"z {NOMBRE[red]} [nm]")
        ax.grid(True, color="#e6e5e0", lw=0.6); ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Yeast6_1, canal derecho: z de las redes v2 frente a la z triangulada (phi {phi:+.1f} deg, 98.48 nm/px)",
                 fontsize=10)
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("partes", nargs="*", default=["convergencia", "evaluacion"],
                   choices=("convergencia", "evaluacion", "reales"))
    p.add_argument("--tif", default="/home/decar/Datos Alex/Yeast6_1_MMStack_Pos0.ome.tif")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dispositivo", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--salida", default=str(FIG / "celula_v2"))
    a = p.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    if "convergencia" in a.partes:
        convergencia(salida)
    if "evaluacion" in a.partes:
        evaluacion(salida, rng)
    if "reales" in a.partes:
        sys.path.insert(0, str(RAIZ / "localizacion"))
        from utils import elegir_dispositivo
        reales(a, elegir_dispositivo(a.dispositivo), rng, salida)


if __name__ == "__main__":
    main()
