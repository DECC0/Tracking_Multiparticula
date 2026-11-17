"""
Simulador v2 (configs/celula_v2.yaml) frente a celula.yaml, y linea base de la
red cerca actual sobre el test v2. Solo lee los tests pregenerados y las
evaluaciones de evaluate.py; no corre la red.

Partes:
    datos      distribuciones de z, fondo, autofluorescencia, fotones y PSF de
               los dos tests (GT y muestras.csv), galeria de PSF de la pupila
               v2 y ejemplos de imagenes -> fig_simulador_v2.png
    linea_base metricas de la red sobre el test v1, el test v2 y el control v2
               con la PSF fija (misma semilla: mismas posiciones, fotones y
               fondo; solo cambia la pupila), por rango de z, de amp_esferica,
               de astigmatismo, de coma, de fotones y de fondo, y FP por imagen
               por amp_esferica y por contraste de la autofluorescencia
               -> simulador_v2_linea_base.csv, fig_simulador_v2_linea_base.png

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/simulador_v2.py [datos] [linea_base] [--corrida C] [--salida figuras/simulador_v2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "localizacion"))
from config import cargar_config  # noqa: E402
from data.simulacion import sim, config_simulador  # noqa: E402

DATOS = RAIZ / "localizacion" / "datos"
CORRIDA = RAIZ / "localizacion" / "corridas" / "hypatia_celula_b45_cerca_20261110-234627"
TESTS = {"v1": "test_celula_b45_cerca", "v2": "test_celula_v2_b45_cerca", "v2_psf_fija": "test_celula_v2_psf_fija_b45_cerca"}
COLOR = {"v1": "#2a78d6", "v2": "#eb6834", "v2_psf_fija": "#52514e"}
NOMBRE = {"v1": "test v1 (celula.yaml)", "v2": "test v2", "v2_psf_fija": "test v2 con la PSF fija (4.71 rad)"}
R_XY, R_Z = 150.0, 300.0
_V2 = cargar_config(RAIZ / "localizacion" / "configs" / "celula_v2.yaml").datos
BINS_PSF = {"amp_esferica": np.linspace(*_V2.psf_amp_esferica_rango, 6), "amp_astig": np.linspace(*_V2.psf_astig_rango, 6),
            "amp_coma": np.linspace(*_V2.psf_coma_rango, 7)}
BINS_CONTRASTE = [0, 0.25, 0.5, 1, 2, 4, 10]       # fondo_celula / fondo medio de la imagen


def leer_test(nombre):
    c = DATOS / TESTS[nombre]
    return pd.read_csv(c / "ground_truth.csv"), pd.read_csv(c / "muestras.csv"), np.load(c / "imagenes.npy", mmap_mode="r")


def densidad_z(gt, paso=250.0, lim=3000.0):
    """Densidad relativa de emisores evaluables por |z| (a la del primer bin)."""
    z = np.abs(gt.z_nm[gt.en_fov])
    h, b = np.histogram(z, bins=np.arange(0, lim + paso, paso))
    return b, h / max(h[0], 1)


def parte_datos(salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    tests = {k: leer_test(k) for k in ("v1", "v2")}
    print("== distribuciones (GT de los tests, emisores en el FOV) ==")
    for k, (gt, m, _) in tests.items():
        g = gt[gt.en_fov]
        b, d = densidad_z(gt)
        print(f"{k}: {len(m)} imagenes, {len(g)} emisores en el FOV; |z| > 2 um: {np.mean(np.abs(g.z_nm) > 2000):.3f}, "
              f"|z| > 2.5 um: {np.mean(np.abs(g.z_nm) > 2500):.3f}; fondo medio mediana {m.fondo_medio.median():.1f} "
              f"(p10 {m.fondo_medio.quantile(0.1):.1f}, p90 {m.fondo_medio.quantile(0.9):.1f}) e-/px; fotones mediana "
              f"{g.photons.median():.0f}, max {g.photons.max():.0f}")
        print("   densidad relativa por |z| (250 nm):", " ".join(f"{v:.2f}" for v in d))
    m2 = tests["v2"][1]
    print(f"v2: fondo_celula (amplitud en el centro) mediana {m2.fondo_celula.median():.1f}, p90 "
          f"{m2.fondo_celula.quantile(0.9):.1f} e-/px; ruido de lectura {m2.read_noise.min():.2f}-{m2.read_noise.max():.2f} ADU")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 6, hspace=0.45, wspace=0.45)
    ax = fig.add_subplot(gs[0, 0:2])
    for k, (gt, _, _) in tests.items():
        b, d = densidad_z(gt)
        ax.step(b[:-1], d, where="post", color=COLOR[k], lw=1.6, label=NOMBRE[k])
    ax.axvline(2500, color="#8a8984", lw=0.8, ls=":")
    ax.text(2520, 0.05, "rango de la red v1", fontsize=6.5, color="#8a8984")
    ax.set_xlabel("|z| [nm]"); ax.set_ylabel("emisores por 250 nm (relativo a |z| < 250)")
    ax.set_title("densidad de emisores en z", fontsize=9); ax.legend(frameon=False, fontsize=7)
    ax = fig.add_subplot(gs[0, 2:4])
    bins = np.logspace(0, np.log10(120), 40)
    for k, (_, m, _) in tests.items():
        ax.hist(m.fondo_medio, bins=bins, histtype="step", color=COLOR[k], lw=1.6, label=NOMBRE[k], density=True)
    ax.axvspan(1.3, 2.3, color="#8a8984", alpha=0.25, lw=0)
    ax.text(1.35, ax.get_ylim()[1] * 0.9, "Yeast6_1\nlejos", fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlabel("fondo medio de la imagen [e-/px]"); ax.set_ylabel("densidad")
    ax.set_title("fondo (con la autofluorescencia en v2)", fontsize=9); ax.legend(frameon=False, fontsize=7)
    ax = fig.add_subplot(gs[0, 4:6])
    bins = np.logspace(3, np.log10(5e4), 40)
    for k, (gt, _, _) in tests.items():
        ax.hist(gt.photons[gt.en_fov], bins=bins, histtype="step", color=COLOR[k], lw=1.6, label=NOMBRE[k], density=True)
    ax.set_xscale("log"); ax.set_xlabel("fotones por emisor"); ax.set_ylabel("densidad")
    ax.set_title("fotones (log-uniforme por subrango)", fontsize=9); ax.legend(frameon=False, fontsize=7)

    # galeria de PSF: amp 4.0 / 6.5, con y sin aberraciones maximas, a tres z
    cd = cargar_config(RAIZ / "localizacion" / "configs" / "celula_v2.yaml").datos
    cs = config_simulador(cd, 3)
    zs = (-2500.0, 0.0, 2500.0)
    casos = [("4.0 rad", dict(amp_esferica=4.0)), ("6.5 rad", dict(amp_esferica=6.5)),
             (f"4.0 + astig {_V2.psf_astig_rango[1]:.1f} + coma {_V2.psf_coma_rango[1]:.1f}",
              dict(amp_esferica=4.0, amp_astig=_V2.psf_astig_rango[1], amp_coma=_V2.psf_coma_rango[1], angulo_coma_deg=90.0)),
             ("v1: 4.71 rad", dict(amp_esferica=1.5 * np.pi))]
    from dataclasses import replace
    sub = gs[1, :].subgridspec(1, 12, wspace=0.08)
    k = 0
    for tit, kw in casos:
        c = replace(cs, **kw)
        for z in zs:
            p = sim.psf_directa(c, z)
            M = p.shape[0]
            r = p[M // 2 - 24:M // 2 + 24, M // 2 - 24:M // 2 + 24]
            a = fig.add_subplot(sub[0, k]); k += 1
            a.imshow(r ** 0.5, cmap="magma"); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{tit}\nz = {z / 1000:+.1f} um" if z == zs[0] else f"z = {z / 1000:+.1f} um", fontsize=6.5)
    # ejemplos de imagenes v2
    sub = gs[2, :].subgridspec(1, 6, wspace=0.1)
    gt2, m2, im2 = tests["v2"]
    elegidos = list(np.argsort(m2.fondo_medio.to_numpy())[[0, 400, 1000, 1600]]) + [int(m2.fondo_celula.idxmax())]
    gt1, m1, im1 = tests["v1"]
    for j, idx in enumerate([0] + elegidos):
        a = fig.add_subplot(sub[0, j])
        img = (im1 if j == 0 else im2)[idx, 0].astype(float)
        lo, hi = np.percentile(img, [1, 99.8])
        a.imshow(img, cmap="gray", vmin=lo, vmax=hi); a.set_xticks([]); a.set_yticks([])
        if j == 0:
            a.set_title(f"v1, muestra 0\nfondo {m1.fondo_medio[0]:.0f} e-/px\nesf 4.71 rad", fontsize=6)
        else:
            f = m2.loc[idx]
            a.set_title(f"v2, muestra {idx}\nfondo {f.fondo_medio:.1f}, celula {f.fondo_celula:.1f} e-/px\n"
                        f"esf {f.amp_esferica:.1f}, astig {f.amp_astig:.1f}, coma {f.amp_coma:.1f} rad", fontsize=6)
        g = (gt1 if j == 0 else gt2)
        g = g[(g.video_id == idx) & g.en_fov]
        a.plot(g.x_px, g.y_px, "o", mfc="none", mec="#2ad6c4", ms=5, mew=0.7)
    fig.suptitle("Simulador v2: z, fondo y fotones de los tests (2000 imagenes cada uno), PSF de la pupila v2 "
                 "(raiz de la intensidad, 48 x 48 px) y ejemplos (circulos: reales en el FOV)", fontsize=9)
    fig.savefig(salida / "fig_simulador_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def metricas(g, n_fp=None):
    """recall, RMSE lateral y z y sesgo z de los emisores evaluables de g (gt_eval)."""
    d = g[g.detectado]
    out = dict(n=len(g), recall=len(d) / max(len(g), 1),
               rmse_lat_nm=float(np.sqrt(np.mean(d.dx_nm ** 2 + d.dy_nm ** 2))) if len(d) else np.nan,
               rmse_z_nm=float(np.sqrt(np.mean(d.dz_nm ** 2))) if len(d) else np.nan,
               sesgo_z_nm=float(d.dz_nm.mean()) if len(d) else np.nan)
    if n_fp is not None:
        out["fp"] = n_fp
        out["precision"] = len(d) / max(len(d) + n_fp, 1)
    return out


def cargar_eval(corrida, nombre):
    e = corrida / f"evaluacion_{TESTS[nombre]}"
    if not e.exists():
        return None
    gt = pd.read_csv(e / "gt_eval.csv")
    pred = pd.read_csv(e / "predicciones.csv")
    m = pd.read_csv(DATOS / TESTS[nombre] / "muestras.csv")
    gt = gt[gt.en_fov & gt.en_rango_z].merge(m, on="video_id", how="left", suffixes=("", "_m"))
    return gt, pred, m


def parte_linea_base(corrida, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    ev = {k: cargar_eval(corrida, k) for k in TESTS}
    ev = {k: v for k, v in ev.items() if v is not None}
    filas = []
    for k, (gt, pred, m) in ev.items():
        fp = int((~pred.tp & ~pred.fuera_fov).sum()) if "fuera_fov" in pred else int((~pred.tp).sum())
        filas.append(dict(test=k, factor="todo", desde=np.nan, hasta=np.nan, **metricas(gt, fp)))
        filas.append(dict(test=k, factor="|z| <= 2500", desde=0, hasta=2500, **metricas(gt[gt.z_nm.abs() <= 2500])))
        filas.append(dict(test=k, factor="|z| > 2500", desde=2500, hasta=3000, **metricas(gt[gt.z_nm.abs() > 2500])))
        cortes = {"z_nm": np.arange(-3000, 3001, 500), "photons": [1000, 2000, 5000, 10000, 20000, 30000, 50000],
                  "fondo_medio": [0, 3, 6, 12, 25, 50, 200]}
        if "amp_esferica" in gt:
            cortes.update(BINS_PSF)
        if "fondo_celula" in gt:
            gt = gt.assign(contraste_celula=gt.fondo_celula / np.maximum(gt.fondo_medio, 1e-9))
        for f, b in cortes.items():
            for lo, hi in zip(b[:-1], b[1:]):
                s = gt[(gt[f] >= lo) & (gt[f] < hi)]
                if len(s) >= 20:
                    filas.append(dict(test=k, factor=f, desde=lo, hasta=hi, **metricas(s)))
    # FP por imagen segun factores de la imagen (muestras.csv del test v2: el control comparte los
    # video_id, el fondo y la autofluorescencia, y la PSF del v2 se usa para agrupar)
    if "v2" in ev:
        mv2 = ev["v2"][2].assign(contraste_celula=lambda d: d.fondo_celula / np.maximum(d.fondo_medio, 1e-9))
        for k, (gt, pred, m) in ev.items():
            if k == "v1":
                continue
            fp = pred[~pred.tp & ~pred.fuera_fov].groupby("video_id").size()
            mm = mv2.assign(fp=mv2.video_id.map(fp).fillna(0))
            for f, b in ({"amp_esferica": BINS_PSF["amp_esferica"], "contraste_celula": BINS_CONTRASTE}).items():
                for lo, hi in zip(b[:-1], b[1:]):
                    s = mm[(mm[f] >= lo) & (mm[f] < hi)]
                    if len(s):
                        filas.append(dict(test=k, factor=f"fp_por_imagen:{f}", desde=lo, hasta=hi, n=len(s),
                                          fp=float(s.fp.mean())))
    t = pd.DataFrame(filas)
    t.to_csv(salida / "simulador_v2_linea_base.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    # pareado: v2 frente a v2_psf_fija, mismos emisores (misma semilla)
    if "v2" in ev and "v2_psf_fija" in ev:
        a, b = ev["v2"][0], ev["v2_psf_fija"][0]
        cl = ["video_id", "particle_id"]
        p = a[cl + ["detectado", "dz_nm", "z_nm", "amp_esferica", "amp_astig", "amp_coma"]].merge(
            b[cl + ["detectado", "dz_nm", "z_nm"]], on=cl, suffixes=("", "_fija"))
        assert np.allclose(p.z_nm, p.z_nm_fija)
        ambos = p[p.detectado & p.detectado_fija]
        print(f"\npareado v2 / v2 con PSF fija ({len(p)} emisores): recall {p.detectado.mean():.3f} / "
              f"{p.detectado_fija.mean():.3f}; detectados solo con la PSF fija {np.mean(~p.detectado & p.detectado_fija):.3f}, "
              f"solo con la v2 {np.mean(p.detectado & ~p.detectado_fija):.3f}; RMSE z de los detectados por las dos "
              f"{np.sqrt(np.mean(ambos.dz_nm ** 2)):.1f} / {np.sqrt(np.mean(ambos.dz_nm_fija ** 2)):.1f} nm ({len(ambos)})")
        for f, b in BINS_PSF.items():
            for lo, hi in zip(b[:-1], b[1:]):
                s = p[(p[f] >= lo) & (p[f] < hi)]
                sa = s[s.detectado & s.detectado_fija]
                print(f"   {f} [{lo:.2f}, {hi:.2f}): n {len(s)}, recall {s.detectado.mean():.3f} / {s.detectado_fija.mean():.3f}, "
                      f"RMSE z (ambos) {np.sqrt(np.mean(sa.dz_nm ** 2)):.0f} / {np.sqrt(np.mean(sa.dz_nm_fija ** 2)):.0f} nm")

    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    paneles = [("z_nm", "z [nm]"), ("photons", "fotones"), ("fondo_medio", "fondo medio [e-/px]"),
               ("amp_esferica", "amp_esferica [rad]"), ("amp_astig", "astigmatismo [rad]"), ("amp_coma", "coma [rad]")]
    for ax, (f, et) in zip(axs.flat, paneles):
        for k in ev:
            s = t[(t.test == k) & (t.factor == f)]
            if not len(s):
                continue
            x = 0.5 * (s.desde + s.hasta)
            ax.plot(x, s.recall, "o-", color=COLOR[k], lw=1.6, ms=4, label=f"{NOMBRE[k]}: recall")
            ax.plot(x, s.rmse_z_nm / 1000, "s--", color=COLOR[k], lw=1.2, ms=3.5, label=f"{NOMBRE[k]}: RMSE z [um]")
        if f in ("photons", "fondo_medio"):
            ax.set_xscale("log")
        if f == "z_nm":
            for v in (-2500, 2500):
                ax.axvline(v, color="#8a8984", lw=0.8, ls=":")
        ax.set_xlabel(et); ax.set_ylim(0, 1.02); ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_ylabel("recall (circulos) / RMSE z en um (cuadrados)")
    axs[0, 0].legend(frameon=False, fontsize=6, loc="lower center")
    fig.suptitle(f"Linea base: red cerca actual ({corrida.name}) sobre los tests v1, v2 y v2 con la PSF fija "
                 f"(tolerancia {R_XY:.0f} / {R_Z:.0f} nm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(salida / "fig_simulador_v2_linea_base.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("partes", nargs="*", default=["datos", "linea_base"])
    p.add_argument("--corrida", default=str(CORRIDA))
    p.add_argument("--salida", default=str(Path(__file__).resolve().parent / "simulador_v2"))
    a = p.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    if "datos" in a.partes:
        parte_datos(salida)
    if "linea_base" in a.partes:
        parte_linea_base(Path(a.corrida), salida)


if __name__ == "__main__":
    main()
