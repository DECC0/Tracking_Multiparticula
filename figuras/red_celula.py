"""
Las redes con celula (hypatia_celula_b45_cerca y _lejos) sobre entradas que
no son su test: corre la red (GPU si hay).

fisher
    sqrt(1/I_zz) de un emisor solo contra z para las dos variantes (sin red): la
    PSF no tiene degeneraciones en z que expliquen los agujeros de la red.
transferencia
    Curva z predicha vs z real de emisores aislados: un emisor por imagen de
    96 x 96, a distancia d = 2 um de un espejo a 45 grados (phi = 0, linea en
    x = 35 px), fondo uniforme de 20 fotones/px, N = 3000 y 10000 fotones, z en
    [-2500, 2500] cada 50 nm, con el reflejo (ratio 0.7) y sin el (ratio 0).
    Real y reflejo con el recorte de pupila (PSFRecortada) y la PSF de la
    variante de cada red. Separa lo que hace la red con un emisor solo de lo
    que pasa en el test (vecinos, celula, fondo no uniforme).
reales
    Yeast6_1 (adquisicion anterior con espejo, Micro-Manager), canal derecho
    (columnas desde el corte de figuras/analisis_datos_reales.py, cuya
    deteccion, pistas, pares y franja reutiliza), 98.48 nm/px. La red corre
    cuadro a cuadro sobre una ventana fija de 96 x 96 centrada en el punto
    medio medio del par real/reflejo, con el canal de distancia al espejo de
    la linea estimada de los datos: orientacion = direccion media reflejo ->
    real, posicion = borde nitido de la franja del espejo (supuesto: es la
    interseccion espejo/plano focal), beta = 45 grados. La z de la red (la
    deteccion a <= 3 px del real) se compara con la triangulada de las
    posiciones del real y del reflejo con esa geometria:
        z_real = -s_M / tan(beta) + (L / 2) tan(beta)
    con s_M la coordenada a lo largo de m del punto medio (desde la linea) y L
    la separacion a lo largo de m; a 45 grados z_real = -s_reflejo. La
    correlacion y la pendiente no dependen de la posicion de la linea; el
    desfase si (1 um de la linea = 1 um de desfase a 45 grados). Tambien
    corre la red con la linea corrida +-10 px y con la ventana centrada en el
    real (sensibilidad), y separa los dos modos de la red (z de signo opuesto).
    Sin la red: perfil radial medio (cuadros 0-29, alineados) del real y del
    reflejo contra el de PSFRecortada de cada variante, con z' - z = -L
    tan(beta), beta en [40, 50] (chi2 con amplitud y fondo libres), y la
    fraccion de la senal en el anillo de 6-18 px (por cuadro) del real y del
    reflejo contra la simulada de cada variante (el reflejo esta siempre mas
    lejos del objetivo). La forma es simetrica entre variantes bajo z -> -z, asi
    que cada variante tiene una rama que la reproduce: lo que las distingue es
    la z absoluta que implican (comparar con la triangulada).

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    LD_PRELOAD=$HOME/miniforge3/envs/tesis/lib/libstdc++.so.6 $PY figuras/red_celula.py transferencia [--n-rep 8]
    LD_PRELOAD=... $PY figuras/red_celula.py reales [--tif RUTA] [--beta-deg 45]
Escribe en --salida (por defecto figuras/): red_celula_fisher_z.csv; red_celula_transferencia.csv,
fig_red_celula_transferencia.png/.pdf; red_celula_reales.csv (por cuadro),
red_celula_reales_resumen.csv, red_celula_reales_perfiles.csv, fig_red_celula_reales.png/.pdf y
fig_red_celula_reales_psf.png/.pdf. reales usa figuras/analisis_datos_reales.py.
"""
import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "localizacion"))
sys.path.insert(1, str(Path(__file__).resolve().parent))
from utils import cargar_modelo, elegir_dispositivo  # noqa: E402
from infer import localizar  # noqa: E402
from data.simulacion import sim, config_simulador, config_geometria, COLUMNAS_GEOMETRIA  # noqa: E402

CORRIDAS = RAIZ / "localizacion" / "corridas"
DATOS = RAIZ / "localizacion" / "datos"
CORRIDA = {"cerca": "hypatia_celula_b45_cerca_20261110-234627", "lejos": "hypatia_celula_b45_lejos_20261110-234617"}
COLOR = {"cerca": "#2a78d6", "lejos": "#eb6834", "tri": "#1baf7a", "gris": "#52514e"}
H = W = 96
X_LINEA, Y_EMISOR = 35.0, 48.0
D_NM, FONDO, RATIO = 2000.0, 20.0, 0.7
N_FOT = (3000.0, 10000.0)
Z_BARRIDO = np.arange(-2500.0, 2500.1, 50.0)
R_ASOC_NM = 150.0
R_ASOC_REAL_PX = 3.0
TIF_DEF = "/home/decar/Datos Alex/Yeast6_1_MMStack_Pos0.ome.tif"


def cargar_red(var, dispositivo, checkpoint=None):
    modelo, cfg, _ = cargar_modelo(checkpoint or CORRIDAS / CORRIDA[var] / "mejor.pt", dispositivo)
    cs = config_simulador(cfg.datos, cfg.datos.seed_train)
    return modelo, cfg, cs


def k_psf(var):
    """K_psf del test de la variante (el auto del simulador, sin construir el banco)."""
    return int(json.loads((DATOS / f"test_celula_b45_{var}" / "config_simulador.json").read_text())["K_psf_usado"])


def geometria(beta, phi, x0, y0, n=1):
    return pd.DataFrame([[beta, phi, x0, y0]] * n, columns=COLUMNAS_GEOMETRIA)


# ----------------------------------------------------------------------------
# Curva de transferencia
# ----------------------------------------------------------------------------

def condiciones_base():
    """Las de la curva de transferencia: N en N_FOT, con reflejo (RATIO) y sin el."""
    return [dict(nombre="base", d_nm=D_NM, fondo=FONDO, fotones=N, ratio=r, beta=45.0) for N in N_FOT for r in (RATIO, 0.0)]


def transferencia(var, n_rep, dispositivo, rng, checkpoint=None, zs=Z_BARRIDO, condiciones=None):
    """Un emisor aislado por imagen para cada condicion (d_nm, fondo, fotones, ratio,
    beta), z en zs y n_rep repeticiones; la red y su deteccion mas cercana (<= 150 nm)."""
    modelo, cfg, cs = cargar_red(var, dispositivo, checkpoint)
    psfs, filas, imgs, geos = {}, [], [], []
    for c in condiciones or condiciones_base():
        beta = c["beta"]
        geo = geometria(beta, 0.0, X_LINEA, Y_EMISOR)
        cg = config_geometria(cs, geo.iloc[0])
        assert cg.recorte_pupila_real and cg.recorte_pupila_reflejo
        if beta not in psfs:
            psfs[beta] = sim.PSFRecortada(cg, k_psf(var))
            assert not psfs[beta].completa
        psf, esp, b = psfs[beta], sim.Espejo(cg), np.radians(beta)
        for z in zs:
            for r in range(n_rep):
                x = X_LINEA + (c["d_nm"] - np.sin(b) * z) / np.cos(b) / cg.pixel_nm + rng.uniform(-0.5, 0.5)
                y = Y_EMISOR + rng.uniform(-0.5, 0.5)
                p = np.array([x, y, z])
                xr, yr, zr = esp.reflejar(p)
                assert abs(esp.distancia_nm(esp.a_nm(p)) - c["d_nm"]) < 40.0      # +-0.5 px de jitter en x
                img = np.full((H, W), float(c["fondo"]))
                psf.colocar(img, x, y, z, c["fotones"])
                psf.colocar(img, xr, yr, zr, c["ratio"] * c["fotones"])
                imgs.append(sim.aplicar_camara(img, cg, rng))
                geos.append(geo)
                filas.append(dict(variante=var, condicion=c["nombre"], d_nm=c["d_nm"], fondo=c["fondo"], beta=beta,
                                  fotones=c["fotones"], ratio=c["ratio"], z_nm=z, rep=r, x_px=x, y_px=y,
                                  x_reflejo_px=xr, y_reflejo_px=yr, z_reflejo_nm=zr))
    t = pd.DataFrame(filas)
    det = localizar(modelo, np.stack(imgs), cfg, cg, dispositivo, geometria=pd.concat(geos, ignore_index=True))
    px = cg.pixel_nm
    pred = np.full((len(t), 4), np.nan)                          # x, y, z, conf de la deteccion del emisor
    n_det, sobre_reflejo = np.zeros(len(t), int), np.zeros(len(t), int)
    for i, d in det.groupby("t"):
        dl = np.hypot(d.x_px - t.x_px.iat[i], d.y_px - t.y_px.iat[i]) * px
        dr = np.hypot(d.x_px - t.x_reflejo_px.iat[i], d.y_px - t.y_reflejo_px.iat[i]) * px
        n_det[i] = len(d)
        sobre_reflejo[i] = int((dr <= R_ASOC_NM).sum())
        if (dl <= R_ASOC_NM).any():
            j = np.argmin(dl.to_numpy())
            pred[i] = d.x_px.iat[j], d.y_px.iat[j], d.z_nm.iat[j], d.conf.iat[j]
    return t.assign(x_pred_px=pred[:, 0], y_pred_px=pred[:, 1], z_pred_nm=pred[:, 2], conf=pred[:, 3], n_det=n_det,
                    det_sobre_reflejo=sobre_reflejo)


Z_AGUJERO = np.arange(0.0, 800.1, 10.0)
BASE_AGUJERO = dict(nombre="base", d_nm=D_NM, fondo=FONDO, fotones=10000.0, ratio=RATIO, beta=45.0)
VARIACIONES_AGUJERO = [("d_nm", 500.0), ("d_nm", 1000.0), ("d_nm", 3500.0), ("fondo", 5.0), ("fondo", 40.0),
                       ("fotones", 3000.0), ("fotones", 30000.0), ("ratio", 0.0), ("beta", 40.0), ("beta", 50.0)]


def bordes_agujero(s, imgs):
    """De una condicion: intervalo de z real donde la deteccion baja de 0.9 (y su minimo),
    z predicha mediana justo antes y justo despues de ese intervalo, y el mayor hueco entre
    z predichas consecutivas (de todas las imagenes, z predicha en [100, 800]): las z que
    la red no devuelve nunca."""
    s = s.sort_values("z_nm")
    bajo = s.z_nm[s.detectado < 0.9]
    out = dict(z_det_bajo_desde=bajo.min() if len(bajo) else np.nan, z_det_bajo_hasta=bajo.max() if len(bajo) else np.nan,
               det_min=s.detectado.min(), z_det_min=s.z_nm.iat[int(np.argmin(s.detectado.to_numpy()))])
    if len(bajo):
        antes, despues = s[s.z_nm < bajo.min()], s[s.z_nm > bajo.max()]
        out.update(z_pred_antes=antes.z_pred_mediana.iat[-1] if len(antes) else np.nan,
                   z_pred_despues=despues.z_pred_mediana.iat[0] if len(despues) else np.nan)
    zp = np.sort(imgs.z_pred_nm.dropna().to_numpy())
    zp = zp[(zp >= 100) & (zp <= 800)]
    if len(zp) > 1:
        k = int(np.argmax(np.diff(zp)))
        out.update(hueco_pred_desde=zp[k], hueco_pred_hasta=zp[k + 1], hueco_pred_nm=zp[k + 1] - zp[k])
    return out


def agujero(a, dispositivo, rng, salida):
    """Agujero de +400 nm de cerca: curva de transferencia fina (z en [0, 800] cada 10 nm)
    en la condicion base y variando una cosa a la vez (d, fondo, fotones, ratio, beta)."""
    conds = [BASE_AGUJERO] + [dict(BASE_AGUJERO, **{k: v}, nombre=f"{k}={v:g}") for k, v in VARIACIONES_AGUJERO]
    ck = dict(a.checkpoint or []).get("cerca")
    previo = salida / "red_celula_agujero_imagenes.csv"
    if a.reusar and previo.exists():
        t = pd.read_csv(previo)
    else:
        t = transferencia("cerca", a.n_rep, dispositivo, rng, ck, Z_AGUJERO, conds)
        t.to_csv(previo, index=False, float_format="%.6g")
    r = resumen_transferencia(t)
    r.to_csv(salida / "red_celula_agujero.csv", index=False)
    filas = [dict(condicion=c, **bordes_agujero(s, t[t.condicion == c])) for c, s in r.groupby("condicion", sort=False)]
    b = pd.DataFrame(filas)
    b.to_csv(salida / "red_celula_agujero_bordes.csv", index=False)
    print(f"checkpoint: {ck or 'mejor.pt'}; {int(t.groupby(['condicion', 'z_nm']).size().max())} imagenes por (condicion, z)")
    with pd.option_context("display.width", 200):
        print(b.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    figura_agujero(r, salida / "fig_red_celula_agujero")


def figura_agujero(r, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    grupos = [("base", ["base"]), ("d al espejo", ["d_nm=500", "d_nm=1000", "d_nm=3500"]),
              ("fondo", ["fondo=5", "fondo=40"]), ("fotones", ["fotones=3000", "fotones=30000"]),
              ("ratio y beta", ["ratio=0", "beta=40", "beta=50"])]
    paleta = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(2, len(grupos), figsize=(3.6 * len(grupos), 6.2), height_ratios=[2, 1], sharex=True)
    base = r[r.condicion == "base"]
    for j, (titulo, conds) in enumerate(grupos):
        ax, ax2 = axes[0, j], axes[1, j]
        if titulo != "base":
            ax.plot(base.z_nm, base.z_pred_mediana, "-", color="#8a8984", lw=1.2, label="base")
            ax2.plot(base.z_nm, base.detectado, "-", color="#8a8984", lw=1.2)
        for k, c in enumerate(conds):
            s = r[r.condicion == c]
            ax.plot(s.z_nm, s.z_pred_mediana, "-", color=paleta[k], lw=1.6, label=c)
            ax2.plot(s.z_nm, s.detectado, "-", color=paleta[k], lw=1.6)
        ax.plot([0, 800], [0, 800], ":", color="#52514e", lw=0.8)
        for v in (300, 600):
            ax.axhline(v, color="#e6e5e0", lw=0.8)
        ax.set_title(titulo, fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        ax2.set_ylim(0, 1.03)
        ax2.set_xlabel("z real [nm]")
        for x in (ax, ax2):
            x.grid(True, color="#e6e5e0", lw=0.6)
    axes[0, 0].set_ylabel("z predicha [nm] (mediana)")
    axes[1, 0].set_ylabel("detectado")
    fig.suptitle("Agujero de +400 nm de la red cerca: emisor aislado (base: d = 2 um, fondo 20, N = 1e4, ratio 0.7, "
                 "beta 45), una variacion por panel", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(ruta.with_suffix(".png"), dpi=300)
    plt.close(fig)


def fisher_z(salida):
    """sqrt(1/I_zz) [nm] (solo z desconocido) y pico de la PSF de un emisor con el
    recorte de pupila del real a 45 grados, N = 1e4, fondo 20, por variante y z: si la
    PSF tuviera una degeneracion en z (forma estacionaria) aqui se veria un pico."""
    filas = []
    for var, an in (("cerca", True), ("lejos", False)):
        cfg = sim.Config(anillos_hacia_objetivo=an, usar_espejo=True, angulo_espejo_eje_deg=45.0,
                         recorte_pupila_real=True, recorte_pupila_reflejo=True)
        psf, v = sim.PSFRecortada(cfg, 128), sim._var_lectura_e2(cfg)
        N = 1e4
        for z in Z_BARRIDO:
            g, dg = psf.psf(z), (psf.psf(z + 5.0) - psf.psf(z - 5.0)) / 10.0
            filas.append(dict(variante=var, z_nm=z, sqrt_inv_Izz_nm=1 / np.sqrt(((N * dg) ** 2 / (N * g + FONDO + v)).sum()),
                              pico=g.max()))
    f = pd.DataFrame(filas)
    f.to_csv(salida / "red_celula_fisher_z.csv", index=False)
    p = f.pivot(index="z_nm", columns="variante", values="sqrt_inv_Izz_nm")
    dif = np.abs(p.cerca.to_numpy() / p.lejos.to_numpy()[::-1] - 1).max()
    print(f"max |cerca(z) / lejos(-z) - 1| = {dif:.1e}")
    print("sqrt(1/I_zz) [nm], N = 1e4, fondo 20:")
    print(p[p.index % 250 == 0].T.to_string(float_format=lambda x: f"{x:.1f}"))


def resumen_transferencia(t):
    def agg(s):
        ok = s.z_pred_nm.notna()
        dz = (s.z_pred_nm - s.z_real)[ok]
        q = np.nanpercentile(s.z_pred_nm, [16, 50, 84]) if ok.any() else [np.nan] * 3
        return pd.Series(dict(detectado=ok.mean(), z_pred_p16=q[0], z_pred_mediana=q[1], z_pred_p84=q[2],
                              rmse_z=np.sqrt(np.mean(dz ** 2)) if ok.any() else np.nan,
                              fp_por_imagen=(s.n_det - ok).mean(), fp_sobre_reflejo=s.det_sobre_reflejo.mean()))
    claves = [c for c in ("variante", "condicion", "fotones", "ratio", "z_nm") if c in t]
    return (t.assign(z_real=t.z_nm).groupby(claves)
            .apply(agg, include_groups=False).reset_index())


def figura_transferencia(r, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    vars_ = list(dict.fromkeys(r.variante))
    fig, axes = plt.subplots(2, 2 * len(vars_), figsize=(4.2 * 2 * len(vars_), 7.2), height_ratios=[2, 1])
    for k, var in enumerate(vars_):
        for j, N in enumerate(N_FOT):
            ax, ax2 = axes[0, 2 * k + j], axes[1, 2 * k + j]
            for ratio, color, etiq in ((RATIO, COLOR[var], f"con reflejo ({RATIO})"), (0.0, COLOR["gris"], "sin reflejo")):
                s = r[(r.variante == var) & (r.fotones == N) & (r.ratio == ratio)]
                ax.plot(s.z_nm, s.z_pred_mediana, "-", color=color, lw=1.8, label=etiq)
                ax.fill_between(s.z_nm, s.z_pred_p16, s.z_pred_p84, color=color, alpha=0.2, lw=0)
                ax2.plot(s.z_nm, s.detectado, "-", color=color, lw=1.8)
            ax.plot([-2500, 2500], [-2500, 2500], ":", color="#8a8984", lw=1)
            ax.set_title(f"red {var}, N = {N:.0f}", fontsize=10)
            ax.set_ylabel("z predicha [nm] (mediana, p16-p84)")
            ax2.set_ylabel("detectado (<= 150 nm lateral)")
            ax2.set_xlabel("z real [nm]")
            ax2.set_ylim(0, 1.03)
            for a in (ax, ax2):
                a.grid(True, color="#e6e5e0", lw=0.6)
                a.set_xlim(-2550, 2550)
            ax.set_ylim(-2600, 2600)
            if k == 0 and j == 0:
                ax.legend(frameon=False, loc="upper left")
    fig.suptitle(f"Emisor aislado: d = {D_NM / 1000:g} um de un espejo a 45 grados, fondo {FONDO:g} fotones/px, "
                 "recorte de pupila del real y del reflejo, PSF de la variante de cada red", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta.with_suffix(".png"), dpi=200)
    fig.savefig(ruta.with_suffix(".pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Datos reales
# ----------------------------------------------------------------------------

def z_triangulada(real, refl, q, m, beta, px):
    """z [nm] del real por cuadro: real, refl (T, 2) [px], q punto de la linea en el plano focal [px], m (2,)."""
    s_r, s_f = (real - q) @ m * px, (refl - q) @ m * px
    s_M, L = (s_r + s_f) / 2, s_r - s_f
    return -s_M / np.tan(beta) + L / 2 * np.tan(beta)


def verificar_triangulacion():
    """La formula de z_triangulada contra GeometriaEspejo (reflejo del simulador)."""
    from geometria_espejo import GeometriaEspejo
    for beta, phi in ((45.0, 30.0), (38.0, -120.0)):
        geo = GeometriaEspejo(beta, phi, [1000.0, -500.0, 0.0], -2500.0, 2500.0)
        for z in (-1500.0, 0.0, 900.0):
            p = geo.p0 + 2500.0 * geo.m + 300.0 * geo.t + np.array([0.0, 0.0, z])
            pr = geo.reflejar_nm(p)
            zt = z_triangulada(p[None, :2], pr[None, :2], geo.p0[:2], geo.m[:2] / np.hypot(*geo.m[:2]),
                               np.radians(beta), 1.0)[0]
            assert abs(zt - z) < 1e-6, (beta, z, zt)


def ajuste(x, y, rng, bloque=10, n_boot=2000):
    """Correlacion, pendiente OLS (y ~ x), ordenada y desfase medio (y - x), con IC 95% por bootstrap de bloques."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)

    def est(xx, yy):
        c = np.corrcoef(xx, yy)[0, 1]
        pend, ord_ = np.polyfit(xx, yy, 1)
        return np.array([c, pend, ord_, np.mean(yy - xx), np.std(yy - xx)])
    e = est(x, y)
    nb = int(np.ceil(n / bloque))
    bs = []
    for _ in range(n_boot):
        ini = rng.integers(0, n - bloque + 1, nb)
        idx = (ini[:, None] + np.arange(bloque)).ravel()[:n]
        bs.append(est(x[idx], y[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    claves = ["correlacion", "pendiente", "ordenada_nm", "desfase_nm", "std_dif_nm"]
    return {**{k: v for k, v in zip(claves, e)}, **{k + "_lo": v for k, v in zip(claves, lo)},
            **{k + "_hi": v for k, v in zip(claves, hi)}, "n": n}


R_PERFIL = 25                           # px del perfil radial
Z_PERFIL = np.arange(-8000.0, 8001.0, 100.0)
CUADROS_PERFIL = 30                     # cuadros promediados (antes del fotoblanqueo fuerte)


def perfil_radial(img, x, y, rmax=R_PERFIL):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    k = np.floor(np.hypot(xx - x, yy - y)).astype(int)
    return np.array([img[k == i].mean() for i in range(rmax)])


def fraccion_anillo(perfil, fondo=None, r_in=6, r_out=18, r_fondo=(20, 25)):
    """Fraccion de la senal (perfil radial menos el fondo, ponderado por el area del
    anillo de 1 px) en r_in <= r < r_out respecto de r < r_out; el fondo es la mediana
    del perfil en r_fondo (o el dado). Mide cuanto pesan los anillos frente al nucleo."""
    r = np.arange(len(perfil)) + 0.5
    b = np.median(perfil[r_fondo[0]:r_fondo[1]]) if fondo is None else fondo
    s = (np.asarray(perfil) - b) * 2 * np.pi * r
    return float(s[r_in:r_out].sum() / s[:r_out].sum())


def media_alineada(vid, pos, r=30):
    """Media de los primeros CUADROS_PERFIL cuadros recortados (2r+1) alrededor de pos[t] y alineados (subpixel, lineal)."""
    from scipy import ndimage as ndi
    acc = []
    for t in range(CUADROS_PERFIL):
        x, y = pos[t]
        ix, iy = int(round(x)), int(round(y))
        c = vid[t, iy - r - 2: iy + r + 3, ix - r - 2: ix + r + 3].astype(float)
        acc.append(ndi.shift(c, (iy - y, ix - x), order=1)[2:-2, 2:-2])
    return np.mean(acc, 0)


def ajuste_perfiles(vid, real, refl, phi, L_nm, betas=(40.0, 50.0)):
    """Perfil radial medio del real y del reflejo contra el de PSFRecortada de cada
    variante (geometria de los datos), con amplitud y fondo libres: chi2(z) de cada
    uno y el minimo de chi2_real(z) + chi2_reflejo(z') con z' - z = -L tan(beta),
    beta en betas (el reflejo esta siempre mas lejos del objetivo)."""
    img_r, img_f = media_alineada(vid, real), media_alineada(vid, refl)
    c = img_r.shape[0] // 2
    datos = {"real": perfil_radial(img_r, c, c), "reflejo": perfil_radial(img_f, c, c)}
    # fraccion en el anillo, por cuadro (perfil de 26 px alrededor del centroide de cada cuadro)
    frac = {n: np.array([fraccion_anillo(perfil_radial(vid[t], *pos[t], rmax=26)) for t in range(len(vid))])
            for n, pos in (("real", real), ("reflejo", refl))}
    out, modelos = [], {}
    for var, an in (("cerca", True), ("lejos", False)):
        cfg = sim.Config(anillos_hacia_objetivo=an, usar_espejo=True, angulo_espejo_eje_deg=45.0,
                         orientacion_espejo_deg=phi, recorte_pupila_real=True, recorte_pupila_reflejo=True)
        psf = sim.PSFRecortada(cfg, 128)
        imgs = np.array([psf.psf(z) for z in Z_PERFIL])
        perf = np.array([perfil_radial(p, 64, 64) for p in imgs])
        frac_sim = np.array([fraccion_anillo(perfil_radial(p, 64, 64, rmax=26), fondo=0.0) for p in imgs])
        # mejor par (z, z' = z - L tan beta) para las fracciones medianas de los datos
        fr, ff = np.median(frac["real"]), np.median(frac["reflejo"])
        mejor_f = (np.inf, np.nan, np.nan)
        for bb in np.arange(min(betas), max(betas) + 0.1, 0.5):
            zp = Z_PERFIL - L_nm * np.tan(np.radians(bb))
            f_zp = np.interp(zp, Z_PERFIL, frac_sim, left=np.nan)
            e = (frac_sim - fr) ** 2 + (f_zp - ff) ** 2
            if np.nanmin(e) < mejor_f[0]:
                k = int(np.nanargmin(e))
                mejor_f = (float(e[k]), Z_PERFIL[k], bb)
        chi, ajus = {}, {}
        for nombre, y in datos.items():
            A = np.stack([np.c_[p, np.ones_like(p)] for p in perf])
            coef = np.array([np.linalg.lstsq(a, y, rcond=None)[0] for a in A])
            pred = np.einsum("zij,zj->zi", A, coef)
            chi[nombre] = np.where(coef[:, 0] > 0, (((pred - y) ** 2) / y).sum(1), np.inf)
            ajus[nombre] = pred
        dz = np.subtract.outer(Z_PERFIL, Z_PERFIL)               # dz[j, i] = z'_j - z_i
        lo, hi = -L_nm * np.tan(np.radians(max(betas))), -L_nm * np.tan(np.radians(min(betas)))
        tot = np.where((dz >= lo) & (dz <= hi), chi["reflejo"][:, None] + chi["real"][None, :], np.inf)
        j, i = np.unravel_index(np.argmin(tot), tot.shape)
        d45 = np.argmin(np.abs(Z_PERFIL[None, :] - L_nm - Z_PERFIL[:, None]), axis=0)  # z' mas cercano a z - L (45 grados)
        tot45 = chi["real"] + chi["reflejo"][d45]
        k45 = int(np.argmin(tot45))
        out.append(dict(variante=var, z_real_libre=Z_PERFIL[np.argmin(chi["real"])], chi2_real_libre=chi["real"].min(),
                        z_reflejo_libre=Z_PERFIL[np.argmin(chi["reflejo"])], chi2_reflejo_libre=chi["reflejo"].min(),
                        z_real=Z_PERFIL[i], z_reflejo=Z_PERFIL[j], chi2_total=tot[j, i],
                        z_real_45=Z_PERFIL[k45], z_reflejo_45=Z_PERFIL[d45[k45]], chi2_total_45=tot45[k45],
                        frac_anillo_real=fr, frac_anillo_reflejo=ff,
                        frac_anillo_real_p5=np.percentile(frac["real"], 5), frac_anillo_real_p95=np.percentile(frac["real"], 95),
                        frac_anillo_reflejo_p5=np.percentile(frac["reflejo"], 5),
                        frac_anillo_reflejo_p95=np.percentile(frac["reflejo"], 95),
                        frac_error_min=np.sqrt(mejor_f[0]), frac_z_real=mejor_f[1], frac_beta=mejor_f[2]))
        modelos[var] = dict(chi=chi, ajus=ajus, i=i, j=j, imgs=imgs, tot45=tot45, frac_sim=frac_sim)
    return pd.DataFrame(out), datos, (img_r, img_f), modelos, frac


def figura_perfiles(res, datos, imgs, modelos, frac, L_nm, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.1], hspace=0.45, wspace=0.75)
    r = 30
    for k, (nombre, im) in enumerate(zip(("real", "reflejo"), imgs)):
        ax = fig.add_subplot(gs[k, 0])
        ax.imshow(im, cmap="magma")
        ax.set_title(f"{nombre} (datos, cuadros 0-{CUADROS_PERFIL - 1})", fontsize=8.5)
        ax.axis("off")
        for v, var in enumerate(("cerca", "lejos")):
            mdl = modelos[var]
            idx = mdl["i"] if nombre == "real" else mdl["j"]
            ax = fig.add_subplot(gs[k, 1 + v])
            ax.imshow(mdl["imgs"][idx][64 - r: 64 + r + 1, 64 - r: 64 + r + 1], cmap="magma")
            ax.set_title(f"{var}, z = {Z_PERFIL[idx]:.0f} nm", fontsize=8.5)
            ax.axis("off")
        ax = fig.add_subplot(gs[k, 3:])
        ax.plot(datos[nombre], "o", color="#52514e", ms=3.5, label="datos")
        for var in ("cerca", "lejos"):
            mdl = modelos[var]
            idx = mdl["i"] if nombre == "real" else mdl["j"]
            ax.plot(mdl["ajus"][nombre][idx], "-", color=COLOR[var], lw=1.8,
                    label=f"{var} (z = {Z_PERFIL[idx]:.0f} nm, ajuste conjunto)")
        ax.set_ylabel(f"{nombre}: ADU medio")
        ax.set_xlabel("radio [px]")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.legend(frameon=False, fontsize=7.5)
    ax = fig.add_subplot(gs[2, 4:])
    for var in ("cerca", "lejos"):
        ax.plot(Z_PERFIL, modelos[var]["frac_sim"], "-", color=COLOR[var], lw=1.8, label=f"simulada, {var}")
    for nombre, ls in (("real", "-"), ("reflejo", "--")):
        q = np.percentile(frac[nombre], [5, 50, 95])
        ax.axhspan(q[0], q[2], color="#8a8984", alpha=0.18, lw=0)
        ax.axhline(q[1], color="#52514e", ls=ls, lw=1.2, label=f"datos: {nombre} (mediana, p5-p95 por cuadro)")
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("fraccion en el anillo 6-18 px")
    ax.set_xticks(np.arange(-8000, 8001, 4000))
    ax.set_title(f"el reflejo esta {L_nm * np.tan(np.radians(40)) / 1000:.1f}-{L_nm * np.tan(np.radians(50)) / 1000:.1f} um "
                 "mas lejos del objetivo que el real (beta 40-50)", fontsize=8.5)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=7)
    ax = fig.add_subplot(gs[2, :2])
    for var in ("cerca", "lejos"):
        ax.plot(Z_PERFIL, modelos[var]["chi"]["real"], "-", color=COLOR[var], lw=1.6, label=f"{var}: real")
        ax.plot(Z_PERFIL, modelos[var]["chi"]["reflejo"], "--", color=COLOR[var], lw=1.6, label=f"{var}: reflejo")
    ax.set_yscale("log")
    ax.set_xlabel("z del modelo [nm]")
    ax.set_xticks(np.arange(-8000, 8001, 4000))
    ax.set_ylabel("chi2 del perfil")
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax = fig.add_subplot(gs[2, 2:4])
    for var in ("cerca", "lejos"):
        ax.plot(Z_PERFIL, modelos[var]["tot45"], "-", color=COLOR[var], lw=1.8, label=f"{var}")
    ax.set_yscale("log")
    ax.set_xlabel("z del real [nm] (reflejo en z - L, beta = 45)")
    ax.set_xticks(np.arange(-8000, 8001, 4000))
    ax.set_ylabel("chi2 real + reflejo")
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False)
    t = res.set_index("variante")
    fig.suptitle("Perfiles radiales de Yeast6_1 (canal derecho) contra la PSF esferica simulada: chi2 total con "
                 f"z' - z en [-L tan 50, -L tan 40]: cerca {t.chi2_total['cerca']:.1f}, lejos {t.chi2_total['lejos']:.1f}",
                 fontsize=10.5)
    fig.savefig(ruta.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def reales(a, dispositivo, rng, salida):
    import analisis_datos_reales as adr
    verificar_triangulacion()
    px = adr.PIXEL_NM
    beta = np.radians(a.beta_deg)
    video, info = adr.cargar(a.tif)
    T = len(video)
    c = adr.buscar_corte(video.mean(0))
    corte = c["corte"]
    vid = video[:, :, corte:].astype(np.float32)
    crudo = video[:, :, corte:]
    del video
    ap = c["ap_der"]
    dets = [adr.detectar(vid[t], ap) for t in range(T)]
    pares = adr.pares_real_reflejo(adr.enlazar(dets, T), T)
    if len(pares) != 1:
        raise SystemExit(f"se esperaba un par real/reflejo en el canal derecho; hay {len(pares)}")
    real, refl = pares[0]["real"], pares[0]["refl"]
    seg = np.nanmean(real - refl, 0)
    m = seg / np.hypot(*seg)
    t_dir = np.array([-m[1], m[0]])
    centro = np.nanmean(np.r_[real, refl], 0)
    _, _, _, fr = adr.perfil_franja(vid.mean(0), centro, m, t_dir, ap, [np.nanmean(real, 0), np.nanmean(refl, 0)])
    q = centro + fr["v_borde"] * m                               # linea espejo/plano focal (supuesto), px locales
    phi = float(np.degrees(np.arctan2(m[1], m[0])))
    z_tri = z_triangulada(real, refl, q, m, beta, px)
    print(f"Yeast6_1, canal derecho (columnas {corte}..): {T} cuadros; par real/reflejo en "
          f"{np.isfinite(real[:, 0] + refl[:, 0]).sum()} cuadros; phi = {phi:+.1f} deg, separacion "
          f"{np.nanmean(np.hypot(*(real - refl).T)) * px:.0f} nm; linea (borde de la franja) a "
          f"{fr['v_borde']:+.1f} px del punto medio; beta = {a.beta_deg:g}")
    print(f"  z triangulada: media {np.nanmean(z_tri):+.0f} nm, std {np.nanstd(z_tri):.0f} nm, "
          f"rango {np.nanmin(z_tri):+.0f}..{np.nanmax(z_tri):+.0f} nm")
    L_nm = float(np.nanmean((real - refl) @ m)) * px
    perf, datos, imgs_p, modelos, frac = ajuste_perfiles(vid, real, refl, phi, L_nm)
    perf.to_csv(salida / "red_celula_reales_perfiles.csv", index=False)
    figura_perfiles(perf, datos, imgs_p, modelos, frac, L_nm, salida / "fig_red_celula_reales_psf")
    print(f"  perfiles radiales (cuadros 0-{CUADROS_PERFIL - 1}), L = {L_nm:.0f} nm:")
    with pd.option_context("display.width", 250):
        print(perf.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    for bb in (40.0, 50.0):
        zb = z_triangulada(real, refl, q, m, np.radians(bb), px)
        print(f"    con beta = {bb:g}: media {np.nanmean(zb):+.0f}, std {np.nanstd(zb):.0f} nm, "
              f"corr con la de {a.beta_deg:g} {np.corrcoef(zb[np.isfinite(zb)], z_tri[np.isfinite(z_tri)])[0, 1]:.3f}")

    # ventanas: (nombre, centro de la ventana [px locales], corrimiento de la linea a lo largo de m [px])
    medio = np.nanmean((real + refl) / 2, 0)
    ventanas = [("medio", medio, 0.0), ("medio_linea-10px", medio, -10.0), ("medio_linea+10px", medio, 10.0),
                ("real", np.nanmean(real, 0), 0.0)]
    filas, resumen, imagenes = [], [], {}
    for var in a.variantes:
        modelo, cfg, cs = cargar_red(var, dispositivo)
        for nombre, cen, dlin in ventanas:
            x0, y0 = int(round(cen[0])) - W // 2, int(round(cen[1])) - H // 2
            assert 0 <= x0 and x0 + W <= crudo.shape[2] and 0 <= y0 and y0 + H <= crudo.shape[1]
            assert ap[y0:y0 + H, x0:x0 + W].all(), "la ventana se sale de la apertura de campo"
            ims = np.ascontiguousarray(crudo[:, y0:y0 + H, x0:x0 + W])
            qv = q + dlin * m - np.array([x0, y0])
            geo = geometria(a.beta_deg, phi, qv[0], qv[1], T)
            det = localizar(modelo, ims, cfg, replace(cs, H=H, W=W), dispositivo, geometria=geo)
            rv, fv = real - np.array([x0, y0]), refl - np.array([x0, y0])
            z_red, conf, n_det, n_refl = np.full(T, np.nan), np.full(T, np.nan), np.zeros(T, int), np.zeros(T, int)
            for t, d in det.groupby("t"):
                dl = np.hypot(d.x_px - rv[t, 0], d.y_px - rv[t, 1])
                n_det[t] = len(d)
                n_refl[t] = int((np.hypot(d.x_px - fv[t, 0], d.y_px - fv[t, 1]) <= R_ASOC_REAL_PX).sum())
                if (dl <= R_ASOC_REAL_PX).any():
                    j = np.argmin(dl.to_numpy())
                    z_red[t], conf[t] = d.z_nm.iat[j], d.conf.iat[j]
            r = ajuste(z_tri, z_red, rng)
            dif = ajuste(np.diff(z_tri), np.diff(z_red), rng)
            # dos modos (z de signo opuesto): fraccion de cada uno y correlacion dentro del dominante
            ok = np.isfinite(z_red)
            signo = np.sign(np.nanmedian(z_red))
            dom = ok & (np.sign(z_red) == signo)
            rd = ajuste(np.where(dom, z_tri, np.nan), np.where(dom, z_red, np.nan), rng)
            # el modo (signo) contra z_tri: diferencia de z_tri media entre los cuadros de cada modo
            otro = ok & ~dom
            modo = dict(signo_dominante=signo, frac_dominante=dom.sum() / ok.sum(),
                        z_red_mediana_dominante=np.nanmedian(z_red[dom]), z_red_mediana_otro=np.nanmedian(z_red[otro])
                        if otro.any() else np.nan, z_tri_media_dominante=np.nanmean(z_tri[dom]),
                        z_tri_media_otro=np.nanmean(z_tri[otro]) if otro.any() else np.nan,
                        **{f"dom_{k}": v for k, v in rd.items()})
            resumen.append(dict(variante=var, ventana=nombre, x0=x0, y0=y0, detectado=np.isfinite(z_red).mean(), **modo,
                                z_red_media=np.nanmean(z_red), z_red_std=np.nanstd(z_red), z_tri_media=np.nanmean(z_tri),
                                conf_mediana=np.nanmedian(conf), otras_det_por_cuadro=(n_det - np.isfinite(z_red)).mean(),
                                det_sobre_reflejo=n_refl.mean(), **r,
                                **{f"dif_{k}": v for k, v in dif.items() if k.startswith(("correlacion", "pendiente"))}))
            print(f"  red {var}, ventana {nombre} (x0 {x0}, y0 {y0}): detectado {np.isfinite(z_red).mean():.2f}, "
                  f"z red {np.nanmean(z_red):+.0f} +- {np.nanstd(z_red):.0f} nm; corr {r['correlacion']:+.3f} "
                  f"[{r['correlacion_lo']:+.2f}, {r['correlacion_hi']:+.2f}], pendiente {r['pendiente']:+.2f} "
                  f"[{r['pendiente_lo']:+.2f}, {r['pendiente_hi']:+.2f}], desfase (red - tri) {r['desfase_nm']:+.0f} "
                  f"[{r['desfase_nm_lo']:+.0f}, {r['desfase_nm_hi']:+.0f}] nm; pasos: corr {dif['correlacion']:+.3f}, "
                  f"pendiente {dif['pendiente']:+.2f}; modo dominante z {modo['z_red_mediana_dominante']:+.0f} nm en "
                  f"{modo['frac_dominante']:.2f} de los cuadros (otro {modo['z_red_mediana_otro']:+.0f}); z_tri media en el "
                  f"dominante {modo['z_tri_media_dominante']:+.0f}, en el otro {modo['z_tri_media_otro']:+.0f}; dentro del "
                  f"dominante corr {rd['correlacion']:+.2f} [{rd['correlacion_lo']:+.2f}, {rd['correlacion_hi']:+.2f}], "
                  f"pendiente {rd['pendiente']:+.2f} [{rd['pendiente_lo']:+.2f}, {rd['pendiente_hi']:+.2f}]; otras detecciones/cuadro {(n_det - np.isfinite(z_red)).mean():.2f} "
                  f"(sobre el reflejo {n_refl.mean():.2f}); conf mediana {np.nanmedian(conf):.2f}")
            filas.append(pd.DataFrame(dict(variante=var, ventana=nombre, frame=np.arange(T), z_tri_nm=z_tri, z_red_nm=z_red,
                                           conf=conf, n_det=n_det, x_real_px=real[:, 0], y_real_px=real[:, 1],
                                           x_reflejo_px=refl[:, 0], y_reflejo_px=refl[:, 1])))
            if nombre == "medio":
                imagenes[var] = (ims, det, rv, fv, qv)
    pd.concat(filas).to_csv(salida / "red_celula_reales.csv", index=False)
    res = pd.DataFrame(resumen)
    res.to_csv(salida / "red_celula_reales_resumen.csv", index=False)
    figura_reales(pd.concat(filas), imagenes, m, phi, a.beta_deg, salida / "fig_red_celula_reales")
    return res


def figura_reales(f, imagenes, m, phi, beta_deg, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.7, 1, 1], wspace=0.35)
    var0 = next(iter(imagenes))
    ims, det, rv, fv, qv = imagenes[var0]
    ax = fig.add_subplot(gs[0])
    med = ims[:10].mean(0)
    ax.imshow(med, cmap="gray", vmin=np.percentile(med, 1), vmax=np.percentile(med, 99.7))
    ax.plot(rv[:, 0], rv[:, 1], "-", color=COLOR["tri"], lw=0.8)
    ax.plot(fv[:, 0], fv[:, 1], "-", color="#eda100", lw=0.8)
    tl = np.array([-m[1], m[0]])
    lin = qv[None] + np.linspace(-80, 80, 2)[:, None] * tl
    ax.plot(lin[:, 0], lin[:, 1], "--", color="#e87ba4", lw=1, label="linea del espejo (borde de la franja)")
    for var, (_, d, *_r) in imagenes.items():
        d0 = d[d.t < 10]
        ax.scatter(d0.x_px, d0.y_px, marker="+", s=30, color=COLOR[var], lw=1, label=f"detecciones red {var} (cuadros 0-9)")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_title("ventana 96 x 96 (media cuadros 0-9)\nverde: real, amarillo: reflejo", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax = fig.add_subplot(gs[1])
    s = f[(f.ventana == "medio") & (f.variante == var0)]
    ax.plot(s.frame, s.z_tri_nm, "-", color=COLOR["tri"], lw=1.8, label=f"triangulada (beta {beta_deg:g})")
    for var in dict.fromkeys(f.variante):
        s = f[(f.ventana == "medio") & (f.variante == var)]
        ax.plot(s.frame, s.z_red_nm, "o-", color=COLOR[var], lw=1, ms=2.5, label=f"red {var}")
    ax.set_xlabel("cuadro (10 Hz)")
    ax.set_ylabel("z del real [nm]")
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    for k, var in enumerate(dict.fromkeys(f.variante)):
        ax = fig.add_subplot(gs[2 + k])
        s = f[(f.ventana == "medio") & (f.variante == var)]
        ok = s.z_red_nm.notna()
        ax.scatter(s.z_tri_nm[ok], s.z_red_nm[ok], s=10, color=COLOR[var])
        if ok.sum() > 2:
            p = np.polyfit(s.z_tri_nm[ok], s.z_red_nm[ok], 1)
            xx = np.array([s.z_tri_nm.min(), s.z_tri_nm.max()])
            ax.plot(xx, np.polyval(p, xx), "-", color=COLOR[var], lw=1.5,
                    label=f"pendiente {p[0]:+.2f}, r = {np.corrcoef(s.z_tri_nm[ok], s.z_red_nm[ok])[0, 1]:+.2f}")
            ax.plot(xx, xx, ":", color="#8a8984", lw=1, label="y = x")
        ax.set_xlabel("z triangulada [nm]")
        ax.set_ylabel(f"z red {var} [nm]")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Yeast6_1, canal derecho: z de las redes con celula vs z triangulada (phi {phi:+.1f} deg, "
                 f"beta {beta_deg:g} deg, 98.48 nm/px)", fontsize=11)
    fig.savefig(ruta.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("modo", choices=("transferencia", "reales", "fisher", "agujero"))
    p.add_argument("--checkpoint", nargs=2, action="append", metavar=("VARIANTE", "RUTA"),
                   help="transferencia / agujero: checkpoint en lugar de mejor.pt de la corrida de la variante")
    p.add_argument("--variantes", nargs="+", default=["cerca", "lejos"], choices=list(CORRIDA))
    p.add_argument("--n-rep", type=int, default=8, help="transferencia: imagenes por (z, N, ratio)")
    p.add_argument("--reusar", action="store_true", help="agujero: resume red_celula_agujero_imagenes.csv sin correr la red")
    p.add_argument("--tif", default=TIF_DEF)
    p.add_argument("--beta-deg", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dispositivo", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--salida", default=str(Path(__file__).resolve().parent))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    salida = Path(a.salida)
    dispositivo = elegir_dispositivo(a.dispositivo)
    if a.modo == "agujero":
        agujero(a, dispositivo, rng, salida)
        return
    if a.modo == "fisher":
        fisher_z(salida)
        return
    if a.modo == "transferencia":
        t = pd.concat([transferencia(v, a.n_rep, dispositivo, rng, dict(a.checkpoint or []).get(v))
                       for v in a.variantes], ignore_index=True)
        t.to_csv(salida / "red_celula_transferencia_imagenes.csv", index=False, float_format="%.6g")
        r = resumen_transferencia(t)
        r.to_csv(salida / "red_celula_transferencia.csv", index=False)
        figura_transferencia(r, salida / "fig_red_celula_transferencia")
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            for (var, N, ratio), s in r.groupby(["variante", "fotones", "ratio"]):
                print(f"\n{var}, N {N:.0f}, ratio {ratio}: detectado medio {s.detectado.mean():.3f}, "
                      f"RMSE z (z en [-2000, 2000]) {np.sqrt(np.nanmean(s[s.z_nm.abs() <= 2000].rmse_z ** 2)):.0f} nm")
                print(s[s.z_nm % 250 == 0][["z_nm", "detectado", "z_pred_mediana", "z_pred_p16", "z_pred_p84", "rmse_z",
                                            "fp_por_imagen"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    else:
        reales(a, dispositivo, rng, salida)


if __name__ == "__main__":
    main()
