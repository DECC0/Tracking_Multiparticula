"""
Piso de error de las redes con celula: es cuantizacion de la rejilla de salida?
La salida de la red tiene planos en z = z_min + j dz (dz = 100 nm, z_min = -2500)
y voxeles laterales de pixel / 4 (24.62 nm) con centros en x_px = (v + 0.5) / 4
- 0.5; el post-procesado (localizacion/postproc.py) da la posicion continua por
centro de masa en una ventana de +-2 planos y +-2 voxeles alrededor de cada
maximo local. Las cotas de redondeo puro al nodo mas cercano son dz / sqrt(12) =
28.9 nm en z y (pixel / 4) / sqrt(12) = 7.1 nm por eje lateral (10.0 nm en 2D).

fases
    1. Distancia de cada prediccion al plano / nodo mas cercano (z_pred mod 100,
       x_pred y y_pred mod 24.62 nm) en los tests de cerca y lejos (predicciones.csv
       de evaluate.py) y en la curva de transferencia (red_celula_transferencia_
       imagenes.csv): histograma y fraccion a < 5 nm (uniforme: 0.10 en z, 0.41 en
       x, y), y la exactamente sobre un plano (< 0.01 nm).
    2. Sesgo contra la fase de la posicion real dentro de la celda de la rejilla
       (tests). La transferencia va cada 50 nm: solo dos fases en z (sobre un plano
       y a mitad de celda), donde el redondeo daria RMSE 0 y 50 nm.
       (error medio por rango de fase): una red/estimador sin cuantizacion da
       sesgo 0; redondeo puro da una rampa de pendiente -1. RMSE a > 19600 fotones
       de los emisores aislados (> 1 um de otro real y de otro reflejo) frente a
       28.9 / 7.1 nm, y la parte del RMSE que explica el sesgo por fase.
subplano (corre la red, GPU)
    2. Por que el centro de masa da exactamente un plano: con la salida de la red,
       por deteccion, si el maximo esta saturado (hardtanh en escala), cuantos
       vecinos en z de la ventana valen 0 y si sum(w dk) = 0.
    3. Estimadores subplano alternativos sobre la misma salida y los mismos maximos
       (las mismas detecciones y confianzas que el post-procesado actual):
           cm         el actual (centro de masa, +-2 planos y +-2 voxeles)
           cm1        centro de masa con +-1
           parabola   parabola por 3 puntos en cada eje a traves del maximo
           gauss      parabola en log (gaussiana de 3 puntos: exacta para el target)
       RMSE z y lateral por fotones (todos y aislados) y sesgo por fase, para cerca
       y lejos. cm reproduce evaluate.py (se verifica).

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/cuantizacion.py fases
    LD_PRELOAD=$HOME/miniforge3/envs/tesis/lib/libstdc++.so.6 $PY figuras/cuantizacion.py subplano
Escribe en --salida (por defecto figuras/cuantizacion/): cuantizacion_fases.csv, cuantizacion_sesgo_fase.csv,
fig_cuantizacion_fases.png, cuantizacion_subplano.csv, cuantizacion_subplano_sesgo.csv,
cuantizacion_cm_plano.csv y fig_cuantizacion_subplano.png (300 dpi).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "localizacion"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

CORRIDAS = RAIZ / "localizacion" / "corridas"
DATOS = RAIZ / "localizacion" / "datos"
CORRIDA = {"cerca": "hypatia_celula_b45_cerca_20261110-234627", "lejos": "hypatia_celula_b45_lejos_20261110-234617"}
COLOR = {"cerca": "#2a78d6", "lejos": "#eb6834"}
COLOR_EST = {"cm": "#52514e", "cm1": "#eda100", "parabola": "#1baf7a", "gauss": "#e87ba4"}
PX = 6500.0 / 66.0                      # nm por pixel
S = 4                                   # voxeles por pixel
DZ, Z_MIN = 100.0, -2500.0
NODO = PX / S                           # 24.62 nm
COTA_Z, COTA_LAT = DZ / np.sqrt(12), NODO / np.sqrt(12)
CERCA_NM = 5.0
FOTONES_ALTOS = 19600.0
BORDES_FOT = np.geomspace(1000.0, 30000.0, 9)
AISLADO_NM = 1000.0
ESTIMADORES = ("cm", "cm1", "parabola", "gauss")


# ----------------------------------------------------------------------------
# Fases
# ----------------------------------------------------------------------------

def fase_z(z):
    """Posicion dentro de la celda en z, en [0, 1) (0 = sobre un plano)."""
    return np.mod((np.asarray(z, float) - Z_MIN) / DZ, 1.0)


def fase_lat(x_px):
    """Posicion dentro de la celda lateral en [0, 1) (0 = sobre un nodo)."""
    return np.mod((np.asarray(x_px, float) + 0.5) * S - 0.5, 1.0)


def dist_nodo(f, paso):
    """Distancia [nm] al nodo mas cercano desde una fase en [0, 1)."""
    return np.minimum(f, 1.0 - f) * paso


def rms_atraccion(fase, paso):
    """RMS [nm] de la atraccion hacia los nodos implicada por la distribucion de fases:
    si la posicion real tiene fase uniforme en la celda (la celula es suave a escala de
    100 nm) y la prediccion es p(s) aplicada a una estimacion s de fase uniforme, p es la
    funcion cuantil de las fases predichas; RMS(p(s) - s) es lo que la atraccion agrega
    al error (en cuadratura, si no se correlaciona con el resto). 0 si son uniformes."""
    f = np.sort(np.where(fase >= 0.5, fase - 1.0, fase))
    u = (np.arange(len(f)) + 0.5) / len(f) - 0.5
    return float(np.sqrt(np.mean((f - u) ** 2)) * paso)


def sesgo_cm_target(radio, sigma=1.0, n=2001):
    """Sesgo del centro de masa en una ventana de +-radio sobre la gaussiana exacta del
    target (sigma en voxeles) centrada a delta del voxel del maximo, delta uniforme en
    [-0.5, 0.5]: (sesgo maximo, RMS) en voxeles. Es lo que el post-procesado agrega aunque
    la red reproduzca el target perfecto."""
    d = np.linspace(-0.5, 0.5, n)
    j = np.arange(-radio, radio + 1)
    w = np.exp(-(j[None, :] - d[:, None]) ** 2 / (2 * sigma ** 2))
    err = (w * j).sum(1) / w.sum(1) - d
    return float(np.abs(err).max()), float(np.sqrt(np.mean(err ** 2)))


def fila_fases(fuente, var, x, y, z):
    fz, fx, fy = fase_z(z), fase_lat(x), fase_lat(y)
    dz, dx, dy = dist_nodo(fz, DZ), dist_nodo(fx, NODO), dist_nodo(fy, NODO)
    return dict(fuente=fuente, variante=var, n=len(z), z_a_5nm=(dz < CERCA_NM).mean(), z_exacto=(dz < 0.01).mean(),
                x_a_5nm=(dx < CERCA_NM).mean(), y_a_5nm=(dy < CERCA_NM).mean(),
                x_exacto=(dx < 0.01).mean(), y_exacto=(dy < 0.01).mean(),
                uniforme_z=2 * CERCA_NM / DZ, uniforme_lat=2 * CERCA_NM / NODO,
                atraccion_z_nm=rms_atraccion(fz, DZ), atraccion_x_nm=rms_atraccion(fx, NODO),
                atraccion_y_nm=rms_atraccion(fy, NODO))


def gt_eval_aislado(var, carpeta_eval=None):
    """gt_eval.csv de la evaluacion de la variante + d al vecino real / reflejo mas cercano."""
    import diagnostico_celula as dc
    e = Path(carpeta_eval or CORRIDAS / CORRIDA[var] / f"evaluacion_test_celula_b45_{var}")
    g = pd.read_csv(e / "gt_eval.csv")
    g["d_vecino_nm"], g["d_reflejo_vecino_nm"] = dc.vecinos(g, PX)
    g["aislado"] = (g.d_vecino_nm > AISLADO_NM) & (g.d_reflejo_vecino_nm > AISLADO_NM)
    return g, pd.read_csv(e / "predicciones.csv")


def sesgo_por_fase(g, n_bins=10):
    """Error medio (z y x, y) por rango de fase de la posicion REAL, de los detectados en el FOV."""
    d = g[g.en_fov & g.detectado]
    filas = []
    for eje, fase, err, paso in (("z", fase_z(d.z_nm), d.dz_nm, DZ), ("x", fase_lat(d.x_px), d.dx_nm, NODO),
                                 ("y", fase_lat(d.y_px), d.dy_nm, NODO)):
        f = np.where(fase >= 0.5, fase - 1.0, fase)                 # en [-0.5, 0.5): 0 = sobre el nodo
        k = np.clip(((f + 0.5) * n_bins).astype(int), 0, n_bins - 1)
        for i in range(n_bins):
            s = err.to_numpy()[k == i]
            filas.append(dict(eje=eje, fase_centro=(i + 0.5) / n_bins - 0.5, desplazamiento_nm=((i + 0.5) / n_bins - 0.5) * paso,
                              n=len(s), sesgo_nm=s.mean() if len(s) else np.nan,
                              rmse_nm=np.sqrt(np.mean(s ** 2)) if len(s) else np.nan))
    return pd.DataFrame(filas)


def rmse_altos(g, sel_extra=None):
    """RMSE z, x, y y lateral 2D de los detectados en el FOV con > FOTONES_ALTOS fotones (y sel_extra)."""
    s = g.en_fov & g.detectado & (g.photons > FOTONES_ALTOS)
    if sel_extra is not None:
        s &= sel_extra
    d = g[s]
    r = lambda a: float(np.sqrt(np.mean(np.square(a))))      # noqa: E731
    return dict(n=len(d), rmse_z=r(d.dz_nm), rmse_x=r(d.dx_nm), rmse_y=r(d.dy_nm), rmse_lat=r(np.hypot(d.dx_nm, d.dy_nm)))


def parte_explicada(g, sel, n_bins=10):
    """RMS del sesgo por fase (la parte sistematica del error ligada a la rejilla) en z y en x, y."""
    t = sesgo_por_fase(g[sel], n_bins)
    out = {}
    for eje in ("z", "x", "y"):
        s = t[(t.eje == eje) & (t.n > 0)]
        out[f"rms_sesgo_fase_{eje}"] = float(np.sqrt(np.average(s.sesgo_nm ** 2, weights=s.n)))
    return out


def fases(a, salida):
    filas, sesgos, resumen, rmse_fase = [], [], [], []
    hist = {}
    for var in ("cerca", "lejos"):
        g, p = gt_eval_aislado(var)
        filas.append(fila_fases("test", var, p.x_px, p.y_px, p.z_nm))
        tp = g[g.en_fov & g.detectado]
        for nombre, sel in (("test, TP", np.ones(len(tp), bool)), ("test, TP > 8380 fotones", tp.photons > 8380),
                            ("test, GT de los TP (referencia)", None)):
            if sel is None:
                filas.append(fila_fases(nombre, var, tp.x_px, tp.y_px, tp.z_nm))
            else:
                t_ = tp[sel]
                filas.append(fila_fases(nombre, var, t_.x_px + t_.dx_nm / PX, t_.y_px + t_.dy_nm / PX, t_.z_nm + t_.dz_nm))
        hist[("test", var)] = p[["x_px", "y_px", "z_nm"]].to_numpy()
        t = sesgo_por_fase(g[g.photons > 8380])
        sesgos.append(t.assign(variante=var, seleccion="> 8380 fotones"))
        alto = (g.photons > FOTONES_ALTOS) & g.aislado
        sesgos.append(sesgo_por_fase(g[alto]).assign(variante=var, seleccion="> 19600 fotones, aislados"))
        for nombre, sel in (("todos", None), ("aislados", g.aislado)):
            r = rmse_altos(g, sel)
            exp = parte_explicada(g, g.en_fov & g.detectado & (g.photons > FOTONES_ALTOS) & (sel if sel is not None else True))
            resumen.append(dict(variante=var, seleccion=f"> {FOTONES_ALTOS:.0f} fotones, {nombre}", **r, **exp,
                                cota_z=COTA_Z, cota_por_eje=COTA_LAT, cota_lat_2d=COTA_LAT * np.sqrt(2)))
    ti = Path(a.transferencia)
    if ti.exists():
        t = pd.read_csv(ti)
        for var, s in t.groupby("variante"):
            s = s[s.z_pred_nm.notna()]
            filas.append(fila_fases("transferencia", var, s.x_pred_px, s.y_pred_px, s.z_pred_nm))
            hist[("transferencia", var)] = s[["x_pred_px", "y_pred_px", "z_pred_nm"]].to_numpy()
            # sesgo por fase en la transferencia (emisor aislado, N = 1e4, con reflejo; |z| <= 2000)
            b = s[(s.fotones == 10000) & (s.ratio > 0) & (s.z_nm.abs() <= 2000)]
            gg = pd.DataFrame(dict(en_fov=True, detectado=True, z_nm=b.z_nm, x_px=b.x_px, y_px=b.y_px,
                                   dz_nm=b.z_pred_nm - b.z_nm, dx_nm=(b.x_pred_px - b.x_px) * PX,
                                   dy_nm=(b.y_pred_px - b.y_px) * PX))
            sesgos.append(sesgo_por_fase(gg).assign(variante=var, seleccion="transferencia, N = 1e4"))
            f = fase_z(b.z_nm)
            for nombre, sel in (("sobre un plano", f < 0.01), ("a mitad de celda", np.abs(f - 0.5) < 0.01)):
                e = (b.z_pred_nm - b.z_nm)[sel]
                rmse_fase.append(dict(variante=var, fase=nombre, n=int(sel.sum()), rmse_z=float(np.sqrt(np.mean(e ** 2))),
                                      sesgo_z=float(e.mean()), redondeo_rmse=0.0 if nombre == "sobre un plano" else DZ / 2))
    else:
        print(f"aviso: no existe {ti} (red_celula.py transferencia); solo los tests")
    ta = Path(a.agujero)
    if ta.exists():                                 # z cada 10 nm en [0, 800]: todas las fases (cerca, condicion base)
        t = pd.read_csv(ta)
        b = t[(t.condicion == "base") & t.z_pred_nm.notna() & ((t.z_nm < 250) | (t.z_nm > 650))]
        filas.append(fila_fases("barrido fino (cerca, base, fuera de 250-650 nm)", "cerca", b.x_pred_px, b.y_pred_px,
                                b.z_pred_nm))
        # (su sesgo contra la fase no se usa: z en [0, 250] y [650, 800] tiene el sesgo dependiente de z del
        # borde del agujero, no de la fase)
    else:
        print(f"aviso: no existe {ta} (red_celula.py agujero)")
    f = pd.DataFrame(filas)
    f.to_csv(salida / "cuantizacion_fases.csv", index=False)
    sz = pd.concat(sesgos, ignore_index=True)
    sz.to_csv(salida / "cuantizacion_sesgo_fase.csv", index=False)
    rf = pd.DataFrame(rmse_fase)
    rf.to_csv(salida / "cuantizacion_transferencia_fase.csv", index=False)
    r = pd.DataFrame(resumen)
    r.to_csv(salida / "cuantizacion_rmse_altos.csv", index=False)
    for radio in (2, 1):
        mx, rm = sesgo_cm_target(radio)
        print(f"centro de masa +-{radio} sobre el target exacto (sigma 1 voxel): sesgo max {mx * DZ:.2f} nm en z "
              f"({mx * NODO:.2f} lateral), RMS {rm * DZ:.2f} nm en z ({rm * NODO:.2f} lateral)")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("fraccion de predicciones a < 5 nm de un plano (z) o nodo (x, y):")
        print(f.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nRMSE a > {FOTONES_ALTOS:.0f} fotones (cotas de redondeo: z {COTA_Z:.1f}, por eje {COTA_LAT:.2f}, "
              f"lateral 2D {COTA_LAT * np.sqrt(2):.2f} nm) y RMS del sesgo por fase:")
        print(r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("\ntransferencia (N = 1e4, con reflejo, |z| <= 2000): RMSE z con el emisor sobre un plano y a mitad de celda")
        print(rf.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        print("\nsesgo por fase (z), > 19600 fotones aislados y transferencia:")
        print(sz[(sz.eje == "z") & sz.seleccion.isin(["> 19600 fotones, aislados", "transferencia, N = 1e4"])]
              .pivot_table(index="desplazamiento_nm", columns=["seleccion", "variante"], values="sesgo_nm").round(1).to_string())
    figura_fases(hist, sz, salida / "fig_cuantizacion_fases.png")


def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def figura_fases(hist, sesgos, ruta):
    plt = _estilo()
    fuentes = list(hist)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    for j, (eje, idx, paso) in enumerate((("z", 2, DZ), ("x", 0, NODO))):
        ax = axes[j, 0]
        for (fuente, var) in fuentes:
            v = hist[(fuente, var)][:, idx]
            f = fase_z(v) if eje == "z" else fase_lat(v)
            d = np.where(f >= 0.5, f - 1.0, f) * paso
            ax.hist(d, np.linspace(-paso / 2, paso / 2, 41), histtype="step", density=True, lw=1.6,
                    ls="-" if fuente == "test" else "--", color=COLOR[var], label=f"{fuente}, {var}")
        ax.axhline(1 / paso, color="#8a8984", lw=0.8, ls=":")
        ax.set_xlabel(f"{eje} predicha - {'plano' if eje == 'z' else 'nodo'} mas cercano [nm]")
        ax.set_ylabel("densidad")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, color="#e6e5e0", lw=0.6)
    for k, (sel, titulo) in enumerate((("> 8380 fotones", "test, > 8380 fotones"),
                                       ("> 19600 fotones, aislados", "test, > 19600 fotones, aislados"),
                                       ("transferencia, N = 1e4", "transferencia, N = 1e4"))):
        for j, eje in enumerate(("z", "x")):
            ax = axes[j, 1 + k]
            for var in ("cerca", "lejos"):
                s = sesgos[(sesgos.seleccion == sel) & (sesgos.variante == var) & (sesgos.eje == eje)]
                if len(s):
                    ax.plot(s.desplazamiento_nm, s.sesgo_nm, "o-", color=COLOR[var], lw=1.6, ms=3.5, label=var)
            paso = DZ if eje == "z" else NODO
            xx = np.array([-paso / 2, paso / 2])
            ax.plot(xx, -xx, ":", color="#52514e", lw=1, label="redondeo al nodo")
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.set_xlabel(f"{eje} real - {'plano' if eje == 'z' else 'nodo'} mas cercano [nm]")
            ax.set_ylabel(f"error medio en {eje} [nm]")
            ax.set_title(titulo, fontsize=9)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Cuantizacion de la rejilla de salida (planos cada 100 nm, voxeles laterales de 24.6 nm): posicion de las "
                 "predicciones respecto de la rejilla (izq.) y error medio segun la posicion real dentro de la celda", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Estimadores subplano sobre la salida de la red
# ----------------------------------------------------------------------------

def _tres_puntos(fm, f0, fp, log):
    """Desplazamiento del vertice de la parabola por (-1, fm), (0, f0), (1, fp), en [-0.5, 0.5];
    0 si no es un maximo. log: parabola en log (gaussiana)."""
    import torch
    if log:
        e = 1e-3
        fm, f0, fp = torch.log(fm.clamp_min(e)), torch.log(f0.clamp_min(e)), torch.log(fp.clamp_min(e))
    den = fm - 2 * f0 + fp
    d = torch.where(den < 0, 0.5 * (fm - fp) / den, torch.zeros_like(den))
    return d.clamp(-0.5, 0.5)


def detectar_multi(pred, rejilla, cfg_pp):
    """Como postproc.detectar (mismos maximos, ventana, confianza y umbral), con las
    posiciones de cada estimador de ESTIMADORES y diagnosticos del centro de masa."""
    import torch
    import torch.nn.functional as F
    from postproc import masa_gaussiana, _empates
    p = pred.detach().float()
    rz, rxy = int(cfg_pp.radio_z_planos), int(cfg_pp.radio_xy_vox)
    cz, cxy = int(cfg_pp.radio_cm_z_planos), int(cfg_pp.radio_cm_xy_vox)
    n_ventana = (2 * cz + 1) * (2 * cxy + 1) ** 2
    masa_ref = masa_gaussiana(rejilla, cfg_pp)
    pico_min = cfg_pp.umbral * (rejilla.escala if cfg_pp.confianza == "pico" else masa_ref / n_ventana)
    maximo = F.max_pool3d(p[:, None], (2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1), stride=1, padding=(rz, rxy, rxy))[:, 0]
    cand = (p == maximo) & (p > 0) & (p >= pico_min)
    b, k, u, v = cand.nonzero(as_tuple=True)
    pico = p[b, k, u, v]
    queda = ~_empates(b, k, u, v, pico, rz, rxy)
    b, k, u, v, pico = b[queda], k[queda], u[queda], v[queda], pico[queda]
    pp = F.pad(p, (cxy, cxy, cxy, cxy, cz, cz))

    def ventana(rk, rl):
        dk, du, dv = torch.meshgrid(torch.arange(-rk, rk + 1, device=p.device), torch.arange(-rl, rl + 1, device=p.device),
                                    torch.arange(-rl, rl + 1, device=p.device), indexing="ij")
        dk, du, dv = dk.reshape(-1), du.reshape(-1), dv.reshape(-1)
        w = pp[b[:, None], k[:, None] + cz + dk, u[:, None] + cxy + du, v[:, None] + cxy + dv]
        return w, dk, du, dv

    w, dk, du, dv = ventana(cz, cxy)
    masa = w.sum(1)
    conf = pico / rejilla.escala if cfg_pp.confianza == "pico" else masa / masa_ref
    pos = {"cm": (k + (w * dk).sum(1) / masa, u + (w * du).sum(1) / masa, v + (w * dv).sum(1) / masa)}
    w1, dk1, du1, dv1 = ventana(1, 1)
    m1 = w1.sum(1)
    pos["cm1"] = (k + (w1 * dk1).sum(1) / m1, u + (w1 * du1).sum(1) / m1, v + (w1 * dv1).sum(1) / m1)
    f = lambda a, bb, c: pp[b, k + cz + a, u + cxy + bb, v + cxy + c]   # noqa: E731
    f0 = f(0, 0, 0)
    for nombre, log in (("parabola", False), ("gauss", True)):
        pos[nombre] = (k + _tres_puntos(f(-1, 0, 0), f0, f(1, 0, 0), log), u + _tres_puntos(f(0, -1, 0), f0, f(0, 1, 0), log),
                       v + _tres_puntos(f(0, 0, -1), f0, f(0, 0, 1), log))
    # diagnosticos del cm en z: maximo saturado, vecinos en z (misma columna) nulos, sum(w dk) == 0
    col = torch.stack([f(d, 0, 0) for d in range(-cz, cz + 1) if d != 0], 1)
    diag = dict(saturado=(pico >= rejilla.escala - 1e-3), vecinos_z_nulos=(col == 0).sum(1),
                cm_en_plano=((w * dk).sum(1) == 0), borde_z=((k < cz) | (k >= p.shape[1] - cz)))
    sel = conf >= cfg_pp.umbral
    out = {}
    for nombre, (kc, uc, vc) in pos.items():
        x, y, z = rejilla.desde_voxel(vc[sel].double(), uc[sel].double(), kc[sel].double())
        out[nombre] = pd.DataFrame({"b": b[sel].cpu().numpy(), "x_px": x.cpu().numpy(), "y_px": y.cpu().numpy(),
                                    "z_nm": z.cpu().numpy(), "conf": conf[sel].double().cpu().numpy(),
                                    **{c: t[sel].cpu().numpy() for c, t in diag.items()}})
    return out


def correr_red(var, dispositivo, carpeta_test=None):
    """{estimador: detecciones (video_id, frame, x_px, y_px, z_nm, conf, diagnosticos)} del test de la variante."""
    from utils import cargar_modelo, rejilla_de
    from infer import leer_entradas, camara_de, predecir, batch_por_defecto
    modelo, cfg, _ = cargar_modelo(CORRIDAS / CORRIDA[var] / "mejor.pt", dispositivo)
    e = leer_entradas([str(carpeta_test or DATOS / f"test_celula_b45_{var}")])[0]
    cs = camara_de(e, cfg, avisar=lambda m: None)
    T, H, W = e["imagenes"].shape
    rejilla = rejilla_de(cfg, H, W)
    batch = batch_por_defecto(H, W)
    partes = {n: [] for n in ESTIMADORES}
    for i0 in range(0, T, batch):
        geo = e["geometria"].iloc[i0:i0 + batch] if e["geometria"] is not None else None
        out = detectar_multi(predecir(modelo, e["imagenes"][i0:i0 + batch], cfg, cs, dispositivo, "fp16", geo), rejilla,
                             cfg.postproc)
        for n, d in out.items():
            d["b"] += i0
            partes[n].append(d)
    dets = {}
    for n, ps in partes.items():
        d = pd.concat(ps, ignore_index=True)
        t = d.pop("b").to_numpy()
        d.insert(0, "video_id", e["video_id"][t])
        d.insert(1, "frame", e["frame"][t])
        dets[n] = d
    return dets, e, cs


def tabla_fotones(g, var, est):
    filas = []
    k = np.clip(np.digitize(g.photons, BORDES_FOT[1:-1]), 0, len(BORDES_FOT) - 2)
    r = lambda a: float(np.sqrt(np.mean(np.square(a)))) if len(a) else np.nan   # noqa: E731
    for sub, base in (("todos", g.en_fov), ("aislados", g.en_fov & g.aislado)):
        for i in range(len(BORDES_FOT) - 1):
            s = g[base & (k == i)]
            d = s[s.detectado]
            filas.append(dict(variante=var, estimador=est, subconjunto=sub, fotones_desde=BORDES_FOT[i],
                              fotones_hasta=BORDES_FOT[i + 1], n=len(s), recall=len(d) / len(s) if len(s) else np.nan,
                              rmse_z=r(d.dz_nm), rmse_x=r(d.dx_nm), rmse_y=r(d.dy_nm), rmse_lat=r(np.hypot(d.dx_nm, d.dy_nm)),
                              sesgo_z=float(d.dz_nm.mean()) if len(d) else np.nan))
    return filas


def subplano(a, salida):
    import diagnostico_celula as dc
    from utils import elegir_dispositivo
    from evaluate import ground_truth
    from metricas import evaluar, resumen
    dispositivo = elegir_dispositivo(a.dispositivo)
    filas, sesgos, cm_plano = [], [], []
    for var in a.variantes:
        dets, e, cs = correr_red(var, dispositivo)
        gt = ground_truth(e)
        dv, dr = dc.vecinos(gt, cs.pixel_nm)
        gt = gt.assign(aislado=(dv > AISLADO_NM) & (dr > AISLADO_NM))
        ref = yaml_resumen(var)
        for est, d in dets.items():
            ge, pe = evaluar(d, gt, cs.pixel_nm)
            r = resumen(ge, pe)
            if est == "cm":                              # reproduce evaluate.py (la de Hypatia: misma red, otra GPU)
                print(f"{var}: cm reproduce la evaluacion: TP {r['tp']} (Hypatia {ref['tp']}), FP {r['fp']} ({ref['fp']}), "
                      f"RMSE z {r['rmse_z_nm']:.2f} ({ref['rmse_z_nm']:.2f}) nm")
                assert abs(r["tp"] - ref["tp"]) <= 5 and abs(r["rmse_z_nm"] - ref["rmse_z_nm"]) < 0.5
                dd = d.assign(tp=pe.tp.to_numpy())
                for nombre, s in (("todas", dd), ("TP", dd[dd.tp])):
                    cm_plano.append(dict(variante=var, detecciones=nombre, n=len(s), cm_en_plano=s.cm_en_plano.mean(),
                                         saturado=s.saturado.mean(), todos_vecinos_z_nulos=(s.vecinos_z_nulos == 4).mean(),
                                         vecinos_z_nulos_medio=s.vecinos_z_nulos.mean(), borde_z=s.borde_z.mean(),
                                         en_plano_y_vecinos_nulos=(s.cm_en_plano & (s.vecinos_z_nulos == 4)).mean(),
                                         en_plano_y_saturado=(s.cm_en_plano & s.saturado).mean()))
            filas += tabla_fotones(ge, var, est)
            print(f"  {var} {est:9s}: TP {r['tp']}, FP {r['fp']}, RMSE z {r['rmse_z_nm']:.1f}, x {r['rmse_x_nm']:.2f}, "
                  f"y {r['rmse_y_nm']:.2f} nm")
            alto = ge.en_fov & ge.detectado & (ge.photons > FOTONES_ALTOS) & ge.aislado
            sesgos.append(sesgo_por_fase(ge[alto]).assign(variante=var, estimador=est))
            if est == "cm":
                sesgos.append(sesgo_por_fase(ge[ge.en_fov & ge.detectado & (ge.photons > 8380)])
                              .assign(variante=var, estimador="cm, > 8380 fotones"))
    t = pd.DataFrame(filas)
    t.to_csv(salida / "cuantizacion_subplano.csv", index=False)
    sz = pd.concat(sesgos, ignore_index=True)
    sz.to_csv(salida / "cuantizacion_subplano_sesgo.csv", index=False)
    cp = pd.DataFrame(cm_plano)
    cp.to_csv(salida / "cuantizacion_cm_plano.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\ncentro de masa exactamente en un plano (sum w dk = 0) y por que:")
        print(cp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        for sub in ("todos", "aislados"):
            print(f"\nRMSE z / x / y por fotones ({sub}):")
            s = t[t.subconjunto == sub]
            print(s.pivot_table(index=["variante", "fotones_desde"], columns="estimador", values=["rmse_z", "rmse_x"])
                  .round(1).to_string())
    figura_subplano(t, sz, salida / "fig_cuantizacion_subplano.png")


def yaml_resumen(var):
    """Metricas de resumen.yaml de la evaluacion de Hypatia de la variante."""
    import yaml
    e = CORRIDAS / CORRIDA[var] / f"evaluacion_test_celula_b45_{var}"
    return yaml.safe_load(open(e / "resumen.yaml"))["metricas"]


def figura_subplano(t, sz, ruta):
    plt = _estilo()
    vars_ = list(dict.fromkeys(t.variante))
    fig, axes = plt.subplots(len(vars_), 4, figsize=(15, 3.6 * len(vars_)), squeeze=False)
    for i, var in enumerate(vars_):
        for j, (col, etiq, cota) in enumerate((("rmse_z", "RMSE z [nm]", COTA_Z), ("rmse_x", "RMSE x [nm]", COTA_LAT))):
            ax = axes[i, j]
            for est in ESTIMADORES:
                for sub, ls, mk in (("todos", "-", "o"), ("aislados", "--", "s")):
                    s = t[(t.variante == var) & (t.estimador == est) & (t.subconjunto == sub)]
                    x = np.sqrt(s.fotones_desde * s.fotones_hasta)
                    ax.plot(x, s[col], ls, marker=mk, ms=3, color=COLOR_EST[est], lw=1.4,
                            label=f"{est} ({sub})" if j == 0 else None)
            ax.axhline(cota, color="#8a8984", lw=0.9, ls=":", label=f"redondeo {cota:.1f} nm" if j == 0 else None)
            ax.set_xscale("log")
            ax.set_yscale("log")
            from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
            for eje in (ax.xaxis, ax.yaxis):
                eje.set_major_locator(LogLocator(subs=(1, 2, 5)))
                eje.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                eje.set_minor_formatter(NullFormatter())
            ax.set_xlabel("fotones del emisor")
            ax.set_ylabel(etiq)
            ax.set_title(f"red {var}", fontsize=9)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            if j == 0:
                ax.legend(frameon=False, fontsize=6, ncol=2)
        for j, eje in enumerate(("z", "x")):
            ax = axes[i, 2 + j]
            paso = DZ if eje == "z" else NODO
            for est in ESTIMADORES:
                s = sz[(sz.variante == var) & (sz.estimador == est) & (sz.eje == eje)]
                ax.plot(s.desplazamiento_nm, s.sesgo_nm, "o-", color=COLOR_EST[est], lw=1.4, ms=3, label=est)
            xx = np.array([-paso / 2, paso / 2])
            ax.plot(xx, -xx, ":", color="#52514e", lw=1, label="redondeo al nodo")
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.set_xlabel(f"{eje} real - {'plano' if eje == 'z' else 'nodo'} [nm]")
            ax.set_ylabel(f"error medio en {eje} [nm]")
            ax.set_title(f"red {var}: > 19600 fotones, aislados", fontsize=9)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Estimadores subplano sobre la misma salida de la red y las mismas detecciones (cm = el actual)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("modo", choices=("fases", "subplano"))
    p.add_argument("--variantes", nargs="+", default=["cerca", "lejos"], choices=list(CORRIDA))
    p.add_argument("--transferencia", default=str(Path(__file__).resolve().parent / "red_celula" / "red_celula_transferencia_imagenes.csv"))
    p.add_argument("--agujero", default=str(Path(__file__).resolve().parent / "red_celula" / "red_celula_agujero_imagenes.csv"))
    p.add_argument("--dispositivo", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--salida", default=str(Path(__file__).resolve().parent / "cuantizacion"))
    a = p.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    if a.modo == "fases":
        fases(a, salida)
    else:
        subplano(a, salida)


if __name__ == "__main__":
    main()
