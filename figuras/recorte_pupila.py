"""
Recorte de pupila del reflejo (simulador_nuevo, recorte_pupila_reflejo): cuanto
pesa y que hace con la z. Convencion del repo: +z hacia el objetivo, beta desde
el eje optico (45 por defecto), m = direccion linea del espejo -> muestra. El
modelo (espejo plano infinito): el reflejo solo recibe los rayos con w . n > 0,
n = cos(beta) m + sin(beta) z; la PSF recortada no se renormaliza.

Partes (--partes, por defecto todas):
  perdida      luz perdida vs beta: onda (fraccion de la pupila de PSFRecortada,
               = integral de la mascara por Parseval) frente a optica geometrica
               con pupila uniforme (la del simulador) y aplanatica (1 / cos(theta)).
  forma        por z' y variante (lejos / cerca = anillos_hacia_objetivo False /
               True; y "ninguna" como referencia geometrica): corrimiento del
               centroide a lo largo de m (rejilla completa y centroide iterativo
               r = 1 um como en los datos reales), asimetria (skewness del perfil
               en m), cociente de anchos m / t, diferencia L2 relativa y luz en la
               apertura de 1.5 um. Figura de PSF completa / recortada.
  sesgo        ajuste MLE (Poisson, x, y, z', N, b libres, ROI 31 x 31 px) del
               reflejo recortado (sin ruido: el sesgo asintotico) con el modelo
               de PSF completa, como haria un localizador que no lo modela. El
               sesgo lateral a lo largo de m entra en la z triangulada como
               dz = -dm / sin(2 beta) (exacto: m y z' del reflejo dependen solo
               de d; ver resultados_calibracion_espejo.md); beta 40 / 45 / 50.
  crlb         CRLB conjunta de z (real + reflejo, figuras/crlb_espejo.py) con el
               reflejo completo y recortado y con real y reflejo recortados
               (modelo correcto), frente al emisor solo, a beta = 45, por
               distancia al espejo.
  calibracion  la fotometria de los datos reales (disco r = 1.5 um, fondo =
               mediana del anillo [r+3, r+10] px sin el otro punto, centros por
               centroide iterativo; copia de figuras/analisis_datos_reales.py)
               sobre imagenes sin ruido con ratio_reflejo = 1 y real y reflejo
               recortados (tambien sin recorte y solo el reflejo), real a z en
               [-1.5, 1.5] um y la separacion medida (4.5 um): Q1 = cociente
               medido por unidad de ratio. ratio = Q_medido / Q1.
  red          (con --red SIN CON, repetible) compara dos evaluaciones de
               evaluate.py de la misma red sobre el mismo test sin y con recorte
               (mismos emisores, pareado): recall, RMSE y sesgo de z, por reflejo
               en el FOV y por distancia al espejo.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/recorte_pupila.py                                  # ~5 min
    $PY figuras/recorte_pupila.py --partes perdida forma
    $PY figuras/recorte_pupila.py --partes --red <corrida>/evaluacion_<sin> <corrida>/evaluacion_<con> [--red ...]
Escribe en figuras/: fig_recorte_pupila.png/.pdf, fig_recorte_pupila_psf.png/.pdf y
recorte_pupila_<parte>.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "simulador_nuevo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulador as sim  # noqa: E402
import crlb_espejo as ce  # noqa: E402

SALIDA = Path(__file__).resolve().parent
VARIANTES = {"lejos": False, "cerca": True}
BETAS_PERDIDA = np.arange(0.0, 60.01, 1.0)
ZP = np.arange(-5500.0, 1500.1, 250.0)            # z' del reflejo [nm]
ZP_FIGURA = (-4500.0, -2500.0, -1000.0, 0.0)
BETAS_SESGO = (40.0, 45.0, 50.0)
N_SESGO, B_SESGO = 5000.0, 20.0                   # fotones (del reflejo) y fondo del ajuste
ROIS_SESGO = (15, 30)                             # medio lado de la ROI [px]: 31 y 61 px
D_CRLB = np.arange(0.0, 4000.1, 250.0)
Z_CRLB = (-800.0, 0.0, 800.0)
N_CRLB, B_CRLB = (2000.0, 5000.0, 20000.0), 20.0
SEPARACION_UM = 4.5                               # real - reflejo medida (resultados_datos_reales.md)
Z_REAL_CAL = np.arange(-2500.0, 2500.1, 100.0)
R_FOT_UM = 1.5
# cociente reflejo/real medido por radio (resultados_datos_reales.md, seccion 3, fondo anillo): cociente por
# cuadro y regresion a lo largo del fotoblanqueo. El canal derecho es el robusto (0.42-0.46 a 1.5 um)
RADIOS_CAL_UM = (0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2)
Q_MEDIDOS = {"der, cociente": (0.70, 0.63, 0.52, 0.43, 0.41, 0.43, 0.45),
             "der, regresion": (0.81, 0.69, 0.53, 0.42, 0.35, 0.37, 0.39),
             "izq, cociente": (1.00, 0.79, 0.62, 0.53, 0.51, 0.51, 0.52),
             "izq, regresion": (0.89, 0.78, 0.50, 0.37, 0.29, 0.31, 0.38)}
R_CENTROIDE = 10
COLOR = {"lejos": "#2a78d6", "cerca": "#eb6834", "ninguna": "#52514e", "uniforme": "#1baf7a",
         "aplanatica": "#eda100", "solo": "#52514e", "completo": "#2a78d6", "recortado": "#eb6834",
         "ambos": "#1baf7a"}
ESTILO_BETA = {40.0: ":", 45.0: "-", 50.0: "--"}


def cfg_base(variante, mascara="esferica", **kw):
    return sim.Config(usar_espejo=True, recorte_pupila_reflejo=True, mascara=mascara, orientacion_espejo_deg=0.0,
                      anillos_hacia_objetivo=VARIANTES.get(variante, False), **kw)


# ----------------------------------------------------------------------------
# fotometria de los datos reales (copia de figuras/analisis_datos_reales.py)
# ----------------------------------------------------------------------------

def ventana(img, x, y, r):
    H, W = img.shape
    x0, x1 = max(int(round(x)) - r, 0), min(int(round(x)) + r + 1, W)
    y0, y1 = max(int(round(y)) - r, 0), min(int(round(y)) + r + 1, H)
    Y, X = np.mgrid[y0:y1, x0:x1]
    return img[y0:y1, x0:x1], X, Y


def centroide(img, x, y, r=R_CENTROIDE, iters=20):
    """Centroide iterativo con fondo = mediana del anillo [r+3, r+8]."""
    for _ in range(iters):
        sub, X, Y = ventana(img, x, y, r + 9)
        d2 = (X - x) ** 2 + (Y - y) ** 2
        fondo = np.median(sub[(d2 > (r + 3) ** 2) & (d2 <= (r + 8) ** 2)])
        w = np.clip(sub - fondo, 0, None) * (d2 <= r * r)
        s = w.sum()
        if s <= 0:
            break
        xn, yn = (w * X).sum() / s, (w * Y).sum() / s
        hecho = np.hypot(xn - x, yn - y) < 1e-3
        x, y = xn, yn
        if hecho:
            break
    return x, y


def fotometria(img, c, otro, r):
    """Disco de radio r [px] en c menos la mediana del anillo [r+3, r+10] sin los pixeles a < r+3 del otro punto."""
    sub, X, Y = ventana(img, c[0], c[1], int(r) + 12)
    d2 = (X - c[0]) ** 2 + (Y - c[1]) ** 2
    disco = d2 <= r * r
    lejos = (X - otro[0]) ** 2 + (Y - otro[1]) ** 2 > (r + 3) ** 2
    anillo = (d2 > (r + 3) ** 2) & (d2 <= (r + 10) ** 2) & lejos
    return (sub[disco] - np.median(sub[anillo])).sum()


# ----------------------------------------------------------------------------
# 1. perdida de luz vs beta
# ----------------------------------------------------------------------------

def fraccion_geometrica(beta_deg, aplanatica, n=1500):
    rho = (np.arange(n) + 0.5) / n
    psi = (np.arange(2 * n) + 0.5) / (2 * n) * 2 * np.pi
    R, P = np.meshgrid(rho, psi)
    st = 1.1 / 1.333 * R
    ct = np.sqrt(1 - st ** 2)
    b = np.radians(beta_deg)
    peso = R / ct if aplanatica else R
    return float(((np.cos(b) * st * np.cos(P) + np.sin(b) * ct > 0) * peso).sum() / peso.sum())


def parte_perdida():
    filas = []
    for b in BETAS_PERDIDA:
        f = sim.PSFRecortada(cfg_base("lejos", angulo_espejo_eje_deg=b), 32).fraccion
        filas.append(dict(beta_deg=b, perdida_onda=1 - f, perdida_geo_uniforme=1 - fraccion_geometrica(b, False),
                          perdida_geo_aplanatica=1 - fraccion_geometrica(b, True)))
    t = pd.DataFrame(filas)
    print("\n[perdida] luz del reflejo que no llega al objetivo (espejo infinito)")
    print("  beta   onda    geo uniforme   geo aplanatica")
    for b in (0, 20, 30, 35, 40, 45, 50, 55):
        f = t[t.beta_deg == b].iloc[0]
        print(f"  {b:4.0f}  {100*f.perdida_onda:5.2f}%   {100*f.perdida_geo_uniforme:5.2f}%        "
              f"{100*f.perdida_geo_aplanatica:5.2f}%")
    return t


# ----------------------------------------------------------------------------
# 2. forma y centroide vs z'
# ----------------------------------------------------------------------------

def momentos(psf, cx, cy, mascara=None):
    M = psf.shape[0]
    Y, X = np.mgrid[0:M, 0:M]
    w = psf if mascara is None else psf * mascara
    s = w.sum()
    mx, my = (w * X).sum() / s, (w * Y).sum() / s
    vx, vy = (w * (X - mx) ** 2).sum() / s, (w * (Y - my) ** 2).sum() / s
    sk = (w * (X - mx) ** 3).sum() / s / vx ** 1.5
    return mx - cx, my - cy, np.sqrt(vx), np.sqrt(vy), sk


def parte_forma():
    filas, imagenes = [], {}
    for variante, masc in (("lejos", "esferica"), ("cerca", "esferica"), ("ninguna", "ninguna")):
        c = cfg_base(variante, masc)
        px, M = c.pixel_nm, c.M_pupila
        cc = M // 2
        Y, X = np.mgrid[0:M, 0:M]
        r_fot = R_FOT_UM * 1000 / px
        disco = (X - cc) ** 2 + (Y - cc) ** 2 <= r_fot ** 2
        for zp in ZP:
            a = sim.psf_directa(c, zp, recorte=True)
            b = sim.psf_directa(c, zp)
            f = a.sum()
            ma, mb = momentos(a, cc, cc), momentos(b, cc, cc)
            ap, bp = momentos(a, cc, cc, disco), momentos(b, cc, cc, disco)
            xa, ya = centroide(a, cc, cc)
            xb, yb = centroide(b, cc, cc)
            filas.append(dict(variante=variante, z_reflejo_nm=zp, fraccion=f,
                              dm_completo_nm=(ma[0] - mb[0]) * px, dt_completo_nm=(ma[1] - mb[1]) * px,
                              dm_centroide_nm=(xa - xb) * px, dt_centroide_nm=(ya - yb) * px,
                              skew_m_rec=ap[4], skew_m_completa=bp[4],
                              anchos_m_t_rec=ap[2] / ap[3], anchos_m_t_completa=bp[2] / bp[3],
                              dif_l2=np.linalg.norm(a / f - b) / np.linalg.norm(b),
                              luz_apertura_rec=(a * disco).sum(), luz_apertura_completa=(b * disco).sum()))
            if zp in ZP_FIGURA and variante != "ninguna":
                imagenes[(variante, zp)] = (b[cc - 40: cc + 41, cc - 40: cc + 41], a[cc - 40: cc + 41, cc - 40: cc + 41])
    t = pd.DataFrame(filas)
    print("\n[forma] reflejo recortado vs completo, beta 45 (m = +x): corrimiento del centroide a lo largo de m [nm]")
    print("        (rejilla completa | centroide iterativo r = 1 um), skewness en m (r 1.5 um), luz en r = 1.5 um")
    for v in ("lejos", "cerca", "ninguna"):
        print(f"  {v}:")
        for zp in (-5500, -4500, -3500, -2500, -1500, -500, 0, 500, 1500):
            f = t[(t.variante == v) & (t.z_reflejo_nm == zp)].iloc[0]
            print(f"    z' {zp:+6.0f}: dm {f.dm_completo_nm:+6.0f} | {f.dm_centroide_nm:+6.0f} nm, skew {f.skew_m_rec:+.3f}"
                  f" (completa {f.skew_m_completa:+.3f}), anchos m/t {f.anchos_m_t_rec:.3f}, dif L2 "
                  f"{100*f.dif_l2:4.1f}%, luz r1.5 {f.luz_apertura_rec:.3f} / {f.luz_apertura_completa:.3f}")
    return t, imagenes


# ----------------------------------------------------------------------------
# 3. sesgo del ajuste con PSF completa y de la z triangulada
# ----------------------------------------------------------------------------

def banco_ajuste(variante):
    c = sim.Config(mascara="esferica", anillos_hacia_objetivo=VARIANTES[variante], z_rango_nm=(-800.0, 800.0),
                   z_banco_nm=(ZP.min() - 1000.0, ZP.max() + 1000.0), K_psf=128)
    return sim.BancoPSF(c)


def ajustar(banco, obs, x0, y0, z0, N0, b0):
    """MLE Poisson (x, y, z', N, b) de obs con el banco (PSF completa); arranca en el valor verdadero."""
    forma = obs.shape
    zg = banco.z_grid

    def modelo(p):
        x, y, z, lN, b = p
        img = np.full(forma, b)
        banco.colocar(img, x, y, float(np.clip(z * 100.0, zg[0], zg[-1])), np.exp(lN))
        return img

    def nll(p):
        mu = np.clip(modelo(p), 1e-9, None)
        return float((mu - obs * np.log(mu)).sum())

    p0 = np.array([x0, y0, z0 / 100.0, np.log(N0), b0])
    r = minimize(nll, p0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-9, maxiter=20000, maxfev=20000))
    r = minimize(nll, r.x, method="Powell", options=dict(xtol=1e-6, ftol=1e-12, maxiter=20000))
    x, y, z, lN, b = r.x
    return x, y, z * 100.0, np.exp(lN), b


def parte_sesgo():
    filas = []
    for variante in VARIANTES:
        banco = banco_ajuste(variante)
        for beta, roi in [(b, r) for b in BETAS_SESGO for r in ROIS_SESGO]:
            if roi != ROIS_SESGO[0] and beta != 45.0:
                continue
            L = 2 * roi + 1
            x0 = y0 = roi + 0.3
            c = cfg_base(variante, angulo_espejo_eje_deg=beta)
            pr = sim.PSFRecortada(c, banco.K)
            for zp in ZP:
                obs = np.full((L, L), B_SESGO)
                pr.colocar(obs, x0, y0, zp, N_SESGO)
                x, y, z, N, b = ajustar(banco, obs, x0, y0, zp, N_SESGO * pr.fraccion, B_SESGO)
                ref = np.full((L, L), B_SESGO)                    # control: la PSF completa se recupera sin sesgo
                banco.colocar(ref, x0, y0, zp, N_SESGO)
                xr, yr, zr, _, _ = ajustar(banco, ref, x0, y0, zp, N_SESGO, B_SESGO)
                dm = (x - x0) * c.pixel_nm
                filas.append(dict(variante=variante, beta_deg=beta, roi_px=L, z_reflejo_nm=zp, dm_nm=dm,
                                  dt_nm=(y - y0) * c.pixel_nm, dz_reflejo_nm=z - zp, N_rel=N / N_SESGO,
                                  db=b - B_SESGO, dz_triangulada_nm=-dm / np.sin(2 * np.radians(beta)),
                                  control_dm_nm=(xr - x0) * c.pixel_nm, control_dz_nm=zr - zp))
        print(f"  [sesgo] {variante} listo", flush=True)
    t = pd.DataFrame(filas)
    print(f"\n[sesgo] MLE con la PSF completa del reflejo recortado ({N_SESGO:.0f} fotones, fondo {B_SESGO:.0f}): "
          "dm lateral, dz' del ajuste y dz de la triangulacion = -dm / sin(2 beta) [nm]")
    print(f"  control (PSF completa): |dm| max {t.control_dm_nm.abs().max():.2f} nm, |dz'| max "
          f"{t.control_dz_nm.abs().max():.1f} nm")
    for v in VARIANTES:
        for (beta, roi), s in t[t.variante == v].groupby(["beta_deg", "roi_px"]):
            txt = ", ".join(f"{f.z_reflejo_nm:+.0f}: {f.dz_triangulada_nm:+.0f} ({f.dz_reflejo_nm:+.0f})"
                            for f in s.itertuples() if f.z_reflejo_nm in (-5500, -4500, -3500, -2500, -1500, -500, 500))
            print(f"  {v} beta {beta:.0f} ROI {roi}: z' -> dz_tri (dz'): {txt}")
    return t


# ----------------------------------------------------------------------------
# 4. CRLB conjunta con el reflejo recortado
# ----------------------------------------------------------------------------

class _Recortado:
    """Adaptador de PSFRecortada para crlb_espejo.derivadas (paso en z de 10 nm)."""

    def __init__(self, pr):
        self.colocar = pr.colocar
        self.z_grid = np.array([-1e6, -1e6 + 20.0, 1e6])


def parte_crlb():
    filas = []
    for variante in VARIANTES:
        c = sim.Config(mascara="esferica", anillos_hacia_objetivo=VARIANTES[variante], z_rango_nm=(-800.0, 800.0),
                       z_banco_nm=(-7000.0, 1000.0))
        banco = sim.BancoPSF(c)
        pr = sim.PSFRecortada(cfg_base(variante), banco.K)
        rec = _Recortado(pr)
        v = sim._var_lectura_e2(c)
        geo = ce.geometria_canonica()
        assert np.isclose(np.degrees(geo.beta), 45.0)
        T = ce.transformacion_px(geo, c.pixel_nm)
        for d in D_CRLB:
            for z in Z_CRLB:
                p, prf = ce.real_y_reflejo(geo, d, z)
                forma, (cx, cy), (rx, ry) = ce.roi_conjunta(p, prf, c.pixel_nm)
                g1, G1 = ce.derivadas(banco, cx, cy, z, forma)
                g2, G2 = ce.derivadas(banco, rx, ry, prf[2], forma)
                g3, G3 = ce.derivadas(rec, rx, ry, prf[2], forma)
                g4, G4 = ce.derivadas(rec, cx, cy, z, forma)              # real recortado
                for N in N_CRLB:
                    vs = ce.var_parametros(g1, G1, N, B_CRLB, v)
                    vc = ce.var_parametros(g1 + ce.RATIO * g2, G1 + ce.RATIO * (T.T @ G2), N, B_CRLB, v)
                    vr = ce.var_parametros(g1 + ce.RATIO * g3, G1 + ce.RATIO * (T.T @ G3), N, B_CRLB, v)
                    va = ce.var_parametros(g4 + ce.RATIO * g3, G4 + ce.RATIO * (T.T @ G3), N, B_CRLB, v)
                    vsr = ce.var_parametros(g4, G4, N, B_CRLB, v)
                    filas.append(dict(variante=variante, d_nm=d, z_nm=z, z_reflejo_nm=prf[2], fotones=N,
                                      crlb_z_solo_nm=np.sqrt(vs[2]), crlb_z_completo_nm=np.sqrt(vc[2]),
                                      crlb_z_recortado_nm=np.sqrt(vr[2]), crlb_z_ambos_nm=np.sqrt(va[2]),
                                      crlb_z_solo_recortado_nm=np.sqrt(vsr[2])))
        print(f"  [crlb] {variante} listo", flush=True)
    t = pd.DataFrame(filas)
    print(f"\n[crlb] sqrt(CRLB z) [nm], RMS sobre z en {Z_CRLB}, ratio {ce.RATIO}, fondo {B_CRLB:.0f}, beta 45: "
          "solo | conjunto con reflejo completo | con reflejo recortado | real y reflejo recortados (solo recortado)")
    for v in VARIANTES:
        for N in N_CRLB:
            s = t[(t.variante == v) & (t.fotones == N)]
            g = s.groupby("d_nm")[["crlb_z_solo_nm", "crlb_z_completo_nm", "crlb_z_recortado_nm", "crlb_z_ambos_nm",
                                   "crlb_z_solo_recortado_nm"]].apply(lambda x: np.sqrt((x ** 2).mean()))
            txt = ", ".join(f"{d/1000:.1f}: {r.crlb_z_solo_nm:.0f}|{r.crlb_z_completo_nm:.0f}|{r.crlb_z_recortado_nm:.0f}"
                            f"|{r.crlb_z_ambos_nm:.0f} ({r.crlb_z_solo_recortado_nm:.0f})"
                            for d, r in g.iterrows() if d in (0, 500, 1000, 2000, 3000, 4000))
            print(f"  {v} {N:.0f} fot, d [um]: {txt}")
    return t


# ----------------------------------------------------------------------------
# 5. calibracion del ratio por fotometria
# ----------------------------------------------------------------------------

CASOS_CAL = ("sin", "reflejo", "ambos")                 # recortes: ninguno, el del reflejo, los dos


def col_q1(caso, r_um):
    return f"Q1_{caso}_r{round(r_um * 1000)}nm"


def parte_calibracion():
    """Q1(r, z_real): cociente de apertura por unidad de ratio_reflejo, sin
    recorte, con el del reflejo y con el de los dos (el modelo:
    recorte_pupila_reflejo y recorte_pupila_real). Despues, por canal y estimador, el ajuste de la curva medida
    Q(r) = rho Q1(r, z_real) (minimos cuadrados en r, rho cerrado) da z_real y
    rho = ratio_reflejo calibrado: la forma de Q(r) fija z_real."""
    filas = []
    W = 260
    for variante in VARIANTES:
        c = sim.Config(mascara="esferica", anillos_hacia_objetivo=VARIANTES[variante], z_rango_nm=(-2500.0, 2500.0),
                       z_banco_nm=(-9000.0, 2500.0), z_paso_nm=50.0, K_psf=256, H=W, W=W)
        banco = sim.BancoPSF(c)
        px = c.pixel_nm
        for beta in BETAS_SESGO:
            pr = sim.PSFRecortada(cfg_base(variante, angulo_espejo_eje_deg=beta), 256)
            d = SEPARACION_UM * 1000 / (2 * np.cos(np.radians(beta)))
            for z in Z_REAL_CAL:
                zp = z - 2 * d * np.sin(np.radians(beta))
                xr_, yr_ = W / 2 + SEPARACION_UM * 500 / px + 0.3, W / 2 + 0.4      # real (+m)
                xf_, yf_ = xr_ - SEPARACION_UM * 1000 / px, yr_                    # reflejo, hacia -m
                fila = dict(variante=variante, beta_deg=beta, z_real_nm=z, z_reflejo_nm=zp, d_nm=d, fraccion=pr.fraccion)
                for caso in CASOS_CAL:
                    img = np.full((W, W), 100.0)
                    (pr.colocar if caso == "ambos" else banco.colocar)(img, xr_, yr_, z, 1e4)
                    (banco.colocar if caso == "sin" else pr.colocar)(img, xf_, yf_, zp, 1e4)
                    cr = centroide(img, xr_, yr_)
                    cf = centroide(img, xf_, yf_)
                    for r in RADIOS_CAL_UM:
                        rp = r * 1000 / px
                        fila[col_q1(caso, r)] = fotometria(img, cf, cr, rp) / fotometria(img, cr, cf, rp)
                filas.append(fila)
        print(f"  [calibracion] {variante} listo", flush=True)
    t = pd.DataFrame(filas)
    print(f"\n[calibracion] Q1 = cociente de apertura (fondo anillo) por unidad de ratio_reflejo a r = {R_FOT_UM} um, "
          f"separacion {SEPARACION_UM} um, con los dos recortes (solo el del reflejo; sin recorte)")
    for v in VARIANTES:
        for beta in BETAS_SESGO:
            s = t[(t.variante == v) & (t.beta_deg == beta)]
            txt = ", ".join(f"{f.z_real_nm/1000:+.1f}: {getattr(f, col_q1('ambos', R_FOT_UM)):.2f} "
                            f"({getattr(f, col_q1('reflejo', R_FOT_UM)):.2f}; {getattr(f, col_q1('sin', R_FOT_UM)):.2f})"
                            for f in s.itertuples() if f.z_real_nm % 500 == 0)
            print(f"  {v} beta {beta:.0f}, z real [um]: {txt}")
    print("\n  intervalo de rho = Q_medido / Q1 (los dos recortes) con z real en [-1.5, 1.5] um y rho <= 1:")
    sel = t[(t.variante == "cerca") & (t.z_real_nm.abs() <= 1500)]
    for beta in BETAS_SESGO + ("todos",):
        s = sel if beta == "todos" else sel[sel.beta_deg == beta]
        q1 = s[col_q1("ambos", R_FOT_UM)]
        print(f"    cerca beta {beta}: Q1 {q1.min():.3f}-{q1.max():.3f}; canal der (0.42-0.46): rho {0.42 / q1.max():.2f}-"
              f"{min(1.0, 0.46 / q1.min()):.2f}; canal izq (0.37-0.44): rho {0.37 / q1.max():.2f}-{min(1.0, 0.44 / q1.min()):.2f}")
    print("\n  ajuste de la curva Q(r) medida: rho = ratio_reflejo calibrado (los dos recortes), z real, residuo rms;"
          " rho a z real fijo")
    ajustes = []
    for v in VARIANTES:
        for beta in BETAS_SESGO:
            s = t[(t.variante == v) & (t.beta_deg == beta)].reset_index(drop=True)
            Q1 = s[[col_q1("ambos", r) for r in RADIOS_CAL_UM]].to_numpy()                    # (nz, nr)
            for canal, q in Q_MEDIDOS.items():
                q = np.asarray(q)
                rho = (Q1 @ q) / (Q1 ** 2).sum(1)
                res = np.sqrt(((q[None] - rho[:, None] * Q1) ** 2).mean(1))
                k = int(np.argmin(res))
                buenos = res <= 1.5 * res[k]                     # z compatibles (residuo hasta 1.5 veces el minimo)
                ajustes.append(dict(variante=v, beta_deg=beta, canal=canal, z_real_nm=s.z_real_nm[k], rho=rho[k],
                                    residuo=res[k], z_min_nm=s.z_real_nm[buenos].min(), z_max_nm=s.z_real_nm[buenos].max(),
                                    rho_min=rho[buenos].min(), rho_max=rho[buenos].max()))
                f = ajustes[-1]
                print(f"    {v} beta {beta:.0f} {canal:15s}: rho {f['rho']:.2f} ({f['rho_min']:.2f}-{f['rho_max']:.2f}), "
                      f"z real {f['z_real_nm']/1000:+.1f} um ({f['z_min_nm']/1000:+.1f} a {f['z_max_nm']/1000:+.1f}), "
                      f"residuo {f['residuo']:.3f}")
    t.attrs["ajustes"] = pd.DataFrame(ajustes)
    return t


# ----------------------------------------------------------------------------
# 6. red sin recorte sobre datos con recorte
# ----------------------------------------------------------------------------

CLAVES = ["video_id", "frame", "particle_id"]
BORDES_D_RED = np.arange(0.0, 4000.1, 1000.0)


def parte_red(pares):
    """Pareado por emisor (gt_eval.csv de evaluate.py; mismos emisores en los dos
    tests): recall y, sobre los detectados en ambos, RMSE y sesgo de z, por
    reflejo en el FOV o no y por distancia al espejo."""
    filas = []
    print("\n[red] misma red, mismo test sin / con recorte de pupila del reflejo (pareado por emisor)")
    for sin, con in pares:
        a = pd.read_csv(Path(sin) / "gt_eval.csv")
        b = pd.read_csv(Path(con) / "gt_eval.csv")
        g = a.merge(b, on=CLAVES, suffixes=("_sin", "_con"))
        assert len(g) == len(a) == len(b) and np.allclose(g.z_nm_sin, g.z_nm_con)
        g = g[g.en_fov_sin]
        nombre = f"{Path(sin).parent.name}: {Path(sin).name} -> {Path(con).name}"
        grupos = [("todos", np.ones(len(g), bool)), ("reflejo en el FOV", g.reflejo_en_fov_sin.to_numpy()),
                  ("reflejo fuera del FOV", ~g.reflejo_en_fov_sin.to_numpy())]
        for lo, hi in zip(BORDES_D_RED[:-1], BORDES_D_RED[1:]):
            grupos.append((f"reflejo en el FOV, d {lo/1000:.0f}-{hi/1000:.0f} um",
                           (g.reflejo_en_fov_sin & (g.d_espejo_nm_sin >= lo) & (g.d_espejo_nm_sin < hi)).to_numpy()))
        for etiqueta, sel in grupos:
            s = g[sel]
            ambos = s[s.detectado_sin & s.detectado_con]
            dz_s, dz_c = ambos.dz_nm_sin, ambos.dz_nm_con
            filas.append(dict(corrida=Path(sin).parent.name, sin=Path(sin).name, con=Path(con).name, par=nombre,
                              grupo=etiqueta, n=len(s), recall_sin=s.detectado_sin.mean(),
                              recall_con=s.detectado_con.mean(), n_ambos=len(ambos),
                              rmse_z_sin=np.sqrt((dz_s ** 2).mean()), rmse_z_con=np.sqrt((dz_c ** 2).mean()),
                              sesgo_z_sin=dz_s.mean(), sesgo_z_con=dz_c.mean(),
                              dif_dz_media=(dz_c - dz_s).mean(), dif_dz_rms=np.sqrt(((dz_c - dz_s) ** 2).mean()),
                              dif_dz_err=(dz_c - dz_s).std() / np.sqrt(max(len(ambos), 1)),
                              rmse_lat_sin=np.sqrt((ambos.dx_nm_sin ** 2 + ambos.dy_nm_sin ** 2).mean()),
                              rmse_lat_con=np.sqrt((ambos.dx_nm_con ** 2 + ambos.dy_nm_con ** 2).mean())))
        print(f"  {nombre}:")
        for f in filas:
            if f["par"] == nombre:
                print(f"    {f['grupo']:34s} n {f['n']:5d}: recall {f['recall_sin']:.3f} -> {f['recall_con']:.3f}; "
                      f"z (n {f['n_ambos']}) RMSE {f['rmse_z_sin']:.1f} -> {f['rmse_z_con']:.1f} nm, sesgo "
                      f"{f['sesgo_z_sin']:+.1f} -> {f['sesgo_z_con']:+.1f}, dz con - sin {f['dif_dz_media']:+.1f} "
                      f"+- {f['dif_dz_err']:.1f} (rms {f['dif_dz_rms']:.1f}); lateral {f['rmse_lat_sin']:.1f} -> "
                      f"{f['rmse_lat_con']:.1f}")
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# figuras
# ----------------------------------------------------------------------------

def figura(perdida, forma, sesgo, crlb, cal, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    a = axs[0, 0]
    if perdida is not None:
        a.plot(perdida.beta_deg, 100 * perdida.perdida_onda, color=COLOR["lejos"], lw=2, label="onda (simulador)")
        a.plot(perdida.beta_deg, 100 * perdida.perdida_geo_uniforme, color=COLOR["uniforme"], lw=1.5, ls="--",
               label="geometrica, pupila uniforme")
        a.plot(perdida.beta_deg, 100 * perdida.perdida_geo_aplanatica, color=COLOR["aplanatica"], lw=1.5, ls=":",
               label="geometrica, aplanatica")
        a.axvspan(40, 50, color="0.9", zorder=0)
        a.set_xlabel("beta [deg] (desde el eje optico)"), a.set_ylabel("luz perdida del reflejo [%]")
        a.set_title("luz del reflejo que no llega (espejo infinito)"), a.legend(fontsize=8)
    a = axs[0, 1]
    if forma is not None:
        for v in ("lejos", "cerca", "ninguna"):
            s = forma[forma.variante == v]
            a.plot(s.z_reflejo_nm / 1000, s.dm_completo_nm, color=COLOR[v], lw=2, label=f"{v}, rejilla completa")
            a.plot(s.z_reflejo_nm / 1000, s.dm_centroide_nm, color=COLOR[v], lw=1.2, ls="--",
                   label=f"{v}, centroide r = 1 um")
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlabel("z' del reflejo [um]"), a.set_ylabel("corrimiento del centroide a lo largo de m [nm]")
        a.set_title("corrimiento del centroide (beta 45)"), a.legend(fontsize=7)
    a = axs[0, 2]
    if sesgo is not None:
        for v in VARIANTES:
            for beta in BETAS_SESGO:
                s = sesgo[(sesgo.variante == v) & (sesgo.beta_deg == beta) & (sesgo.roi_px == 2 * ROIS_SESGO[0] + 1)]
                a.plot(s.z_reflejo_nm / 1000, s.dz_triangulada_nm, color=COLOR[v], ls=ESTILO_BETA[beta], lw=1.6,
                       label=f"{v}, beta {beta:.0f}")
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlabel("z' del reflejo [um]"), a.set_ylabel("sesgo de la z triangulada [nm]")
        a.set_title(f"sesgo de la z triangulada\n(ajuste con PSF completa, ROI {2 * ROIS_SESGO[0] + 1} px)")
        a.legend(fontsize=7, ncol=2)
    a = axs[1, 0]
    if sesgo is not None:
        for v in VARIANTES:
            for roi, ls in zip(ROIS_SESGO, ("-", "--")):
                L = 2 * roi + 1
                s = sesgo[(sesgo.variante == v) & (sesgo.beta_deg == 45.0) & (sesgo.roi_px == L)]
                a.plot(s.z_reflejo_nm / 1000, s.dz_reflejo_nm, color=COLOR[v], lw=1.6, ls=ls, label=f"{v}, z', ROI {L}")
                a.plot(s.z_reflejo_nm / 1000, s.dz_triangulada_nm, color=COLOR[v], lw=0.9, ls=ls, alpha=0.6,
                       label=f"{v}, z triangulada, ROI {L}")
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlabel("z' del reflejo [um]"), a.set_ylabel("sesgo [nm]")
        a.set_title("sesgo de z' del ajuste y de la z triangulada (beta 45)"), a.legend(fontsize=6.5, ncol=2)
    a = axs[1, 1]
    if crlb is not None:
        for v, ls in (("lejos", "-"), ("cerca", "--")):
            s = crlb[(crlb.variante == v) & (crlb.fotones == 5000.0)]
            g = s.groupby("d_nm")[["crlb_z_solo_nm", "crlb_z_completo_nm", "crlb_z_recortado_nm",
                                   "crlb_z_ambos_nm"]].apply(lambda x: np.sqrt((x ** 2).mean()))
            for col, clave in (("crlb_z_solo_nm", "solo"), ("crlb_z_completo_nm", "completo"),
                               ("crlb_z_recortado_nm", "recortado"), ("crlb_z_ambos_nm", "ambos")):
                a.plot(g.index / 1000, g[col], color=COLOR[clave], ls=ls, lw=1.6, label=f"{v}, {clave}")
        a.set_xlabel("distancia al espejo d [um]"), a.set_ylabel("sqrt(CRLB z) [nm]")
        a.set_title("CRLB z, 5000 fotones, fondo 20, beta 45"), a.legend(fontsize=7, ncol=2)
    a = axs[1, 2]
    if cal is not None:
        for v in VARIANTES:
            for beta in BETAS_SESGO:
                s = cal[(cal.variante == v) & (cal.beta_deg == beta)]
                a.plot(s.z_real_nm / 1000, s[col_q1("ambos", R_FOT_UM)], color=COLOR[v], ls=ESTILO_BETA[beta], lw=1.6,
                       label=f"{v}, beta {beta:.0f}")
        a.axhspan(0.42, 0.46, color="0.88", zorder=0, label="medido, canal der")
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlabel("z del real [um]"), a.set_ylabel("Q1 = cociente de apertura / ratio_reflejo")
        a.set_ylim(-0.5, 1.0)
        a.set_title(f"fotometria r = {R_FOT_UM} um, separacion {SEPARACION_UM} um\n(real y reflejo recortados)")
        a.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(ruta.with_suffix(ext), dpi=200)
    plt.close(fig)


def figura_psf(imagenes, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, len(ZP_FIGURA), figsize=(3 * len(ZP_FIGURA), 12))
    for j, zp in enumerate(ZP_FIGURA):
        for i, (v, cual) in enumerate((("lejos", 0), ("lejos", 1), ("cerca", 0), ("cerca", 1))):
            img = imagenes[(v, zp)][cual]
            a = axs[i, j]
            a.imshow(img ** 0.5, cmap="magma", vmin=0, vmax=imagenes[(v, zp)][0].max() ** 0.5)
            a.axvline(40, color="w", lw=0.4, alpha=0.5), a.axhline(40, color="w", lw=0.4, alpha=0.5)
            a.set_xticks([]), a.set_yticks([])
            a.set_title(f"{v}, {'recortada' if cual else 'completa'}, z' {zp/1000:+.1f} um", fontsize=9)
    fig.suptitle("PSF del reflejo (raiz de la intensidad, 81 x 81 px; beta 45, m = +x: el espejo queda a la izquierda)")
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(ruta.with_suffix(ext), dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partes", nargs="*", default=["perdida", "forma", "sesgo", "crlb", "calibracion"])
    p.add_argument("--red", nargs=2, action="append", metavar=("EVAL_SIN", "EVAL_CON"), default=None,
                   help="dos carpetas de evaluate.py (sin y con recorte); se puede repetir")
    p.add_argument("--sufijo", default="")
    args = p.parse_args(argv)
    t0 = time.time()
    res = {}
    if "perdida" in args.partes:
        res["perdida"] = parte_perdida()
    imagenes = None
    if "forma" in args.partes:
        res["forma"], imagenes = parte_forma()
    if "sesgo" in args.partes:
        res["sesgo"] = parte_sesgo()
    if "crlb" in args.partes:
        res["crlb"] = parte_crlb()
    if "calibracion" in args.partes:
        res["calibracion"] = parte_calibracion()
    if args.red is not None:
        res["red"] = parte_red(args.red)
    for k, t in res.items():
        t.to_csv(SALIDA / f"recorte_pupila_{k}{args.sufijo}.csv", index=False)
        if "ajustes" in t.attrs:
            t.attrs["ajustes"].to_csv(SALIDA / f"recorte_pupila_{k}_ajustes{args.sufijo}.csv", index=False)
    if len(set(res) & {"perdida", "forma", "sesgo", "crlb", "calibracion"}) == 5:
        figura(res["perdida"], res["forma"], res["sesgo"], res["crlb"], res["calibracion"],
               SALIDA / f"fig_recorte_pupila{args.sufijo}.png")
    if imagenes is not None:
        figura_psf(imagenes, SALIDA / f"fig_recorte_pupila_psf{args.sufijo}.png")
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
