"""
Calibracion de la pupila de la PSF esferica de simulador_nuevo con Yeast6_1
(canal derecho, real y reflejo de los 100 cuadros), sin cambiar de modelo:
la misma pupila escalar (NA 1.1, agua n = 1.333, 530 nm, blur 0.6 px,
integracion sobre el pixel, recorte de pupila del espejo para el real y el
reflejo) con la fase

    fase = a40 Z4^0 + a22 Z2^2 + a2m2 Z2^-2 + a31 Z3^1 + a3m1 Z3^-1

con la normalizacion de fase_esferica (valor maximo 1 en el disco unidad):
Z4^0 = 6 r^4 - 6 r^2 + 1, Z2^2 = r^2 cos 2t, Z2^-2 = r^2 sin 2t,
Z3^1 = (3 r^3 - 2 r) cos t, Z3^-1 = (3 r^3 - 2 r) sin t, con t el azimut de la
frecuencia (fx, fy) en el marco del canal derecho (x = columna - corte,
y = fila). La fase de desenfoque es la fisica, q = -z (+z hacia el
objetivo), y la mascara del espejo se toma en +f (mascara_recorte con
anillos_hacia_objetivo False). Con eso el signo de a40 es la variante:
    a40 > 0  <->  anillos_hacia_objetivo False (lejos), amp_esferica = a40
    a40 < 0  <->  anillos_hacia_objetivo True (cerca), amp_esferica = |a40|
(el simulador con True usa q = +z y la mascara en -f, que es la PSF de -a40
con q = -z reflejada r -> -r: para los terminos impares, a31 y a3m1, el
equivalente en esa representacion lleva el signo cambiado).

Datos: canal derecho en e- = (ADU - nivel oscuro del cuadro) x 0.45, con el
nivel oscuro y el ruido de lectura (mediana, diferencias de cuadros / sqrt 2)
medidos fuera de las aperturas como en psf_real.py; sin interpolar. Region
del ajuste por cuadro: discos de R_DISCO px alrededor del real y del reflejo
detectados (analisis_datos_reales.py), dentro de la apertura.

Geometria: GeometriaEspejo con beta (compartido, en [40, 50] grados),
phi = direccion media reflejo -> real (fija) y la linea espejo/plano focal en
el borde de la franja (analisis_datos_reales.perfil_franja) corrida delta a lo
largo de m (compartido). Por cuadro: x, y del real, su distancia al espejo d
(equivalente a su z: z = (d - s cos beta) / sin beta, con s la coordenada a lo
largo de m desde la linea; con z libre por cuadro, delta seria una direccion
plana del ajuste, porque correr la linea delta equivale a z -> z + delta /
tan beta en todos los cuadros), fotones del real, fotones del reflejo y fondo
uniforme. El reflejo sale de la geometria: lateral p - 2 d cos(beta) m,
z' = z - 2 d sin(beta).

Verosimilitud: Poisson + ruido de lectura gaussiano con la aproximacion de
Poisson desplazada (d + s^2 ~ Poisson(mu + s^2), Huang et al. 2013). Ajuste por
maxima verosimilitud (L-BFGS en torch, GPU si hay) sobre los cuadros pares,
desde varios arranques: un barrido barato de la z media del real (la
posicion de la linea) con la pupila actual elige, por signo de a40, los
mejores minimos locales, y desde cada uno se ajusta con varios beta.
Validacion: recuperacion en una pelicula sintetica igual con los Zernike
calibrados, cuadros impares con la pupila calibrada frente a la actual
(+-1.5 pi sin otros terminos, beta y delta ajustados en los pares),
incertidumbre por la Hessiana (complemento de Schur) y por bootstrap de
bloques de cuadros. Con --perfil-ratio, solo el perfil de la verosimilitud en la
reflectividad (fotones del reflejo / del real, comun a todos los cuadros)
desde el mejor arranque de cada signo (lee calibracion_pupila_arranques.csv).
Con --solo-esferica, el modelo que se simula (a40 sola, signo libre, con beta
y delta compartidos y los parametros por cuadro de siempre): arranques en
--a40-arranques x --betas (z inicial por signo), minimos distintos con su
-log L en pares e impares, perfil en a40 por continuacion desde las ramas de
cerca, los cinco Zernike desde el arranque exploratorio que dio a40 = -6.40 y
desde el mejor minimo de cerca, y bootstrap de bloques del ganador y del
mejor de cerca (~3 h).

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    LD_PRELOAD=$HOME/miniforge3/envs/tesis/lib/libstdc++.so.6 $PY figuras/calibracion_pupila.py \
        [--tif RUTA] [--cache NPZ] [--betas 41 45 49] [--z-barrido -4000 8000 1000] [--n-minimos 2]
        [--n-boot 30] [--n-sint 2] [--desde-arranques] [--salida figuras/calibracion_pupila]
    LD_PRELOAD=... $PY figuras/calibracion_pupila.py --perfil-ratio 0.1 0.2 0.45 0.7 0.9   # despues de la corrida
    LD_PRELOAD=... $PY figuras/calibracion_pupila.py --solo-esferica [--a40-arranques ...] [--perfil-a40 -8 -3.5 0.25]
Escribe en --salida: calibracion_pupila_barrido.csv, calibracion_pupila_arranques.csv,
calibracion_pupila_validacion.csv, calibracion_pupila_residuos_impares.csv, calibracion_pupila.json,
por variante calibracion_pupila_{parametros,bootstrap,correlaciones,sintetico,cuadros}_{cerca,lejos}.csv,
fig_calibracion_pupila_imagenes.png, fig_calibracion_pupila_signo.png y
fig_calibracion_pupila_medidas_{cerca,lejos}.png (300 dpi); los de cada variante llevan el sufijo _cerca / _lejos;
con --perfil-ratio, calibracion_pupila_ratio.csv; con --solo-esferica,
calibracion_pupila_esferica_{arranques,minimos,perfil,cinco}.csv,
calibracion_pupila_esferica_bootstrap_{cerca,lejos}.csv, calibracion_pupila_esferica.json y
fig_calibracion_pupila_esferica.png.
Resultados en resultados/resultados_calibracion_pupila.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(RAIZ / "simulador_nuevo"))
sys.path.append(str(RAIZ))
import analisis_datos_reales as adr  # noqa: E402
import simulador as sim  # noqa: E402
from geometria_espejo import GeometriaEspejo  # noqa: E402

PX = adr.PIXEL_NM
GANANCIA = 0.45                 # e-/ADU (transferencia de fotones, resultados/resultados_psf_real.md)
R_DISCO = 22.0                  # radio de cada disco del ajuste [px]
B = 112                         # lado de la ventana por cuadro [px] (contiene los dos discos)
ZERNIKE = ("a40", "a22", "a2m2", "a31", "a3m1")
COMPARTIDOS = ZERNIKE + ("beta", "delta_um")
POR_CUADRO = ("dx_px", "dy_px", "d_um", "lnN_real", "lnN_reflejo", "ln_fondo")
NC, NP = len(COMPARTIDOS), len(POR_CUADRO)
BETA_C, BETA_H = 45.0, 5.0      # beta = 45 + 5 tanh(u): [40, 50] grados
A_ACTUAL = 1.5 * np.pi
CHUNK = 10                      # cuadros por bloque en la GPU
RAMPA = 0.25                    # ancho del borde de la mascara continua, en pasos de la rejilla de w


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------

def preparar(tif, cache=None):
    """Canal derecho: ventanas (T, B, B) en e-, origen de cada ventana, mascara
    del ajuste, real/reflejo detectados, m, punto de la linea (borde de la
    franja), nivel oscuro y ruido de lectura. Se guarda en cache (npz)."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import psf_real as pr
    from scipy import ndimage as ndi
    video, _ = adr.cargar(tif)
    T = len(video)
    c = adr.buscar_corte(video.mean(0))
    corte = c["corte"]
    vid = video[:, :, corte:].astype(np.float32)
    del video
    ap = c["ap_der"]
    dets = [adr.detectar(vid[t], ap) for t in range(T)]
    pares = adr.pares_real_reflejo(adr.enlazar(dets, T), T)
    if len(pares) != 1:
        raise SystemExit(f"se esperaba un par real/reflejo en el canal derecho; hay {len(pares)}")
    real, refl = pares[0]["real"], pares[0]["refl"]
    if not np.isfinite(real + refl).all():
        raise SystemExit("faltan posiciones del real o del reflejo en algun cuadro")
    seg = np.mean(real - refl, 0)
    m = seg / np.hypot(*seg)
    t_dir = np.array([-m[1], m[0]])
    centro = np.mean(np.r_[real, refl], 0)
    _, _, _, fr = adr.perfil_franja(vid.mean(0), centro, m, t_dir, ap, [real.mean(0), refl.mean(0)])
    q = centro + fr["v_borde"] * m
    # nivel oscuro por cuadro y ruido de lectura, fuera de las aperturas (como psf_real.py)
    union = np.c_[c["ap_izq"], ap]
    fuera = ndi.binary_erosion(~ndi.binary_dilation(union, iterations=15), iterations=5)
    fuera[:, corte - 20:corte + 20] = False
    fuera = fuera[:, corte:]
    oscuro = np.array([pr.media_recortada(vid[t][fuera]) for t in range(T)])
    var_f = np.var(np.diff(vid, axis=0)[:, fuera], axis=0) / 2
    rn_adu = float(np.sqrt(np.median(var_f)))
    medio = (real + refl) / 2
    o = np.round(medio).astype(int) - B // 2                       # origen (x0, y0) de cada ventana
    Hc, Wc = vid.shape[1:]
    assert (o >= 0).all() and (o[:, 0] + B <= Wc).all() and (o[:, 1] + B <= Hc).all()
    ventanas = np.stack([vid[t, o[t, 1]:o[t, 1] + B, o[t, 0]:o[t, 0] + B] for t in range(T)])
    e = (ventanas - oscuro[:, None, None]) * GANANCIA
    yy, xx = np.mgrid[0:B, 0:B]
    masc = np.zeros((T, B, B), bool)
    for t in range(T):
        for p in (real[t], refl[t]):
            masc[t] |= np.hypot(xx - (p[0] - o[t, 0]), yy - (p[1] - o[t, 1])) <= R_DISCO
        masc[t] &= ap[o[t, 1]:o[t, 1] + B, o[t, 0]:o[t, 0] + B]
    d = dict(e=e.astype(np.float64), o=o, masc=masc, real=real, refl=refl, m=m, q=q, oscuro=oscuro,
             rn_adu=np.array(rn_adu), rn_rms_adu=np.array(np.sqrt(var_f.mean())), v_borde=np.array(fr["v_borde"]),
             angulo_borde_t=np.array(fr["angulo_vs_t_deg"]), corte=np.array(corte))
    if cache is not None:
        np.savez_compressed(cache, **d)
    return d


# ----------------------------------------------------------------------------
# Modelo directo (torch)
# ----------------------------------------------------------------------------

class Modelo:
    """Pupila de simulador_nuevo (Config por defecto) con los cinco Zernike, el
    recorte del espejo y la geometria, en torch. psf(...) da la PSF de cada
    emisor en una ventana (B, B) con el emisor en (B // 2 + dx, B // 2 + dy),
    normalizada como PSFRecortada.psf (la PSF entera suma la fraccion de la
    pupila que pasa la mascara). binaria=False usa una mascara con borde en
    rampa de un paso de la rejilla de la pupila (continua en beta)."""

    def __init__(self, dispositivo, dtype=torch.float64, binaria=False):
        cfg = sim.Config()
        self.cfg, self.dev, self.dt, self.binaria = cfg, dispositivo, dtype, binaria
        M, px = cfg.M_pupila, cfg.pixel_nm
        f = np.fft.fftfreq(M, d=px)
        fx, fy = np.meshgrid(f, f, indexing="xy")
        rho = np.hypot(fx, fy) * cfg.lambda_nm / cfg.NA
        ap = rho <= 1.0
        r, th = np.clip(rho, 0, 1), np.arctan2(fy, fx)
        rad3 = 3 * r ** 3 - 2 * r
        Z = np.stack([6 * r ** 4 - 6 * r ** 2 + 1, r ** 2 * np.cos(2 * th), r ** 2 * np.sin(2 * th),
                      rad3 * np.cos(th), rad3 * np.sin(th)]) * ap
        sin_th = np.clip(cfg.NA * rho / cfg.n_medio, 0, 1)
        kcos = np.where(ap, 2 * np.pi * cfg.n_medio / cfg.lambda_nm * np.sqrt(1 - sin_th ** 2), 0.0)
        sig = cfg.blur_extra_px * px
        filtro = np.exp(-2 * (np.pi * sig) ** 2 * (fx ** 2 + fy ** 2)) * np.sinc(fx * px) * np.sinc(fy * px)
        esc = cfg.lambda_nm / cfg.n_medio                          # q = -z: w_xy = +(lambda / n) f
        wx, wy = esc * fx, esc * fy
        wz = np.sqrt(np.clip(1 - wx ** 2 - wy ** 2, 0, None))
        T_ = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dispositivo)  # noqa: E731
        self.ap, self.Z, self.kcos = T_(ap.astype(float)), T_(Z), T_(kcos)
        self.filtro = T_(filtro[:, :M // 2 + 1])
        self.w = T_(np.stack([wx, wy, wz]))
        self.paso_w = esc / (M * px)                               # paso de la rejilla en w
        self.gx, self.gy = T_(-2 * np.pi * fx * px), T_(-2 * np.pi * fy * px)
        self.norm = M * M / float(ap.sum())
        self.idx = torch.as_tensor((np.arange(B) - B // 2) % M, device=dispositivo)
        self.M = M

    def mascara(self, beta_rad, phi_rad):
        n = torch.stack([torch.cos(beta_rad) * torch.cos(phi_rad), torch.cos(beta_rad) * torch.sin(phi_rad),
                         torch.sin(beta_rad)])
        wn = torch.einsum("i,ijk->jk", n, self.w)
        if self.binaria:
            return torch.where(wn > 0, 1.0, torch.where(wn == 0, np.sqrt(0.5), 0.0)).to(self.dt)
        return torch.clamp(0.5 + wn / (RAMPA * self.paso_w), 0.0, 1.0)

    def pupila(self, coef, beta_rad, phi_rad):
        fase = torch.einsum("i,ijk->jk", coef, self.Z)
        return self.ap * self.mascara(beta_rad, phi_rad) * torch.exp(1j * fase)

    def psf(self, P0, z_nm, dx, dy):
        """P0 (M, M) complejo; z_nm, dx, dy (n,) -> (n, B, B)."""
        arg = self.kcos * (-z_nm)[:, None, None] + self.gx * dx[:, None, None] + self.gy * dy[:, None, None]
        P = P0[None] * torch.exp(1j * arg)
        I = torch.fft.ifft2(P).abs() ** 2
        I = torch.fft.irfft2(torch.fft.rfft2(I) * self.filtro, s=(self.M, self.M)).clamp_min(0.0)
        return I[:, self.idx][:, :, self.idx] * self.norm


def beta_de(u):
    return BETA_C + BETA_H * torch.tanh(u)


def u_de(beta_deg):
    return float(np.arctanh((beta_deg - BETA_C) / BETA_H))


class Problema:
    """Datos de un conjunto de cuadros + el modelo: nll(theta) con theta =
    [compartidos (NC), por cuadro (T, NP)] y la geometria de cada cuadro."""

    def __init__(self, d, cuadros, modelo, e=None):
        dev, dt = modelo.dev, modelo.dt
        self.mod, self.cuadros = modelo, np.asarray(cuadros)
        T_ = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)  # noqa: E731
        c = self.cuadros
        self.e = T_((d["e"] if e is None else e)[c])
        self.masc = torch.as_tensor(d["masc"][c], device=dev)
        self.o = T_(d["o"][c].astype(float))
        self.x0 = T_(d["real"][c])                                  # posicion inicial del real [px, canal]
        self.m = T_(d["m"])
        self.q = T_(d["q"])
        self.phi = torch.atan2(self.m[1], self.m[0])
        self.s2 = float(d["rn_adu"]) ** 2 * GANANCIA ** 2           # varianza de lectura [e-^2]
        self.T = len(c)
        self.d = d
        self.ln_ratio = None                                        # None: fotones del reflejo libres por cuadro

    # -- geometria ---------------------------------------------------------
    def geometria(self, th):
        """Real (x, y) [px, canal], z [nm]; reflejo (x', y'), z'; beta [rad]."""
        sh = th[:NC]
        pc = th[NC:].view(self.T, NP)
        beta = torch.deg2rad(beta_de(sh[5]))
        delta_px = sh[6] * 1000.0 / PX
        xy = self.x0 + pc[:, :2]
        dist = pc[:, 2] * 1000.0                                    # [nm]
        s = ((xy - (self.q + delta_px * self.m)) @ self.m) * PX     # [nm] a lo largo de m desde la linea
        z = (dist - s * torch.cos(beta)) / torch.sin(beta)
        xy_f = xy - (2 * dist * torch.cos(beta) / PX)[:, None] * self.m
        z_f = z - 2 * dist * torch.sin(beta)
        return xy, z, xy_f, z_f, beta

    def mu(self, th, sel=slice(None), P0=None):
        """Imagen esperada [e-] (n, B, B) de los cuadros sel (indices locales)."""
        sh = th[:NC]
        pc = th[NC:].view(self.T, NP)[sel]
        xy, z, xy_f, z_f, beta = (g[sel] if g.dim() else g for g in self.geometria(th))
        if P0 is None:
            P0 = self.mod.pupila(sh[:5], beta, self.phi)
        o = self.o[sel]
        n = len(z)
        rel = torch.cat([xy - o - B // 2, xy_f - o - B // 2])
        I = self.mod.psf(P0, torch.cat([z, z_f]), rel[:, 0], rel[:, 1])
        Nr, b = torch.exp(pc[:, 3]), torch.exp(pc[:, 5])
        Nf = torch.exp(pc[:, 4]) if self.ln_ratio is None else Nr * float(np.exp(self.ln_ratio))
        return Nr[:, None, None] * I[:n] + Nf[:, None, None] * I[n:] + b[:, None, None], I[:n], I[n:]

    def nll_bloque(self, th, sel):
        mu = self.mu(th, sel)[0]
        mp, dp = mu + self.s2, self.e[sel] + self.s2
        return ((mp - dp * torch.log(mp)) * self.masc[sel]).sum()

    def bloques(self):
        return [slice(i, min(i + CHUNK, self.T)) for i in range(0, self.T, CHUNK)]

    def nll(self, th):
        with torch.no_grad():
            return float(sum(self.nll_bloque(th, s) for s in self.bloques()))

    def deviance(self, th):
        """2 sum [d' log(d'/mu') - (d' - mu')] en la mascara (d' = d + s^2) y el numero de pixeles."""
        tot, npx = 0.0, 0
        with torch.no_grad():
            for s in self.bloques():
                mp, dp = self.mu(th, s)[0] + self.s2, self.e[s] + self.s2
                dv = torch.where(dp > 0, dp * torch.log(dp.clamp_min(1e-12) / mp), torch.zeros_like(dp)) - (dp - mp)
                tot += float(2 * (dv * self.masc[s]).sum())
                npx += int(self.masc[s].sum())
        return tot, npx


# ----------------------------------------------------------------------------
# Ajuste
# ----------------------------------------------------------------------------

def theta_inicial(prob, a40, beta_deg, delta_um):
    """Arranque: Zernike (a40, 0...), beta, delta; por cuadro la posicion
    detectada, d de la separacion a lo largo de m y fotones/fondo nominales."""
    d = prob.d
    c = prob.cuadros
    L = ((d["real"][c] - d["refl"][c]) @ d["m"]) * PX               # [nm]
    dist = L / (2 * np.cos(np.radians(beta_deg)))
    pc = np.zeros((prob.T, NP))
    pc[:, 2] = dist / 1000.0
    pc[:, 3], pc[:, 4], pc[:, 5] = np.log(3e4), np.log(1.3e4), np.log(5.0)
    sh = np.array([a40, 0, 0, 0, 0, u_de(beta_deg), delta_um])
    return torch.as_tensor(np.r_[sh, pc.ravel()], dtype=prob.mod.dt, device=prob.mod.dev)


def indices(prob, compartidos=COMPARTIDOS, por_cuadro=POR_CUADRO):
    """Mascara booleana de los parametros libres."""
    li = np.zeros(NC + prob.T * NP, bool)
    for k in compartidos:
        li[COMPARTIDOS.index(k)] = True
    for k in por_cuadro:
        li[NC + POR_CUADRO.index(k)::NP] = True
    return torch.as_tensor(li, device=prob.mod.dev)


def ajustar(prob, th0, libres, max_iter=400, tol=1e-9, verbose=False):
    """L-BFGS (torch, strong Wolfe) sobre los parametros libres; los demas fijos."""
    th_fijo = th0.detach().clone()
    x = th0.detach()[libres].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=50, history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-12)

    def completo():
        th = th_fijo.clone()
        th[libres] = x
        return th

    def cierre():
        opt.zero_grad()
        tot = 0.0
        for s in prob.bloques():
            v = prob.nll_bloque(completo(), s)
            v.backward()
            tot += float(v.detach())
        return tot

    prev = np.inf
    for it in range(max_iter // 50):
        val = opt.step(cierre)
        val = float(val) if not torch.is_tensor(val) else float(val.detach())
        actual = prob.nll(completo().detach())
        if verbose:
            print(f"      iter {50 * (it + 1)}: nll {actual:.3f}")
        if prev - actual < tol * max(1.0, abs(actual)):
            break
        prev = actual
    return completo().detach()


def ajuste_completo(prob, th0, verbose=False, compartidos=COMPARTIDOS):
    """Por etapas: fotones y fondo; + posicion; + compartidos."""
    th = ajustar(prob, th0, indices(prob, (), ("lnN_real", "lnN_reflejo", "ln_fondo")), max_iter=100)
    th = ajustar(prob, th, indices(prob, (), POR_CUADRO), max_iter=200)
    th = ajustar(prob, th, indices(prob, compartidos, POR_CUADRO), max_iter=1500, verbose=verbose)
    return th


def resumen_theta(prob, th):
    sh = th[:NC].cpu().numpy()
    xy, z, xy_f, z_f, beta = (g.detach().cpu().numpy() for g in prob.geometria(th))
    pc = th[NC:].view(prob.T, NP).cpu().numpy()
    return dict(a40=sh[0], a22=sh[1], a2m2=sh[2], a31=sh[3], a3m1=sh[4], beta=float(np.degrees(beta)),
                delta_um=sh[6], z_real_media=float(z.mean()), z_real_std=float(z.std()),
                z_reflejo_media=float(z_f.mean()), N_real_media=float(np.exp(pc[:, 3]).mean()),
                N_reflejo_media=float(np.exp(pc[:, 4]).mean()), fondo_media=float(np.exp(pc[:, 5]).mean()))


# ----------------------------------------------------------------------------
# Verificaciones del modelo
# ----------------------------------------------------------------------------

def verificar(mod_bin, mod, d):
    """Modelo torch con mascara binaria y sin Zernike extra frente a
    PSFRecortada de simulador_nuevo (los dos signos de a40 con su
    anillos_hacia_objetivo); la geometria frente a GeometriaEspejo; y la
    diferencia de la mascara en rampa."""
    dev, dt = mod.dev, mod.dt
    phi = float(np.degrees(np.arctan2(d["m"][1], d["m"][0])))
    peor, peor_rampa = 0.0, 0.0
    for beta in (41.0, 47.5):
        for a40, cerca in ((A_ACTUAL, False), (-A_ACTUAL, True)):
            cfg = sim.Config(H=B, W=B, usar_espejo=True, angulo_espejo_eje_deg=beta, orientacion_espejo_deg=phi,
                             espejo_punto_px=(10.0, 10.0), recorte_pupila_reflejo=True, recorte_pupila_real=True,
                             amp_esferica=abs(a40), anillos_hacia_objetivo=cerca)
            ref = sim.PSFRecortada(cfg, B)
            for z, dx, dy in ((-1200.0, 0.3, -0.7), (-5800.0, -0.45, 0.2), (2500.0, 0.0, 0.0)):
                r = ref.psf(z, dx, dy)
                coef = torch.tensor([a40, 0, 0, 0, 0], dtype=dt, device=dev)
                br, pr_ = torch.tensor(np.radians(beta), dtype=dt, device=dev), torch.tensor(np.radians(phi), dtype=dt, device=dev)
                t = lambda v: torch.tensor([v], dtype=dt, device=dev)  # noqa: E731
                a = mod_bin.psf(mod_bin.pupila(coef, br, pr_), t(z), t(dx), t(dy))[0].cpu().numpy()
                b = mod.psf(mod.pupila(coef, br, pr_), t(z), t(dx), t(dy))[0].cpu().numpy()
                peor = max(peor, np.abs(a - r).max() / r.max())
                peor_rampa = max(peor_rampa, np.abs(b - r).max() / r.max())
    assert peor < 1e-9, f"el modelo torch no reproduce PSFRecortada: {peor:.2e}"
    # geometria
    prob = Problema(d, np.arange(3), mod)
    th = theta_inicial(prob, A_ACTUAL, 43.0, 0.8)
    th[NC:].view(3, NP)[:, :2] += torch.tensor([[0.3, -0.2], [1.0, 0.5], [-0.4, 0.0]], dtype=dt, device=dev)
    xy, z, xy_f, z_f, beta = (g.detach().cpu().numpy() for g in prob.geometria(th))
    q = d["q"] + 0.8 * 1000 / PX * d["m"]
    geo = GeometriaEspejo(43.0, phi, [q[0] * PX, q[1] * PX, 0.0], -1e9, 1e9)
    p = np.c_[xy * PX, z]
    pf = geo.reflejar_nm(p)
    dist = th[NC:].view(3, NP)[:, 2].cpu().numpy() * 1000
    assert np.allclose(geo.distancia_nm(p), dist, atol=1e-6)
    assert np.allclose(pf[:, :2] / PX, xy_f, atol=1e-9) and np.allclose(pf[:, 2], z_f, atol=1e-6)
    return peor, peor_rampa


# ----------------------------------------------------------------------------
# Incertidumbre
# ----------------------------------------------------------------------------

def hessiana(prob, th):
    """Hessiana (n, n) de la nll en theta, por bloques de cuadros: la matriz es
    de flecha (los parametros de un cuadro solo se acoplan con los suyos y con
    los compartidos), asi que basta con NC + NP productos Hessiana-vector por
    bloque."""
    n = NC + prob.T * NP
    H = np.zeros((n, n))
    for s in prob.bloques():
        th_ = th.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(prob.nll_bloque(th_, s), th_, create_graph=True)[0]
        ks = np.arange(s.start, s.stop)
        cols = (NC + ks[:, None] * NP + np.arange(NP)).ravel()
        for j in range(NC):
            h = torch.autograd.grad(g[j], th_, retain_graph=True)[0].cpu().numpy()
            H[j, :NC] += h[:NC]
            H[j, cols] = h[cols]
            H[cols, j] = h[cols]
        for j in range(NP):
            v = torch.zeros(n, dtype=th.dtype, device=th.device)
            v[NC + ks * NP + j] = 1.0
            h = torch.autograd.grad((g * v).sum(), th_, retain_graph=True)[0].cpu().numpy()
            for k in ks:
                fil = NC + k * NP + np.arange(NP)
                H[fil, NC + k * NP + j] = h[fil]
    return (H + H.T) / 2


def jacobiano_derivados(prob, th):
    """Derivados (beta [grados], z media del real, z' media del reflejo) y su
    gradiente respecto de theta, para el metodo delta."""
    th_ = th.detach().clone().requires_grad_(True)
    _, z, _, z_f, beta = prob.geometria(th_)
    vals = [torch.rad2deg(beta), z.mean(), z_f.mean()]
    J = np.stack([torch.autograd.grad(v, th_, retain_graph=True)[0].cpu().numpy() for v in vals])
    return np.array([float(v.detach()) for v in vals]), J


def covarianza(prob, th, libres):
    """Covarianza (inversa de la Hessiana de los parametros libres), errores de
    los compartidos y de los derivados, y correlaciones."""
    H = hessiana(prob, th)
    li = libres.cpu().numpy()
    C = np.zeros_like(H)
    Hl = H[np.ix_(li, li)]
    C[np.ix_(li, li)] = np.linalg.inv(Hl)
    vals, J = jacobiano_derivados(prob, th)
    Cd = J @ C @ J.T
    return H, C, vals, Cd, float(np.linalg.eigvalsh(Hl).min())


# ----------------------------------------------------------------------------
# Residuos
# ----------------------------------------------------------------------------

def discos(prob):
    """Mascaras (T, B, B) del disco del real y del reflejo (posiciones detectadas)."""
    d, c = prob.d, prob.cuadros
    yy, xx = np.mgrid[0:B, 0:B]
    out = []
    for pos in (d["real"][c], d["refl"][c]):
        rel = pos - d["o"][c]
        out.append((np.hypot(xx[None] - rel[:, 0, None, None], yy[None] - rel[:, 1, None, None]) <= R_DISCO)
                   & d["masc"][c])
    return out


def residuos(prob, th):
    """Residuo relativo |datos - modelo| / |datos - fondo| (norma cuadratica) en
    el disco del real, el del reflejo y los dos, con ruido y sin ruido (restando
    la varianza esperada mu + s^2 del numerador y del denominador), sumado sobre
    los cuadros; y por cuadro (sin ruido)."""
    with torch.no_grad():
        mu = torch.cat([prob.mu(th, s)[0] for s in prob.bloques()]).cpu().numpy()
    e = prob.e.cpu().numpy()
    b = np.exp(th[NC:].view(prob.T, NP)[:, 5].cpu().numpy())[:, None, None]
    var = mu + prob.s2
    dr, df = discos(prob)
    out, por_cuadro = {}, {}
    for nom, mk in (("real", dr), ("reflejo", df), ("total", dr | df)):
        num, den, v = ((e - mu) ** 2 * mk), ((e - b) ** 2 * mk), var * mk
        out[f"res_{nom}_con_ruido"] = float(np.sqrt(num.sum() / den.sum()))
        out[f"res_{nom}"] = float(np.sqrt(max(num.sum() - v.sum(), 0) / (den.sum() - v.sum())))
        por_cuadro[nom] = np.sqrt(np.clip(num.sum((1, 2)) - v.sum((1, 2)), 0, None) / (den.sum((1, 2)) - v.sum((1, 2))))
    return out, por_cuadro, mu


def recortes_medios(prob, th, mus, r=20):
    """Suma de los cuadros, alineados en el pixel entero del real y del reflejo
    ajustados (sin interpolar), de los datos y de cada modelo (mus: lista de
    (T, B, B)); recortes (2r+1)^2 menos el fondo de cada cuadro."""
    xy, _, xy_f, _, _ = (g.detach().cpu().numpy() for g in prob.geometria(th))
    o = prob.d["o"][prob.cuadros]
    e = prob.e.cpu().numpy()
    out = {}
    for nom, pos in (("real", xy), ("reflejo", xy_f)):
        c = np.round(pos - o).astype(int)
        acc = np.zeros((1 + len(mus), 2 * r + 1, 2 * r + 1))
        for k in range(prob.T):
            sl = (slice(c[k, 1] - r, c[k, 1] + r + 1), slice(c[k, 0] - r, c[k, 0] + r + 1))
            for i, img in enumerate([e] + list(mus)):
                acc[i] += img[k][sl]
        out[nom] = acc
    return out


def residuo_medio(rec, fondo_total, r=R_DISCO):
    """Residuo relativo de la imagen suma (casi sin ruido) en el disco de radio r."""
    n = rec.shape[-1] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    mk = np.hypot(xx, yy) <= r
    dat = rec[0]
    return [float(np.sqrt(((dat - m) ** 2)[mk].sum() / ((dat - fondo_total) ** 2)[mk].sum())) for m in rec[1:]]


# ----------------------------------------------------------------------------
# Arranques multiples
# ----------------------------------------------------------------------------

def delta_para_z(prob, beta_deg, z_obj_nm):
    """delta [um] que pone la z media del real (de la separacion detectada) en z_obj_nm."""
    th = theta_inicial(prob, A_ACTUAL, beta_deg, 0.0)
    z = prob.geometria(th)[1].mean().item()
    return (z_obj_nm - z) * np.tan(np.radians(beta_deg)) / 1000.0


def barrido_z(prob, signos, z_objs, beta=45.0, a_abs=A_ACTUAL, verbose=True):
    """Perfil barato del paisaje en z: pupila actual (a40 = signo * a_abs, sin
    otros terminos), beta fijo y la linea puesta para que la z media del real
    sea z0; solo se ajustan los parametros por cuadro. DataFrame (signo, z0, nll)."""
    filas = []
    for sg in signos:
        for z0 in z_objs:
            t0 = time.time()
            th = theta_inicial(prob, sg * a_abs, beta, delta_para_z(prob, beta, z0))
            th = ajustar(prob, th, indices(prob, (), ("lnN_real", "lnN_reflejo", "ln_fondo")), max_iter=100)
            th = ajustar(prob, th, indices(prob, (), POR_CUADRO), max_iter=200)
            nll = prob.nll(th)
            filas.append(dict(signo=int(sg), z0=float(z0), nll=nll if np.isfinite(nll) else np.inf,
                              z_real_media=float(prob.geometria(th)[1].mean())))
            if verbose:
                print(f"    a40 {sg:+d}, z0 {z0:+6.0f}: nll {nll:.1f}, z real ajustada {filas[-1]['z_real_media']:+.0f} nm "
                      f"({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(filas)


def minimos_locales(barr, n=2):
    """Por signo, los z0 de los n mejores minimos locales del perfil del barrido."""
    out = []
    for sg, g in barr.groupby("signo"):
        g = g.sort_values("z0").reset_index(drop=True)
        v = g.nll.to_numpy()
        loc = [i for i in range(len(v)) if (i == 0 or v[i] <= v[i - 1]) and (i == len(v) - 1 or v[i] <= v[i + 1])]
        loc = sorted(loc, key=lambda i: v[i])[:n]
        out += [(int(sg), float(g.z0[i])) for i in loc]
    return out


def arranques(prob, puntos, betas, compartidos=COMPARTIDOS, a_abs=A_ACTUAL, etiqueta="", verbose=True):
    """Ajusta desde cada (signo de a40, z media inicial del real) de puntos y
    cada beta inicial; lista de (fila, theta). Un ajuste que da NaN queda con
    nll = inf."""
    res = []
    for sg, z0 in puntos:
        for b0 in betas:
            t0 = time.time()
            d0 = delta_para_z(prob, b0, z0)
            th = ajuste_completo(prob, theta_inicial(prob, sg * a_abs, b0, d0), compartidos=compartidos)
            nll = prob.nll(th)
            ok = bool(np.isfinite(nll) and torch.isfinite(th).all())
            fila = dict(ajuste=etiqueta, signo_inicial=int(sg), beta_inicial=b0, z_real_inicial=z0,
                        delta_inicial_um=d0, nll=nll if ok else np.inf,
                        nll_mascara_binaria=nll_binaria(prob, th) if ok else np.inf,
                        **resumen_theta(prob, th), segundos=time.time() - t0)
            if verbose:
                print(f"  [{etiqueta}] a40 {sg:+d}, beta0 {b0:g}, z0 {z0:+.0f}: nll {fila['nll']:.2f}; a40 {fila['a40']:+.3f}, "
                      f"a22 {fila['a22']:+.3f}, a2m2 {fila['a2m2']:+.3f}, a31 {fila['a31']:+.3f}, a3m1 {fila['a3m1']:+.3f}, "
                      f"beta {fila['beta']:.2f}, delta {fila['delta_um']:+.3f} um, z real {fila['z_real_media']:+.0f}, "
                      f"z' {fila['z_reflejo_media']:+.0f} nm ({fila['segundos']:.0f} s)", flush=True)
            if ok:
                res.append((fila, th))
    return res


def nll_binaria(prob, th):
    """nll con la mascara binaria exacta de mascara_recorte (la del simulador)."""
    mod = prob.mod
    prob.mod = Modelo(mod.dev, mod.dt, binaria=True)
    try:
        return prob.nll(th)
    finally:
        prob.mod = mod


def ajustar_cuadros(prob, sh, beta_deg):
    """Con los compartidos fijos (sh, NC), ajusta solo los parametros por cuadro."""
    th = theta_inicial(prob, 0.0, beta_deg, 0.0)
    th[:NC] = sh
    th = ajustar(prob, th, indices(prob, (), ("lnN_real", "lnN_reflejo", "ln_fondo")), max_iter=100)
    return ajustar(prob, th, indices(prob, (), POR_CUADRO), max_iter=600)


def describir(sh):
    """Diccionario legible de los compartidos (beta en grados)."""
    sh = np.asarray(sh, float)
    return dict(zip(ZERNIKE, sh[:5]), beta=BETA_C + BETA_H * np.tanh(sh[5]), delta_um=sh[6])


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------

def perfil_ratio(P, arr, ratios, salida):
    """Prueba fisica: la reflectividad del espejo es una sola. Con los fotones
    del reflejo = R x los del real en todos los cuadros (R fijo), reajusta todo
    lo demas desde el mejor arranque de cada signo (pupila calibrada) y da la
    nll frente a R."""
    print(f"\n[4] perfil en la reflectividad R = N_reflejo / N_real (comun a todos los cuadros), cuadros pares")
    filas = []
    cal = arr[arr.ajuste == "calibrada"]
    for sg in (-1, 1):
        f0 = cal[np.sign(cal.a40) == sg].sort_values("nll").iloc[0]
        for R in ratios:
            t0 = time.time()
            P.ln_ratio = float(np.log(R))
            th = theta_inicial(P, sg * A_ACTUAL, f0.beta_inicial, delta_para_z(P, f0.beta_inicial, f0.z_real_inicial))
            th = ajuste_completo(P, th)
            nll = P.nll(th)
            filas.append(dict(signo=sg, R=R, nll=nll, nll_libre=f0.nll, **resumen_theta(P, th)))
            filas[-1]["N_reflejo_media"] = R * filas[-1]["N_real_media"]
            print(f"  a40 {sg:+d}, R {R:g}: nll {nll:.1f} (con R libre por cuadro {f0.nll:.1f}); a40 {filas[-1]['a40']:+.3f}, "
                  f"beta {filas[-1]['beta']:.2f}, delta {filas[-1]['delta_um']:+.3f} um, z real {filas[-1]['z_real_media']:+.0f}, "
                  f"z' {filas[-1]['z_reflejo_media']:+.0f} nm, N real {filas[-1]['N_real_media']:.0f} ({time.time() - t0:.0f} s)",
                  flush=True)
    P.ln_ratio = None
    t = pd.DataFrame(filas)
    t.to_csv(salida / "calibracion_pupila_ratio.csv", index=False)
    return t


def distintos(filas, tol_nll=1.0):
    """Agrupa los arranques en minimos distintos (nll a menos de tol_nll y mismos parametros redondeados)."""
    grupos = []
    for i, f in enumerate(sorted(filas, key=lambda f: f["nll"])):
        clave = (np.sign(f["a40"]), round(f["beta"], 1), round(f["delta_um"], 1))
        for g in grupos:
            if abs(g["nll"] - f["nll"]) < tol_nll and g["clave"] == clave:
                g["n"] += 1
                break
        else:
            grupos.append(dict(f, clave=clave, n=1))
    return grupos


def bootstrap_bloques(d, pares, th, mod, compartidos, n_boot, bloque, rng):
    """Bootstrap de bloques de `bloque` cuadros pares consecutivos: reajusta los
    compartidos libres y los parametros por cuadro desde th (el minimo)."""
    nb = len(pares) // bloque
    T = len(pares)
    pc = th[NC:].view(T, NP)
    boot = []
    for r in range(n_boot):
        t0 = time.time()
        bl = rng.integers(0, nb, nb)
        loc = np.concatenate([np.arange(b * bloque, (b + 1) * bloque) for b in bl])
        Pb = Problema(d, pares[loc], mod)
        th0 = torch.cat([th[:NC], pc[loc].reshape(-1)])
        thb = ajustar(Pb, th0, indices(Pb, compartidos, POR_CUADRO), max_iter=1500)
        boot.append(dict(remuestreo=r, nll=Pb.nll(thb), **describir(thb[:NC].cpu().numpy()),
                         z_real_media=float(Pb.geometria(thb)[1].mean())))
        print(f"  {r}: " + ", ".join(f"{k} {v:+.3f}" for k, v in boot[-1].items() if k in compartidos)
              + f" ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(boot)


# ----------------------------------------------------------------------------
# Modelo que se simula: solo la esferica (--solo-esferica)
# ----------------------------------------------------------------------------

ESFERICA = ("a40", "beta", "delta_um")


def agrupar_minimos(res, tol_nll=2.0):
    """res: lista de (fila, th). Minimos distintos: misma nll (tol_nll) y mismos
    a40, beta y delta redondeados. Lista de (fila del mejor, th, arranques)."""
    grupos = []
    for f, th in sorted(res, key=lambda x: x[0]["nll"]):
        clave = (round(f["a40"], 2), round(f["beta"], 1), round(f["delta_um"], 2))
        for g in grupos:
            if abs(g[0]["nll"] - f["nll"]) < tol_nll and g[3] == clave:
                g[2].append(f)
                break
        else:
            grupos.append([f, th, [f], clave])
    return [(f, th, fs) for f, th, fs, _ in grupos]


def perfil_a40(P, ramas, grilla):
    """Perfil de la -log L en a40 (fijo; beta, delta y por cuadro libres; los
    otros Zernike en 0) por continuacion desde el minimo de cada rama, hacia
    arriba y hacia abajo de la grilla (cada punto arranca del anterior)."""
    filas = []
    for nombre, th0 in ramas:
        a0 = float(th0[0])
        tramos = (sorted(a for a in grilla if a >= a0), sorted((a for a in grilla if a < a0), reverse=True))
        for tramo in tramos:
            th = th0.clone()
            for a40 in tramo:
                t0 = time.time()
                prueba = th.clone()
                prueba[0] = a40
                prueba = ajustar(P, prueba, indices(P, ("beta", "delta_um"), POR_CUADRO), max_iter=600)
                nll = P.nll(prueba)
                ok = bool(np.isfinite(nll) and torch.isfinite(prueba).all())
                if ok:
                    th = prueba
                fila = dict(rama=nombre, a40_fijo=float(a40), nll=nll if ok else np.inf,
                            **{k: v for k, v in resumen_theta(P, prueba if ok else th).items() if k != "a40"},
                            segundos=time.time() - t0)
                filas.append(fila)
                print(f"  rama {nombre}, a40 {a40:+.2f}: nll {fila['nll']:.1f}; beta {fila['beta']:.2f}, delta "
                      f"{fila['delta_um']:+.3f} um, z real {fila['z_real_media']:+.0f} nm ({fila['segundos']:.0f} s)", flush=True)
    return pd.DataFrame(filas)


def solo_esferica(a, d, P, Pi, mod, salida, rng):
    """El modelo que se simula: fase a40 Z4^0 sola (signo libre), beta y delta
    compartidos y los parametros por cuadro de siempre. Arranques en los a40 y
    beta pedidos (z inicial por signo), minimos distintos con su -log L en
    pares e impares, perfil en a40 de las ramas de cerca, los cinco Zernike
    desde el arranque que dio a40 = -6.40 en la corrida exploratoria, y
    bootstrap de bloques del ganador."""
    t_ini = time.time()
    pares = P.cuadros
    print(f"\n[E1] solo esferica: arranques a40 {a.a40_arranques} x beta0 {a.betas}; z0 cerca {a.z_arranques_cerca}, "
          f"lejos {a.z_arranques_lejos} (cuadros pares)")
    res = []
    for a40 in a.a40_arranques:
        sg = int(np.sign(a40))
        for z0 in (a.z_arranques_cerca if sg < 0 else a.z_arranques_lejos):
            for f, th in arranques(P, [(sg, z0)], a.betas, compartidos=ESFERICA, a_abs=abs(a40), etiqueta="esferica"):
                f["a40_inicial"] = float(a40)
                res.append((f, th))
            pd.DataFrame([f for f, _ in res]).to_csv(salida / "calibracion_pupila_esferica_arranques.csv", index=False)
    grupos = agrupar_minimos(res)
    print("\n[E2] minimos distintos; cuadros impares con los compartidos de cada uno")
    filas, ths = [], []
    for i, (f, th, fs) in enumerate(grupos):
        thi = ajustar_cuadros(Pi, th[:NC], f["beta"])
        pci = thi[NC:].view(Pi.T, NP).cpu().numpy()
        pcp = th[NC:].view(P.T, NP).cpu().numpy()
        fila = dict(minimo=i, variante=VARIANTE[int(np.sign(f["a40"]))], a40=f["a40"], beta=f["beta"], delta_um=f["delta_um"],
                    z_real_media=f["z_real_media"], z_reflejo_media=f["z_reflejo_media"],
                    nll_pares=f["nll"], nll_pares_mascara_binaria=f["nll_mascara_binaria"], nll_impares=Pi.nll(thi),
                    z_real_media_impares=float(Pi.geometria(thi)[1].mean()),
                    N_real_media=f["N_real_media"], reflejo_sobre_real=float(np.mean(np.exp(pcp[:, 4] - pcp[:, 3]))),
                    reflejo_sobre_real_impares=float(np.mean(np.exp(pci[:, 4] - pci[:, 3]))), fondo_media=f["fondo_media"],
                    n_arranques=len(fs),
                    arranques=" ".join(f"({g['a40_inicial']:+.2f},{g['beta_inicial']:g},{g['z_real_inicial'] / 1000:+g})"
                                       for g in fs))
        filas.append(fila)
        ths.append(th)
    mins = pd.DataFrame(filas)
    mins["delta_logL_pares"] = mins.nll_pares.min() - mins.nll_pares
    mins["delta_logL_impares"] = mins.nll_impares.min() - mins.nll_impares
    mins.to_csv(salida / "calibracion_pupila_esferica_minimos.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(mins.drop(columns=["arranques"]).to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    # ------------------------------------------------ perfil en a40 de las ramas de cerca
    grilla = np.round(np.arange(a.perfil_a40[0], a.perfil_a40[1] + 1e-9, a.perfil_a40[2]), 6)
    cerca = mins[mins.variante == "cerca"].sort_values("nll_pares")
    ramas = []
    for _, f in cerca.iterrows():               # ramas de cerca distintas en z (la rama la marca la z del real)
        if all(abs(f.z_real_media - g.z_real_media) > 250 or abs(f.a40 - g.a40) > 0.4 for _, g in ramas):
            ramas.append((f"{int(f.minimo)}", f))
    ramas = ramas[:a.ramas_perfil]
    print(f"\n[E3] perfil en a40 ({grilla[0]:g} a {grilla[-1]:g}) desde los minimos de cerca {[r for r, _ in ramas]}")
    perfil = perfil_a40(P, [(r, ths[int(f.minimo)]) for r, f in ramas], grilla)
    perfil.to_csv(salida / "calibracion_pupila_esferica_perfil.csv", index=False)

    # ------------------------------------------------ los cinco Zernike desde el arranque de a40 = -6.40
    print("\n[E4] cinco Zernike: arranque de la corrida exploratoria (a40 -1.5 pi, beta0 41, z0 -1 um) y desde "
          "el mejor minimo de cerca de la esferica sola")
    cinco = []
    for f, th in arranques(P, [(-1, -1000.0)], [41.0], compartidos=COMPARTIDOS, etiqueta="cinco, exploratorio"):
        cinco.append(dict(desde="exploratorio (a40 -1.5 pi, beta0 41, z0 -1 um)", **f))
    j = int(cerca.iloc[0].minimo)
    t0 = time.time()
    th5 = ajustar(P, ths[j], indices(P), max_iter=1500)
    cinco.append(dict(desde=f"minimo {j} de la esferica sola (cerca)", ajuste="cinco", nll=P.nll(th5),
                      nll_mascara_binaria=nll_binaria(P, th5), **resumen_theta(P, th5), segundos=time.time() - t0))
    print(f"  desde el minimo {j}: nll {cinco[-1]['nll']:.2f}; " + ", ".join(
        f"{k} {cinco[-1][k]:+.3f}" for k in ZERNIKE + ("beta", "delta_um", "z_real_media")))
    cinco = pd.DataFrame(cinco)
    cinco.to_csv(salida / "calibracion_pupila_esferica_cinco.csv", index=False)

    # ------------------------------------------------ incertidumbre del ganador (y del mejor de cerca)
    ganadores = [int(mins.sort_values("nll_pares").iloc[0].minimo)]
    if int(cerca.iloc[0].minimo) not in ganadores:
        ganadores.append(int(cerca.iloc[0].minimo))
    resumen = dict(minimos=mins.drop(columns=["arranques"]).to_dict("records"), incertidumbre={})
    for j in ganadores:
        f = mins.set_index("minimo").loc[j]
        print(f"\n[E5] minimo {j} ({f.variante}): Hessiana y bootstrap de bloques de {a.bloque} cuadros pares "
              f"({a.n_boot} remuestreos)")
        _, C, vals, Cd, ev_min = covarianza(P, ths[j], indices(P, ESFERICA, POR_CUADRO))
        dv, npx = P.deviance(ths[j])
        sobre = dv / (npx - (len(ESFERICA) + P.T * NP))
        boot = bootstrap_bloques(d, pares, ths[j], mod, ESFERICA, a.n_boot, a.bloque, rng)
        boot.to_csv(salida / f"calibracion_pupila_esferica_bootstrap_{f.variante}.csv", index=False)
        inc = {}
        for k, ee in (("a40", np.sqrt(C[0, 0])), ("beta", np.sqrt(Cd[0, 0])), ("delta_um", np.sqrt(C[6, 6])),
                      ("z_real_media", np.sqrt(Cd[1, 1]))):
            inc[k] = dict(valor=float(f[k]), ee_hessiana=float(ee), ee_hessiana_sobredispersion=float(ee * np.sqrt(sobre)),
                          ee_bootstrap=float(boot[k].std(ddof=1)), p2_5=float(boot[k].quantile(0.025)),
                          p97_5=float(boot[k].quantile(0.975)), mediana_bootstrap=float(boot[k].median()))
            print(f"  {k}: {inc[k]['valor']:+.4g}, ee Hessiana {ee:.3g} (x sobredispersion {inc[k]['ee_hessiana_sobredispersion']:.3g}), "
                  f"bootstrap {inc[k]['ee_bootstrap']:.3g} [{inc[k]['p2_5']:+.4g}, {inc[k]['p97_5']:+.4g}]")
        inc["deviance_nu"], inc["autovalor_minimo"] = sobre, ev_min
        resumen["incertidumbre"][f"{j} {f.variante}"] = inc
    resumen["segundos"] = time.time() - t_ini
    (salida / "calibracion_pupila_esferica.json").write_text(json.dumps(resumen, indent=1, default=float))
    figura_esferica(pd.DataFrame([f for f, _ in res]), mins, perfil,
                    {j: pd.read_csv(salida / f"calibracion_pupila_esferica_bootstrap_{mins.set_index('minimo').loc[j].variante}.csv")
                     for j in ganadores}, salida / "fig_calibracion_pupila_esferica.png")
    print(f"\nlisto en {(time.time() - t_ini) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--tif", default=adr.TIF_DEF)
    p.add_argument("--cache", default=None, help="npz con los datos preparados (se crea si no existe)")
    p.add_argument("--betas", type=float, nargs="+", default=[41.0, 45.0, 49.0])
    p.add_argument("--z-barrido", type=float, nargs=3, default=[-4000.0, 8000.0, 1000.0], metavar=("INI", "FIN", "PASO"),
                   help="z media del real [nm] del barrido con la pupila actual que elige los arranques")
    p.add_argument("--n-minimos", type=int, default=2, help="minimos locales del barrido por signo que se ajustan")
    p.add_argument("--n-boot", type=int, default=30)
    p.add_argument("--bloque", type=int, default=5, help="cuadros pares por bloque del bootstrap")
    p.add_argument("--n-sint", type=int, default=2)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--cuadros", type=int, default=None, help="usa solo los primeros N cuadros (pruebas)")
    p.add_argument("--perfil-ratio", type=float, nargs="+", default=None, metavar="R",
                   help="solo el perfil en la reflectividad: fotones del reflejo = R x los del real en todos los "
                        "cuadros, desde el mejor arranque de cada signo de calibracion_pupila_arranques.csv")
    p.add_argument("--desde-arranques", action="store_true",
                   help="no repite el barrido ni los arranques: reajusta el mejor de cada signo de "
                        "calibracion_pupila_arranques.csv y sigue con la validacion")
    p.add_argument("--solo-esferica", action="store_true",
                   help="el modelo que se simula: solo a40 (signo libre), beta y delta compartidos; arranques, minimos, "
                        "perfil en a40, cinco Zernike desde el arranque exploratorio y bootstrap del ganador")
    p.add_argument("--a40-arranques", type=float, nargs="+", default=[-4.71, -5.16, -6.40, 4.71, 5.16, 6.40])
    p.add_argument("--z-arranques-cerca", type=float, nargs="+", default=[-1000.0, 0.0],
                   help="z media inicial del real [nm] de los arranques con a40 < 0 (--solo-esferica)")
    p.add_argument("--z-arranques-lejos", type=float, nargs="+", default=[6000.0, -2000.0],
                   help="idem con a40 > 0 (los minimos locales del barrido)")
    p.add_argument("--perfil-a40", type=float, nargs=3, default=[-8.0, -3.5, 0.25], metavar=("INI", "FIN", "PASO"))
    p.add_argument("--ramas-perfil", type=int, default=3, help="minimos de cerca (distintos en z o a40) con perfil")
    p.add_argument("--salida", default=str(AQUI / "calibracion_pupila"))
    a = p.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.semilla)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_ini = time.time()
    d = preparar(a.tif, a.cache)
    if a.cuadros:
        d = {k: (v[:a.cuadros] if k in ("e", "o", "masc", "real", "refl", "oscuro") else v) for k, v in d.items()}
    T = len(d["e"])
    mod, mod_bin = Modelo(dev), Modelo(dev, binaria=True)
    err, err_rampa = verificar(mod_bin, mod, d)
    phi = float(np.degrees(np.arctan2(d["m"][1], d["m"][0])))
    print(f"Yeast6_1, canal derecho: {T} cuadros, ventanas {B} x {B}, discos de {R_DISCO:g} px "
          f"({d['masc'].sum((1, 2)).mean():.0f} px por cuadro); phi = {phi:+.1f} deg; linea en el borde de la franja a "
          f"{float(d['v_borde']):+.1f} px del punto medio (borde a {float(d['angulo_borde_t']):+.1f} deg de t); nivel oscuro "
          f"{d['oscuro'].mean():.2f} ADU; ruido de lectura {float(d['rn_adu']):.2f} ADU = {float(d['rn_adu']) * GANANCIA:.2f} e- "
          f"(mediana; rms {float(d['rn_rms_adu']):.2f} ADU)")
    print(f"modelo torch vs PSFRecortada (mascara binaria): {err:.1e} del pico; mascara en rampa: {err_rampa:.1e}")
    pares, impares = np.arange(0, T, 2), np.arange(1, T, 2)
    P = Problema(d, pares, mod)
    Pi = Problema(d, impares, mod)
    if a.perfil_ratio:
        perfil_ratio(P, pd.read_csv(salida / "calibracion_pupila_arranques.csv"), a.perfil_ratio, salida)
        return
    if a.solo_esferica:
        solo_esferica(a, d, P, Pi, mod, salida, rng)
        return

    # ---------------------------------------------------- 2. arranques (calibrada y actual)
    if a.desde_arranques:
        tabla = pd.read_csv(salida / "calibracion_pupila_arranques.csv")
        barr = pd.read_csv(salida / "calibracion_pupila_barrido.csv")
        print("\n[2] desde calibracion_pupila_arranques.csv: se reajusta el mejor arranque de cada signo")
        mejores = {}
        for nombre, comp in (("calibrada", COMPARTIDOS), ("actual", ("beta", "delta_um"))):
            for sg in (-1, 1):
                t = tabla[(tabla.ajuste == nombre) & (np.sign(tabla.a40) == sg)].sort_values("nll").iloc[0]
                (f, th), = arranques(P, [(int(t.signo_inicial), float(t.z_real_inicial))], [float(t.beta_inicial)],
                                     compartidos=comp, etiqueta=nombre)
                print(f"    (en la tabla: nll {t.nll:.2f})")
                mejores[nombre, sg] = (f, th)
    else:
        zb = np.arange(a.z_barrido[0], a.z_barrido[1] + 1e-6, a.z_barrido[2])
        print(f"\n[2] barrido en z (pupila actual, beta 45, solo parametros por cuadro), cuadros pares")
        barr = barrido_z(P, (-1, 1), zb)
        barr.to_csv(salida / "calibracion_pupila_barrido.csv", index=False)
        puntos = minimos_locales(barr, a.n_minimos)
        print(f"  arranques (signo, z0): {puntos} x beta0 {a.betas}")
        print("\n[2] ajuste en los cuadros pares desde varios arranques")
        todos = arranques(P, puntos, a.betas, etiqueta="calibrada")
        print("  pupila actual (a40 = +-1.5 pi fijo, sin otros terminos; beta y delta libres):")
        actual = arranques(P, puntos, a.betas, compartidos=("beta", "delta_um"), etiqueta="actual")
        tabla = pd.DataFrame([f for f, _ in todos + actual])
        tabla.to_csv(salida / "calibracion_pupila_arranques.csv", index=False)
        mejores = {}
        for nombre, lista in (("calibrada", todos), ("actual", actual)):
            for sg in (-1, 1):
                cand = [(f, th) for f, th in lista if np.sign(f["a40"]) == sg]
                if cand:
                    mejores[nombre, sg] = min(cand, key=lambda x: x[0]["nll"])
        print("  minimos distintos (calibrada):")
        for g in distintos([f for f, _ in todos]):
            print(f"    nll {g['nll']:.2f} (x{g['n']}): a40 {g['a40']:+.3f}, beta {g['beta']:.2f}, delta {g['delta_um']:+.3f} um, "
                  f"z real {g['z_real_media']:+.0f}, z' {g['z_reflejo_media']:+.0f}")
    ruta_ratio = salida / "calibracion_pupila_ratio.csv"
    ratio = pd.read_csv(ruta_ratio) if ruta_ratio.exists() else None

    # ---------------------------------------------------- 3b. cuadros impares
    print("\n[3b] cuadros impares: solo los parametros por cuadro, con los compartidos de cada ajuste")
    val, mus, ths_imp = [], {}, {}
    res_imp = pd.DataFrame(dict(cuadro=impares))
    for (nombre, sg), (f, th) in sorted(mejores.items()):
        thi = ajustar_cuadros(Pi, th[:NC], f["beta"])
        nll_i = Pi.nll(thi)
        dv, npx_i = Pi.deviance(thi)
        res, por_cuadro, mu = residuos(Pi, thi)
        res_p, _, _ = residuos(P, th)
        mus[nombre, sg], ths_imp[nombre, sg] = mu, thi
        for k_, v_ in por_cuadro.items():
            res_imp[f"residuo_{k_}_{nombre}_{VARIANTE[sg]}"] = v_
        pci_ = thi[NC:].view(Pi.T, NP).cpu().numpy()
        val.append(dict(pupila=nombre, variante=VARIANTE[sg], a40=f["a40"], beta=f["beta"], delta_um=f["delta_um"],
                        nll_pares=f["nll"], nll_impares=nll_i, deviance_nu_impares=dv / (npx_i - Pi.T * NP),
                        **{k + "_impares": v for k, v in res.items()}, **{k + "_pares": v for k, v in res_p.items()},
                        z_real_media_impares=float(Pi.geometria(thi)[1].mean()),
                        N_real_media_impares=float(np.exp(pci_[:, 3]).mean()),
                        reflejo_sobre_real_impares=float(np.mean(np.exp(pci_[:, 4] - pci_[:, 3]))),
                        fondo_medio_impares=float(np.exp(pci_[:, 5]).mean())))
        print(f"  {nombre} {VARIANTE[sg]}: nll impares {nll_i:.1f}; residuo sin ruido real {res['res_real']:.3f}, reflejo "
              f"{res['res_reflejo']:.3f}, total {res['res_total']:.3f} (con ruido {res['res_total_con_ruido']:.3f})", flush=True)
    val = pd.DataFrame(val)
    val["delta_logL_impares"] = val.nll_impares.min() - val.nll_impares
    val["delta_logL_pares"] = val.nll_pares.min() - val.nll_pares
    res_imp.to_csv(salida / "calibracion_pupila_residuos_impares.csv", index=False)
    # imagenes medias de los impares (alineadas en el pixel entero del real/reflejo de la calibrada cerca)
    orden = [("actual", -1), ("actual", 1), ("calibrada", -1), ("calibrada", 1)]
    th_ref = ths_imp["calibrada", -1]
    rec = recortes_medios(Pi, th_ref, [mus[k] for k in orden])
    fondo_tot = float(np.exp(th_ref[NC:].view(Pi.T, NP)[:, 5].cpu().numpy()).sum())
    res_medio = {nom: residuo_medio(rec[nom], fondo_tot) for nom in rec}
    for i, (nombre, sg) in enumerate(orden):
        sel = (val.pupila == nombre) & (val.variante == VARIANTE[sg])
        val.loc[sel, "res_real_imagen_media"] = res_medio["real"][i]
        val.loc[sel, "res_reflejo_imagen_media"] = res_medio["reflejo"][i]
    val.to_csv(salida / "calibracion_pupila_validacion.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(val.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    figura_imagenes(rec, orden, res_medio, Pi.T, salida / "fig_calibracion_pupila_imagenes.png")
    figura_signo(tabla, barr, ratio, res_imp, salida / "fig_calibracion_pupila_signo.png")

    # ---------------------------------------------------- 3c y 3a por variante
    resumen = dict(deviance_nu={}, phi_deg=phi, ruido_lectura_e=float(d["rn_adu"]) * GANANCIA,
                   error_modelo_vs_psfrecortada=err, error_rampa=err_rampa,
                   actual={VARIANTE[k[1]]: describir(v[1][:NC].cpu().numpy()) for k, v in mejores.items() if k[0] == "actual"},
                   calibrada={})
    for sg in (-1, 1):
        f, th = mejores["calibrada", sg]
        sobre = analizar_variante(a, d, P, Pi, mod, tabla, f, th, ths_imp["calibrada", sg], rng, salida)
        resumen["calibrada"][VARIANTE[sg]] = describir(th[:NC].cpu().numpy())
        resumen["deviance_nu"][VARIANTE[sg]] = sobre
    resumen["segundos"] = time.time() - t_ini
    (salida / "calibracion_pupila.json").write_text(json.dumps(resumen, indent=1, default=float))
    print(f"\nlisto en {(time.time() - t_ini) / 60:.1f} min")


VARIANTE = {-1: "cerca", 1: "lejos"}


def analizar_variante(a, d, P, Pi, mod, tabla, fcal, th_cal, th_imp, rng, salida):
    """Para el minimo calibrado de una variante: Hessiana, bootstrap de bloques,
    recuperacion en sinteticos, tabla por cuadro y figura. Devuelve deviance / nu."""
    sg = int(np.sign(fcal["a40"]))
    var = VARIANTE[sg]
    dev = mod.dev
    T = len(d["e"])
    pares, impares = P.cuadros, Pi.cuadros
    print(f"\n[3c] {var}: incertidumbre por la Hessiana")
    lib = indices(P)
    H, C, vals, Cd, ev_min = covarianza(P, th_cal, lib)
    sh = th_cal[:NC].cpu().numpy()
    se = np.sqrt(np.diag(C)[:NC])
    se_beta = np.sqrt(Cd[0, 0])
    corr = C[:NC, :NC] / np.outer(se, se)
    dev_tot, npx = P.deviance(th_cal)
    nlib = int(lib.sum())
    sobre = dev_tot / (npx - nlib)
    print(f"  autovalor minimo de la Hessiana {ev_min:.3g}; deviance / nu = {sobre:.2f} ({npx} px, {nlib} parametros)")

    print(f"[3c] {var}: bootstrap de bloques de {a.bloque} cuadros pares ({a.n_boot} remuestreos)")
    boot = bootstrap_bloques(d, pares, th_cal, mod, COMPARTIDOS, a.n_boot, a.bloque, rng)
    boot.to_csv(salida / f"calibracion_pupila_bootstrap_{var}.csv", index=False)

    params = []
    for i, k in enumerate(COMPARTIDOS):
        e_h = se[i] if k != "beta" else se_beta
        params.append(dict(parametro=k, valor=fcal[k], ee_hessiana=e_h, ee_hessiana_sobredispersion=e_h * np.sqrt(sobre),
                           ee_bootstrap=float(boot[k].std(ddof=1)) if len(boot) > 1 else np.nan,
                           p2_5=float(boot[k].quantile(0.025)) if len(boot) else np.nan,
                           p97_5=float(boot[k].quantile(0.975)) if len(boot) else np.nan))
    params.append(dict(parametro="z_real_media", valor=vals[1], ee_hessiana=np.sqrt(Cd[1, 1]),
                       ee_hessiana_sobredispersion=np.sqrt(Cd[1, 1] * sobre),
                       ee_bootstrap=float(boot.z_real_media.std(ddof=1)) if len(boot) > 1 else np.nan,
                       p2_5=float(boot.z_real_media.quantile(0.025)) if len(boot) else np.nan,
                       p97_5=float(boot.z_real_media.quantile(0.975)) if len(boot) else np.nan))
    params = pd.DataFrame(params)
    params.to_csv(salida / f"calibracion_pupila_parametros_{var}.csv", index=False)
    cb = boot[list(COMPARTIDOS)].corr().to_numpy() if len(boot) > 2 else np.full((NC, NC), np.nan)
    _, J = jacobiano_derivados(P, th_cal)
    cz = (C @ J[1])[:NC] / (se * np.sqrt(Cd[1, 1]))
    corr_t = pd.DataFrame([dict(p1=COMPARTIDOS[i], p2=COMPARTIDOS[j], corr_hessiana=corr[i, j], corr_bootstrap=cb[i, j])
                           for i in range(NC) for j in range(i + 1, NC)]
                          + [dict(p1=k, p2="z_real_media", corr_hessiana=cz[i],
                                  corr_bootstrap=float(np.corrcoef(boot[k], boot.z_real_media)[0, 1]) if len(boot) > 2 else np.nan)
                             for i, k in enumerate(COMPARTIDOS)])
    corr_t.to_csv(salida / f"calibracion_pupila_correlaciones_{var}.csv", index=False)
    print(params.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("  correlaciones con |r| > 0.5 (Hessiana / bootstrap):")
    for _, f in corr_t[(corr_t.corr_hessiana.abs() > 0.5) | (corr_t.corr_bootstrap.abs() > 0.5)].iterrows():
        print(f"    {f.p1} - {f.p2}: {f.corr_hessiana:+.2f} / {f.corr_bootstrap:+.2f}")

    # ---------------------------------------------------- 3a. recuperacion en sintetico
    print(f"\n[3a] {var}: recuperacion en {a.n_sint} peliculas sinteticas con esta pupila calibrada")
    cal = tabla[tabla.ajuste == "calibrada"]
    puntos = [(int(s_), float(cal[np.sign(cal.a40) == s_].sort_values("nll").iloc[0].z_real_inicial)) for s_ in (-1, 1)]
    sint = []
    Pt = Problema(d, np.arange(T), mod)
    pc_all = np.zeros((T, NP))
    pc_all[pares] = th_cal[NC:].view(P.T, NP).cpu().numpy()
    pc_all[impares] = th_imp[NC:].view(Pi.T, NP).cpu().numpy()
    th_all = torch.cat([th_cal[:NC], torch.as_tensor(pc_all.ravel(), dtype=mod.dt, device=dev)])
    with torch.no_grad():
        mu_all = torch.cat([Pt.mu(th_all, s_)[0] for s_ in Pt.bloques()]).cpu().numpy()
    verdad = describir(sh)
    zt = float(Pt.geometria(th_all)[1][pares].mean())
    for r in range(a.n_sint):
        g = np.random.default_rng(1000 + r)
        el = g.poisson(mu_all) + g.normal(0, float(d["rn_adu"]) * GANANCIA, mu_all.shape)
        adu = np.round(el / GANANCIA + d["oscuro"][:, None, None])                 # cuantizacion de la camara
        e_s = (adu - d["oscuro"][:, None, None]) * GANANCIA
        Ps = Problema(d, pares, mod, e=e_s)
        rs = arranques(Ps, puntos, [45.0], etiqueta=f"sintetico {var} {r}", verbose=True)
        fs, ths = min(rs, key=lambda x: x[0]["nll"])
        otro = [f_["nll"] for f_, _ in rs if np.sign(f_["a40"]) != np.sign(fs["a40"])]
        _, Cs, vs, Cds, _ = covarianza(Ps, ths, indices(Ps))
        ses = np.sqrt(np.diag(Cs)[:NC])
        for i, k in enumerate(COMPARTIDOS):
            e_h = ses[i] if k != "beta" else np.sqrt(Cds[0, 0])
            sint.append(dict(pelicula=r, parametro=k, verdad=verdad[k], ajuste=fs[k], error=fs[k] - verdad[k],
                             ee_hessiana=e_h, error_en_ee=(fs[k] - verdad[k]) / e_h, signo_recuperado=np.sign(fs["a40"]) == sg,
                             nll_mejor=fs["nll"], nll_otro_signo=min(otro) if otro else np.nan))
        sint.append(dict(pelicula=r, parametro="z_real_media", verdad=zt, ajuste=fs["z_real_media"],
                         error=fs["z_real_media"] - zt, ee_hessiana=np.sqrt(Cds[1, 1]),
                         error_en_ee=(fs["z_real_media"] - zt) / np.sqrt(Cds[1, 1]), signo_recuperado=np.sign(fs["a40"]) == sg,
                         nll_mejor=fs["nll"], nll_otro_signo=min(otro) if otro else np.nan))
    sint = pd.DataFrame(sint)
    sint.to_csv(salida / f"calibracion_pupila_sintetico_{var}.csv", index=False)
    print(sint.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ---------------------------------------------------- por cuadro y figura
    xy, z, xy_f, z_f, beta = (g_.detach().cpu().numpy() for g_ in P.geometria(th_cal))
    xyi, zi, xyfi, zfi, _ = (g_.detach().cpu().numpy() for g_ in Pi.geometria(th_imp))
    pci = th_imp[NC:].view(Pi.T, NP).cpu().numpy()
    pcp = th_cal[NC:].view(P.T, NP).cpu().numpy()
    s_r = (d["real"] - d["q"]) @ d["m"] * PX
    s_f = (d["refl"] - d["q"]) @ d["m"] * PX
    tri = {bb: -(s_r + s_f) / 2 / np.tan(np.radians(bb)) + (s_r - s_f) / 2 * np.tan(np.radians(bb))
           for bb in (45.0, float(np.degrees(beta)))}
    filas_c = []
    for paridad, cs, (xy_, z_, xyf_, zf_, pc_) in (("par", pares, (xy, z, xy_f, z_f, pcp)),
                                                   ("impar", impares, (xyi, zi, xyfi, zfi, pci))):
        for k, t in enumerate(cs):
            filas_c.append(dict(cuadro=t, paridad=paridad, x_real_px=xy_[k, 0], y_real_px=xy_[k, 1], z_real_nm=z_[k],
                                x_reflejo_px=xyf_[k, 0], y_reflejo_px=xyf_[k, 1], z_reflejo_nm=zf_[k],
                                N_real=np.exp(pc_[k, 3]), N_reflejo=np.exp(pc_[k, 4]), fondo=np.exp(pc_[k, 5]),
                                z_triangulada_borde_45=tri[45.0][t], z_triangulada_borde_beta=list(tri.values())[-1][t]))
    cuadros = pd.DataFrame(filas_c).sort_values("cuadro")
    cuadros.to_csv(salida / f"calibracion_pupila_cuadros_{var}.csv", index=False)
    fase = torch.einsum("i,ijk->jk", th_cal[1:5], mod.Z[1:]).cpu().numpy() * mod.ap.cpu().numpy()   # sin a40
    figura_medidas(var, fcal, cuadros, corr, sint, fase, mod, salida / f"fig_calibracion_pupila_medidas_{var}.png")
    return sobre


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------

COLOR = {-1: "#2a78d6", 1: "#eb6834", "gris": "#52514e", "rejilla": "#e6e5e0", "tri": "#1baf7a"}
NOMBRE_SIGNO = {-1: "a40 < 0 (cerca)", 1: "a40 > 0 (lejos)"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def figura_imagenes(rec, orden, res_medio, n_cuadros, ruta):
    """Real y reflejo (suma de los cuadros impares, alineados en el pixel entero):
    datos, modelo actual (los dos signos), modelo calibrado y residuos."""
    plt = _plt()
    titulos = [f"{n} {VARIANTE[s]}" + (f" ({'-' if s < 0 else '+'}1.5 pi)" if n == "actual" else "") for n, s in orden]
    fig, axes = plt.subplots(4, 1 + len(orden), figsize=(2.3 * (1 + len(orden)), 9.6))
    for i, nom in enumerate(("real", "reflejo")):
        r = rec[nom]
        n = r.shape[-1] // 2
        fondo = np.median(np.r_[r[0][0], r[0][-1], r[0][:, 0], r[0][:, -1]])
        vmax = np.percentile(r[0], 99.8)
        for j in range(1 + len(orden)):
            ax = axes[2 * i, j]
            ax.imshow(r[j], cmap="magma", vmin=fondo, vmax=vmax)
            ax.set_title(("datos" if j == 0 else titulos[j - 1]) + f"\n{nom}", fontsize=7.5)
        lim = np.percentile(np.abs(r[0] - r[1:]), 99.5)
        axes[2 * i + 1, 0].imshow(np.log10(np.clip(r[0] - fondo, 1, None)), cmap="magma")
        axes[2 * i + 1, 0].set_title(f"datos (log10, sin fondo)\n{nom}", fontsize=7.5)
        for j in range(len(orden)):
            ax = axes[2 * i + 1, j + 1]
            im = ax.imshow(r[0] - r[j + 1], cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"datos - modelo\nresiduo relativo {res_medio[nom][j]:.2f}", fontsize=7.5)
        fig.colorbar(im, ax=axes[2 * i + 1, -1], fraction=0.046, label="e- (suma de cuadros)")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"Yeast6_1, canal derecho: suma de los {n_cuadros} cuadros impares (41 x 41 px, alineados en el pixel entero; "
                 "parametros por cuadro reajustados)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def figura_signo(tabla, barr, ratio, res_imp, ruta):
    """El signo de a40: minimos de los arranques, barrido en z, perfil en la
    reflectividad y residuo por cuadro en los impares."""
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    nmin = tabla.nll.min()
    # a) minimos
    ax = axes[0, 0]
    for sg in (-1, 1):
        s_ = tabla[(tabla.ajuste == "calibrada") & (np.sign(tabla.a40) == sg)]
        ax.scatter(s_.z_real_media / 1000, s_.nll - nmin, s=30, color=COLOR[sg], edgecolor="white", linewidth=0.6,
                   zorder=3, label=f"calibrada, {NOMBRE_SIGNO[sg]}")
        s_ = tabla[(tabla.ajuste == "actual") & (np.sign(tabla.a40) == sg)]
        ax.scatter(s_.z_real_media / 1000, s_.nll - nmin, s=30, marker="s", facecolor="none", edgecolor=COLOR[sg],
                   label=f"actual (1.5 pi), {NOMBRE_SIGNO[sg]}")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("z media del real en el minimo [um]")
    ax.set_ylabel("-log L - minimo global (pares)")
    ax.set_title("a) minimos desde cada arranque (cinco Zernike, beta y linea libres)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    # b) barrido
    ax = axes[0, 1]
    for sg in (-1, 1):
        s_ = barr[(barr.signo == sg) & np.isfinite(barr.nll)].sort_values("z0")
        ax.plot(s_.z0 / 1000, s_.nll - nmin, "o-", color=COLOR[sg], ms=4, lw=1.4, label=NOMBRE_SIGNO[sg])
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("z media inicial del real [um] (fija la linea)")
    ax.set_ylabel("-log L - minimo global (pares)")
    ax.set_title("b) barrido con la pupila actual (beta 45, solo parametros por cuadro)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    # c) perfil en la reflectividad
    ax = axes[1, 0]
    if ratio is not None:
        for sg in (-1, 1):
            s_ = ratio[ratio.signo == sg].sort_values("R")
            ax.plot(s_.R, s_.nll - nmin, "o-", color=COLOR[sg], ms=4, lw=1.4, label=NOMBRE_SIGNO[sg])
            ax.axhline(s_.nll_libre.iloc[0] - nmin, color=COLOR[sg], lw=0.8, ls=":")
        ax.axvspan(0.45, 1.0, color=COLOR["rejilla"], zorder=0)
        ax.text(0.47, 0.97, "ratio_reflejo_rango de celula.yaml", transform=ax.get_xaxis_transform(), fontsize=6.5,
                va="top", color=COLOR["gris"])
        ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("R = fotones del reflejo / del real (comun a los cuadros)")
    ax.set_ylabel("-log L - minimo global (pares)")
    ax.set_title("c) perfil en la reflectividad (punteado: R libre por cuadro)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    # d) residuo por cuadro en los impares
    ax = axes[1, 1]
    for nombre, ls in (("actual", "--"), ("calibrada", "-")):
        for sg in (-1, 1):
            col = f"residuo_total_{nombre}_{VARIANTE[sg]}"
            if col in res_imp:
                ax.plot(res_imp.cuadro, res_imp[col], ls, color=COLOR[sg], lw=1.3, label=f"{nombre}, {NOMBRE_SIGNO[sg]}")
    ax.set_xlabel("cuadro (impares)")
    ax.set_ylabel("residuo relativo sin ruido (real + reflejo)")
    ax.set_title("d) validacion en los cuadros impares", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    for ax in axes.flat:
        ax.grid(True, color=COLOR["rejilla"], lw=0.6)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def figura_medidas(var, fcal, cuadros, corr, sint, fase, mod, ruta):
    """Una variante: z por cuadro, fase calibrada sin a40, recuperacion en sintetico y correlaciones."""
    plt = _plt()
    sg = int(np.sign(fcal["a40"]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    # a) z por cuadro
    ax = axes[0, 0]
    for par, mk in (("par", "o"), ("impar", "^")):
        c = cuadros[cuadros.paridad == par]
        ax.plot(c.cuadro, c.z_real_nm / 1000, mk, color=COLOR[sg], ms=3, label=f"real, ajuste ({par}es)")
        ax.plot(c.cuadro, c.z_reflejo_nm / 1000, mk, color=COLOR[sg], ms=3, mfc="none", label=f"reflejo, ajuste ({par}es)")
    ax.plot(cuadros.cuadro, cuadros.z_triangulada_borde_45 / 1000, "-", color=COLOR["tri"], lw=1.2,
            label="real, triangulada (borde de la franja, beta 45)")
    ax.plot(cuadros.cuadro, cuadros.z_triangulada_borde_beta / 1000, "--", color=COLOR["tri"], lw=1.2,
            label=f"real, triangulada (borde, beta {fcal['beta']:.1f})")
    ax.set_xlabel("cuadro"); ax.set_ylabel("z [um] (+ hacia el objetivo)")
    ax.set_title(f"a) z por cuadro, {var} calibrada (linea a {fcal['delta_um']:+.2f} um del borde)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    # b) fase sin a40
    ax = axes[0, 1]
    M = mod.M
    f = np.fft.fftshift(fase)
    apm = np.fft.fftshift(mod.ap.cpu().numpy()) > 0
    c0 = M // 2
    r = int(np.ceil(np.sqrt(apm.sum() / np.pi))) + 2
    img = np.where(apm, f, np.nan)[c0 - r:c0 + r + 1, c0 - r:c0 + r + 1]
    lim = np.nanmax(np.abs(img))
    im = ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", extent=(-1, 1, -1, 1))
    fig.colorbar(im, ax=ax, fraction=0.046, label="fase [rad]")
    ax.set_xlabel("fx (pupila normalizada; marco del canal derecho)"); ax.set_ylabel("fy")
    ax.set_title(f"b) fase calibrada sin a40 ({fcal['a40']:+.2f} rad)", fontsize=9)
    # c) sintetico
    ax = axes[1, 0]
    ks = list(COMPARTIDOS) + ["z_real_media"]
    for r_ in sorted(sint.pelicula.unique()):
        s_ = sint[sint.pelicula == r_].set_index("parametro").reindex(ks)
        ax.plot(np.arange(len(ks)) + 0.12 * r_, s_.error_en_ee, "o", color=COLOR[sg], ms=4,
                label="peliculas sinteticas" if r_ == 0 else None)
    ax.axhspan(-2, 2, color=COLOR["rejilla"], zorder=0)
    ax.axhline(0, color=COLOR["gris"], lw=0.6)
    ax.set_xticks(range(len(ks)), ks, rotation=40, ha="right")
    ax.set_ylabel("(ajuste - verdad) / ee de la Hessiana")
    ax.set_title("c) recuperacion en sintetico (banda: +-2 ee)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, color=COLOR["rejilla"], lw=0.6)
    axes[0, 0].grid(True, color=COLOR["rejilla"], lw=0.6)
    # d) correlaciones
    ax = axes[1, 1]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(NC), COMPARTIDOS, rotation=40, ha="right"); ax.set_yticks(range(NC), COMPARTIDOS)
    for i in range(NC):
        for j in range(NC):
            ax.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("d) correlaciones de los compartidos (Hessiana)", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def figura_esferica(arr, mins, perfil, boots, ruta):
    """--solo-esferica: minimos de los arranques, perfil en a40 (nll, beta, z),
    pares frente a impares y bootstrap de a40."""
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    nmin = mins.nll_pares.min()
    mk_beta = {41.0: "v", 45.0: "o", 49.0: "^"}
    # a) arranques
    ax = axes[0, 0]
    for sg in (-1, 1):
        for b0, mk in mk_beta.items():
            s_ = arr[(np.sign(arr.a40) == sg) & (arr.beta_inicial == b0)]
            if len(s_):
                ax.scatter(s_.a40, s_.nll - nmin, s=34, marker=mk, color=COLOR[sg], edgecolor="white", linewidth=0.6,
                           zorder=3, label=f"{VARIANTE[sg]}, beta0 {b0:g}")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("a40 en el minimo [rad]")
    ax.set_ylabel("-log L - minimo global (pares)")
    ax.set_title("a) minimos desde cada arranque (solo a40, beta y linea)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    # b-d) perfil
    colores = ["#2a78d6", "#1baf7a", "#8a5cd1", "#52514e"]
    for i, (rama, s_) in enumerate(perfil.groupby("rama", sort=False)):
        s_ = s_[np.isfinite(s_.nll)].sort_values("a40_fijo")
        c = colores[i % len(colores)]
        f = mins.set_index("minimo").loc[int(rama)]
        etiqueta = f"desde el minimo {rama} (a40 {f.a40:+.2f}, z {f.z_real_media / 1000:+.2f} um)"
        axes[0, 1].plot(s_.a40_fijo, s_.nll - nmin, "o-", color=c, ms=3.5, lw=1.4, label=etiqueta)
        axes[0, 2].plot(s_.a40_fijo, s_.beta, "o-", color=c, ms=3.5, lw=1.4, label=etiqueta)
        axes[1, 0].plot(s_.a40_fijo, s_.z_real_media / 1000, "o-", color=c, ms=3.5, lw=1.4, label=etiqueta)
    cer = mins[mins.variante == "cerca"]
    axes[0, 1].plot(cer.a40, cer.nll_pares - nmin, "*", color="black", ms=8, label="minimos con a40 libre")
    axes[0, 2].plot(cer.a40, cer.beta, "*", color="black", ms=8)
    axes[1, 0].plot(cer.a40, cer.z_real_media / 1000, "*", color="black", ms=8)
    axes[0, 1].set_ylabel("-log L - minimo global (pares)")
    axes[0, 1].set_title("b) perfil en a40 (beta, linea y por cuadro libres)", fontsize=9)
    axes[0, 2].set_ylabel("beta [grados]")
    axes[0, 2].set_title("c) beta a lo largo del perfil", fontsize=9)
    axes[1, 0].set_ylabel("z media del real [um]")
    axes[1, 0].set_title("d) z del real a lo largo del perfil", fontsize=9)
    for ax in (axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.set_xlabel("a40 fijo [rad] (cerca: amp_esferica = -a40)")
        ax.legend(frameon=False, fontsize=6.5)
    # e) pares frente a impares
    ax = axes[1, 1]
    for sg in (-1, 1):
        s_ = mins[mins.variante == VARIANTE[sg]]
        ax.scatter(-s_.delta_logL_pares, -s_.delta_logL_impares, s=34, color=COLOR[sg], edgecolor="white", linewidth=0.6,
                   zorder=3, label=VARIANTE[sg])
    lim = max(1.0, float(-mins.delta_logL_pares.min()), float(-mins.delta_logL_impares.min()))
    ax.plot([0, lim], [0, lim], "-", color=COLOR["gris"], lw=0.7)
    ax.set_xscale("symlog", linthresh=100); ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("-log L - mejor (pares, ajuste)")
    ax.set_ylabel("-log L - mejor (impares, validacion)")
    ax.set_title("e) minimos distintos: pares frente a impares", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    # f) bootstrap de a40
    ax = axes[1, 2]
    for j, b in boots.items():
        f = mins.set_index("minimo").loc[j]
        sg = int(np.sign(f.a40))
        ax.hist(b.a40 - f.a40, bins=15, color=COLOR[sg], alpha=0.6,
                label=f"minimo {j} ({f.variante}, a40 {f.a40:+.3f}): ee {b.a40.std(ddof=1):.3f}")
    ax.axvline(0, color=COLOR["gris"], lw=0.7)
    ax.set_xlabel("a40 del remuestreo - a40 del minimo [rad]")
    ax.set_ylabel("remuestreos")
    ax.set_title("f) bootstrap de bloques de a40", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    for ax in axes.flat:
        ax.grid(True, color=COLOR["rejilla"], lw=0.6)
    fig.suptitle("Yeast6_1, canal derecho: el modelo que se simula (fase a40 Z4^0 sola, beta y linea compartidos)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
