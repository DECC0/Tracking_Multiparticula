"""
Generador de videos sinteticos de fluorescencia (un solo canal) con emisores
en movimiento browniano 3D. Version nueva, escrita a partir de los tests de
tests/ y basada en sim_imagenesPSF.py (misma optica, camara y rangos).

Montaje: NA 1.1, agua (n = 1.333), emision 530 nm, pixel de camara 6.5 um,
magnificacion 66x  ->  98.48 nm/px en la muestra.

PSF: modelo escalar de pupila (sin desajuste de indice, inmersion = muestra = agua):

    PSF(x, y, z) = |IFFT{ circ(rho) * exp(i*(Phi(rho, phi) + k z cos(theta))) }|^2,
    sin(theta) = NA * rho / n,  k = 2 pi n / lambda

    y despues, en Fourier: blur gaussiano extra (tamano del emisor, aberraciones)
    e integracion sobre el area del pixel (OTF * sinc(fx px) * sinc(fy px)).

Mascaras de fase Phi (Castelblanco, tesis Uniandes 2026):
    "corkscrew" : analitica, Ec. 8.1  -> lobulo principal que rota con z (codifica z)
    "esferica"  : Zernike Z4^0, Ec. 4.1.1 (tipo SPED) -> PSF compacta con DOF extendido
    "ninguna"   : PSF limpia (Airy desenfocado)

Movimiento: difusion browniana 3D, dx = sqrt(2 D dtau) * N(0, 1) por eje.
    - z con paredes reflectantes en z_rango_nm (la PSF solo esta definida ahi)
    - x, y libres: las particulas pueden salir/entrar del FOV (columna en_fov)
    - desenfoque de movimiento opcional: n_subpasos posiciones durante la
      exposicion t_exp_s; el ground truth es la posicion media en la exposicion

Cadena de formacion de imagen (el orden importa):
    1. suma de PSFs * fotones por emisor (repartidos entre los subpasos)
    2. + fondo (fijo por video)
    3. ruido de Poisson          <- dominante en bajo SNR
    4. / ganancia -> ruido de lectura -> offset -> redondeo -> uint16

Convencion de coordenadas: el centro del pixel (fila i, columna j) es
(x = j, y = i). El FOV es [-0.5, W - 0.5) x [-0.5, H - 0.5).

Aleatoriedad: cada video tiene flujos independientes por componente (muestra,
trayectorias, fotones, fondo, camara) derivados de (seed, video_id). Asi un
video se puede regenerar solo, y cambiar el ruido, el fondo o los fotones no
cambia las trayectorias.

Salidas (generar_dataset):
    <carpeta>/video_XXX.tif     (T, H, W) uint16, un archivo por video
    <carpeta>/ground_truth.csv  video_id, frame, particle_id, x_px, y_px, z_nm,
                                photons, D_um2_s, en_fov
    <carpeta>/config.json       Config usada + K_psf efectivo
"""

import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


MASCARAS = ("corkscrew", "esferica", "ninguna")
COLUMNAS_GT = ["video_id", "frame", "particle_id", "x_px", "y_px", "z_nm",
               "photons", "D_um2_s", "en_fov"]
COMPONENTES_RNG = ("muestra", "trayectorias", "fotones", "fondo", "camara")
_RANGOS = {"n_emisores_rango": int, "z_rango_nm": float, "fotones_rango": float,
           "fondo_rango": float, "D_rango_um2_s": float}


# ----------------------------------------------------------------------------

@dataclass
class Config:
    # --- optica ---
    NA: float = 1.1
    n_medio: float = 1.333
    lambda_nm: float = 530.0
    pixel_camara_um: float = 6.5
    magnificacion: float = 66.0

    # --- video ---
    H: int = 64
    W: int = 64
    n_videos: int = 10
    n_frames: int = 50
    dt_s: float = 0.1               # 10 Hz, como en la tesis
    t_exp_s: float = None           # exposicion; None -> dt_s
    n_subpasos: int = 1             # posiciones por exposicion (desenfoque de movimiento); 1 = instantaneo

    # --- camara (Orca Flash 4.0, como sim_imagenesPSF.py) ---
    gain: float = 0.46              # e- por ADU
    offset: float = 99.0            # ADU
    read_noise: float = 3.26        # ADU rms (se suma despues de e/gain) = 1.5 e- rms

    # --- PSF (numerico) ---
    mascara: str = "esferica"       # "corkscrew", "esferica" o "ninguna"
    anillos_corkscrew: int = 3      # N en la Ec. 8.1
    amp_esferica: float = 1.5 * np.pi  # amplitud maxima de Z4^0 [rad] (tesis: 1.5 pi)
    M_pupila: int = 256             # rejilla FFT
    K_psf: int = None               # recorte de la PSF [px]; None -> el menor que pierda <= energia_max_recortada
    energia_max_recortada: float = 0.02
    z_paso_nm: float = 20.0         # resolucion maxima del banco precalculado
    blur_extra_px: float = 0.6      # tamano finito del emisor + aberraciones
    integrar_pixel: bool = True     # integra la intensidad sobre el area del pixel

    # --- muestra (rangos: se muestrean por video / por particula) ---
    n_emisores_rango: tuple = (1, 12)
    z_rango_nm: tuple = (-800.0, 800.0)
    fotones_rango: tuple = (1000.0, 30000.0)
    muestreo_fotones: str = "log"   # "log" (log-uniforme en fotones_rango) o "uniforme"
    fondo_rango: tuple = (5.0, 40.0)
    fondo_no_uniforme: bool = True
    D_rango_um2_s: tuple = (0.01, 0.3)
    margen_px: float = 3.0          # las posiciones iniciales pueden caer hasta margen_px fuera del FOV

    seed: int = 0

    def __post_init__(self):
        # desde JSON los rangos llegan como listas
        for campo, tipo in _RANGOS.items():
            setattr(self, campo, tuple(tipo(v) for v in getattr(self, campo)))
        self.validar()

    def validar(self):
        def exige(cond, msg):
            if not cond:
                raise ValueError(msg)

        exige(self.mascara in MASCARAS, f"mascara desconocida: {self.mascara}")
        exige(self.muestreo_fotones in ("log", "uniforme"),
              f"muestreo_fotones desconocido: {self.muestreo_fotones}")
        exige(0 < self.NA < self.n_medio, "se requiere 0 < NA < n_medio")
        exige(min(self.lambda_nm, self.pixel_camara_um, self.magnificacion) > 0,
              "lambda, pixel y magnificacion deben ser positivos")
        exige(self.pixel_nm < self.lambda_nm / (2 * self.NA),
              f"pixel de {self.pixel_nm:.0f} nm >= lambda/(2 NA): la pupila no cabe en la rejilla FFT")
        exige(min(self.H, self.W, self.n_videos, self.n_frames) >= 1,
              "H, W, n_videos y n_frames deben ser >= 1")
        exige(self.dt_s > 0, "dt_s debe ser positivo")
        exige(self.t_exp_s is None or 0 < self.t_exp_s <= self.dt_s, "se requiere 0 < t_exp_s <= dt_s")
        exige(self.n_subpasos >= 1, "n_subpasos debe ser >= 1")
        exige(self.gain > 0 and self.read_noise >= 0 and self.offset >= 0,
              "se requiere gain > 0, read_noise >= 0 y offset >= 0")
        exige(self.M_pupila % 2 == 0 and self.M_pupila >= 32, "M_pupila debe ser par y >= 32")
        exige(self.K_psf is None or (self.K_psf % 2 == 0 and 8 <= self.K_psf <= self.M_pupila),
              "K_psf debe ser par y estar entre 8 y M_pupila")
        exige(0 < self.energia_max_recortada < 1, "energia_max_recortada debe estar en (0, 1)")
        exige(self.z_paso_nm > 0, "z_paso_nm debe ser positivo")
        exige(self.blur_extra_px >= 0, "blur_extra_px debe ser >= 0")
        for campo in _RANGOS:
            lo, hi = getattr(self, campo)
            exige(lo <= hi, f"{campo}: el minimo es mayor que el maximo")
        exige(self.n_emisores_rango[0] >= 0, "n_emisores_rango debe ser >= 0")
        exige(self.fondo_rango[0] >= 0, "fondo_rango debe ser >= 0")
        exige(self.D_rango_um2_s[0] >= 0, "D_rango_um2_s debe ser >= 0")
        if self.muestreo_fotones == "log":
            exige(self.fotones_rango[0] > 0, "el muestreo log-uniforme requiere fotones_rango > 0")
        else:
            exige(self.fotones_rango[0] >= 0, "fotones_rango debe ser >= 0")
        exige(min(self.W, self.H) + 2 * self.margen_px > 0, "margen_px deja una region inicial vacia")

    @property
    def pixel_nm(self):
        return self.pixel_camara_um * 1000.0 / self.magnificacion

    @property
    def sigma0_nm(self):
        return 0.21 * self.lambda_nm / self.NA

    @property
    def dof_nm(self):
        return self.n_medio * self.lambda_nm / self.NA ** 2

    @property
    def nyquist_nm(self):
        """Pixel maximo sin aliasing de la PSF de intensidad (corte 2 NA / lambda)."""
        return self.lambda_nm / (4 * self.NA)

    @property
    def t_exp(self):
        return self.dt_s if self.t_exp_s is None else self.t_exp_s

    def a_dict(self):
        return asdict(self)

    @classmethod
    def desde_dict(cls, d):
        return cls(**d)

    def resumen(self):
        px, nyq = self.pixel_nm, self.nyquist_nm
        paso_max = np.sqrt(2 * self.D_rango_um2_s[1] * self.dt_s) * 1000.0
        desliz = np.sqrt(2 * self.D_rango_um2_s[1] * self.t_exp) * 1000.0
        return (f"pixel en muestra : {px:.2f} nm\n"
                f"sigma0 en foco   : {self.sigma0_nm:.1f} nm = {self.sigma0_nm/px:.2f} px\n"
                f"Nyquist (l/4NA)  : {nyq:.1f} nm -> {nyq/px:.2f}x "
                f"({'OK' if px < nyq else 'SUBMUESTREADO'})\n"
                f"DOF              : {self.dof_nm:.0f} nm\n"
                f"FOV              : {self.W*px/1000:.2f} x {self.H*px/1000:.2f} um\n"
                f"paso rms max/eje : {paso_max:.0f} nm = {paso_max/px:.2f} px por frame\n"
                f"exposicion       : {1000*self.t_exp:.0f} ms, {self.n_subpasos} subpaso(s) "
                f"(desplazamiento rms max en la exposicion {desliz/px:.2f} px)")


def guardar_config(cfg: Config, ruta, banco=None):
    meta = {"config": cfg.a_dict()}
    if banco is not None:
        meta["K_psf_usado"] = int(banco.K)
        meta["energia_recortada_max"] = float(banco.energia_recortada.max())
    Path(ruta).write_text(json.dumps(meta, indent=2))


def cargar_config(ruta):
    return Config.desde_dict(json.loads(Path(ruta).read_text())["config"])


# ----------------------------------------------------------------------------
# PSF
# ----------------------------------------------------------------------------

def fase_corkscrew(rho, phi, N):
    """Ec. 8.1: Phi_ck = {ceil[N * (r/r0)^(4/3)]} * phi  mod 2pi.
    Mascara analitica con saltos duros entre anillos. NO es la mascara
    optimizada iterativamente de Lew et al. 2011."""
    zona = np.ceil(N * np.clip(rho, 1e-9, 1.0)**(4.0 / 3.0))
    return np.mod(zona * phi, 2 * np.pi)


def fase_esferica(rho, amp):
    """Ec. 4.1.1: Z4^0 = sqrt(5)(6 rho^4 - 6 rho^2 + 1), escalado para que el
    valor maximo de la fase sea amp. Mascara suave, no codifica z."""
    z40 = np.sqrt(5.0) * (6 * rho**4 - 6 * rho**2 + 1)
    return amp * z40 / np.sqrt(5.0)  # max |Z4^0| = sqrt(5) en rho = 0 y rho = 1


def rejilla_z(cfg: Config):
    """Planos z del banco: incluye ambos extremos, paso uniforme <= z_paso_nm."""
    zlo, zhi = cfg.z_rango_nm
    n = int(np.ceil((zhi - zlo) / cfg.z_paso_nm - 1e-9)) + 1
    return np.linspace(zlo, zhi, n)


def _rejilla_pupila(cfg: Config, S=1):
    """Pupila y filtros en una rejilla de M*S puntos con paso pixel/S. Todo en
    unidades fisicas (ciclos/nm), asi que no depende de S."""
    M, px = cfg.M_pupila * S, cfg.pixel_nm
    f = np.fft.fftfreq(M, d=px / S)                              # ciclos/nm
    fx, fy = np.meshgrid(f, f, indexing="xy")
    rho = np.hypot(fx, fy) * cfg.lambda_nm / cfg.NA              # radio normalizado (r/r0)
    apertura = rho <= 1.0
    sin_th = np.clip(cfg.NA * rho / cfg.n_medio, 0.0, 1.0)
    kcos = 2 * np.pi * cfg.n_medio / cfg.lambda_nm * np.sqrt(1.0 - sin_th**2)

    if cfg.mascara == "corkscrew":
        fase = fase_corkscrew(rho, np.arctan2(fy, fx), cfg.anillos_corkscrew)
    elif cfg.mascara == "esferica":
        fase = fase_esferica(np.clip(rho, 0.0, 1.0), cfg.amp_esferica)
    else:
        fase = np.zeros_like(rho)

    sigma_nm = cfg.blur_extra_px * px
    filtro = np.exp(-2 * (np.pi * sigma_nm)**2 * (fx**2 + fy**2))   # OTF del blur gaussiano
    if S == 1 and cfg.integrar_pixel:
        filtro = filtro * np.sinc(fx * px) * np.sinc(fy * px)        # OTF del pixel cuadrado
    return dict(fx=fx, fy=fy, P0=apertura * np.exp(1j * fase), kcos=np.where(apertura, kcos, 0.0),
                filtro=filtro, S=S)


def _psf_de_rejilla(cfg: Config, rej, z_nm, dx_px=0.0, dy_px=0.0):
    P = rej["P0"] * np.exp(1j * rej["kcos"] * z_nm)
    if dx_px or dy_px:                                           # desplaza el campo: exacto
        P = P * np.exp(-2j * np.pi * (rej["fx"] * dx_px + rej["fy"] * dy_px) * cfg.pixel_nm)
    psf = np.abs(np.fft.ifft2(P))**2
    psf = np.real(np.fft.ifft2(np.fft.fft2(psf) * rej["filtro"]))
    psf = np.fft.fftshift(np.clip(psf, 0.0, None))               # centro en M*S // 2
    S = rej["S"]
    if S > 1:                                                    # integra cada pixel con S x S muestras
        M = cfg.M_pupila
        psf = np.roll(psf, (S - 1) // 2, axis=(0, 1)).reshape(M, S, M, S).sum(axis=(1, 3))
    return psf / psf.sum()


def psf_directa(cfg: Config, z_nm, dx_px=0.0, dy_px=0.0, sobremuestreo=1):
    """PSF completa (M_pupila x M_pupila, suma 1) con el emisor en el pixel
    M_pupila // 2 + (dx, dy). Camino lento y exacto, usado para construir el
    banco y como referencia en los tests. Con sobremuestreo = S > 1 (impar) la
    integracion sobre el pixel se hace numericamente (ignora integrar_pixel)."""
    if sobremuestreo < 1 or sobremuestreo % 2 == 0:
        raise ValueError("sobremuestreo debe ser un entero impar >= 1")
    return _psf_de_rejilla(cfg, _rejilla_pupila(cfg, sobremuestreo), z_nm, dx_px, dy_px)


class BancoPSF:
    """PSF precalculada en una rejilla de z; se coloca con shift subpixel exacto
    (fase lineal en Fourier), sin interpolacion espacial."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.z_grid = rejilla_z(cfg)
        rej = _rejilla_pupila(cfg)
        completas = np.stack([_psf_de_rejilla(cfg, rej, z) for z in self.z_grid])
        M = cfg.M_pupila
        c = M // 2

        def recorte(K):
            return completas[:, c - K // 2: c + K // 2, c - K // 2: c + K // 2]

        if cfg.K_psf is not None:
            K = cfg.K_psf
        else:
            # el menor recorte (multiplo de 16) que pierde <= energia_max_recortada en todo z
            K = next((K for K in range(16, M + 1, 16)
                      if 1.0 - recorte(K).sum(axis=(1, 2)).min() <= cfg.energia_max_recortada), M)
        self.K = K
        self.stack = recorte(K).copy()
        self.energia_recortada = 1.0 - self.stack.sum(axis=(1, 2))

        self.kx, self.ky = np.meshgrid(np.fft.rfftfreq(K), np.fft.fftfreq(K), indexing="xy")

    def psf_en_z(self, z_nm):
        zg = self.z_grid
        if len(zg) == 1:
            return self.stack[0].copy()
        z = float(np.clip(z_nm, zg[0], zg[-1]))
        i = int(np.clip(np.searchsorted(zg, z) - 1, 0, len(zg) - 2))
        w = (z - zg[i]) / (zg[i + 1] - zg[i])
        return (1 - w) * self.stack[i] + w * self.stack[i + 1]

    def colocar(self, img, x, y, z_nm, N):
        """Suma N * PSF(z) centrada en (x, y) [px, continuo] sobre img (in place)."""
        if N <= 0:
            return
        K = self.K
        ix, iy = int(np.floor(x)), int(np.floor(y))
        y0, x0 = iy - K // 2, ix - K // 2
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(img.shape[0], y0 + K), min(img.shape[1], x0 + K)
        if ys >= ye or xs >= xe:
            return                                               # no toca la imagen
        parche = self.psf_en_z(z_nm)
        fx, fy = x - ix, y - iy
        if fx or fy:
            fase = np.exp(-2j * np.pi * (self.kx * fx + self.ky * fy))
            parche = np.fft.irfft2(np.fft.rfft2(parche) * fase, s=(K, K))
        img[ys:ye, xs:xe] += N * parche[ys - y0: ye - y0, xs - x0: xe - x0]


# ----------------------------------------------------------------------------
# Movimiento browniano
# ----------------------------------------------------------------------------

def reflejar(z, lo, hi):
    """Paredes reflectantes en [lo, hi] (funciona aunque el paso cruce varias veces)."""
    L = hi - lo
    if L <= 0:
        return np.full_like(np.asarray(z, dtype=float), lo)
    u = np.mod(np.asarray(z) - lo, 2 * L)
    return lo + np.where(u > L, 2 * L - u, u)


def generar_trayectorias(cfg: Config, n, rng):
    """Trayectorias 3D en [px, px, nm].

    Devuelve
        tray : (n, T, 3)     posicion media durante la exposicion de cada frame (ground truth)
        sub  : (n, T, S, 3)  posiciones en los S subpasos de la exposicion
        D    : (n,)          coeficiente de difusion por particula [um^2/s]

    El subpaso k del frame t ocurre en t*dt + (k + 1/2) * t_exp / S. Los pasos
    se sortean en orden temporal, asi que alargar el video no cambia los
    primeros frames."""
    T, S, px = cfg.n_frames, cfg.n_subpasos, cfg.pixel_nm
    h = cfg.t_exp / S
    m = cfg.margen_px

    D = rng.uniform(*cfg.D_rango_um2_s, n)
    pos0 = np.stack([rng.uniform(-0.5 - m, cfg.W - 0.5 + m, n),
                     rng.uniform(-0.5 - m, cfg.H - 0.5 + m, n),
                     rng.uniform(*cfg.z_rango_nm, n)], axis=1)   # (n, 3) en px, px, nm

    dtau = np.full((T, S), h)
    dtau[:, 0] = cfg.dt_s - cfg.t_exp + h                        # del ultimo subpaso al primero del frame siguiente
    dtau[0, 0] = 0.0                                             # el primer subpaso es pos0
    sigma_nm = np.sqrt(2 * D[None, :] * dtau.reshape(-1, 1)) * 1000.0   # (T*S, n)
    escala = np.array([1 / px, 1 / px, 1.0])                     # nm -> px en x, y

    pasos = rng.normal(size=(T * S, n, 3)) * sigma_nm[..., None] * escala
    sub = pos0[None] + np.cumsum(pasos, axis=0)                  # (T*S, n, 3)
    sub = sub.reshape(T, S, n, 3).transpose(2, 0, 1, 3)          # (n, T, S, 3)
    sub[..., 2] = reflejar(sub[..., 2], *cfg.z_rango_nm)
    return sub.mean(axis=2), sub, D


# ----------------------------------------------------------------------------
# Fondo y camara
# ----------------------------------------------------------------------------

def generar_fondo(cfg: Config, rng):
    base = rng.uniform(*cfg.fondo_rango)
    if not cfg.fondo_no_uniforme:
        return np.full((cfg.H, cfg.W), base)
    pico = rng.uniform(*cfg.fondo_rango)
    x0, y0 = rng.uniform(0.3, 0.7) * cfg.W, rng.uniform(0.3, 0.7) * cfg.H
    sx, sy = rng.uniform(0.3, 0.8) * cfg.W, rng.uniform(0.3, 0.8) * cfg.H
    yy, xx = np.mgrid[0:cfg.H, 0:cfg.W]
    return base + pico * np.exp(-(((xx - x0) / sx)**2 + ((yy - y0) / sy)**2) / 2.0)


def aplicar_camara(fotones, cfg: Config, rng):
    """Poisson -> / ganancia -> ruido de lectura -> offset -> redondeo -> uint16."""
    e = rng.poisson(np.clip(fotones, 0, None))
    adu = e / cfg.gain + rng.normal(0.0, cfg.read_noise, e.shape) + cfg.offset
    return np.clip(np.round(adu), 0, 65535).astype(np.uint16)


def adu_a_electrones(adu, cfg: Config):
    """Inversa (en media) de aplicar_camara: (ADU - offset) * ganancia."""
    return (np.asarray(adu, dtype=float) - cfg.offset) * cfg.gain


# ----------------------------------------------------------------------------
# Generacion
# ----------------------------------------------------------------------------

def rngs_video(seed, video_id):
    """Un generador independiente por componente, derivado de (seed, video_id)."""
    return {c: np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(video_id, i)))
            for i, c in enumerate(COMPONENTES_RNG)}


def muestrear_fotones(cfg: Config, n, rng):
    """Fotones por emisor. "log": N = exp(U(log Nmin, log Nmax)), cubre por
    igual cada decada del rango; "uniforme": U(Nmin, Nmax)."""
    Nmin, Nmax = cfg.fotones_rango
    if cfg.muestreo_fotones == "log":
        return np.exp(rng.uniform(np.log(Nmin), np.log(Nmax), n))
    return rng.uniform(Nmin, Nmax, n)


def en_fov(cfg: Config, x, y):
    """True si (x, y) [px] cae sobre algun pixel: [-0.5, W - 0.5) x [-0.5, H - 0.5)."""
    return (x >= -0.5) & (x < cfg.W - 0.5) & (y >= -0.5) & (y < cfg.H - 0.5)


def _tabla_gt(cfg: Config, video_id, tray, Ns, D):
    n, T = tray.shape[:2]
    pid = np.tile(np.arange(n), T)                               # orden: frame, luego particula
    pos = tray.transpose(1, 0, 2).reshape(-1, 3)
    x, y = pos[:, 0], pos[:, 1]
    return pd.DataFrame({
        "video_id": np.full(n * T, video_id, dtype=np.int64),
        "frame": np.repeat(np.arange(T, dtype=np.int64), n),
        "particle_id": pid.astype(np.int64),
        "x_px": x, "y_px": y, "z_nm": pos[:, 2],
        "photons": Ns[pid], "D_um2_s": D[pid],
        "en_fov": en_fov(cfg, x, y),
    }, columns=COLUMNAS_GT)


def generar_video(cfg: Config, banco: BancoPSF, video_id=0, extra=False):
    """Un video (T, H, W) uint16 y su ground truth (DataFrame).
    Con extra=True devuelve ademas {"limpio": fotones esperados por pixel
    (T, H, W), "fondo": (H, W), "sub": subposiciones (n, T, S, 3)}."""
    zg = banco.z_grid
    if cfg.z_rango_nm[0] < zg[0] or cfg.z_rango_nm[1] > zg[-1]:
        raise ValueError(f"z_rango_nm {cfg.z_rango_nm} se sale del banco de PSF [{zg[0]}, {zg[-1]}]")

    r = rngs_video(cfg.seed, video_id)
    # randomizacion por video: numero de emisores, brillo, D y fondo
    n = int(r["muestra"].integers(cfg.n_emisores_rango[0], cfg.n_emisores_rango[1] + 1))
    tray, sub, D = generar_trayectorias(cfg, n, r["trayectorias"])
    Ns = muestrear_fotones(cfg, n, r["fotones"])                 # fijo por particula en todo el video
    fondo = generar_fondo(cfg, r["fondo"])                       # fijo en el video

    T, S = cfg.n_frames, cfg.n_subpasos
    video = np.empty((T, cfg.H, cfg.W), dtype=np.uint16)
    limpio = np.empty((T, cfg.H, cfg.W)) if extra else None
    for t in range(T):
        img = np.zeros((cfg.H, cfg.W))
        for p in range(n):
            for k in range(S):
                x, y, z = sub[p, t, k]
                banco.colocar(img, x, y, z, Ns[p] / S)
        img += fondo
        video[t] = aplicar_camara(img, cfg, r["camara"])          # ruido nuevo por frame
        if extra:
            limpio[t] = img

    gt = _tabla_gt(cfg, video_id, tray, Ns, D)
    if extra:
        return video, gt, dict(limpio=limpio, fondo=fondo, sub=sub)
    return video, gt


def generar_dataset(cfg: Config, carpeta=None, banco=None):
    """Escribe video_XXX.tif, ground_truth.csv y config.json en carpeta
    (por defecto simulador_nuevo/salida/) y devuelve el ground truth."""
    carpeta = Path(carpeta) if carpeta is not None else Path(__file__).resolve().parent / "salida"
    carpeta.mkdir(parents=True, exist_ok=True)
    banco = banco or BancoPSF(cfg)

    tablas = []
    for v in range(cfg.n_videos):
        video, gt = generar_video(cfg, banco, v)
        tifffile.imwrite(carpeta / f"video_{v:03d}.tif", video, imagej=True,
                         metadata={"axes": "TYX", "finterval": cfg.dt_s})
        tablas.append(gt)
    gt = pd.concat(tablas, ignore_index=True)
    gt.to_csv(carpeta / "ground_truth.csv", index=False)
    guardar_config(cfg, carpeta / "config.json", banco)
    return gt


# ----------------------------------------------------------------------------
# Diagnosticos
# ----------------------------------------------------------------------------

def angulo_lobulo(banco, z_nm):
    """Angulo [grados] y radio [px] del lobulo principal (centroide 3x3
    alrededor del maximo). Para un PSF rotatorio el angulo codifica z."""
    p = banco.psf_en_z(z_nm)
    K = p.shape[0]
    iy, ix = np.unravel_index(p.argmax(), p.shape)
    v = p[iy - 1: iy + 2, ix - 1: ix + 2]
    yy, xx = np.mgrid[-1:2, -1:2]
    dy = iy + (v * yy).sum() / v.sum() - K // 2
    dx = ix + (v * xx).sum() / v.sum() - K // 2
    return np.degrees(np.arctan2(dy, dx)), np.hypot(dy, dx)


def energia_encerrada(banco, z_nm, radio_px=2.0):
    """Fraccion de energia dentro de un circulo de radio_px alrededor del pico."""
    p = banco.psf_en_z(z_nm)
    K = p.shape[0]
    iy, ix = np.unravel_index(p.argmax(), p.shape)
    yy, xx = np.mgrid[0:K, 0:K]
    return p[(xx - ix)**2 + (yy - iy)**2 <= radio_px**2].sum()


def _var_lectura_e2(cfg: Config):
    """Varianza de lectura + cuantizacion (1/12 ADU^2) en e-^2."""
    return (cfg.read_noise**2 + 1.0 / 12.0) * cfg.gain**2


def snr_pico(banco, N, b):
    """SNR del pixel pico por plano z del banco (emisor centrado en un pixel,
    el mejor caso).
        p_max      = fraccion de fotones en el pixel pico
        senal_pico = N * p_max
        snr_pico   = senal_pico / sqrt(senal_pico + b + var_lectura)
    N [fotones por emisor], b [fotones de fondo por pixel]."""
    p_max = banco.stack.max(axis=(1, 2))
    senal = N * p_max
    snr = senal / np.sqrt(senal + b + _var_lectura_e2(banco.cfg))
    return p_max, senal, snr


def imprimir_snr_pico(banco, paso_nm=100.0):
    """Tabla por z con p_max, senal pico y SNR pico en el peor caso (N min,
    b max) y el mejor caso (N max, b min) de los rangos de Config."""
    cfg = banco.cfg
    N_min, N_max = cfg.fotones_rango
    b_min, b_max = cfg.fondo_rango
    p_max, s_min, snr_min = snr_pico(banco, N_min, b_max)
    _, s_max, snr_max = snr_pico(banco, N_max, b_min)
    print(f"\nSNR pico, mascara '{cfg.mascara}'  (N = {N_min:.0f}..{N_max:.0f} fotones, "
          f"b = {b_min:.0f}..{b_max:.0f} fotones/px)")
    print("   z [nm]    p_max    senal [N min]  senal [N max]   SNR peor   SNR mejor")
    for i, z in enumerate(banco.z_grid):
        if not np.isclose(np.mod(z - banco.z_grid[0], paso_nm), 0.0):
            continue
        print(f"  {z:6.0f}   {p_max[i]:7.4f}   {s_min[i]:12.1f}   {s_max[i]:12.1f}   "
              f"{snr_min[i]:8.2f}   {snr_max[i]:9.2f}")


def modelo_roi(banco, x, y, z_nm, N, b, tam):
    """Imagen esperada (fotones) de un emisor en una ROI tam x tam con fondo b."""
    img = np.full((tam, tam), float(b))
    banco.colocar(img, x, y, z_nm, N)
    return img


def _fisher_roi(banco, N, b, z_nm, tam_roi, dx, dy, var_lectura_e2, con_z):
    """Matriz de Fisher de un emisor en el centro de una ROI tam_roi x tam_roi
    para (x, y[, z], N, b), con ruido Poisson + lectura gaussiana (var = mu +
    var_lectura). Derivadas por diferencias centradas: en x, y con paso 1e-3 px
    (el shift de colocar es exacto); en z con medio paso de la rejilla del banco,
    que interpola linealmente entre planos, y de un solo lado en los extremos."""
    if var_lectura_e2 is None:
        var_lectura_e2 = _var_lectura_e2(banco.cfg)
    x0, y0, h = tam_roi // 2 + dx, tam_roi // 2 + dy, 1e-3
    m = lambda x, y, z: modelo_roi(banco, x, y, z, N, b, tam_roi)
    mu = m(x0, y0, z_nm)
    J = [(m(x0 + h, y0, z_nm) - m(x0 - h, y0, z_nm)) / (2 * h),
         (m(x0, y0 + h, z_nm) - m(x0, y0 - h, z_nm)) / (2 * h)]
    if con_z:
        zg = banco.z_grid
        if len(zg) < 2:
            raise ValueError("la CRLB en z necesita un banco con mas de un plano z")
        hz = np.diff(zg).min() / 2
        z1, z2 = max(z_nm - hz, zg[0]), min(z_nm + hz, zg[-1])
        J.append((m(x0, y0, z2) - m(x0, y0, z1)) / (z2 - z1))
    J += [(mu - b) / N, np.ones_like(mu)]
    J = np.stack(J).reshape(len(J), -1)
    return (J / (mu.ravel() + var_lectura_e2)) @ J.T


def crlb(banco, N, b, z_nm=0.0, tam_roi=15, dx=0.0, dy=0.0, var_lectura_e2=None):
    """Cota de Cramer-Rao (desviacion estandar minima) para (x, y [px], N, b
    [fotones]) de un emisor en el centro de una ROI tam_roi x tam_roi, con z
    conocido. Ruido: Poisson + lectura gaussiana, var = mu + var_lectura."""
    fisher = _fisher_roi(banco, N, b, z_nm, tam_roi, dx, dy, var_lectura_e2, con_z=False)
    return dict(zip(("x", "y", "N", "b"), np.sqrt(np.diag(np.linalg.inv(fisher)))))


def crlb_3d(banco, N, b, z_nm=0.0, tam_roi=15, dx=0.0, dy=0.0, var_lectura_e2=None):
    """Como crlb, pero con z tambien desconocido: cota para (x, y [px], z [nm],
    N, b [fotones]). Estimar z no puede mejorar la cota lateral (x, y >= las de
    crlb); con una PSF simetrica y el emisor centrado en un pixel no la cambia.
    Si la PSF casi no cambia con z (mascara "ninguna" cerca de z = 0) la cota en
    z diverge."""
    fisher = _fisher_roi(banco, N, b, z_nm, tam_roi, dx, dy, var_lectura_e2, con_z=True)
    return dict(zip(("x", "y", "z", "N", "b"), np.sqrt(np.diag(np.linalg.inv(fisher)))))


if __name__ == "__main__":
    cfg = Config(mascara="esferica")    # <- "corkscrew", "esferica" o "ninguna"
    print(cfg.resumen())

    banco = BancoPSF(cfg)
    print(f"\nmascara: {cfg.mascara}, K_psf = {banco.K} px "
          f"(energia fuera del recorte: max {100*banco.energia_recortada.max():.2f}%)")

    p0 = banco.psf_en_z(0).max()
    zs = [-800, -600, -400, -200, 0, 200, 400, 600, 800]
    if cfg.mascara == "corkscrew":
        print("\nrotacion del lobulo vs z:")
        print("   z [nm]   angulo [deg]   radio [px]   pico rel.")
        for z in zs:
            a, r = angulo_lobulo(banco, z)
            print(f"  {z:6.0f}   {a:11.1f}   {r:9.1f}   {banco.psf_en_z(z).max()/p0:8.3f}")
    else:
        print("\ncompacidad de la PSF vs z:")
        print("   z [nm]   pico rel.   energia r<=2px")
        for z in zs:
            print(f"  {z:6.0f}   {banco.psf_en_z(z).max()/p0:8.3f}   {energia_encerrada(banco, z):12.3f}")

    imprimir_snr_pico(banco)
    if cfg.mascara != "ninguna":
        imprimir_snr_pico(BancoPSF(replace(cfg, mascara="ninguna", K_psf=None)))

    b_med = float(np.mean(cfg.fondo_rango))
    print(f"\nCRLB de localizacion en x (z = 0, fondo {b_med:.0f} fotones/px):")
    for N in (cfg.fotones_rango[0], np.sqrt(np.prod(cfg.fotones_rango)), cfg.fotones_rango[1]):
        print(f"  N = {N:7.0f}  ->  {crlb(banco, N, b_med)['x'] * cfg.pixel_nm:6.2f} nm")

    gt = generar_dataset(cfg, banco=banco)
    carpeta = Path(__file__).resolve().parent / "salida"
    print(f"\n{carpeta}/: {cfg.n_videos} videos de ({cfg.n_frames}, {cfg.H}, {cfg.W}) uint16, "
          f"ground_truth.csv, config.json")
    print(f"ground truth: {gt.groupby('video_id').particle_id.nunique().sum()} trayectorias, "
          f"{gt.en_fov.mean()*100:.0f}% de puntos dentro del FOV")
