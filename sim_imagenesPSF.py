
"""
Generador de videos sinteticos de fluorescencia (un solo canal) con emisores
en movimiento browniano 3D.

Montaje: NA 1.1, agua (n = 1.333), emision 530 nm, pixel de camara 6.5 um,
magnificacion 66x  ->  98.48 nm/px en la muestra.

PSF: modelo escalar de pupila. Como inmersion y muestra son ambas agua, no hay
desajuste de indice y el modelo escalar basta:

    PSF(x, y, z) = |FFT{ circ(rho) * exp(i*(Phi(rho, phi) + 2*pi*n*z/lambda * cos(theta))) }|^2,
    sin(theta) = NA * rho / n

Mascaras de fase Phi disponibles (Castelblanco, tesis Uniandes 2026):
    "corkscrew" : analitica, Ec. 8.1  -> lobulo principal que rota con z (codifica z)
    "esferica"  : Zernike Z4^0, Ec. 4.1.1 (tipo SPED) -> PSF compacta con DOF extendido
    "ninguna"   : PSF limpia (Airy desenfocado)

Movimiento: difusion browniana 3D, dx = sqrt(2 D dt) * N(0, 1) por eje.
    - z con paredes reflectantes en z_rango_nm (la PSF solo esta definida ahi)
    - x, y libres: las particulas pueden salir/entrar del FOV (columna en_fov)

Cadena de formacion de imagen (el orden importa):
    1. suma de PSFs * fotones por emisor
    2. + fondo
    3. ruido de Poisson          <- dominante en bajo SNR
    4. ganancia -> ruido de lectura -> offset -> uint16

Salidas:
    videos/video_XXX.tif  (T, H, W) uint16, un archivo por video
    ground_truth.csv      video_id, frame, particle_id, x_px, y_px, z_nm,
                          photons, D_um2_s, en_fov
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile


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

    # --- camara ---
    gain: float = 1.0               # e- por ADU
    offset: float = 100.0           # ADU
    read_noise: float = 2.0         # ADU rms

    # --- PSF (numerico) ---
    mascara: str = "esferica"       # "corkscrew", "esferica" o "ninguna"
    anillos_corkscrew: int = 3      # N en la Ec. 8.1
    amp_esferica: float = 1.5 * np.pi  # amplitud maxima de Z4^0 [rad] (tesis: 1.5 pi)
    M_pupila: int = 256             # rejilla FFT
    K_psf: int = None               # recorte de la PSF [px]; None -> 48, o 128 para "esferica"
    z_paso_nm: float = 20.0         # resolucion del banco precalculado
    blur_extra_px: float = 0.6      # tamano finito del emisor + aberraciones

    # --- muestra (rangos: se muestrean por video / por particula) ---
    n_emisores_rango: tuple = (1, 12)
    z_rango_nm: tuple = (-800.0, 800.0)
    fotones_rango: tuple = (300.0, 2000.0)
    fondo_rango: tuple = (5.0, 40.0)
    fondo_no_uniforme: bool = True
    D_rango_um2_s: tuple = (0.01, 0.3)  # coeficiente de difusion por particula

    seed: int = 0

    def __post_init__(self):
        # la esferica deja un halo ancho: con 48 px se pierde ~25% de la energia
        if self.K_psf is None:
            self.K_psf = 128 if self.mascara == "esferica" else 48

    @property
    def pixel_nm(self):
        return self.pixel_camara_um * 1000.0 / self.magnificacion

    @property
    def sigma0_nm(self):
        return 0.21 * self.lambda_nm / self.NA

    @property
    def dof_nm(self):
        return self.n_medio * self.lambda_nm / self.NA ** 2

    def resumen(self):
        px, nyq = self.pixel_nm, self.lambda_nm / (4 * self.NA)
        paso_max = np.sqrt(2 * self.D_rango_um2_s[1] * self.dt_s) * 1000.0
        return (f"pixel en muestra : {px:.2f} nm\n"
                f"sigma0 en foco   : {self.sigma0_nm:.1f} nm = {self.sigma0_nm/px:.2f} px\n"
                f"Nyquist (l/4NA)  : {nyq:.1f} nm -> {nyq/px:.2f}x "
                f"({'OK' if px < nyq else 'SUBMUESTREADO'})\n"
                f"DOF              : {self.dof_nm:.0f} nm\n"
                f"FOV              : {self.W*px/1000:.2f} x {self.H*px/1000:.2f} um\n"
                f"paso rms max/eje : {paso_max:.0f} nm = {paso_max/px:.2f} px por frame")


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


class BancoPSF:
    """PSF precalculada en una rejilla de z; se coloca con shift subpixel exacto
    (fase lineal en Fourier), sin interpolacion espacial."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        M, K, px = cfg.M_pupila, cfg.K_psf, cfg.pixel_nm

        f = np.fft.fftfreq(M, d=px)                              # ciclos/nm
        fx, fy = np.meshgrid(f, f, indexing="xy")
        rho = np.sqrt(fx**2 + fy**2) * cfg.lambda_nm / cfg.NA    # radio normalizado (r/r0)
        apertura = (rho <= 1.0).astype(float)
        sin_th = np.clip(cfg.NA * rho / cfg.n_medio, 0.0, 1.0)
        cos_th = np.sqrt(1.0 - sin_th**2)
        k = 2 * np.pi * cfg.n_medio / cfg.lambda_nm

        # --- mascara de fase ---
        if cfg.mascara == "corkscrew":
            phi = np.arctan2(fy, fx)
            fase_mascara = fase_corkscrew(rho, phi, cfg.anillos_corkscrew)
        elif cfg.mascara == "esferica":
            fase_mascara = fase_esferica(np.clip(rho, 0.0, 1.0), cfg.amp_esferica)
        elif cfg.mascara == "ninguna":
            fase_mascara = np.zeros_like(rho)
        else:
            raise ValueError(f"mascara desconocida: {cfg.mascara}")

        fr2 = (fx**2 + fy**2) * px**2
        blur = np.exp(-2 * (np.pi * cfg.blur_extra_px)**2 * fr2)

        zlo, zhi = cfg.z_rango_nm
        self.z_grid = np.arange(zlo, zhi + cfg.z_paso_nm, cfg.z_paso_nm)

        stack = np.empty((len(self.z_grid), K, K))
        c = M // 2
        for i, z in enumerate(self.z_grid):
            P = apertura * np.exp(1j * (fase_mascara + k * z * cos_th))
            psf = np.abs(np.fft.fft2(P))**2
            psf = np.real(np.fft.ifft2(np.fft.fft2(psf) * blur))
            psf /= psf.sum()                                     # normaliza completa
            psf = np.fft.fftshift(psf)
            stack[i] = psf[c - K//2: c + K//2, c - K//2: c + K//2]
        self.stack = stack
        self.energia_recortada = 1.0 - stack.sum(axis=(1, 2))

        kf = np.fft.fftfreq(K)
        self.kx, self.ky = np.meshgrid(kf, kf, indexing="xy")

    def psf_en_z(self, z_nm):
        z = float(np.clip(z_nm, self.z_grid[0], self.z_grid[-1]))
        i = int(np.clip(np.searchsorted(self.z_grid, z) - 1,
                        0, len(self.z_grid) - 2))
        w = (z - self.z_grid[i]) / (self.z_grid[i+1] - self.z_grid[i])
        return (1 - w) * self.stack[i] + w * self.stack[i+1]

    def colocar(self, img, x, y, z_nm, N):
        """Suma N * PSF(z) centrada en (x, y) [px, continuo]."""
        if N <= 0:
            return
        K = self.cfg.K_psf
        ix, iy = int(np.floor(x)), int(np.floor(y))
        parche = self.psf_en_z(z_nm)
        fase = np.exp(-2j * np.pi * (self.kx * (x - ix) + self.ky * (y - iy)))
        parche = np.real(np.fft.ifft2(np.fft.fft2(parche) * fase))

        y0, x0 = iy - K // 2, ix - K // 2
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(img.shape[0], y0 + K), min(img.shape[1], x0 + K)
        if ys < ye and xs < xe:
            img[ys:ye, xs:xe] += N * parche[ys-y0: ye-y0, xs-x0: xe-x0]


# ----------------------------------------------------------------------------
# Movimiento browniano
# ----------------------------------------------------------------------------

def reflejar(z, lo, hi):
    """Paredes reflectantes en [lo, hi] (funciona aunque el paso cruce varias veces)."""
    L = hi - lo
    u = np.mod(z - lo, 2 * L)
    return lo + np.where(u > L, 2 * L - u, u)


def generar_trayectorias(cfg: Config, n, rng):
    """Trayectorias 3D (n, T, 3) en [px, px, nm] y D por particula [um^2/s]."""
    T, px = cfg.n_frames, cfg.pixel_nm
    margen = 3  # los emisores pueden acercarse al borde

    D = rng.uniform(*cfg.D_rango_um2_s, n)
    sigma_nm = np.sqrt(2 * D * cfg.dt_s) * 1000.0                # rms por eje por frame

    pos0 = np.stack([rng.uniform(-margen, cfg.W + margen, n) * px,
                     rng.uniform(-margen, cfg.H + margen, n) * px,
                     rng.uniform(*cfg.z_rango_nm, n)], axis=1)   # (n, 3) en nm

    pasos = rng.normal(size=(n, T, 3)) * sigma_nm[:, None, None]
    pasos[:, 0] = 0.0                                            # frame 0 = pos0
    tray = pos0[:, None, :] + np.cumsum(pasos, axis=1)

    tray[..., 2] = reflejar(tray[..., 2], *cfg.z_rango_nm)
    tray[..., :2] /= px                                          # x, y a px
    return tray, D


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
    return base + pico * np.exp(-(((xx-x0)/sx)**2 + ((yy-y0)/sy)**2) / 2.0)


def aplicar_camara(fotones, cfg: Config, rng):
    """Poisson -> ganancia -> ruido de lectura -> offset -> uint16."""
    e = rng.poisson(np.clip(fotones, 0, None))
    adu = e / cfg.gain + rng.normal(0.0, cfg.read_noise, e.shape) + cfg.offset
    return np.clip(np.round(adu), 0, 65535).astype(np.uint16)


# ----------------------------------------------------------------------------
# Generacion
# ----------------------------------------------------------------------------

def generar_video(cfg: Config, banco, rng, video_id):
    # randomizacion agresiva por video: numero de emisores, brillo, D y fondo
    n = rng.integers(cfg.n_emisores_rango[0], cfg.n_emisores_rango[1] + 1)
    tray, D = generar_trayectorias(cfg, n, rng)
    Ns = rng.uniform(*cfg.fotones_rango, n)
    fondo = generar_fondo(cfg, rng)                              # fijo en el video

    video = np.zeros((cfg.n_frames, cfg.H, cfg.W), dtype=np.uint16)
    filas = []
    for t in range(cfg.n_frames):
        img = np.zeros((cfg.H, cfg.W))
        for p in range(n):
            x, y, z = tray[p, t]
            banco.colocar(img, x, y, z, Ns[p])
            filas.append(dict(video_id=video_id, frame=t, particle_id=p,
                              x_px=x, y_px=y, z_nm=z, photons=Ns[p],
                              D_um2_s=D[p],
                              en_fov=bool(0 <= x < cfg.W and 0 <= y < cfg.H)))
        video[t] = aplicar_camara(img + fondo, cfg, rng)          # ruido nuevo por frame
    return video, filas


def generar_dataset(cfg: Config, banco=None, carpeta="videos"):
    rng = np.random.default_rng(cfg.seed)
    banco = banco or BancoPSF(cfg)
    Path(carpeta).mkdir(exist_ok=True)

    filas = []
    for v in range(cfg.n_videos):
        video, f = generar_video(cfg, banco, rng, v)
        tifffile.imwrite(Path(carpeta) / f"video_{v:03d}.tif", video,
                         imagej=True, metadata={"axes": "TYX",
                                                "finterval": cfg.dt_s})
        filas += f
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Diagnosticos
# ----------------------------------------------------------------------------

def angulo_lobulo(banco, z_nm):
    """Angulo [grados] y radio [px] del lobulo principal. Para un PSF rotatorio
    el angulo es la variable que codifica z, no el ancho."""
    p = banco.psf_en_z(z_nm)
    K = p.shape[0]
    iy, ix = np.unravel_index(p.argmax(), p.shape)
    dy, dx = iy - K // 2, ix - K // 2
    return np.degrees(np.arctan2(dy, dx)), np.hypot(dy, dx)


def energia_encerrada(banco, z_nm, radio_px=2.0):
    """Fraccion de energia dentro de un circulo de radio_px alrededor del pico.
    Para la esferica mide que tan compacta se mantiene la PSF con z."""
    p = banco.psf_en_z(z_nm)
    K = p.shape[0]
    iy, ix = np.unravel_index(p.argmax(), p.shape)
    yy, xx = np.mgrid[0:K, 0:K]
    return p[(xx - ix)**2 + (yy - iy)**2 <= radio_px**2].sum()


if __name__ == "__main__":
    cfg = Config(mascara="esferica")    # <- "corkscrew", "esferica" o "ninguna"
    print(cfg.resumen())

    banco = BancoPSF(cfg)
    print(f"\nmascara: {cfg.mascara}")
    print(f"energia fuera del recorte de {cfg.K_psf} px: "
          f"max {100*banco.energia_recortada.max():.1f}%")

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
            print(f"  {z:6.0f}   {banco.psf_en_z(z).max()/p0:8.3f}   "
                  f"{energia_encerrada(banco, z):12.3f}")

    gt = generar_dataset(cfg, banco)
    gt.to_csv("ground_truth.csv", index=False)
    print(f"\nvideos/          : {cfg.n_videos} videos de "
          f"({cfg.n_frames}, {cfg.H}, {cfg.W}) uint16")
    print(f"ground_truth.csv : {gt.particle_id.groupby(gt.video_id).nunique().sum()} "
          f"trayectorias, {gt.en_fov.mean()*100:.0f}% de puntos dentro del FOV")
