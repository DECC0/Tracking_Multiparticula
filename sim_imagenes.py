

"""
Generador de imagenes sinteticas de fluorescencia (un solo canal, sin video).

Montaje: NA 1.1, agua (n = 1.333), emision 530 nm, pixel de camara 6.5 um,
magnificacion 66x  ->  98.48 nm/px en la muestra.

PSF: modelo escalar de pupila. Como inmersion y muestra son ambas agua, no hay
desajuste de indice y el modelo escalar basta:

    PSF(x, y, z) = |FFT{ circ(rho) * exp(i * 2*pi*n*z/lambda * cos(theta)) }|^2,
    sin(theta) = NA * rho / n

Cadena de formacion de imagen (el orden importa):
    1. suma de PSFs * fotones por emisor
    2. + fondo
    3. ruido de Poisson          <- dominante en bajo SNR
    4. ganancia -> ruido de lectura -> offset -> uint16

Salidas:
    imagenes.tif      (N, H, W) uint16
    ground_truth.csv  img_id, x_px, y_px, z_nm, photons
"""

from dataclasses import dataclass
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

    # --- imagen ---
    H: int = 64
    W: int = 64
    n_imagenes: int = 200

    # --- camara ---
    gain: float = 1.0               # e- por ADU
    offset: float = 100.0           # ADU
    read_noise: float = 2.0         # ADU rms

    # --- PSF (numerico) ---
    mascara: str = "corkscrew"      # "corkscrew" o "ninguna" (PSF limpia)
    zonas_corkscrew: int = 3        # numero de zonas anulares de igual area
    M_pupila: int = 256             # rejilla FFT
    K_psf: int = 48                 # recorte de la PSF [px]
    z_paso_nm: float = 20.0         # resolucion del banco precalculado
    blur_extra_px: float = 0.6      # tamano finito del emisor + aberraciones

    # --- muestra (rangos: se muestrean por imagen / por emisor) ---
    n_emisores_rango: tuple = (1, 12)
    z_rango_nm: tuple = (-800.0, 800.0)
    fotones_rango: tuple = (300.0, 2000.0)
    fondo_rango: tuple = (5.0, 40.0)
    fondo_no_uniforme: bool = True

    seed: int = 0

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
        return (f"pixel en muestra : {px:.2f} nm\n"
                f"sigma0 en foco   : {self.sigma0_nm:.1f} nm = {self.sigma0_nm/px:.2f} px\n"
                f"Nyquist (l/4NA)  : {nyq:.1f} nm -> {nyq/px:.2f}x "
                f"({'OK' if px < nyq else 'SUBMUESTREADO'})\n"
                f"DOF              : {self.dof_nm:.0f} nm\n"
                f"FOV              : {self.W*px/1000:.2f} x {self.H*px/1000:.2f} um")


# ----------------------------------------------------------------------------
# PSF
# ----------------------------------------------------------------------------

class BancoPSF:
    """PSF precalculada en una rejilla de z; se coloca con shift subpixel exacto
    (fase lineal en Fourier), sin interpolacion espacial."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        M, K, px = cfg.M_pupila, cfg.K_psf, cfg.pixel_nm

        f = np.fft.fftfreq(M, d=px)                              # ciclos/nm
        fx, fy = np.meshgrid(f, f, indexing="xy")
        rho = np.sqrt(fx**2 + fy**2) * cfg.lambda_nm / cfg.NA    # radio normalizado
        apertura = (rho <= 1.0).astype(float)
        sin_th = np.clip(cfg.NA * rho / cfg.n_medio, 0.0, 1.0)
        cos_th = np.sqrt(1.0 - sin_th**2)
        k = 2 * np.pi * cfg.n_medio / cfg.lambda_nm

        # --- mascara de fase ---
        # Corkscrew: espiral por zonas anulares de igual area. La zona l (radios
        # r_l = sqrt(l/L)) lleva carga topologica l, o sea fase l*phi. El resultado
        # es un lobulo unico que rota con z manteniendo radio e intensidad de pico.
        #
        # OJO: NO es la mascara de Lew et al. 2011. La original parte de los modos
        # Gauss-Laguerre (1,1), (2,4), (3,7), (4,10) y se obtiene por optimizacion
        # iterativa de una mascara de solo fase; concentra mas energia en el lobulo
        # principal que esta aproximacion (~45% queda en lobulos secundarios aqui).
        # Sirve como stand-in mientras llega la PSF real del montaje.
        if cfg.mascara == "corkscrew":
            phi = np.arctan2(fy, fx)
            zona = np.ceil(np.clip(rho, 1e-9, 1.0)**2 * cfg.zonas_corkscrew)
            fase_mascara = zona * phi
        else:
            fase_mascara = np.zeros_like(rho)

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

def generar_dataset(cfg: Config, banco=None):
    rng = np.random.default_rng(cfg.seed)
    banco = banco or BancoPSF(cfg)

    pila = np.zeros((cfg.n_imagenes, cfg.H, cfg.W), dtype=np.uint16)
    filas = []
    margen = 3  # los emisores pueden acercarse al borde

    for i in range(cfg.n_imagenes):
        # randomizacion agresiva: numero de emisores, brillo y fondo por imagen
        n = rng.integers(cfg.n_emisores_rango[0], cfg.n_emisores_rango[1] + 1)
        xs = rng.uniform(-margen, cfg.W + margen, n)
        ys = rng.uniform(-margen, cfg.H + margen, n)
        zs = rng.uniform(*cfg.z_rango_nm, n)
        Ns = rng.uniform(*cfg.fotones_rango, n)

        img = np.zeros((cfg.H, cfg.W))
        for x, y, z, N in zip(xs, ys, zs, Ns):
            banco.colocar(img, x, y, z, N)
        pila[i] = aplicar_camara(img + generar_fondo(cfg, rng), cfg, rng)

        for x, y, z, N in zip(xs, ys, zs, Ns):
            filas.append(dict(img_id=i, x_px=x, y_px=y, z_nm=z, photons=N))

    return pila, pd.DataFrame(filas)


def angulo_lobulo(banco, z_nm):
    """Angulo [grados] y radio [px] del lobulo principal. Para un PSF rotatorio
    el angulo es la variable que codifica z, no el ancho."""
    p = banco.psf_en_z(z_nm)
    K = p.shape[0]
    iy, ix = np.unravel_index(p.argmax(), p.shape)
    dy, dx = iy - K // 2, ix - K // 2
    return np.degrees(np.arctan2(dy, dx)), np.hypot(dy, dx)


if __name__ == "__main__":
    cfg = Config()
    print(cfg.resumen())

    banco = BancoPSF(cfg)
    print(f"\nmascara: {cfg.mascara}")
    print(f"energia fuera del recorte de {cfg.K_psf} px: "
          f"max {100*banco.energia_recortada.max():.1f}%")
    print("\nrotacion del lobulo vs z:")
    print("   z [nm]   angulo [deg]   radio [px]   pico rel.")
    p0 = banco.psf_en_z(0).max()
    for z in [-800, -600, -400, -200, 0, 200, 400, 600, 800]:
        a, r = angulo_lobulo(banco, z)
        print(f"  {z:6.0f}   {a:11.1f}   {r:9.1f}   {banco.psf_en_z(z).max()/p0:8.3f}")

    pila, gt = generar_dataset(cfg, banco)
    tifffile.imwrite("imagenes.tif", pila, imagej=True, metadata={"axes": "ZYX"})
    gt.to_csv("ground_truth.csv", index=False)
    print(f"\nimagenes.tif     : {pila.shape} {pila.dtype}, "
          f"rango [{pila.min()}, {pila.max()}]")
    print(f"ground_truth.csv : {len(gt)} emisores en {cfg.n_imagenes} imagenes "
          f"({len(gt)/cfg.n_imagenes:.1f} por imagen)")
