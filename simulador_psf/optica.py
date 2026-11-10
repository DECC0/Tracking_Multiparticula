"""
Sistema optico: pupila escalar con mascara de fase, banco de PSF precalculado
en z y colocacion de emisores con desplazamiento subpixel exacto.

Modelo escalar de pupila. Como inmersion y muestra son ambas agua, no hay
desajuste de indice y el modelo escalar basta:

    PSF(x, y, q) = |FFT{ circ(rho) * exp(i*(Phi(rho) + k q cos(theta))) }|^2,
    sin(theta) = NA * rho / n,   k = 2 pi n / lambda

q es la coordenada de desenfoque del modelo, no la z fisica: q = signo_q * z,
con signo_q = -1 si cfg.anillos_hacia_objetivo es False (por defecto) y +1 si
es True (+z hacia el objetivo, z = 0 en foco). Con la mascara esferica la PSF
del modelo gana anillos hacia +q; cual lado del foco es en el microscopio no
lo fija la fase k q cos(theta) y se decidira con un z-stack de beads (README).
Con q = -z las imagenes son bit a bit las de la convencion anterior (+z lejos
del objetivo, signo_z = +1).

Mascaras de fase Phi disponibles (Castelblanco, tesis Uniandes 2026):
    "esferica" : Zernike Z4^0, Ec. 4.1.1 (tipo SPED) -> PSF compacta con DOF extendido
    "ninguna"  : PSF limpia (Airy desenfocado)

La pupila (Pupila) esta separada del banco: un segundo canal (p. ej. la vista
del espejo de un esquema biplano) seria otra Pupila con sus propios
parametros y su propio BancoPSF, sin tocar este codigo. No esta implementado.
"""

from dataclasses import dataclass
import numpy as np

from .config import Config
from .espejo import rango_banco_q


def fase_esferica(rho, amp):
    """Ec. 4.1.1: Z4^0 = sqrt(5)(6 rho^4 - 6 rho^2 + 1), escalado para que el
    valor maximo de la fase sea amp. Mascara suave, no codifica z.
    rho: radio normalizado de la pupila en [0, 1]; amp [rad]. Retorna la fase [rad]."""
    z40 = np.sqrt(5.0) * (6 * rho**4 - 6 * rho**2 + 1)
    return amp * z40 / np.sqrt(5.0)  # max |Z4^0| = sqrt(5) en rho = 0 y rho = 1


@dataclass
class Pupila:
    """Pupila muestreada en la rejilla FFT (M, M) de frecuencias espaciales.
    apertura    : circ(rho), 1 dentro de NA
    fase_mascara: Phi(rho) [rad]
    cos_th      : cos(theta) por frecuencia, para el desenfoque k z cos(theta)
    k           : numero de onda en el medio [rad/nm]
    otf_blur    : OTF gaussiana del blur extra (tamano del emisor + aberraciones)"""
    apertura: np.ndarray
    fase_mascara: np.ndarray
    cos_th: np.ndarray
    k: float
    otf_blur: np.ndarray


def fase_de_mascara(cfg: Config, rho):
    """Fase [rad] de la mascara cfg.mascara sobre el radio normalizado rho."""
    if cfg.mascara == "esferica":
        return fase_esferica(np.clip(rho, 0.0, 1.0), cfg.amp_esferica)
    if cfg.mascara == "ninguna":
        return np.zeros_like(rho)
    if cfg.mascara == "corkscrew":
        raise ValueError("la mascara 'corkscrew' ya no esta disponible en simulador_psf; "
                         "usar 'esferica' o 'ninguna'")
    raise ValueError(f"mascara desconocida: {cfg.mascara}")


def construir_pupila(cfg: Config):
    """Pupila de cfg en una rejilla FFT de cfg.M_pupila puntos con paso de
    frecuencia 1 / (M pixel_nm): la PSF resultante queda muestreada a 1 px."""
    M, px = cfg.M_pupila, cfg.pixel_nm

    f = np.fft.fftfreq(M, d=px)                              # ciclos/nm
    fx, fy = np.meshgrid(f, f, indexing="xy")
    rho = np.sqrt(fx**2 + fy**2) * cfg.lambda_nm / cfg.NA    # radio normalizado (r/r0): corte en NA/lambda
    apertura = (rho <= 1.0).astype(float)
    sin_th = np.clip(cfg.NA * rho / cfg.n_medio, 0.0, 1.0)
    cos_th = np.sqrt(1.0 - sin_th**2)
    k = 2 * np.pi * cfg.n_medio / cfg.lambda_nm

    fase_mascara = fase_de_mascara(cfg, rho)

    # blur gaussiano de sigma blur_extra_px aplicado como producto en Fourier:
    # OTF = exp(-2 pi^2 sigma^2 f^2), con f en ciclos/px
    fr2 = (fx**2 + fy**2) * px**2
    otf_blur = np.exp(-2 * (np.pi * cfg.blur_extra_px)**2 * fr2)
    return Pupila(apertura, fase_mascara, cos_th, k, otf_blur)


def psf_completa(pupila: Pupila, q_nm):
    """PSF (M, M) con desenfoque del modelo q_nm [nm] (q = cfg.signo_q * z; ver
    el docstring del modulo), centrada en (M//2, M//2) y normalizada a suma 1
    sobre toda la rejilla (antes de recortar)."""
    # aberracion: la fase de la mascara y el desenfoque k q cos(theta) se suman en la pupila
    P = pupila.apertura * np.exp(1j * (pupila.fase_mascara + pupila.k * q_nm * pupila.cos_th))
    psf = np.abs(np.fft.fft2(P))**2                          # intensidad = |campo|^2
    psf = np.real(np.fft.ifft2(np.fft.fft2(psf) * pupila.otf_blur))
    psf /= psf.sum()                                         # normaliza completa: 1 foton en total
    return np.fft.fftshift(psf)


class BancoPSF:
    """PSF precalculada en una rejilla de z; se coloca con shift subpixel exacto
    (fase lineal en Fourier), sin interpolacion espacial.

    Atributos publicos, en z fisica (+z hacia el objetivo) y ordenados por z:
    z_grid [nm] (espejo.rango_banco_nm(cfg): cfg.z_rango_nm, salvo con espejo
    o z_banco_nm); stack (nz, K, K), cada plano es la PSF normalizada sobre la
    rejilla completa y luego recortada a K = cfg.K_psf px, asi que suma
    1 - energia_recortada. Internamente el banco se construye e interpola en
    la coordenada q del modelo (q_grid, stack_q; q = cfg.signo_q * z):
    z_grid y stack son la misma rejilla vista en z."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        M, K = cfg.M_pupila, cfg.K_psf
        pupila = construir_pupila(cfg)

        qlo, qhi = rango_banco_q(cfg)      # = cfg.z_rango_nm en q sin espejo; ampliado con espejo
        self.q_grid = np.arange(qlo, qhi + cfg.z_paso_nm, cfg.z_paso_nm)

        stack = np.empty((len(self.q_grid), K, K))
        c = M // 2
        for i, q in enumerate(self.q_grid):
            psf = psf_completa(pupila, q)
            stack[i] = psf[c - K//2: c + K//2, c - K//2: c + K//2]
        self.stack_q = stack
        orden = slice(None, None, -1) if cfg.signo_q < 0 else slice(None)   # z ascendente
        self.z_grid = cfg.signo_q * self.q_grid[orden]
        self.stack = stack[orden]
        self.energia_recortada = 1.0 - self.stack.sum(axis=(1, 2))   # fraccion que queda fuera de K x K

        kf = np.fft.fftfreq(K)
        self.kx, self.ky = np.meshgrid(kf, kf, indexing="xy")   # ciclos/px del parche

    def psf_en_z(self, z_nm):
        """PSF (K, K) en z_nm [nm] (+z hacia el objetivo), interpolada
        linealmente en q = signo_q * z entre los dos planos vecinos del banco;
        fuera del rango se recorta a los extremos."""
        q = float(np.clip(self.cfg.signo_q * z_nm, self.q_grid[0], self.q_grid[-1]))
        i = int(np.clip(np.searchsorted(self.q_grid, q) - 1,
                        0, len(self.q_grid) - 2))
        w = (q - self.q_grid[i]) / (self.q_grid[i+1] - self.q_grid[i])
        return (1 - w) * self.stack_q[i] + w * self.stack_q[i+1]

    def colocar(self, img, x, y, z_nm, N):
        """Suma N * PSF(z) centrada en (x, y) [px, continuo] sobre img (H, W),
        en el lugar. N en fotones; la parte del parche fuera de img se pierde."""
        if N <= 0:
            return
        K = self.cfg.K_psf
        ix, iy = int(np.floor(x)), int(np.floor(y))
        parche = self.psf_en_z(z_nm)
        # desplazamiento subpixel (x - ix, y - iy) por el teorema de traslacion:
        # multiplicar la FFT por exp(-2 pi i k d) desplaza d px sin interpolar
        fase = np.exp(-2j * np.pi * (self.kx * (x - ix) + self.ky * (y - iy)))
        parche = np.real(np.fft.ifft2(np.fft.fft2(parche) * fase))

        y0, x0 = iy - K // 2, ix - K // 2                        # esquina del parche en img
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(img.shape[0], y0 + K), min(img.shape[1], x0 + K)
        if ys < ye and xs < xe:
            img[ys:ye, xs:xe] += N * parche[ys-y0: ye-y0, xs-x0: xe-x0]
