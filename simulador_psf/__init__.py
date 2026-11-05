"""
simulador_psf: videos sinteticos de fluorescencia (un solo canal) con
emisores en movimiento browniano 3D y PSF con aberracion esferica.
"""

from .config import Config
from .posiciones import muestrear_n_emisores, posiciones_iniciales, muestrear_fotones
