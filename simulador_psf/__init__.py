"""
simulador_psf: videos sinteticos de fluorescencia (un solo canal) con
emisores en movimiento browniano 3D y PSF con aberracion esferica.
"""

from .config import Config
from .posiciones import muestrear_n_emisores, posiciones_iniciales, muestrear_fotones
from .browniano import reflejar, generar_trayectorias
from .optica import fase_esferica, Pupila, fase_de_mascara, construir_pupila, psf_completa, BancoPSF
from .fondo import generar_fondo
from .ruido import aplicar_camara
from .io_gt import COLUMNAS_GT, nombre_video, guardar_video, guardar_gt
