"""
simulador_psf: videos sinteticos de fluorescencia (un solo canal) con
emisores en movimiento browniano 3D y PSF con aberracion esferica.

    posiciones -> browniano -> optica -> + fondo -> ruido -> io_gt
                (generacion orquesta; config tiene los parametros)

Modulos:
    config        Config y constantes fijas
    posiciones    numero de emisores, posiciones iniciales, fotones
    browniano     trayectorias 3D con paredes reflectantes en z
    optica        pupila, banco de PSF en z, colocacion subpixel
    fondo         fondo de fluorescencia (fotones/px)
    ruido         Poisson -> ganancia -> lectura -> offset -> uint16
    io_gt         escritura de videos .tif y ground_truth.csv
    generacion    generar_video, generar_dataset
    diagnosticos  energia encerrada y SNR pico del banco
    espejo        vista reflejada: plano del espejo, reflexion, pared para el browniano

Todo lo publico se importa desde aqui: from simulador_psf import Config, BancoPSF, ...
"""

from .config import Config
from .posiciones import muestrear_n_emisores, sortear_z, posiciones_iniciales, muestrear_fotones
from .browniano import reflejar, reflejar_z, generar_trayectorias
from .optica import fase_esferica, Pupila, fase_de_mascara, construir_pupila, psf_completa, BancoPSF
from .fondo import generar_fondo
from .ruido import aplicar_camara
from .io_gt import COLUMNAS_GT, COLUMNAS_REFLEJO, COLUMNAS_GT_ESPEJO, nombre_video, guardar_video, guardar_gt
from .espejo import Espejo, matriz_R, rango_z_reflejo, a_q, rango_banco_q, rango_banco_nm
from .diagnosticos import energia_encerrada, snr_pico, imprimir_snr_pico
from .generacion import generar_video, generar_dataset
