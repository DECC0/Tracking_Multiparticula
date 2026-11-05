"""
CAPA DE COMPATIBILIDAD: este archivo ya no contiene codigo del simulador.
Solo reexporta desde el paquete simulador_psf, para que los imports antiguos
(from sim_imagenesPSF import Config, BancoPSF, ...) sigan funcionando.

    - El codigo vive en simulador_psf/ (ver el README de la raiz).
    - El punto de entrada es ahora generar_dataset.py; ejecutar este archivo
      no hace nada.
    - La mascara "corkscrew", fase_corkscrew y angulo_lobulo se eliminaron:
      BancoPSF lanza ValueError con mascara="corkscrew".
"""

from simulador_psf import (  # noqa: F401
    Config,
    fase_esferica, BancoPSF,
    reflejar, generar_trayectorias,
    generar_fondo,
    aplicar_camara,
    muestrear_fotones,
    generar_video, generar_dataset,
    energia_encerrada, snr_pico, imprimir_snr_pico,
)
