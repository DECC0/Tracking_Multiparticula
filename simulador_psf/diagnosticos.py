"""
Diagnosticos del banco de PSF: compacidad (energia encerrada) y SNR del
pixel pico por plano z. No intervienen en la generacion de datos.
"""

import numpy as np


def energia_encerrada(banco, z_nm, radio_px=2.0):
    """Fraccion de energia dentro de un circulo de radio_px alrededor del pico.
    Para la esferica mide que tan compacta se mantiene la PSF con z.
    banco: optica.BancoPSF; z_nm [nm]; radio_px [px]. Retorna un float en [0, 1]."""
    p = banco.psf_en_z(z_nm)
    K = p.shape[0]
    iy, ix = np.unravel_index(p.argmax(), p.shape)
    yy, xx = np.mgrid[0:K, 0:K]
    return p[(xx - ix)**2 + (yy - iy)**2 <= radio_px**2].sum()


def snr_pico(banco, N, b):
    """SNR del pixel pico por plano z del banco (sin shift subpixel: emisor
    centrado en un pixel, que es el mejor caso).
        p_max      = fraccion de fotones en el pixel pico
        senal_pico = N * p_max
        snr_pico   = senal_pico / sqrt(senal_pico + b + (sigma_r * g)^2)
    N [fotones por emisor], b [fotones de fondo por pixel].
    Retorna (p_max, senal_pico [fotones], snr_pico), cada uno (nz,) sobre banco.z_grid."""
    cfg = banco.cfg
    p_max = banco.stack.max(axis=(1, 2))                         # (nz,), banco ya suavizado y normalizado
    senal = N * p_max
    ruido_lectura_e = cfg.read_noise * cfg.gain                  # ADU rms -> e- rms (gain en e- por ADU)
    snr = senal / np.sqrt(senal + b + ruido_lectura_e**2)
    return p_max, senal, snr


def imprimir_snr_pico(banco, paso_nm=100.0):
    """Tabla por z con p_max, senal pico y SNR pico en el peor caso (N min,
    b max) y el mejor caso (N max, b min) de los rangos de Config. Ojo: con
    fondo_no_uniforme el fondo puede llegar a ~2 * fondo_rango[1] en el
    centro de la mancha, asi que el peor caso real es algo peor.
    Imprime una fila cada paso_nm [nm] de la rejilla del banco."""
    cfg = banco.cfg
    N_min, N_max = cfg.fotones_rango
    b_min, b_max = cfg.fondo_rango
    p_max, s_min, snr_min = snr_pico(banco, N_min, b_max)        # peor caso
    _, s_max, snr_max = snr_pico(banco, N_max, b_min)            # mejor caso

    print(f"\nSNR pico, mascara '{cfg.mascara}'  "
          f"(N = {N_min:.0f}..{N_max:.0f} fotones, b = {b_min:.0f}..{b_max:.0f} "
          f"fotones/px, sigma_r*g = {cfg.read_noise*cfg.gain:.1f} e-)")
    print("   z [nm]    p_max    senal [N min]  senal [N max]   SNR peor   SNR mejor")
    for i, z in enumerate(banco.z_grid):
        if not np.isclose(np.mod(z - banco.z_grid[0], paso_nm), 0.0):
            continue                                             # submuestrea la tabla
        print(f"  {z:6.0f}   {p_max[i]:7.4f}   {s_min[i]:12.1f}   {s_max[i]:12.1f}   "
              f"{snr_min[i]:8.2f}   {snr_max[i]:9.2f}")
