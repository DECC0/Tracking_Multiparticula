
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

from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile

from simulador_psf.config import Config
from simulador_psf.posiciones import muestrear_n_emisores, posiciones_iniciales, muestrear_fotones
from simulador_psf.browniano import reflejar, generar_trayectorias
from simulador_psf.optica import fase_esferica, BancoPSF
from simulador_psf.fondo import generar_fondo


# ----------------------------------------------------------------------------
# Fondo y camara
# ----------------------------------------------------------------------------

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
    n = muestrear_n_emisores(cfg, rng)
    tray, D = generar_trayectorias(cfg, n, rng)
    Ns = muestrear_fotones(cfg, n, rng)                          # fijo por particula en todo el video
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


def snr_pico(banco, N, b):
    """SNR del pixel pico por plano z del banco (sin shift subpixel: emisor
    centrado en un pixel, que es el mejor caso).
        p_max      = fraccion de fotones en el pixel pico
        senal_pico = N * p_max
        snr_pico   = senal_pico / sqrt(senal_pico + b + (sigma_r * g)^2)
    N [fotones por emisor], b [fotones de fondo por pixel]."""
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
    centro de la mancha, asi que el peor caso real es algo peor."""
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

    # SNR pico por z, y el mismo calculo sin aberracion (Phi = 0) para comparar
    imprimir_snr_pico(banco)
    if cfg.mascara != "ninguna":
        imprimir_snr_pico(BancoPSF(replace(cfg, mascara="ninguna")))

    gt = generar_dataset(cfg, banco)
    gt.to_csv("ground_truth.csv", index=False)
    print(f"\nvideos/          : {cfg.n_videos} videos de "
          f"({cfg.n_frames}, {cfg.H}, {cfg.W}) uint16")
    print(f"ground_truth.csv : {gt.particle_id.groupby(gt.video_id).nunique().sum()} "
          f"trayectorias, {gt.en_fov.mean()*100:.0f}% de puntos dentro del FOV")
