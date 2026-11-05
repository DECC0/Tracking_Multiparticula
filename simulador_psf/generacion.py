"""
Orquestacion: arma videos completos con los demas modulos.

    posiciones -> browniano -> optica (suma de N * PSF) -> + fondo -> ruido -> guardado

Un solo np.random.Generator (np.random.default_rng(cfg.seed)) se pasa
explicitamente y se comparte EN SECUENCIA por todos los videos del dataset:
el video v depende de todas las llamadas de los videos anteriores. Orden de
consumo dentro de cada video (no cambiarlo: check_e2e.py y
figuras/figuras_documento.py lo repiten para reconstruir el fondo, y
figuras/gif_video.py supone que el numero de llamadas no depende de
fondo_rango, fotones_rango ni read_noise):

    1. numero de emisores             posiciones.muestrear_n_emisores
    2. D por particula                browniano.generar_trayectorias
    3. posiciones iniciales           posiciones.posiciones_iniciales
    4. pasos brownianos               browniano.generar_trayectorias
    5. fotones por particula          posiciones.muestrear_fotones
    6. fondo                          fondo.generar_fondo
    7. por frame: Poisson y lectura   ruido.aplicar_camara
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .config import Config
from .posiciones import muestrear_n_emisores, muestrear_fotones
from .browniano import generar_trayectorias
from .optica import BancoPSF
from .fondo import generar_fondo
from .ruido import aplicar_camara
from .io_gt import nombre_video, guardar_video


def generar_video(cfg: Config, banco, rng, video_id):
    """Un video de cfg.n_frames frames.

    banco: optica.BancoPSF (reutilizarlo entre videos, es caro de construir);
    rng: np.random.Generator, que se consume en el orden del docstring del modulo.
    Retorna (video, filas): video (T, H, W) uint16 en ADU; filas, lista de
    dicts con las columnas io_gt.COLUMNAS_GT, una por (frame, particula)."""
    # randomizacion agresiva por video: numero de emisores, brillo, D y fondo
    n = muestrear_n_emisores(cfg, rng)
    tray, D = generar_trayectorias(cfg, n, rng)
    Ns = muestrear_fotones(cfg, n, rng)                          # fijo por particula en todo el video
    fondo = generar_fondo(cfg, rng)                              # fijo en el video

    video = np.zeros((cfg.n_frames, cfg.H, cfg.W), dtype=np.uint16)
    filas = []
    for t in range(cfg.n_frames):
        img = np.zeros((cfg.H, cfg.W))                           # fotones esperados por pixel
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
    """cfg.n_videos videos escritos en carpeta/video_XXX.tif (la crea si no
    existe), con rng = np.random.default_rng(cfg.seed). Si no se pasa banco
    se construye uno con cfg. Retorna el ground truth de todos los videos
    como DataFrame (columnas io_gt.COLUMNAS_GT); no escribe el CSV."""
    rng = np.random.default_rng(cfg.seed)
    banco = banco or BancoPSF(cfg)
    Path(carpeta).mkdir(exist_ok=True)

    filas = []
    for v in range(cfg.n_videos):
        video, f = generar_video(cfg, banco, rng, v)
        guardar_video(Path(carpeta) / nombre_video(v), video, cfg.dt_s)
        filas += f
    return pd.DataFrame(filas)
