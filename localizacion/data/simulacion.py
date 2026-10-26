"""
Muestras de entrenamiento a partir de simulador_nuevo, sin modificarlo (solo
se importa). Cada muestra es un frame independiente, generado con
simulador.generar_video (n_frames = 1) y determinado por (seed, idx): idx hace
de video_id, asi que la muestra idx se puede regenerar sola.

Randomizacion por muestra (estilo Newby et al. 2018), encima de la del
simulador (numero de emisores, fotones por emisor, fondo):
    - rango de fotones de la imagen: un subrango de fotones_rango de razon
      max/min = fotones_dispersion, con centro log-uniforme. Asi hay imagenes
      enteras con emisores tenues, no solo tenues mezclados con brillantes;
      con probabilidad prob_rango_completo la imagen usa todo fotones_rango
    - fondo uniforme con probabilidad prob_fondo_uniforme
    - ruido de lectura uniforme en read_noise_rango
Usa su propio flujo aleatorio, asi que no cambia las posiciones de los
emisores (las trayectorias tienen su propio flujo en el simulador).
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

DIR_SIMULADOR = Path(__file__).resolve().parents[2] / "simulador_nuevo"
if str(DIR_SIMULADOR) not in sys.path:
    sys.path.insert(0, str(DIR_SIMULADOR))
import simulador as sim  # noqa: E402

CLAVE_RNG = 1000                        # spawn_key del flujo de randomizacion; el simulador usa 0..4
CAMPOS_RESERVADOS = ("n_frames", "n_videos", "seed")
COLUMNAS_GT = sim.COLUMNAS_GT + ["snr"]
# campos de simulador.Config que definen el banco de PSF
CAMPOS_PSF = ("NA", "n_medio", "lambda_nm", "pixel_camara_um", "magnificacion", "mascara",
              "anillos_corkscrew", "amp_esferica", "M_pupila", "K_psf", "energia_max_recortada",
              "z_paso_nm", "blur_extra_px", "integrar_pixel", "z_rango_nm")


def config_simulador(cfg_datos, seed):
    """simulador.Config base de un split: un frame por muestra."""
    reservados = [c for c in CAMPOS_RESERVADOS if c in cfg_datos.simulador]
    if reservados:
        raise ValueError(f"datos.simulador no puede fijar {reservados}: los controla el dataset")
    cfg = sim.Config(**cfg_datos.simulador, n_frames=1, n_videos=1, seed=seed)
    if cfg_datos.fotones_dispersion is not None and cfg.fotones_rango[0] <= 0:
        raise ValueError("fotones_dispersion requiere fotones_rango > 0")
    return cfg


def banco_compatible(banco, cfg_sim):
    """True si el banco se construyo con la misma optica y PSF que cfg_sim."""
    return all(getattr(banco.cfg, c) == getattr(cfg_sim, c) for c in CAMPOS_PSF)


def _sorteos(cfg_sim, idx):
    """Los uniformes de la randomizacion extra de la muestra idx: fotones,
    fondo, ruido de lectura y rango completo. Siempre son los mismos 4, asi
    activar una opcion no cambia las demas; el de prob_rango_completo va al
    final para que las muestras de antes de agregarlo no cambien."""
    rng = np.random.default_rng(np.random.SeedSequence(cfg_sim.seed, spawn_key=(idx, CLAVE_RNG)))
    u_fot, u_fondo, u_ruido = rng.uniform(size=3)
    return u_fot, u_fondo, u_ruido, rng.uniform()


def usa_rango_completo(cfg_datos, cfg_sim, idx):
    """True si la muestra idx usa todo fotones_rango: sin fotones_dispersion, o
    por sorteo con prob_rango_completo."""
    return cfg_datos.fotones_dispersion is None or _sorteos(cfg_sim, idx)[3] < cfg_datos.prob_rango_completo


def config_muestra(cfg_datos, cfg_sim, idx):
    """simulador.Config de la muestra idx, con la randomizacion extra."""
    u_fot, u_fondo, u_ruido, _ = _sorteos(cfg_sim, idx)
    cambios = {}
    d = cfg_datos.fotones_dispersion
    if not usa_rango_completo(cfg_datos, cfg_sim, idx):
        Nmin, Nmax = cfg_sim.fotones_rango
        lo, hi = np.log(Nmin), np.log(Nmax)
        c, w = lo + u_fot * (hi - lo), np.log(d) / 2          # centro log-uniforme y semiancho
        cambios["fotones_rango"] = (max(Nmin, float(np.exp(c - w))), min(Nmax, float(np.exp(c + w))))
    if u_fondo < cfg_datos.prob_fondo_uniforme:
        cambios["fondo_no_uniforme"] = False
    if cfg_datos.read_noise_rango is not None:
        rlo, rhi = cfg_datos.read_noise_rango
        cambios["read_noise"] = float(rlo + u_ruido * (rhi - rlo))
    return replace(cfg_sim, **cambios)


def snr_emisores(cfg, banco, gt, fondo):
    """SNR del pixel pico de cada emisor, como simulador.snr_pico (emisor
    centrado en un pixel): senal = N p_max(z), snr = senal / sqrt(senal + b +
    var_lectura), con b el fondo en el pixel del emisor (el mas cercano dentro
    de la imagen)."""
    if len(gt) == 0:
        return np.zeros(0)
    p_max = np.array([banco.psf_en_z(z).max() for z in gt.z_nm])
    i = np.clip(np.round(gt.y_px.to_numpy()).astype(int), 0, cfg.H - 1)
    j = np.clip(np.round(gt.x_px.to_numpy()).astype(int), 0, cfg.W - 1)
    senal = gt.photons.to_numpy() * p_max
    return senal / np.sqrt(senal + fondo[i, j] + sim._var_lectura_e2(cfg))


def generar_muestra(cfg_datos, cfg_sim, banco, idx):
    """Muestra idx del split de semilla cfg_sim.seed.

    Devuelve
        imagen : (C, H, W) uint16 en ADU
        gt     : DataFrame con COLUMNAS_GT (video_id = idx, frame = 0); incluye
                 los emisores fuera del FOV, con en_fov = False
        meta   : dict con la randomizacion de la muestra"""
    if cfg_datos.canales != 1:
        raise NotImplementedError("C > 1 (biplano) queda pendiente del simulador biplano")
    cfg_m = config_muestra(cfg_datos, cfg_sim, idx)
    video, gt, extra = sim.generar_video(cfg_m, banco, idx, extra=True)
    gt["snr"] = snr_emisores(cfg_m, banco, gt, extra["fondo"])
    meta = dict(video_id=idx, n_emisores=int(gt.en_fov.sum()),
                fotones_min=cfg_m.fotones_rango[0], fotones_max=cfg_m.fotones_rango[1],
                rango_completo=bool(usa_rango_completo(cfg_datos, cfg_sim, idx)),
                fondo_medio=float(extra["fondo"].mean()), fondo_no_uniforme=cfg_m.fondo_no_uniforme,
                read_noise=cfg_m.read_noise)
    return video, gt, meta                                       # video (T = 1, H, W) = (C = 1, H, W)
