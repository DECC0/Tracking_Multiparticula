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

Espejo (datos.simulador.usar_espejo): la geometria se puede sortear por
muestra (alpha en datos.espejo_angulo_rango_deg, phi en [0, 360), punto de la
linea del espejo en el FOV o, con espejo_linea_rango_fov, en el borde de la
imagen y con la muestra hacia adentro) con otro flujo propio (spawn_key (idx,
1001, intento)), que siempre sortea 4 numeros. El banco de PSF cubre entonces las
z' de cualquier geometria (z_banco_espejo). Si aun asi algun reflejo queda
sin dibujar (reflejo_en_banco = False), la muestra se regenera con otro
video_id del simulador y otra geometria (intento 1, 2, ...) y se cuenta en
meta["reintentos"]: en los datos de entrenamiento no hay reflejos sin dibujar.
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
CLAVE_RNG_ESPEJO = 1001                 # spawn_key del flujo de la geometria del espejo
DESFASE_REINTENTO = 2 ** 40             # video_id del simulador en el intento k: idx + k * DESFASE_REINTENTO
CAMPOS_RESERVADOS = ("n_frames", "n_videos", "seed")
COLUMNAS_GT = sim.COLUMNAS_GT + ["snr"]
# campos de simulador.Config que definen el banco de PSF
CAMPOS_PSF = ("NA", "n_medio", "lambda_nm", "pixel_camara_um", "magnificacion", "mascara",
              "anillos_corkscrew", "amp_esferica", "M_pupila", "K_psf", "energia_max_recortada",
              "z_paso_nm", "blur_extra_px", "integrar_pixel", "z_rango_nm")
COLUMNAS_GEOMETRIA = ["espejo_angulo_deg", "espejo_orientacion_deg", "espejo_punto_x_px", "espejo_punto_y_px"]


def config_simulador(cfg_datos, seed):
    """simulador.Config base de un split: un frame por muestra."""
    reservados = [c for c in CAMPOS_RESERVADOS if c in cfg_datos.simulador]
    if reservados:
        raise ValueError(f"datos.simulador no puede fijar {reservados}: los controla el dataset")
    cfg = sim.Config(**cfg_datos.simulador, n_frames=1, n_videos=1, seed=seed)
    if cfg_datos.fotones_dispersion is not None and cfg.fotones_rango[0] <= 0:
        raise ValueError("fotones_dispersion requiere fotones_rango > 0")
    if geometria_aleatoria(cfg_datos) and cfg.z_banco_nm is None:
        cfg = replace(cfg, z_banco_nm=z_banco_espejo(cfg_datos, cfg))
    return cfg


def geometria_aleatoria(cfg_datos):
    return (cfg_datos.espejo_angulo_rango_deg is not None or cfg_datos.espejo_orientacion_aleatoria
            or cfg_datos.espejo_punto_aleatorio)


def z_banco_espejo(cfg_datos, cfg_sim):
    """Rango z [nm] que cubre los reflejos de cualquier geometria sorteable.
    Para un real a distancia d >= 0 del espejo, z' - z = 2 d s cos(alpha), con
    s = signo_inclinacion * signo_z; cos es monotono en (0, 180), asi que basta
    con los extremos del rango de alpha. d <= dist_max_espejo_um, o (sin el)
    la diagonal de la caja de posiciones iniciales. Con un frame por muestra no
    hay browniano que saque a un emisor de ahi."""
    lo_a, hi_a = cfg_datos.espejo_angulo_rango_deg or (cfg_sim.angulo_espejo_deg,) * 2
    s = cfg_sim.signo_inclinacion * cfg_sim.signo_z
    c = s * np.cos(np.radians([lo_a, hi_a]))
    zlo, zhi = cfg_sim.z_rango_nm
    if cfg_sim.dist_max_espejo_um is not None:
        dmax = cfg_sim.dist_max_espejo_um * 1000.0
    else:
        lado = np.array([cfg_sim.W, cfg_sim.H]) + 1 + 2 * cfg_sim.margen_px
        dmax = float(np.sqrt(np.sum((lado * cfg_sim.pixel_nm) ** 2) + (zhi - zlo) ** 2))
    return (float(zlo + 2 * dmax * min(0.0, c.min())), float(zhi + 2 * dmax * max(0.0, c.max())))


def banco_compatible(banco, cfg_sim):
    """True si el banco se construyo con la misma optica y PSF que cfg_sim (y los mismos planos z)."""
    return (all(getattr(banco.cfg, c) == getattr(cfg_sim, c) for c in CAMPOS_PSF)
            and np.array_equal(banco.z_grid, sim.rejilla_z(cfg_sim)))


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


def geometria_muestra(cfg_datos, cfg_sim, idx, intento=0):
    """Campos del espejo de la muestra idx (intento k): los de cfg_sim, salvo los
    que se sortean. Siempre consume 4 uniformes de su flujo, asi activar una
    opcion no cambia las demas.

    Con espejo_linea_rango_fov = (lo, hi), la interseccion del espejo con el
    plano focal es la recta perpendicular a m a distancia u h del centro c del
    FOV, del lado opuesto a m (el centro queda del lado de la muestra): u
    uniforme en [lo, hi], h = W/2 |cos phi| + H/2 |sin phi| el semiancho del FOV
    [-0.5, W - 0.5) x [-0.5, H - 0.5) proyectado sobre m (u = 1: la recta toca
    el borde). espejo_punto_px es el punto de esa recta en el plano de
    referencia z_ref_espejo_nm, corrido en m segun Espejo.punto_en_z_nm."""
    rng = np.random.default_rng(np.random.SeedSequence(cfg_sim.seed, spawn_key=(idx, CLAVE_RNG_ESPEJO, intento)))
    u_a, u_f, (u_x, u_y) = rng.uniform(), rng.uniform(), rng.uniform(size=2)
    g = dict(angulo_espejo_deg=cfg_sim.angulo_espejo_deg, orientacion_espejo_deg=cfg_sim.orientacion_espejo_deg,
             espejo_punto_px=cfg_sim.espejo_punto_px)
    if cfg_datos.espejo_angulo_rango_deg is not None:
        lo, hi = cfg_datos.espejo_angulo_rango_deg
        g["angulo_espejo_deg"] = float(lo + u_a * (hi - lo))
    if cfg_datos.espejo_orientacion_aleatoria:
        g["orientacion_espejo_deg"] = float(360.0 * u_f)
    if cfg_datos.espejo_punto_aleatorio and cfg_datos.espejo_linea_rango_fov is None:
        g["espejo_punto_px"] = (float(-0.5 + u_x * cfg_sim.W), float(-0.5 + u_y * cfg_sim.H))
    elif cfg_datos.espejo_punto_aleatorio:
        lo, hi = cfg_datos.espejo_linea_rango_fov
        f = np.radians(g["orientacion_espejo_deg"])
        m = np.array([np.cos(f), np.sin(f)])
        c = np.array([(cfg_sim.W - 1) / 2, (cfg_sim.H - 1) / 2])
        h = cfg_sim.W / 2 * abs(m[0]) + cfg_sim.H / 2 * abs(m[1])
        q = c - (lo + u_x * (hi - lo)) * h * m                    # punto de la recta en el plano focal [px]
        s = (cfg_sim.signo_inclinacion * cfg_sim.signo_z * (0.0 - cfg_sim.z_ref_espejo_nm)
             / np.tan(np.radians(g["angulo_espejo_deg"])) / cfg_sim.pixel_nm)   # corrimiento en m de z_ref a z = 0
        g["espejo_punto_px"] = tuple(float(v) for v in q - s * m)
    return g


def config_geometria(cfg_sim, fila):
    """cfg_sim con la geometria del espejo de una fila de meta / muestras.csv (COLUMNAS_GEOMETRIA)."""
    return replace(cfg_sim, angulo_espejo_deg=float(fila["espejo_angulo_deg"]),
                   orientacion_espejo_deg=float(fila["espejo_orientacion_deg"]),
                   espejo_punto_px=(float(fila["espejo_punto_x_px"]), float(fila["espejo_punto_y_px"])))


def escala_distancia(cfg_datos, cfg_sim_entreno):
    """Longitud [px] que normaliza el canal de distancia: escala_distancia_px o max(H, W) de entrenamiento."""
    return cfg_datos.escala_distancia_px or float(max(cfg_sim_entreno.H, cfg_sim_entreno.W))


def mapa_distancia_espejo(cfg_sim, H, W, escala_px):
    """(H, W) float32: distancia lateral con signo [px] de cada centro de pixel
    (x = j, y = i) a la interseccion del espejo de cfg_sim con el plano focal
    (z = 0), positiva del lado de la muestra, dividida por escala_px."""
    e = sim.Espejo(cfg_sim)
    q = e.punto_en_z_nm(0.0)[:2] / e.pixel_nm                     # punto de la linea en el plano focal [px]
    yy, xx = np.mgrid[0:H, 0:W]
    d = (xx - q[0]) * e.m[0] + (yy - q[1]) * e.m[1]
    return (d / escala_px).astype(np.float32)


def config_muestra(cfg_datos, cfg_sim, idx, intento=0):
    """simulador.Config de la muestra idx, con la randomizacion extra (y, con
    espejo, la geometria del intento)."""
    u_fot, u_fondo, u_ruido, _ = _sorteos(cfg_sim, idx)
    cambios = geometria_muestra(cfg_datos, cfg_sim, idx, intento) if cfg_sim.usar_espejo else {}
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
    intentos = cfg_datos.max_reintentos_espejo + 1 if cfg_sim.usar_espejo else 1
    for intento in range(intentos):
        cfg_m = config_muestra(cfg_datos, cfg_sim, idx, intento)
        video, gt, extra = sim.generar_video(cfg_m, banco, idx + intento * DESFASE_REINTENTO, extra=True)
        if not cfg_sim.usar_espejo or gt.reflejo_en_banco.all():
            break
    else:
        raise RuntimeError(f"muestra {idx}: quedan reflejos fuera del banco tras {intentos} intentos; "
                           "amplia datos.simulador.z_banco_nm")
    gt["video_id"] = idx
    gt["snr"] = snr_emisores(cfg_m, banco, gt, extra["fondo"])
    meta = dict(video_id=idx, n_emisores=int(gt.en_fov.sum()),
                fotones_min=cfg_m.fotones_rango[0], fotones_max=cfg_m.fotones_rango[1],
                rango_completo=bool(usa_rango_completo(cfg_datos, cfg_sim, idx)),
                fondo_medio=float(extra["fondo"].mean()), fondo_no_uniforme=cfg_m.fondo_no_uniforme,
                read_noise=cfg_m.read_noise)
    if cfg_sim.usar_espejo:
        meta.update(espejo_angulo_deg=cfg_m.angulo_espejo_deg, espejo_orientacion_deg=cfg_m.orientacion_espejo_deg,
                    espejo_punto_x_px=cfg_m.espejo_punto_px[0], espejo_punto_y_px=cfg_m.espejo_punto_px[1],
                    reintentos=intento, n_reflejos_en_fov=int(gt.reflejo_en_fov.sum()))
    return video, gt, meta                                       # video (T = 1, H, W) = (C = 1, H, W)
