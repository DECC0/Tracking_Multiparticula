"""
Generador de videos sinteticos de fluorescencia (un solo canal) para
microscopia de localizacion de una sola molecula (SMLM).

Es una EXTENSION TEMPORAL de sim_imagenes.py: reutiliza exactamente el mismo
modelo optico (PSF con mascara de fase corkscrew) y el mismo modelo de camara
(Poisson -> ganancia -> ruido de lectura -> offset -> uint16). La diferencia
con sim_imagenes.py es que alli cada imagen es independiente (emisores nuevos
y aleatorios en cada frame); aqui los emisores son particulas persistentes
que se siguen de un frame al siguiente y que tienen:

    1. Movimiento  : difusion browniana en (x, y, z) + deriva opcional de
                     platina, para simular particulas que se mueven en el
                     medio (o que estan fijas, segun 'fraccion_inmovil').
    2. Fotofisica  : cada particula conmuta entre tres estados con una
                     cadena de Markov de tiempo continuo:

                         OFF <--k_on/k_off--> ON --k_bleach--> BLEACHED

                     Solo emite fotones mientras esta en estado ON.
                     BLEACHED es absorbente (fotoblanqueo irreversible).
    3. Poblacion   : la cantidad de particulas visibles puede mantenerse
                     mas o menos constante si se activa una tasa de
                     "inmigracion" (nuevas particulas entrando al campo de
                     vision), compensando las que se fotoblanquean o salen
                     del campo de vision por difusion.

Cadena de formacion de cada frame del video (igual que en sim_imagenes.py):
    1. para cada particula ON: actualizar posicion, sumar N_fotones * PSF(z)
    2. + fondo (fijo para todo el video, con su propio grano de Poisson)
    3. ruido de Poisson
    4. ganancia -> ruido de lectura -> offset -> uint16

Salidas:
    video.tif          (T, H, W) uint16   (eje T = tiempo)
    ground_truth_video.csv   una fila por (frame, particula) con su
                              posicion, fotones emitidos y estado.

Para el detalle de cada formula y cada eleccion de diseno ver
DOCUMENTACION_VIDEOS.md.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import tifffile

from sim_imagenes import Config as ConfigImagen, BancoPSF, generar_fondo, aplicar_camara


# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------

@dataclass
class ConfigVideo(ConfigImagen):
    """Hereda toda la parte optica/PSF/camara de Config (sim_imagenes.py):
    NA, n_medio, lambda_nm, pixel_camara_um, magnificacion, H, W, gain,
    offset, read_noise, mascara, zonas_corkscrew, M_pupila, K_psf,
    z_paso_nm, blur_extra_px, z_rango_nm, fondo_rango, fondo_no_uniforme,
    n_emisores_rango, fotones_rango, seed.

    En este contexto de video, 'n_emisores_rango' se reinterpreta como el
    rango de la POBLACION INICIAL (en t=0), y 'fotones_rango' como el rango
    del BRILLO PROMEDIO por particula mientras esta en estado ON (no el
    numero de fotones exacto de cada frame, que fluctua segun
    'fotones_cv'). El campo heredado 'n_imagenes' no se usa aqui.
    """

    # --- temporal ---
    n_frames: int = 300
    fps: float = 50.0                    # cuadros por segundo -> dt = 1/fps

    # --- difusion browniana ---
    D_um2_s: float = 0.05                # coeficiente de difusion lateral (x,y)
    D_z_um2_s: float = 0.0               # idem en z (0 = sin difusion axial)
    fraccion_inmovil: float = 0.3        # fraccion de particulas con D = 0

    # --- deriva de platina (drift), constante durante todo el video ---
    drift_x_nm_s: float = 0.0
    drift_y_nm_s: float = 0.0

    # --- fotofisica: tasas del proceso de Markov de 3 estados, en 1/s ---
    k_on_s: float = 0.5                  # OFF -> ON
    k_off_s: float = 2.0                 # ON  -> OFF (parpadeo reversible)
    k_bleach_s: float = 0.05             # ON  -> BLEACHED (irreversible)
    frac_on_inicial: float = 0.3         # fraccion de la poblacion inicial que arranca ON
    fotones_cv: float = 0.2              # variabilidad frame a frame del brillo (CV)

    # --- dinamica de poblacion ---
    tasa_llegada_s: float = 0.0          # nuevas particulas / s que entran al FOV
    margen_px: float = 3.0               # borde donde pueden aparecer/persistir particulas
    factor_margen_salida: float = 3.0    # margen (x margen_px) para considerar "fuera para siempre"

    def resumen_video(self):
        dt = 1.0 / self.fps
        duracion_s = self.n_frames * dt
        duty_cycle = self.k_on_s / (self.k_on_s + self.k_off_s)
        vida_on_antes_blanqueo_s = 1.0 / self.k_bleach_s if self.k_bleach_s > 0 else float("inf")
        sigma_paso_nm = np.sqrt(2 * self.D_um2_s * dt) * 1000.0
        return (f"frames            : {self.n_frames} @ {self.fps:.1f} fps "
                f"(dt = {dt*1000:.1f} ms, duracion = {duracion_s:.2f} s)\n"
                f"paso difusion xy  : {sigma_paso_nm:.1f} nm/frame "
                f"({sigma_paso_nm/self.pixel_nm:.3f} px/frame), D = {self.D_um2_s} um2/s\n"
                f"duty cycle ON     : {duty_cycle:.2f} (estado estacionario, sin fotoblanqueo)\n"
                f"tiempo ON acumulado antes de blanquear (promedio): {vida_on_antes_blanqueo_s:.1f} s\n"
                f"inmigracion       : {self.tasa_llegada_s} particulas/s")


# ----------------------------------------------------------------------------
# Particula (emisor con estado persistente)
# ----------------------------------------------------------------------------

@dataclass
class Emisor:
    """Una particula fluorescente rastreada a lo largo del video.

    x, y estan en pixeles de la muestra (mismo sistema de BancoPSF.colocar);
    z esta en nm. 'estado' es uno de {"on", "off", "bleached"}.
    'mu_fotones' es el brillo promedio (fotones/frame) mientras esta ON;
    se fija una sola vez al crear la particula (fotones_rango) y el brillo
    real de cada frame se muestrea alrededor de este valor (ver
    fotones_frame).
    """
    id: int
    x: float
    y: float
    z: float
    movil: bool
    mu_fotones: float
    estado: str = "off"


def crear_emisor(id_, cfg: ConfigVideo, rng, x=None, y=None):
    if x is None:
        x = rng.uniform(-cfg.margen_px, cfg.W + cfg.margen_px)
    if y is None:
        y = rng.uniform(-cfg.margen_px, cfg.H + cfg.margen_px)
    z = rng.uniform(*cfg.z_rango_nm)
    movil = rng.uniform() > cfg.fraccion_inmovil
    mu = rng.uniform(*cfg.fotones_rango)
    estado = "on" if rng.uniform() < cfg.frac_on_inicial else "off"
    return Emisor(id_, x, y, z, movil, mu, estado)


# ----------------------------------------------------------------------------
# Dinamica: movimiento
# ----------------------------------------------------------------------------

def paso_difusion(e: Emisor, cfg: ConfigVideo, rng, dt_s: float):
    """Camina aleatoria browniana de un paso de tiempo dt_s.

    Para una difusion libre en 1D, la posicion en t+dt es Normal(pos(t),
    sigma^2) con sigma^2 = 2*D*dt (relacion de Einstein). x, y se manejan en
    pixeles de la muestra; z se maneja en nm porque asi lo espera BancoPSF.

    Las particulas 'inmoviles' (fraccion_inmovil) tienen D = 0 y no se
    mueven; representan, por ejemplo, moleculas ancladas a una superficie.
    Las particulas ya fotoblanqueadas se congelan en su ultima posicion
    (ya no emiten, asi que su posicion deja de ser observable).
    """
    if not e.movil or e.estado == "bleached":
        return

    if cfg.D_um2_s > 0:
        sigma_px = np.sqrt(2 * cfg.D_um2_s * dt_s) * 1000.0 / cfg.pixel_nm
        e.x += rng.normal(0.0, sigma_px) + cfg.drift_x_nm_s * dt_s / cfg.pixel_nm
        e.y += rng.normal(0.0, sigma_px) + cfg.drift_y_nm_s * dt_s / cfg.pixel_nm
    else:
        e.x += cfg.drift_x_nm_s * dt_s / cfg.pixel_nm
        e.y += cfg.drift_y_nm_s * dt_s / cfg.pixel_nm

    if cfg.D_z_um2_s > 0:
        sigma_z_nm = np.sqrt(2 * cfg.D_z_um2_s * dt_s) * 1000.0
        e.z = float(np.clip(e.z + rng.normal(0.0, sigma_z_nm), *cfg.z_rango_nm))


def dentro_del_fov_extendido(e: Emisor, cfg: ConfigVideo) -> bool:
    """True si la particula sigue lo bastante cerca del campo de vision
    como para poder volver a contribuir luz en un frame futuro. Mas alla de
    este margen se elimina de la simulacion (una particula que difunde muy
    lejos tiene probabilidad despreciable de volver a tiempo dentro de un
    video corto; ver DOCUMENTACION_VIDEOS.md, seccion 'Limitaciones')."""
    m = cfg.margen_px * cfg.factor_margen_salida
    return (-m <= e.x <= cfg.W + m) and (-m <= e.y <= cfg.H + m)


# ----------------------------------------------------------------------------
# Dinamica: fotofisica (parpadeo + fotoblanqueo)
# ----------------------------------------------------------------------------

def paso_fotofisica(e: Emisor, cfg: ConfigVideo, rng, dt_s: float):
    """Avanza un paso de la cadena de Markov de tiempo continuo

        OFF <--k_on--/--k_off--> ON --k_bleach--> BLEACHED

    Una tasa k (1/s) en tiempo continuo se convierte en probabilidad de que
    "ya ocurrio" en una ventana dt con  p = 1 - exp(-k * dt)  (tiempo de
    espera exponencial). Desde ON hay dos salidas en competencia (a OFF y a
    BLEACHED); la probabilidad de dejar ON es 1 - exp(-(k_off+k_bleach)*dt),
    y si se deja, la probabilidad de que sea hacia BLEACHED es
    k_bleach / (k_off + k_bleach) (particion de un proceso de Poisson).
    """
    if e.estado == "bleached":
        return

    if e.estado == "on":
        k_total = cfg.k_off_s + cfg.k_bleach_s
        if k_total <= 0:
            return
        p_deja_on = 1.0 - np.exp(-k_total * dt_s)
        if rng.uniform() < p_deja_on:
            e.estado = "bleached" if rng.uniform() < cfg.k_bleach_s / k_total else "off"
    else:  # "off"
        p_on = 1.0 - np.exp(-cfg.k_on_s * dt_s)
        if rng.uniform() < p_on:
            e.estado = "on"


def fotones_frame(e: Emisor, cfg: ConfigVideo, rng) -> float:
    """Fotones esperados que aporta la particula en este frame (antes del
    ruido de Poisson de camara, que se aplica una sola vez sobre la imagen
    completa en aplicar_camara).

    Si esta OFF o BLEACHED, aporta 0. Si esta ON, el valor fluctua alrededor
    de mu_fotones con coeficiente de variacion 'fotones_cv' (variaciones de
    iluminacion, orientacion del dipolo, etc.). Se usa una distribucion
    Gamma(forma=1/cv^2, escala=mu*cv^2) porque tiene media exactamente mu,
    CV exactamente cv, y soporte no negativo (a diferencia de una Normal,
    que podria dar fotones negativos)."""
    if e.estado != "on":
        return 0.0
    if cfg.fotones_cv <= 0:
        return e.mu_fotones
    k_forma = 1.0 / cfg.fotones_cv ** 2
    return float(rng.gamma(k_forma, e.mu_fotones / k_forma))


# ----------------------------------------------------------------------------
# Generacion del video
# ----------------------------------------------------------------------------

def generar_video(cfg: ConfigVideo, banco: BancoPSF = None):
    """Simula el video completo, frame a frame.

    Devuelve:
        video : ndarray (n_frames, H, W) uint16
        gt    : DataFrame con una fila por (frame, particula viva ese frame)
                columnas: frame, track_id, x_px, y_px, z_nm, photons, estado
    """
    rng = np.random.default_rng(cfg.seed)
    banco = banco or BancoPSF(cfg)
    dt_s = 1.0 / cfg.fps

    n0 = rng.integers(cfg.n_emisores_rango[0], cfg.n_emisores_rango[1] + 1)
    emisores = [crear_emisor(i, cfg, rng) for i in range(n0)]
    next_id = n0

    # El fondo se fija una vez para todo el video: representa iluminacion de
    # campo ancho estable durante la adquisicion. Su granularidad de disparo
    # (Poisson) si varia frame a frame, dentro de aplicar_camara.
    fondo = generar_fondo(cfg, rng)

    video = np.zeros((cfg.n_frames, cfg.H, cfg.W), dtype=np.uint16)
    filas = []

    for t in range(cfg.n_frames):
        if cfg.tasa_llegada_s > 0:
            n_nuevas = rng.poisson(cfg.tasa_llegada_s * dt_s)
            for _ in range(n_nuevas):
                emisores.append(crear_emisor(next_id, cfg, rng))
                next_id += 1

        img = np.zeros((cfg.H, cfg.W))
        vivos = []
        for e in emisores:
            paso_difusion(e, cfg, rng, dt_s)
            paso_fotofisica(e, cfg, rng, dt_s)

            N = fotones_frame(e, cfg, rng)
            if N > 0:
                banco.colocar(img, e.x, e.y, e.z, N)

            filas.append(dict(frame=t, track_id=e.id, x_px=e.x, y_px=e.y,
                               z_nm=e.z, photons=N, estado=e.estado))

            # Se descarta la particula si se fotoblanqueo (ya no volvera a
            # emitir, no tiene sentido seguir simulandola) o si difundio
            # demasiado lejos del campo de vision como para volver.
            if e.estado != "bleached" and dentro_del_fov_extendido(e, cfg):
                vivos.append(e)
        emisores = vivos

        video[t] = aplicar_camara(img + fondo, cfg, rng)

    return video, pd.DataFrame(filas)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ConfigVideo()
    print(cfg.resumen())
    print()
    print(cfg.resumen_video())

    banco = BancoPSF(cfg)
    video, gt = generar_video(cfg, banco)

    tifffile.imwrite("video.tif", video, imagej=True, metadata={"axes": "TYX"})
    gt.to_csv("ground_truth_video.csv", index=False)

    n_tracks = gt["track_id"].nunique()
    on_frac = (gt["estado"] == "on").mean()
    print(f"\nvideo.tif              : {video.shape} {video.dtype}, "
          f"rango [{video.min()}, {video.max()}]")
    print(f"ground_truth_video.csv : {len(gt)} filas, {n_tracks} particulas distintas, "
          f"{100*on_frac:.1f}% de las filas en estado ON")
