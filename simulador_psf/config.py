"""
Parametros fisicos y de simulacion del simulador, en un solo lugar.

Config agrupa todo lo que se puede cambiar por corrida (optica, video, camara,
PSF numerica y rangos de la muestra). Las constantes de modulo son los
numeros fijos que antes estaban dispersos por el codigo; no son campos de
Config para no cambiar su interfaz.

Montaje: NA 1.1, agua (n = 1.333), emision 530 nm, pixel de camara 6.5 um,
magnificacion 66x  ->  98.48 nm/px en la muestra.
"""

from dataclasses import dataclass
import numpy as np


# --- constantes fijas (no configurables) ---
MARGEN_PX = 3                   # los emisores pueden aparecer hasta 3 px fuera del FOV
K_PSF_ESFERICA = 128            # recorte por defecto de la esferica: su halo ancho pierde ~25% de energia en 48 px
K_PSF_OTRAS = 48                # recorte por defecto de las demas mascaras
FACTOR_SIGMA0 = 0.21            # sigma0 = 0.21 lambda / NA (aproximacion gaussiana del Airy en foco)
FONDO_CENTRO_FRAC = (0.3, 0.7)  # centro de la mancha de fondo, en fraccion de W y H
FONDO_ANCHO_FRAC = (0.3, 0.8)   # sigma de la mancha de fondo, en fraccion de W y H
ADU_MAX = 65535                 # saturacion de la camara (uint16)


@dataclass
class Config:
    """Configuracion de un dataset. Unidades: nm para longitudes en la
    muestra, px para el sensor, um^2/s para D, s para el tiempo, fotones
    (= e-, eficiencia cuantica 1) para la senal y ADU para la salida.

    orientacion_espejo_deg (phi) desde una medicion en ImageJ
    ---------------------------------------------------------
    phi es el azimut de la direccion que va de la linea del espejo hacia la
    muestra, medido de +x hacia +y con y hacia ABAJO (filas de la imagen): en
    pantalla crece en sentido HORARIO. ImageJ mide los angulos de una seleccion
    de linea recta en sentido ANTIHORARIO en pantalla (y hacia arriba), del
    punto inicial al final. Por eso, para una misma direccion:
        phi = -angulo_ImageJ   (mod 360)
    - Desde un par particula-reflejo (lo mas directo): el segmento que une una
      particula con su reflejo es perpendicular a la linea del espejo (p' - p es
      paralelo a la normal), y va del reflejo hacia la particula en la direccion
      +m. Trazar la linea en ImageJ DEL REFLEJO A LA PARTICULA y usar
      phi = -angulo (mod 360).
    - Desde la linea del espejo: la linea esta a phi + 90 grados, asi que con su
      angulo en ImageJ, phi = -angulo - 90 o -angulo + 90 (mod 360); de los dos,
      el que apunta hacia la muestra.
    Ejemplo: la linea reflejo -> particula mide 30 grados en ImageJ (hacia arriba
    a la derecha en pantalla) -> phi = -30 = 330 grados, m = (0.866, -0.500). La
    linea del espejo mide entonces -60 (o 120) grados en ImageJ, y
    -(-60) - 90 = 330 coincide (la otra opcion, 150, apunta lejos de la muestra).
    Para espejo_punto_px: este simulador centra el pixel (fila i, columna j) en
    (x = j, y = i); ImageJ con coordenadas subpixel pone el pixel j en
    [j, j + 1), asi que hay que restar 0.5 px a x e y (medir en px, no en um)."""

    # --- optica ---
    NA: float = 1.1
    n_medio: float = 1.333
    lambda_nm: float = 530.0
    pixel_camara_um: float = 6.5
    magnificacion: float = 66.0

    # --- video ---
    H: int = 64
    W: int = 64
    n_videos: int = 10
    n_frames: int = 50
    dt_s: float = 0.1               # 10 Hz, como en la tesis

    # --- camara ---
    gain: float = 0.46              # e- por ADU; valor tipico de la Orca Flash 4.0 V3, consistente
                                    # con el ruido medido en el fondo; pendiente de confirmar
    offset: float = 99.0            # ADU, medido en la zona sin luz del sensor
    read_noise: float = 3.26        # ADU rms (se suma despues de e/gain) = 1.5 e- rms / 0.46 e-/ADU

    # --- PSF (numerico) ---
    mascara: str = "esferica"       # "esferica" o "ninguna" (BancoPSF rechaza "corkscrew")
    amp_esferica: float = 1.5 * np.pi  # amplitud maxima de Z4^0 [rad] (tesis: 1.5 pi)
    M_pupila: int = 256             # rejilla FFT
    K_psf: int = None               # recorte de la PSF [px]; None -> 48, o 128 para "esferica"
    z_paso_nm: float = 20.0         # resolucion del banco precalculado
    blur_extra_px: float = 0.6      # tamano finito del emisor + aberraciones

    # --- muestra (rangos: se muestrean por video / por particula) ---
    n_emisores_rango: tuple = (1, 12)
    z_rango_nm: tuple = (-800.0, 800.0)
    fotones_rango: tuple = (1000.0, 30000.0)  # calibrado con muNS reales (microscopio de Alex): offset ~99 ADU,
                                              # ganancia asumida ~0.46 e-/ADU, fondo ~15 fotones/px, particula
                                              # tenue ~1000-2600 fotones, p_max experimental ~0.010-0.019
    muestreo_fotones: str = "log"           # "log" (log-uniforme en fotones_rango) o "uniforme" (comportamiento anterior)
    fondo_rango: tuple = (5.0, 40.0)
    fondo_no_uniforme: bool = True
    D_rango_um2_s: tuple = (0.01, 0.3)  # coeficiente de difusion por particula

    # --- espejo (vista reflejada; geometria y convencion de signos en espejo.py) ---
    usar_espejo: bool = False           # False: salidas identicas a las de antes del espejo
    angulo_espejo_deg: float = 53.0     # angulo espejo-plano focal, dentro de la cuna de la muestra, en (0, 180)
    orientacion_espejo_deg: float = 0.0  # (ver docstring: desde ImageJ) azimut [grados] de la direccion de la linea del espejo hacia la
                                         # muestra, de +x hacia +y; 0 -> muestra hacia +x, linea vertical en la imagen
    espejo_punto_px: tuple = (20.0, 32.0)  # (x, y) [px] de un punto de la linea del espejo en el plano de referencia
    ratio_reflejo: float = 0.8          # fotones del reflejo / fotones del real, en (0, 1]
    z_ref_espejo_nm: float = 0.0        # z [nm] del plano de referencia del espejo respecto del plano focal
                                        # (z = 0): espejo_punto_px es un punto del espejo a esa altura. 0 -> la
                                        # linea es la interseccion del espejo con el plano focal. Sin
                                        # cubreobjetos: z_rango_nm solo delimita el volumen simulado
    dist_max_espejo_um: float = None    # si se da [um]: las posiciones iniciales se muestrean solo hasta esa
                                        # distancia de la base del espejo en la direccion m (posiciones.py);
                                        # None -> en todo el FOV ampliado, como siempre
    z_banco_nm: tuple = None            # rango z del banco de PSF [nm]; None -> z_rango_nm, ampliado
                                        # automaticamente con el espejo para cubrir las z de los reflejos

    seed: int = 0

    def __post_init__(self):
        # la esferica deja un halo ancho: con 48 px se pierde ~25% de la energia
        if self.K_psf is None:
            self.K_psf = K_PSF_ESFERICA if self.mascara == "esferica" else K_PSF_OTRAS
        # validacion del espejo (solo lanza: no cambia ninguna salida)
        if not 0.0 < self.angulo_espejo_deg < 180.0:
            raise ValueError(f"angulo_espejo_deg debe estar en (0, 180): {self.angulo_espejo_deg}")
        if not 0.0 < self.ratio_reflejo <= 1.0:
            raise ValueError(f"ratio_reflejo debe estar en (0, 1]: {self.ratio_reflejo}")
        if len(self.espejo_punto_px) != 2:
            raise ValueError(f"espejo_punto_px debe ser (x, y) en px: {self.espejo_punto_px}")
        if self.dist_max_espejo_um is not None and not self.dist_max_espejo_um > 0:
            raise ValueError(f"dist_max_espejo_um debe ser > 0 o None: {self.dist_max_espejo_um}")
        if self.z_banco_nm is not None and not self.z_banco_nm[0] < self.z_banco_nm[1]:
            raise ValueError(f"z_banco_nm debe ser (z_min, z_max) con z_min < z_max: {self.z_banco_nm}")

    @property
    def pixel_nm(self):
        """Tamano del pixel en la muestra [nm]: pixel de camara / magnificacion."""
        return self.pixel_camara_um * 1000.0 / self.magnificacion

    @property
    def sigma0_nm(self):
        """Sigma de la PSF en foco [nm], aproximacion gaussiana del disco de Airy."""
        return FACTOR_SIGMA0 * self.lambda_nm / self.NA

    @property
    def dof_nm(self):
        """Profundidad de campo [nm]: n lambda / NA^2."""
        return self.n_medio * self.lambda_nm / self.NA ** 2

    def resumen(self):
        """Texto con pixel, sigma0, muestreo de Nyquist, DOF, FOV y paso
        browniano maximo por frame."""
        px, nyq = self.pixel_nm, self.lambda_nm / (4 * self.NA)
        paso_max = np.sqrt(2 * self.D_rango_um2_s[1] * self.dt_s) * 1000.0
        texto = (f"pixel en muestra : {px:.2f} nm\n"
                f"sigma0 en foco   : {self.sigma0_nm:.1f} nm = {self.sigma0_nm/px:.2f} px\n"
                f"Nyquist (l/4NA)  : {nyq:.1f} nm -> {nyq/px:.2f}x "
                f"({'OK' if px < nyq else 'SUBMUESTREADO'})\n"
                f"DOF              : {self.dof_nm:.0f} nm\n"
                f"FOV              : {self.W*px/1000:.2f} x {self.H*px/1000:.2f} um\n"
                f"paso rms max/eje : {paso_max:.0f} nm = {paso_max/px:.2f} px por frame")
        if self.usar_espejo:        # sin espejo el texto queda como antes (la regresion lo compara)
            texto += (f"\nespejo           : alpha {self.angulo_espejo_deg:.1f} deg, orientacion "
                      f"{self.orientacion_espejo_deg:.1f} deg, punto ({self.espejo_punto_px[0]:.1f}, "
                      f"{self.espejo_punto_px[1]:.1f}) px, ratio_reflejo {self.ratio_reflejo:.2f}, "
                      f"plano de referencia z = {self.z_ref_espejo_nm:.0f} nm")
            if self.dist_max_espejo_um is not None:
                texto += f"\n                   posiciones iniciales hasta {self.dist_max_espejo_um:.2f} um de la base del espejo"
        return texto
