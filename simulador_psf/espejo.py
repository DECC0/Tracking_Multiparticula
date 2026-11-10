"""
Espejo plano junto a la muestra: la camara ve cada emisor real y ademas su
imagen virtual (reflejo), mas tenue y con otra z, lo que codifica
informacion axial.

Modelo:
    1. Solo se simulan las trayectorias reales (browniano.generar_trayectorias).
       El espejo es una pared mas para ellas: ningun emisor real lo cruza.
    2. En cada frame el reflejo se calcula geometricamente desde la posicion
       real (nunca tiene browniano propio):
           p' = p - 2 [(p - p0) . n] n = p0 + R (p - p0),   R = I - 2 n n^T
       Como R es constante, dp' = R dp en cada paso: los movimientos quedan
       ligados por construccion, y |dp'| = |dp| (R es ortogonal), asi que el
       MSD 3D del reflejo es el del real.
    3. Real y reflejo se renderizan con la misma optica (optica.BancoPSF), cada
       uno con su z; el reflejo con ratio_reflejo * fotones del real.
Este modulo no consume el rng.

Montaje (grupo de Biofisica): sin cubreobjetos; las celulas estan pegadas al
espejo, que queda detras de la muestra vista desde el objetivo (el reflejo
siempre mas lejos del objetivo que el real). El espejo esta a 45 grados del
eje optico (angulo_espejo_eje_deg = 45, el valor por defecto): segun un
analisis del asesor, la informacion axial del reflejo, proporcional a
S sin^2(2 beta) / (1/R + cos^2(2 beta)), es maxima cerca de 45 grados, a costa
de perder ~6% de la luz del reflejo por el recorte de pupila (a 55.6 grados =
arcsin(NA/n) la coleccion era completa). El recorte no se modela (abajo).

Unidades: la geometria se calcula en nm (x_nm = x_px * pixel_nm, y igual,
z en nm). Las funciones publicas reciben y devuelven posiciones en las
unidades del ground truth, [x_px, y_px, z_nm]; las que terminan en _nm
trabajan en nm en los tres ejes.

Convencion (fija; la del asesor): objetivo ARRIBA, +z HACIA el objetivo, y
el angulo del espejo medido desde el EJE OPTICO.

  Vista lateral (corte perpendicular a la linea del espejo):

                         objetivo (arriba)
       z ^  +z hacia el objetivo
         |
         |               * p   emisor real, del lado de la muestra (d > 0)
         |            .
     ----P0--------.--------------------------> m  (lateral, hacia la muestra)
         |\     .     plano de referencia z = z_ref_espejo_nm (0: el plano focal)
         | \ .
         |  \         p' = p - 2 d n,   d = (p - p0) . n
         |.  \        n = cos(beta) m + sin(beta) z_hat   (normal hacia la muestra)
         o    \  espejo
        p'     \      beta = angulo entre el espejo y el eje optico (la vertical)
     reflejo (mas lejos del objetivo)

  Vista desde la camara (x a la derecha, y hacia abajo, como imshow):

     (0,0) +-------------------------------+  x [px] ->
           |        |                      |
           |        |   --> m (azimut phi  |   phi = orientacion_espejo_deg,
           |        |       de +x hacia +y)|   de +x hacia +y (en pantalla se
           |        P (x0, y0)   muestra   |   ve horario porque y baja)
           |        |                      |
           |  interseccion del espejo con  |
           |  el plano de ref. (dir. t)    |
           +-------------------------------+
     y [px]

  - beta = angulo_espejo_eje_deg: angulo entre el espejo y el eje optico, en
    (-90, 90). beta > 0: el espejo queda detras de la muestra vista desde el
    objetivo (el montaje; 45 por defecto). beta = 0: pared paralela al eje.
    beta < 0: inclinado hacia el otro lado. Los extremos +-90 (espejo paralelo
    al plano focal) quedan excluidos. Internamente alpha = 90 - beta es el
    angulo con el plano focal; hacia afuera solo se usa beta.
  - phi = orientacion_espejo_deg: azimut de m, la direccion lateral desde la
    linea del espejo hacia la muestra. La linea va en la direccion
    t = (-sin phi, cos phi, 0), es decir a phi + 90 grados. Un solo numero
    fija la linea y el lado de la muestra. Como obtenerlo de una medicion en
    ImageJ (que mide antihorario, y hacia arriba): docstring de config.Config.
  - P = espejo_punto_px = (x0, y0): un punto de la interseccion del espejo con
    el plano de referencia z = z_ref_espejo_nm (altura respecto del foco, +z
    hacia el objetivo; 0 por defecto, el plano focal); p0 = (x0 px, y0 px,
    z_ref_espejo_nm) en nm. El montaje no tiene cubreobjetos: el espejo se
    define respecto del plano focal, y z_rango_nm solo delimita el volumen
    simulado (el espejo lo corta a cualquier altura). A otra altura z la linea
    del espejo esta desplazada en m (Espejo.punto_en_z_nm).
  - Con m = (cos phi, sin phi, 0) y n = cos(beta) m + sin(beta) z_hat:
        p' - p = -2 d n:  lateral  -2 d cos(beta) m   (hacia la linea del espejo)
                          axial    -2 d sin(beta)     (lejos del objetivo si beta > 0)
    asi que tan(beta) = diferencia de z / separacion lateral.
  - Casos: beta = 0 -> x' = 2 x0 - x (con phi = 0), y' = y, z' = z;
    beta = 45, phi = 0 -> z' - z0 = -(x - x0) y x' - x0 = -(z - z0) (en nm).
  - De que lado del foco salen los anillos de la PSF esferica es una propiedad
    de la PSF, no de la geometria: Config.anillos_hacia_objetivo (optica.py).

Supuestos (no se modelan):
  - Espejo plano ideal: sin aberraciones propias ni efectos de polarizacion.
  - La perdida de brillo del reflejo es un unico factor constante
    ratio_reflejo (reflectividad y perdidas de coleccion juntas).
  - El reflejo es una fuente puntual vista por la misma optica. No se modela
    el truncamiento de la pupila: a 45 grados parte del cono de coleccion del
    reflejo no pasa por el espejo (~6% de su luz segun el analisis del
    asesor), lo que tambien puede hacer asimetrica su PSF y desplazar su
    centroide con z'. Con beta <= 0 el espejo no manda luz al objetivo a lo
    largo del eje optico; la formula se aplica igual (vista puramente
    geometrica).
  - Espejo infinito: para reflejar hacia el objetivo la luz de un emisor
    lejano, el espejo real tiene que extenderse lo suficiente.
  - Iluminacion uniforme: no se modela la hoja de luz (solo fluorescerian
    los emisores dentro de ella); seria un modulo aparte.
"""

import numpy as np

from geometria_espejo import GeometriaEspejo, matriz_R, MAX_PLIEGUES  # noqa: F401 (reexportados)

from .config import Config, MARGEN_PX


# La geometria (normal, reflexion, plegado, rango de z', franja de dist_max)
# esta en geometria_espejo.py, en la raiz del repo, compartida con
# simulador_nuevo. Este modulo solo la lee de Config y agrega las unidades del
# ground truth.


class Espejo(GeometriaEspejo):
    """Geometria del espejo de cfg (ver el docstring del modulo).

    Atributos (todos en nm salvo pixel_nm), de GeometriaEspejo:
        m        direccion lateral desde la linea del espejo hacia la muestra
        t        direccion de la linea del espejo (horizontal)
        n        normal unitaria del espejo, hacia la muestra
        p0       punto del espejo en el plano de referencia [nm]
        z_ref    z del plano de referencia respecto del plano focal [nm] (= p0[2])
        beta     angulo_espejo_eje_deg [rad] (alpha = pi/2 - beta, interno)
        R        matriz de reflexion
        z_lo, z_hi  paredes en z de los emisores reales (cfg.z_rango_nm)"""

    def __init__(self, cfg: Config):
        self.pixel_nm = cfg.pixel_nm
        x0, y0 = cfg.espejo_punto_px
        z_lo, z_hi = cfg.z_rango_nm
        super().__init__(cfg.angulo_espejo_eje_deg, cfg.orientacion_espejo_deg,
                         [x0 * self.pixel_nm, y0 * self.pixel_nm, float(cfg.z_ref_espejo_nm)],
                         z_lo, z_hi)

    # --- conversion de unidades: [px, px, nm] <-> nm ---
    def a_nm(self, pos):
        """[x_px, y_px, z_nm] (..., 3) -> nm en los tres ejes (copia)."""
        p = np.array(pos, dtype=float)
        p[..., :2] *= self.pixel_nm
        return p

    def a_gt(self, p_nm):
        """nm en los tres ejes (..., 3) -> [x_px, y_px, z_nm] (copia)."""
        p = np.array(p_nm, dtype=float)
        p[..., :2] /= self.pixel_nm
        return p

    def reflejar(self, pos):
        """Reflejo de posiciones [x_px, y_px, z_nm] (..., 3), en las mismas unidades.
        Es una involucion: reflejar(reflejar(p)) == p (salvo redondeo)."""
        return self.a_gt(self.reflejar_nm(self.a_nm(pos)))


def rango_z_reflejo(cfg: Config, espejo=None):
    """Rango (z_min, z_max) [nm] de z' de los reflejos de todos los puntos de
    la caja de posiciones iniciales (x, y en el FOV ampliado MARGEN_PX px por
    lado, z en cfg.z_rango_nm) que estan del lado de la muestra.

    z' es lineal en p, asi que sus extremos sobre ese politopo convexo
    (caja ∩ {d >= 0}) estan en sus vertices: las esquinas de la caja del lado
    de la muestra y los cortes de las aristas de la caja con el plano.
    Si ningun punto de la caja queda del lado de la muestra lanza ValueError."""
    e = espejo or Espejo(cfg)
    px = cfg.pixel_nm
    lo = np.array([-MARGEN_PX * px, -MARGEN_PX * px, cfg.z_rango_nm[0]])
    hi = np.array([(cfg.W + MARGEN_PX) * px, (cfg.H + MARGEN_PX) * px, cfg.z_rango_nm[1]])
    return e.rango_z_reflejo_caja(lo, hi)


def a_q(cfg: Config, z):
    """Rango o valores de z [nm] a la coordenada de desenfoque q del modelo de
    pupila (optica.py): q = cfg.signo_q * z; un par (lo, hi) vuelve ordenado."""
    if isinstance(z, tuple):
        a, b = (cfg.signo_q * float(v) for v in z)
        return (min(a, b), max(a, b))
    return cfg.signo_q * z


def rango_banco_q(cfg: Config):
    """Rango (q_min, q_max) [nm] del banco de PSF, en la coordenada q del
    modelo de pupila (q = -z por defecto; optica.py).

    - cfg.z_banco_nm dado: ese rango (pasado a q).
    - Sin espejo: cfg.z_rango_nm pasado a q (el banco de siempre; la regresion
      depende de esto: con q = -z y un z_rango simetrico es la misma rejilla).
    - Con espejo: ampliado para cubrir rango_z_reflejo, en multiplos de
      cfg.z_paso_nm desde el extremo inferior en q, asi la rejilla contiene los
      mismos planos que el banco sin espejo. Los reflejos de emisores que salen
      de la caja inicial (x, y libres) pueden quedar fuera: se marcan con
      reflejo_en_banco = False y no se renderizan."""
    if cfg.z_banco_nm is not None:
        return a_q(cfg, tuple(cfg.z_banco_nm))
    qlo, qhi = a_q(cfg, tuple(cfg.z_rango_nm))
    if not cfg.usar_espejo:
        return qlo, qhi
    r1, r2 = a_q(cfg, rango_z_reflejo(cfg))
    paso, tol = cfg.z_paso_nm, 1e-6        # tol: que el redondeo de z' (~1e-13 nm) no agregue un plano
    lo = qlo - paso * np.ceil(max(0.0, qlo - r1) / paso - tol)
    hi = qlo + paso * np.ceil((max(qhi, r2) - qlo) / paso - tol)
    return float(lo), float(hi)


def rango_banco_nm(cfg: Config):
    """Rango (z_min, z_max) [nm] del banco de PSF en z (rango_banco_q pasado a z)."""
    return a_q(cfg, rango_banco_q(cfg))
