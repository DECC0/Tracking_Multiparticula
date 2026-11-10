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
espejo, que queda detras de la muestra vista desde el objetivo (z' > z
siempre: signo_inclinacion = +1 confirmado). Angulo de 56 grados respecto
del eje optico (valor medido, redondeado al grado) -> alpha = 34.0 grados
respecto del plano focal. Coincide, dentro de ese redondeo, con el semiangulo
del cono de coleccion del objetivo, arcsin(NA/n) = arcsin(1.1/1.333) = 55.6
grados.

Unidades: la geometria se calcula en nm (x_nm = x_px * pixel_nm, y igual,
z en nm). Las funciones publicas reciben y devuelven posiciones en las
unidades del ground truth, [x_px, y_px, z_nm]; las que terminan en _nm
trabajan en nm en los tres ejes.

Convencion (Config.signo_z y Config.signo_inclinacion, +1 por defecto):

  Vista lateral (corte perpendicular a la linea del espejo):

    altura respecto del plano focal (+z: lejos del objetivo)
       ^
       |              / espejo (plano): con alpha < 90 se inclina SOBRE la muestra
       |    p' o     /
       |      .     /      * p   emisor real, del lado de la muestra (d > 0)
       |        .  /    .
       |          /  .        p' = p - 2 d n,   d = (p - p0) . n
       |         / alpha      n = normal unitaria hacia la muestra
    ---+--------P0-------------------------> m  (lateral, hacia la muestra)
       |       /  plano de referencia z = z_ref_espejo_nm (0: el plano focal)
       |      /                               ^ objetivo abajo

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

  - alpha = angulo_espejo_deg: angulo entre el espejo y el plano focal (o
    cualquier plano z = cte), medido dentro de la cuna donde esta la muestra;
    valido en (0, 180). 90 = pared vertical. Medido desde el plano focal; si
    se mide desde el eje optico (beta = 56 grados en el montaje actual, valor
    medido redondeado al grado), alpha = 90 - beta (34.0 grados por defecto:
    un espejo perpendicular al eje optico, de frente al objetivo, girado 34
    grados).
  - phi = orientacion_espejo_deg: azimut de m, la direccion lateral desde la
    linea del espejo hacia la muestra. La linea va en la direccion
    t = (-sin phi, cos phi, 0), es decir a phi + 90 grados. Un solo numero
    fija la linea y el lado de la muestra. Como obtenerlo de una medicion en
    ImageJ (que mide antihorario, y hacia arriba): docstring de config.Config.
  - P = espejo_punto_px = (x0, y0): un punto de la interseccion del espejo con
    el plano de referencia z = z_ref_espejo_nm (altura respecto del foco; 0 por
    defecto, el plano focal); p0 = (x0 px, y0 px, z_ref_espejo_nm) en nm. El
    montaje no tiene cubreobjetos: el espejo se define respecto del plano
    focal, y z_rango_nm solo delimita el volumen simulado (el espejo lo corta
    a cualquier altura). A otra altura z la linea del espejo esta desplazada
    en m (Espejo.punto_en_z_nm).
  - Con arriba = signo_z * z_hat (direccion que se aleja del objetivo):
        m = (cos phi, sin phi, 0)
        n = sin(alpha) m - signo_inclinacion cos(alpha) arriba
    Una sola expresion para todo alpha, sin ramas. Los dos signos solo
    entran como su producto, asi que signo_z = -1, signo_inclinacion = -1 y
    alpha -> 180 - alpha dan la misma geometria. Consecuencias:
        p' - p = -2 d n:  lateral  -2 d sin(alpha) m       (hacia la linea del espejo)
                          axial    +2 d cos(alpha) arriba  (lejos del objetivo si alpha < 90)
    asi que tan(alpha) = separacion lateral / diferencia de z.
  - Casos: alpha = 90 -> x' = 2 x0 - x (con phi = 0), y' = y, z' = z;
    alpha = 45, phi = 0 -> z' - z0 = x - x0 y x' - x0 = z - z0 (en nm).

Supuestos (no se modelan):
  - Espejo plano ideal: sin aberraciones propias ni efectos de polarizacion.
  - La perdida de brillo del reflejo es un unico factor constante
    ratio_reflejo (reflectividad y perdidas de coleccion juntas).
  - El reflejo es una fuente puntual vista por la misma optica. No se modela
    el truncamiento de la pupila (solo parte del cono de coleccion llega al
    espejo), que podria hacer asimetrica la PSF del reflejo. En particular,
    con alpha >= 90 el espejo no manda luz al objetivo a lo largo del eje
    optico (el rayo que sale hacia el objetivo cumple v . n = -cos alpha);
    la formula se aplica igual (vista puramente geometrica).
  - Espejo infinito: para reflejar hacia el objetivo la luz de un emisor
    lejano, el espejo real tiene que llegar a una altura del orden de su
    distancia al espejo (~6 um a 45 grados para el borde del FOV).
  - Iluminacion uniforme: no se modela la hoja de luz (solo fluorescerian
    los emisores dentro de ella); seria un modulo aparte.
"""

import numpy as np

from geometria_espejo import GeometriaEspejo, matriz_R, MAX_PLIEGUES  # noqa: F401 (reexportados)

from .config import Config, MARGEN_PX


# La geometria (normal, reflexion, plegado, rango de z', franja de dist_max)
# esta en geometria_espejo.py, en la raiz del repo, compartida con
# simulador_nuevo. Este modulo solo la lee de Config y agrega las unidades del
# ground truth. La convencion de signos esta en Config.signo_z y
# Config.signo_inclinacion (+1 por defecto); posiciones.py la toma de una
# instancia de Espejo: no hay constantes de modulo que se copien al importar.


class Espejo(GeometriaEspejo):
    """Geometria del espejo de cfg (ver el docstring del modulo).

    Atributos (todos en nm salvo pixel_nm), de GeometriaEspejo:
        m        direccion lateral desde la linea del espejo hacia la muestra
        t        direccion de la linea del espejo (horizontal)
        arriba   direccion que se aleja del objetivo (signo_z * z_hat)
        signo_z, signo_inclinacion  los de cfg (+1 o -1)
        n        normal unitaria del espejo, hacia la muestra
        p0       punto del espejo en el plano de referencia [nm]
        z_ref    z del plano de referencia respecto del plano focal [nm] (= p0[2])
        alpha    angulo_espejo_deg [rad]
        R        matriz de reflexion
        z_lo, z_hi  paredes en z de los emisores reales (cfg.z_rango_nm)"""

    def __init__(self, cfg: Config):
        self.pixel_nm = cfg.pixel_nm
        x0, y0 = cfg.espejo_punto_px
        z_lo, z_hi = cfg.z_rango_nm
        super().__init__(cfg.angulo_espejo_deg, cfg.orientacion_espejo_deg,
                         [x0 * self.pixel_nm, y0 * self.pixel_nm, float(cfg.z_ref_espejo_nm)],
                         z_lo, z_hi, cfg.signo_z, cfg.signo_inclinacion)

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


def rango_banco_nm(cfg: Config):
    """Rango (z_min, z_max) [nm] del banco de PSF.

    - cfg.z_banco_nm dado: ese rango.
    - Sin espejo: cfg.z_rango_nm (el banco de siempre; la regresion depende de esto).
    - Con espejo: cfg.z_rango_nm ampliado para cubrir rango_z_reflejo, en
      multiplos de cfg.z_paso_nm desde z_rango_nm[0], asi la rejilla contiene
      los mismos planos que el banco sin espejo. Los reflejos de emisores que
      salen de la caja inicial (x, y libres) pueden quedar fuera: se marcan
      con reflejo_en_banco = False y no se renderizan."""
    if cfg.z_banco_nm is not None:
        return tuple(float(v) for v in cfg.z_banco_nm)
    zlo, zhi = cfg.z_rango_nm
    if not cfg.usar_espejo:
        return zlo, zhi
    r1, r2 = rango_z_reflejo(cfg)
    paso, tol = cfg.z_paso_nm, 1e-6        # tol: que el redondeo de z' (~1e-13 nm) no agregue un plano
    lo = zlo - paso * np.ceil(max(0.0, zlo - r1) / paso - tol)
    hi = zlo + paso * np.ceil((max(zhi, r2) - zlo) / paso - tol)
    return float(lo), float(hi)
