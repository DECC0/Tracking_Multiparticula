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

Unidades: la geometria se calcula en nm (x_nm = x_px * pixel_nm, y igual,
z en nm). Las funciones publicas reciben y devuelven posiciones en las
unidades del ground truth, [x_px, y_px, z_nm]; las que terminan en _nm
trabajan en nm en los tres ejes.

Convencion (ver SIGNO_Z y SIGNO_INCLINACION abajo):

  Vista lateral (corte perpendicular a la linea del espejo):

    altura sobre el cubreobjetos (+z: lejos del objetivo)
       ^
       |              / espejo (plano): con alpha < 90 se inclina SOBRE la muestra
       |    p' o     /
       |      .     /      * p   emisor real, del lado de la muestra (d > 0)
       |        .  /    .
       |          /  .        p' = p - 2 d n,   d = (p - p0) . n
       |         / alpha      n = normal unitaria hacia la muestra
    ===+========P0=========================> m  (lateral, hacia la muestra)
       cubreobjetos (z = z_rango_nm[0])     ^ objetivo abajo

  Vista desde la camara (x a la derecha, y hacia abajo, como imshow):

     (0,0) +-------------------------------+  x [px] ->
           |        |                      |
           |        |   --> m (azimut phi  |   phi = orientacion_espejo_deg,
           |        |       de +x hacia +y)|   de +x hacia +y (en pantalla se
           |        P (x0, y0)   muestra   |   ve horario porque y baja)
           |        |                      |
           |  linea del espejo en el       |
           |  cubreobjetos (direccion t)   |
           +-------------------------------+
     y [px]

  - alpha = angulo_espejo_deg: angulo entre el espejo y el cubreobjetos,
    medido dentro de la cuna donde esta la muestra; valido en (0, 180).
    90 = pared vertical. Si el angulo se mide respecto del eje optico (beta),
    alpha = 90 - beta.
  - phi = orientacion_espejo_deg: azimut de m, la direccion lateral desde la
    linea del espejo hacia la muestra. La linea va en la direccion
    t = (-sin phi, cos phi, 0), es decir a phi + 90 grados. Un solo numero
    fija la linea y el lado de la muestra.
  - P = espejo_punto_px = (x0, y0): un punto de la linea del espejo sobre el
    cubreobjetos; p0 = (x0 px, y0 px, z del cubreobjetos) en nm.
  - Con arriba = SIGNO_Z * z_hat (direccion que se aleja del objetivo):
        m = (cos phi, sin phi, 0)
        n = sin(alpha) m - SIGNO_INCLINACION cos(alpha) arriba
    Una sola expresion para todo alpha, sin ramas. Consecuencias:
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

from .config import Config, MARGEN_PX


# --- convencion de signos: UNICO lugar donde se fija ----------------------------
# Si al comparar con datos reales resulta al reves, se cambia aqui y nada mas.
SIGNO_Z = +1            # +1: +z se aleja del objetivo, el cubreobjetos es z_rango_nm[0]
                        # -1: +z se acerca al objetivo, el cubreobjetos es z_rango_nm[1]
SIGNO_INCLINACION = +1  # +1: con alpha < 90 el espejo se inclina SOBRE la muestra
                        # -1: se inclina alejandose de ella (equivale a alpha -> 180 - alpha)

MAX_PLIEGUES = 1000     # reflexiones maximas por punto en plegar_nm (en la practica, unas pocas)


def matriz_R(n):
    """Matriz de reflexion (3, 3) R = I - 2 n n^T del plano de normal unitaria n."""
    n = np.asarray(n, dtype=float)
    return np.eye(3) - 2.0 * np.outer(n, n)


class Espejo:
    """Geometria del espejo de cfg (ver el docstring del modulo).

    Atributos (todos en nm salvo pixel_nm):
        m        direccion lateral desde la linea del espejo hacia la muestra
        t        direccion de la linea del espejo en el cubreobjetos
        arriba   direccion que se aleja del objetivo (SIGNO_Z * z_hat)
        n        normal unitaria del espejo, hacia la muestra
        p0       punto del plano del espejo sobre el cubreobjetos [nm]
        R        matriz de reflexion
        z_lo, z_hi  paredes en z de los emisores reales (cfg.z_rango_nm)"""

    def __init__(self, cfg: Config):
        self.pixel_nm = cfg.pixel_nm
        a = np.radians(cfg.angulo_espejo_deg)
        f = np.radians(cfg.orientacion_espejo_deg)
        self.m = np.array([np.cos(f), np.sin(f), 0.0])
        self.t = np.array([-np.sin(f), np.cos(f), 0.0])
        self.arriba = np.array([0.0, 0.0, float(SIGNO_Z)])
        self.n = np.sin(a) * self.m - SIGNO_INCLINACION * np.cos(a) * self.arriba
        self.R = matriz_R(self.n)
        self.z_lo, self.z_hi = cfg.z_rango_nm
        z_cubre = self.z_lo if SIGNO_Z > 0 else self.z_hi       # el cubreobjetos es la pared del lado del objetivo
        x0, y0 = cfg.espejo_punto_px
        self.p0 = np.array([x0 * self.pixel_nm, y0 * self.pixel_nm, z_cubre])

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

    # --- geometria en nm ---
    def distancia_nm(self, p_nm):
        """Distancia con signo al plano [nm] (...,): > 0 del lado de la muestra."""
        return (np.asarray(p_nm, dtype=float) - self.p0) @ self.n

    def reflejar_nm(self, p_nm):
        """p' = p - 2 [(p - p0) . n] n, todo en nm; p_nm (..., 3)."""
        p = np.asarray(p_nm, dtype=float)
        return p - 2.0 * self.distancia_nm(p)[..., None] * self.n

    def reflejar(self, pos):
        """Reflejo de posiciones [x_px, y_px, z_nm] (..., 3), en las mismas unidades.
        Es una involucion: reflejar(reflejar(p)) == p (salvo redondeo)."""
        return self.a_gt(self.reflejar_nm(self.a_nm(pos)))

    def plegar_nm(self, p_nm):
        """Lleva cada punto (..., 3) [nm] al dominio de los emisores reales:
        z_lo <= z <= z_hi y d >= 0 (lado de la muestra), reflejando en la
        pared o en el espejo que viola, hasta que no viole ninguna.

        Es el mismo criterio que las paredes en z (reflexion especular) con el
        espejo como pared extra. Termina: el dominio es convexo y reflejar un
        punto en un semiespacio que viola acerca el punto a todo punto del
        semiespacio, en particular a todo punto del dominio. No usa el rng."""
        p = np.array(p_nm, dtype=float)
        lo, hi = self.z_lo, self.z_hi
        for _ in range(MAX_PLIEGUES):
            z = p[..., 2]
            mal_z = (z < lo) | (z > hi)
            d = self.distancia_nm(p)
            mal_e = d < 0
            if not (mal_z.any() or mal_e.any()):
                return p
            # paredes en z: z -> 2 hi - z o 2 lo - z (una reflexion por vuelta)
            p[..., 2] = np.where(z > hi, 2 * hi - z, np.where(z < lo, 2 * lo - z, z))
            # espejo: p -> p - 2 d n solo donde d < 0 (d recalculado tras plegar z)
            d = self.distancia_nm(p)
            p -= 2.0 * np.minimum(d, 0.0)[..., None] * self.n
        raise RuntimeError(f"plegar_nm no convergio en {MAX_PLIEGUES} reflexiones")


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
    esq = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    d = e.distancia_nm(esq)
    vert = list(esq[d >= 0])
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(esq[i] != esq[j]) == 1 and d[i] * d[j] < 0:   # arista que cruza el plano
                s = d[i] / (d[i] - d[j])
                vert.append(esq[i] + s * (esq[j] - esq[i]))
    if not vert:
        raise ValueError("el espejo deja toda la caja de posiciones iniciales fuera del lado de la muestra")
    zp = e.reflejar_nm(np.array(vert))[:, 2]
    return float(zp.min()), float(zp.max())


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
