"""
Enlace de detecciones en pistas: linea base clasica tipo LAP (Jaqaman et al.,
Nat. Methods 2008), sin fusiones ni divisiones.

1. Cuadro a cuadro: asignacion hungara entre las detecciones del cuadro t - 1
   y las del cuadro t, con costo d^2 (distancia euclidea en 3D con z
   reescalada) y la matriz aumentada de Jaqaman: cada deteccion puede quedar
   sin enlazar (muerte / nacimiento) con costo b = radio^2, asi que dos
   detecciones se enlazan solo si d < radio y la asignacion es optima en
   conjunto (no voraz).
2. Cierre de huecos: los finales y comienzos de segmento separados por
   1 .. max_hueco cuadros vacios se enlazan con otra asignacion hungara
   aumentada, con costo d^2 / g (g = cuadros entre el final y el comienzo:
   difusion) y umbral radio^2 (es decir d < radio sqrt(g)).

Costos (el parametro costo de enlazar; los dos con el mismo esquema):
    CostoEuclideo(radio)  linea base (tracking_base.py): d^2 / g con umbral
        radio^2. Coordenadas X, Y, Z en nm con Z ya multiplicada por escala_z
        (z reescalada por su error, escala_z = sigma_lateral / sigma_z), asi el
        radio es isotropo.
    CostoNormalizado(sigma, D, dt, r_chi, c_loc)  distancia normalizada por eje
        d^2 = sum_e (a_e - b_e)^2 / (c_loc sigma_e^2 + 2 D g dt), con umbral
        r_chi^2: es el chi^2 del desplazamiento si el paso browniano y el error
        de localizacion son gaussianos e independientes por eje. c_loc = 2
        cuando los errores de dos cuadros son independientes (la diferencia
        tiene 2 sigma^2); 1 es la formula sigma^2 + 2 D dt. Coordenadas en nm
        sin reescalar.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

GRANDE = 1e30           # costo de un enlace prohibido (linear_sum_assignment no acepta inf en todas las filas)


def coordenadas(df, pixel_nm, escala_z):
    """(n, 3) en nm con z reescalada: x_px, y_px, z_nm de df."""
    return np.c_[df.x_px.to_numpy() * pixel_nm, df.y_px.to_numpy() * pixel_nm, df.z_nm.to_numpy() * escala_z]


def asignacion_lap(costo, b):
    """Pares (i, j) de la asignacion de Jaqaman sobre costo (n, m) (np.inf =
    prohibido), con costo b de no enlazar cada fila y cada columna."""
    n, m = costo.shape
    if n == 0 or m == 0:
        return np.zeros((0, 2), int)
    valido = np.isfinite(costo)
    if not valido.any():
        return np.zeros((0, 2), int)
    A = np.full((n + m, m + n), GRANDE)
    A[:n, :m] = np.where(valido, costo, GRANDE)
    A[:n, m:] = np.where(np.eye(n, dtype=bool), b, GRANDE)            # muerte
    A[n:, :m] = np.where(np.eye(m, dtype=bool), b, GRANDE)            # nacimiento
    A[n:, m:] = np.where(valido.T, costo[valido].min(), GRANDE)       # bloque auxiliar (transpuesto)
    f, c = linear_sum_assignment(A)
    sel = (f < n) & (c < m)
    pares = np.c_[f[sel], c[sel]]
    return pares[valido[pares[:, 0], pares[:, 1]]] if len(pares) else pares


class CostoEuclideo:
    """d^2 / g (g = cuadros entre las dos detecciones) con umbral b = radio^2."""

    def __init__(self, radio):
        self.b = float(radio) ** 2

    def __call__(self, a, c, g):
        return ((a[:, None, :] - c[None, :, :]) ** 2).sum(-1) / g


class CostoNormalizado:
    """sum_e (a_e - c_e)^2 / (c_loc sigma_e^2 + 2 D g dt) con umbral b = r_chi^2.

    sigma : (d,) error de localizacion por eje [nm]; D [nm^2/s]; dt [s]."""

    def __init__(self, sigma, D, dt, r_chi, c_loc=2.0):
        self.s2 = c_loc * np.asarray(sigma, float) ** 2
        self.q = 2.0 * D * dt
        self.b = float(r_chi) ** 2

    def __call__(self, a, c, g):
        g = np.asarray(g, float)
        var = self.s2 + self.q * g[..., None]                    # (..., d)
        return ((a[:, None, :] - c[None, :, :]) ** 2 / var).sum(-1)


def enlazar(frames, pos, radio=None, max_hueco=2, costo=None):
    """Pistas de detecciones.

    frames : (n,) int, cuadro de cada deteccion
    pos    : (n, d) coordenadas (ver coordenadas y los costos)
    radio  : desplazamiento maximo entre cuadros consecutivos (mismas unidades);
             equivale a costo=CostoEuclideo(radio)
    costo  : objeto con b (costo de no enlazar) y __call__(a, c, g) -> (n_a, n_c)
    Devuelve (n,) int: id de pista de cada deteccion (0, 1, ...)."""
    if costo is None:
        costo = CostoEuclideo(radio)
    frames = np.asarray(frames)
    n = len(frames)
    pista = -np.ones(n, int)
    if n == 0:
        return pista
    b = costo.b
    orden = {t: np.flatnonzero(frames == t) for t in np.unique(frames)}
    ts = sorted(orden)
    sig = 0
    for t in ts:
        act = orden[t]
        prev = orden.get(t - 1, np.zeros(0, int))
        if len(prev) and len(act):
            d2 = costo(pos[prev], pos[act], 1.0)
            d2[d2 >= b] = np.inf
            for i, j in asignacion_lap(d2, b):
                pista[act[j]] = pista[prev[i]]
        for j in act[pista[act] < 0]:
            pista[j] = sig
            sig += 1
    if max_hueco < 1:
        return pista
    # cierre de huecos sobre los segmentos
    ids = np.unique(pista)
    ini_t = np.array([frames[pista == k].min() for k in ids])
    fin_t = np.array([frames[pista == k].max() for k in ids])
    ini_p = np.array([pos[(pista == k) & (frames == t)][0] for k, t in zip(ids, ini_t)])
    fin_p = np.array([pos[(pista == k) & (frames == t)][0] for k, t in zip(ids, fin_t)])
    g = ini_t[None, :] - fin_t[:, None]                               # (finales, comienzos)
    candidato = (g >= 2) & (g <= max_hueco + 1)
    if not candidato.any():
        return _renumerar(pista)
    d2 = costo(fin_p, ini_p, np.maximum(g, 1))
    d2 = np.where(candidato & (d2 < b), d2, np.inf)
    padre = {int(k): int(k) for k in ids}
    for i, j in asignacion_lap(d2, b):
        padre[int(ids[j])] = int(ids[i])
    raiz = {}
    for k in padre:
        r = k
        while padre[r] != r:
            r = padre[r]
        raiz[k] = r
    return _renumerar(np.array([raiz[int(k)] for k in pista]))


def filtrar_cortas(pista, largo_min):
    """Mascara (n,) de las detecciones cuyas pistas tienen >= largo_min detecciones."""
    _, inv, cuenta = np.unique(pista, return_inverse=True, return_counts=True)
    return cuenta[inv] >= largo_min


def _renumerar(p):
    _, inv = np.unique(p, return_inverse=True)
    return inv
