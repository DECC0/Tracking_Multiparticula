"""
Deteccion clasica de manchas (para los reflejos): filtro paso banda DoG,
maximos locales sobre un umbral de ruido y ajuste gaussiano 2D por minimos
cuadrados en una ventana alrededor de cada maximo.

    imagen (e-) -> DoG = G(sigma_1) - G(sigma_2)  (quita el fondo lento y la
    autofluorescencia de la celula) -> ruido = 1.4826 MAD(DoG) -> maximos
    locales en una vecindad (2 r + 1)^2 con DoG > k ruido -> ajuste de
    A exp(-|r - r0|^2 / (2 s^2)) + B en una ventana (2 w + 1)^2 de la imagen
    original -> se queda con los ajustes con s en [s_min, s_max] y el centro a
    menos de 1.5 px del maximo.

El ajuste usa scipy.optimize.least_squares con el jacobiano analitico. Las
coordenadas siguen la convencion de simulador_nuevo: el pixel (fila i,
columna j) esta centrado en (x = j, y = i).
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares


def _modelo(p, X, Y):
    A, x0, y0, s, B = p
    e = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * s * s))
    return A * e + B, e


def ajustar_gaussiana(img, x, y, w=5, s0=1.5):
    """Ajuste gaussiano 2D en la ventana (2 w + 1)^2 centrada en el pixel (y, x).
    Devuelve (A, x0, y0, s, B, ok) en px y unidades de img."""
    H, W = img.shape
    i0, j0 = int(round(y)), int(round(x))
    a, b = max(i0 - w, 0), min(i0 + w + 1, H)
    c, d = max(j0 - w, 0), min(j0 + w + 1, W)
    v = img[a:b, c:d].astype(float)
    Y, X = np.mgrid[a:b, c:d].astype(float)
    B0 = float(np.percentile(v, 20))
    p0 = [max(float(img[i0, j0]) - B0, 1.0), float(j0), float(i0), s0, B0]

    def res(p):
        return (_modelo(p, X, Y)[0] - v).ravel()

    def jac(p):
        A, x0, y0, s, _ = p
        _, e = _modelo(p, X, Y)
        r2 = (X - x0) ** 2 + (Y - y0) ** 2
        return np.c_[e.ravel(), (A * e * (X - x0) / s ** 2).ravel(), (A * e * (Y - y0) / s ** 2).ravel(),
                     (A * e * r2 / s ** 3).ravel(), np.ones(e.size)]

    lo = [0.0, j0 - 2.5, i0 - 2.5, 0.4, -np.inf]
    hi = [np.inf, j0 + 2.5, i0 + 2.5, 6.0, np.inf]
    try:
        r = least_squares(res, np.clip(p0, lo, hi), jac=jac, bounds=(lo, hi), x_scale="jac", max_nfev=60)
        return (*r.x, bool(r.success))
    except (ValueError, np.linalg.LinAlgError):
        return (*p0, False)


def detectar_manchas(img, sigma1=1.5, sigma2=4.0, k=5.0, r=2, w=5, s_min=0.6, s_max=4.0, borde=2):
    """Manchas de una imagen (H, W) en e-: DataFrame-like dict de arreglos x_px, y_px,
    amplitud, sigma_px, fondo, fotones (2 pi A s^2), snr (pico DoG / ruido)."""
    img = np.asarray(img, float)
    dog = ndimage.gaussian_filter(img, sigma1) - ndimage.gaussian_filter(img, sigma2)
    ruido = 1.4826 * np.median(np.abs(dog - np.median(dog)))
    mx = ndimage.maximum_filter(dog, size=2 * r + 1, mode="nearest")
    cand = (dog == mx) & (dog > k * ruido)
    cand[:borde] = cand[-borde:] = False
    cand[:, :borde] = cand[:, -borde:] = False
    iy, ix = np.nonzero(cand)
    filas = []
    for i, j in zip(iy, ix):
        A, x0, y0, s, B, ok = ajustar_gaussiana(img, j, i, w=w)
        if ok and s_min <= s <= s_max and np.hypot(x0 - j, y0 - i) < 1.5:
            filas.append((x0, y0, A, s, B, 2 * np.pi * A * s * s, dog[i, j] / ruido))
    cols = ("x_px", "y_px", "amplitud", "sigma_px", "fondo", "fotones", "snr")
    a = np.array(filas, float).reshape(-1, len(cols))
    return {c: a[:, n] for n, c in enumerate(cols)}
