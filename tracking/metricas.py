"""
Metricas de tracking.

ISBI 2012 (Chenouard et al., Nat. Methods 11, 281 (2014), "Objective
comparison of particle tracking methods"):
    distancia entre una pista real theta y una estimada omega, sobre la union de
    sus cuadros: sum_t min(||theta(t) - omega(t)||, eps), con eps si solo una de
    las dos esta definida en t. Las pistas reales se emparejan con las
    estimadas o con pistas vacias (costo eps |theta|) por asignacion hungara
    que minimiza la distancia total d(X, Y*).
    alpha = 1 - d(X, Y*) / d(X, vacio)                  en [0, 1], sin penalizar pistas espurias
    beta  = (d(X, vacio) - d(X, Y*)) / (d(X, vacio) + d(Ybar, vacio))
            con Ybar las pistas estimadas sin pareja (espurias): beta <= alpha
    JSC (detecciones): TP = puntos de pares emparejados definidos en los dos a
            distancia < eps; FN = puntos reales que no son TP; FP = puntos
            estimados que no son TP. JSC = TP / (TP + FN + FP)
    JSC_theta (pistas): TP = pistas reales emparejadas con una estimada, FN = las
            emparejadas con una vacia, FP = estimadas sin pareja
Cambios de identidad y fragmentacion (como CLEAR MOT): en cada cuadro los
puntos reales y estimados se emparejan (hungaro, distancia < eps); para cada
pista real, la secuencia de ids estimados emparejados da
    idsw  = veces que el id emparejado cambia respecto del ultimo emparejado
    frag  = interrupciones: tramos emparejados - 1
    n_ids = ids estimados distintos que la cubren
D por MSD: MSD(tau) = <|r(t + tau) - r(t)|^2> con los pares disponibles
(admite huecos), ajuste lineal con ordenada en tau = 1..n_lags (la ordenada
absorbe el error de localizacion): D = pendiente / (2 d dt), d = 3 (3D) o 1 (z).

Para comparar D, cada pista estimada se asocia a la real con la que comparte mas
cuadros emparejados (particula_mayoritaria) y su D se compara con el que da el
mismo estimador sobre esa pista real (confinamiento y lags iguales).

Las distancias se calculan en las coordenadas que se pasan: el llamador elige
la escala de z (p. ej. z * 150/300 para que eps = 150 nm sea el elipsoide de
150 nm lateral y 300 nm axial).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def a_arreglo(ids, frames, pos, T):
    """(n_pistas, T, 3) con nan donde la pista no esta, y la lista de ids."""
    u = np.unique(ids)
    A = np.full((len(u), T, pos.shape[1]), np.nan)
    k = np.searchsorted(u, ids)
    A[k, frames] = pos
    return A, u


def distancias_pistas(X, Y, eps):
    """(n_x, n_y) distancia ISBI entre cada par de pistas de X (n_x, T, 3) y Y (n_y, T, 3)."""
    if len(X) == 0 or len(Y) == 0:
        return np.zeros((len(X), len(Y)))
    dx = np.isfinite(X[..., 0])
    dy = np.isfinite(Y[..., 0])
    d = np.sqrt(((np.nan_to_num(X)[:, None] - np.nan_to_num(Y)[None]) ** 2).sum(-1))       # (n_x, n_y, T)
    ambos = dx[:, None] & dy[None]
    uno = dx[:, None] ^ dy[None]
    return np.where(ambos, np.minimum(d, eps), 0).sum(-1) + eps * uno.sum(-1)


def isbi(X, Y, eps):
    """Sumas de un video para alpha, beta y JSC (se agregan sumando entre videos).

    X : (n_x, T, 3) pistas reales; Y : (n_y, T, 3) pistas estimadas."""
    n_x, n_y = len(X), len(Y)
    len_x = np.isfinite(X[..., 0]).sum(1) if n_x else np.zeros(0)
    len_y = np.isfinite(Y[..., 0]).sum(1) if n_y else np.zeros(0)
    d_vacio = float(eps * len_x.sum())
    pareja = -np.ones(n_x, int)
    d_xy = d_vacio
    if n_x and n_y:
        D = distancias_pistas(X, Y, eps)
        C = np.full((n_x, n_y + n_x), np.inf)
        C[:, :n_y] = D
        C[np.arange(n_x), n_y + np.arange(n_x)] = eps * len_x
        f, c = linear_sum_assignment(np.where(np.isfinite(C), C, 1e30))
        d_xy = float(C[f, c].sum())
        real = c < n_y
        pareja[f[real]] = c[real]
    espurias = np.setdiff1d(np.arange(n_y), pareja[pareja >= 0])
    d_espurias = float(eps * len_y[espurias].sum()) if n_y else 0.0
    tp = 0
    for i in np.flatnonzero(pareja >= 0):
        a, b = X[i], Y[pareja[i]]
        ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
        tp += int((np.sqrt(((a[ok] - b[ok]) ** 2).sum(-1)) < eps).sum())
    return dict(d_xy=d_xy, d_x0=d_vacio, d_ybar0=d_espurias,
                tp_det=tp, fn_det=int(len_x.sum()) - tp, fp_det=int(len_y.sum()) - tp,
                tp_pistas=int((pareja >= 0).sum()), fn_pistas=int((pareja < 0).sum()), fp_pistas=len(espurias),
                n_pistas_gt=n_x, n_pistas_est=n_y)


def de_sumas(s):
    """alpha, beta, JSC y JSC_theta de un dict (o Series) de sumas de isbi()."""
    a = 1 - s["d_xy"] / s["d_x0"] if s["d_x0"] else np.nan
    b = (s["d_x0"] - s["d_xy"]) / (s["d_x0"] + s["d_ybar0"]) if s["d_x0"] else np.nan
    jsc = s["tp_det"] / max(s["tp_det"] + s["fn_det"] + s["fp_det"], 1)
    jsc_t = s["tp_pistas"] / max(s["tp_pistas"] + s["fn_pistas"] + s["fp_pistas"], 1)
    return dict(alpha=a, beta=b, jsc=jsc, jsc_pistas=jsc_t)


def identidades(X, Y, eps, emparejamiento=False):
    """(idsw, frag, n_ids) por pista real (arreglos (n_x,)), con el emparejamiento por cuadro;
    con emparejamiento=True agrega m (n_x, T): indice de la pista estimada emparejada (-1: ninguna)."""
    n_x, T = X.shape[:2]
    m = -np.ones((n_x, T), int)
    for t in range(T):
        i = np.flatnonzero(np.isfinite(X[:, t, 0]))
        j = np.flatnonzero(np.isfinite(Y[:, t, 0])) if len(Y) else np.zeros(0, int)
        if len(i) == 0 or len(j) == 0:
            continue
        d = np.sqrt(((X[i, t][:, None] - Y[j, t][None]) ** 2).sum(-1))
        f, c = linear_sum_assignment(np.where(d < eps, d, 1e30))
        ok = d[f, c] < eps
        m[i[f[ok]], t] = j[c[ok]]
    idsw, frag, n_ids = np.zeros(n_x, int), np.zeros(n_x, int), np.zeros(n_x, int)
    for k in range(n_x):
        s = m[k][np.isfinite(X[k, :, 0])]
        e = s[s >= 0]
        n_ids[k] = len(np.unique(e))
        idsw[k] = int((np.diff(e) != 0).sum()) if len(e) > 1 else 0
        emp = (s >= 0).astype(int)
        tramos = int(((np.diff(np.r_[0, emp]) == 1)).sum())
        frag[k] = max(tramos - 1, 0)
    return (idsw, frag, n_ids, m) if emparejamiento else (idsw, frag, n_ids)


def particula_mayoritaria(m, n_y):
    """(n_y,) indice de la pista real con la que cada pista estimada comparte mas cuadros
    emparejados (-1 si ninguno) y (n_y,) cuantos cuadros comparte, de m de identidades()."""
    cuenta = np.zeros((m.shape[0], n_y), int)
    i, t = np.nonzero(m >= 0)
    np.add.at(cuenta, (i, m[i, t]), 1)
    if n_y == 0 or m.shape[0] == 0:
        return -np.ones(n_y, int), np.zeros(n_y, int)
    k = cuenta.argmax(0)
    c = cuenta[k, np.arange(n_y)]
    return np.where(c > 0, k, -1), c


def d_msd(r, dt, n_lags=4, min_puntos=10, min_pares=3):
    """D [unidades^2 / s] de una pista r (T, d) con nan en los huecos: pendiente del
    ajuste lineal (con ordenada) de MSD(tau), tau = 1..n_lags, / (2 d dt). nan si la
    pista tiene menos de min_puntos puntos o algun lag tiene menos de min_pares pares."""
    ok = np.isfinite(r[:, 0])
    if ok.sum() < min_puntos:
        return np.nan
    taus, msd = [], []
    for tau in range(1, n_lags + 1):
        a, b = r[:-tau], r[tau:]
        v = ok[:-tau] & ok[tau:]
        if v.sum() < min_pares:
            return np.nan
        taus.append(tau * dt)
        msd.append(np.mean(((b[v] - a[v]) ** 2).sum(-1)))
    pend = np.polyfit(taus, msd, 1)[0]
    return pend / (2 * r.shape[1])
