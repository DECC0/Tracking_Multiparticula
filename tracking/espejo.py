"""
Tracking con el espejo: coordenadas respecto de la linea del espejo,
emparejamiento de pistas real <-> reflejo, z por triangulacion y suavizado
Kalman / RTS.

Geometria (geometria_espejo.py; +z hacia el objetivo, beta desde el eje
optico): m = (cos phi, sin phi) es la direccion lateral linea -> muestra y
t = (-sin phi, cos phi) la de la linea; p0 es el punto de la linea en el plano
focal (z_ref = 0). Para un punto p, s = (p - p0) . m y t = (p - p0) . t (nm).
El reflejo de un real a distancia d >= 0 del plano tiene
    t' = t                         (la misma coordenada a lo largo del espejo)
    s' = s - 2 d cos(beta) <= s    (siempre del lado del espejo respecto de su real)
y con el punto medio s_M = (s + s') / 2 y la separacion L = s - s' >= 0,
    z = -s_M / tan(beta) + (L / 2) tan(beta).
A 45 grados, z = -s' (no depende del real). Con beta en (40, 50) s' cae a los
dos lados de la linea, asi que el criterio de lado es relativo (s' < s), no el
signo de s'.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

GRANDE = 1e30


def coordenadas_espejo(x_px, y_px, px, phi_deg, p0_px):
    """(s, t) en nm de puntos (x_px, y_px) respecto de la linea del espejo."""
    f = np.radians(phi_deg)
    dx = (np.asarray(x_px, float) - p0_px[0]) * px
    dy = (np.asarray(y_px, float) - p0_px[1]) * px
    return dx * np.cos(f) + dy * np.sin(f), -dx * np.sin(f) + dy * np.cos(f)


def triangular_z(s_real, s_refl, beta_deg):
    """z [nm] por triangulacion: -s_M / tan(beta) + (L / 2) tan(beta)."""
    tb = np.tan(np.radians(beta_deg))
    s_m = (np.asarray(s_real) + np.asarray(s_refl)) / 2
    L = np.asarray(s_real) - np.asarray(s_refl)
    return -s_m / tb + L / 2 * tb


def sigma_z_triangulada(sigma_s_real, sigma_s_refl, beta_deg):
    """Error [nm] de triangular_z por propagacion lineal con errores independientes:
    dz/ds = (tan b - 1/tan b) / 2 y dz/ds' = -(tan b + 1/tan b) / 2."""
    tb = np.tan(np.radians(beta_deg))
    return np.sqrt(((tb - 1 / tb) / 2 * sigma_s_real) ** 2 + ((tb + 1 / tb) / 2 * sigma_s_refl) ** 2)


def kalman_rts(z, var_obs, q):
    """Suavizado RTS de un paseo aleatorio z_k = z_{k-1} + w (var q) observado con
    ruido de varianza var_obs; z (T,) con nan donde no hay observacion.

    Devuelve (media, varianza) suavizadas (T,), nan antes de la primera
    observacion y despues de la ultima (no se extrapola). Prior difuso: la
    primera observacion inicializa el filtro."""
    z = np.asarray(z, float)
    T = len(z)
    obs = np.flatnonzero(np.isfinite(z))
    mu = np.full(T, np.nan)
    var = np.full(T, np.nan)
    if len(obs) == 0:
        return mu, var
    a, b = obs[0], obs[-1]
    mf, pf, mp, pp = (np.zeros(T) for _ in range(4))
    mf[a], pf[a] = z[a], var_obs
    for k in range(a + 1, b + 1):
        mp[k], pp[k] = mf[k - 1], pf[k - 1] + q                    # prediccion
        if np.isfinite(z[k]):
            g = pp[k] / (pp[k] + var_obs)
            mf[k], pf[k] = mp[k] + g * (z[k] - mp[k]), (1 - g) * pp[k]
        else:
            mf[k], pf[k] = mp[k], pp[k]
    mu[b], var[b] = mf[b], pf[b]
    for k in range(b - 1, a - 1, -1):                               # RTS hacia atras
        c = pf[k] / pp[k + 1]
        mu[k] = mf[k] + c * (mu[k + 1] - mp[k + 1])
        var[k] = pf[k] + c ** 2 * (var[k + 1] - pp[k + 1])
    return mu, var


K_DT = 3.0                                      # cuadros consistentes: |dt| < K_DT sigma_dt
C_LR = float(np.log(2 * K_DT / np.sqrt(2 * np.pi)))    # log(ventana / sigma sqrt(2 pi)): razon de verosimilitud
                                                # de un cuadro con dt = 0 frente a dt uniforme en la ventana


def estadisticas_par(R, F, sigma_dt, libres=None):
    """Estadisticas del par de pistas R (real) y F (reflejo), (T, 2) en (s, t) con nan.

    Cuadros comunes: los dos presentes (y libres, si se da la mascara (T,)).
    Cuadros consistentes: comunes con |dt| < K_DT sigma_dt (dt = t_R - t_F) y el
    reflejo del lado del espejo (s_F < s_R). Sobre ellos:
        puntaje = sum (C_LR - (dt / sigma_dt)^2 / 2), el log de la razon de
                  verosimilitudes de "es su reflejo" (dt ~ N(0, sigma_dt)) frente a
                  "coincidencia" (dt uniforme en la ventana); un par verdadero
                  suma ~0.37 por cuadro y una coincidencia ~ -0.63
        dt_medio, corr (t_R con t_F en el tiempo) y mov_t (desviacion de t_R).
    Devuelve el dict y la mascara (T,) de cuadros consistentes."""
    ok = np.isfinite(R[:, 0]) & np.isfinite(F[:, 0])
    if libres is not None:
        ok &= libres
    n = int(ok.sum())
    dt = np.where(ok, R[:, 1] - F[:, 1], np.nan)
    cons = ok & (np.abs(np.nan_to_num(dt, nan=np.inf)) < K_DT * sigma_dt) & (np.nan_to_num(F[:, 0], nan=np.inf) < np.nan_to_num(R[:, 0], nan=-np.inf))
    k = int(cons.sum())
    e = dict(n=n, n_cons=k, puntaje=-np.inf, dt_medio=np.nan, corr=np.nan, mov_t=np.nan,
             lado=float(np.mean(F[ok, 0] < R[ok, 0])) if n else np.nan)
    if k == 0:
        return e, cons
    tr, tf = R[cons, 1], F[cons, 1]
    u = dt[cons] / sigma_dt
    e.update(puntaje=float(np.sum(C_LR - u ** 2 / 2)), dt_medio=float(dt[cons].mean()), mov_t=float(tr.std()),
             corr=float(np.corrcoef(tr, tf)[0, 1]) if k > 2 and tr.std() > 0 and tf.std() > 0 else np.nan)
    return e, cons


def emparejar_pistas(R, F, sigma_dt, n_min=5, puntaje_min=2.0, corr_min=0.5, mov_min=None, z_par=None, rondas=6):
    """Emparejamiento de pistas reales R (n_R, T, 2) con pistas de reflejo F (n_F, T, 2),
    en (s, t) [nm] con nan donde la pista no esta.

    Por rondas: en cada una, para cada par, los cuadros consistentes
    (estadisticas_par) entre los que ninguno de los dos tiene ya asignados. Un
    par es candidato si tiene >= n_min cuadros consistentes, puntaje >=
    puntaje_min, |dt_medio| < sigma_dt, correlacion >= corr_min cuando R se
    mueve en t mas que mov_min (por defecto 3 sigma_dt; si no se mueve la
    correlacion no informa) y z_par(i, j, cons) (p. ej. z triangulada posible).
    Una asignacion hungara maximiza el puntaje total (no emparejar vale 0) y
    cada par asignado consume solo sus cuadros consistentes: si una pista
    cambia de particula a mitad del video, sus otros cuadros pueden
    emparejarse en una ronda siguiente con el reflejo que corresponde, y una
    pista cortada puede emparejarse con varios fragmentos.

    Devuelve una lista de dicts (i, j, ronda, cuadros: mascara (T,) de cuadros
    asignados, y las estadisticas del par) y la mascara (n_R, T) de cuadros
    reales con reflejo asignado."""
    n_r, n_f = len(R), len(F)
    T = R.shape[1] if n_r else (F.shape[1] if n_f else 0)
    mov_min = 3 * sigma_dt if mov_min is None else mov_min
    usado_r = np.zeros((n_r, T), bool)
    usado_f = np.zeros((n_f, T), bool)
    pares = []
    if n_r == 0 or n_f == 0:
        return pares, usado_r
    presente_r = np.isfinite(R[..., 0])
    presente_f = np.isfinite(F[..., 0])
    for ronda in range(rondas):
        libre_r = presente_r & ~usado_r
        libre_f = presente_f & ~usado_f
        comunes = libre_r.astype(int) @ libre_f.T.astype(int)
        C = np.full((n_r, n_f), np.inf)
        est = {}
        for i, j in zip(*np.nonzero(comunes >= n_min)):
            e, cons = estadisticas_par(R[i], F[j], sigma_dt, libres=libre_r[i] & libre_f[j])
            if e["n_cons"] < n_min or e["puntaje"] < puntaje_min or abs(e["dt_medio"]) >= sigma_dt:
                continue
            if e["mov_t"] > mov_min and not (e["corr"] >= corr_min):
                continue
            if z_par is not None and not z_par(i, j, cons):
                continue
            C[i, j] = -e["puntaje"]
            est[i, j] = (e, cons)
        if not est:
            break
        A = np.full((n_r + n_f, n_f + n_r), GRANDE)
        A[:n_r, :n_f] = np.where(np.isfinite(C), C, GRANDE)
        A[:n_r, n_f:] = np.where(np.eye(n_r, dtype=bool), 0.0, GRANDE)
        A[n_r:, :n_f] = np.where(np.eye(n_f, dtype=bool), 0.0, GRANDE)
        A[n_r:, n_f:] = np.where(np.isfinite(C).T, 0.0, GRANDE)
        f, c = linear_sum_assignment(A)
        nuevos = 0
        for i, j in zip(f, c):
            if i < n_r and j < n_f and (i, j) in est:
                e, cons = est[i, j]
                usado_r[i] |= cons
                usado_f[j] |= cons
                pares.append(dict(i=int(i), j=int(j), ronda=ronda, cuadros=cons, **e))
                nuevos += 1
        if nuevos == 0:
            break
    return pares, usado_r
