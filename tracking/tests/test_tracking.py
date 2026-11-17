"""
Tests del enlace y de las metricas de tracking (assert, sin pytest).

    PY=~/miniforge3/envs/tesis/bin/python
    $PY tracking/tests/test_tracking.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))                 # geometria_espejo.py
from enlace import enlazar, asignacion_lap, CostoEuclideo, CostoNormalizado, filtrar_cortas  # noqa: E402
from espejo import (coordenadas_espejo, triangular_z, sigma_z_triangulada, kalman_rts,  # noqa: E402
                    emparejar_pistas)
from reflejos import detectar_manchas  # noqa: E402
from metricas import a_arreglo, isbi, de_sumas, identidades, d_msd, distancias_pistas, particula_mayoritaria  # noqa: E402


def _pistas(n=5, T=30, sep=1000.0, paso=20.0, seed=0):
    rng = np.random.default_rng(seed)
    r0 = np.c_[np.arange(n) * sep, np.zeros(n), np.zeros(n)]
    r = r0[:, None] + np.cumsum(rng.normal(0, paso, (n, T, 3)), 1)
    ids = np.repeat(np.arange(n), T)
    fr = np.tile(np.arange(T), n)
    return ids, fr, r.reshape(-1, 3), r


def test_lap_no_enlaza_lejos():
    c = np.array([[1.0, np.inf], [np.inf, 50.0]])
    p = asignacion_lap(c, b=10.0)                          # el segundo par cuesta mas que no enlazar
    assert p.tolist() == [[0, 0]], p


def test_lap_optimo_global():
    # voraz enlazaria 0-0 (1) y dejaria 1 sin pareja; el optimo es 0-1 y 1-0
    c = np.array([[1.0, 2.0], [2.0, np.inf]])
    p = sorted(asignacion_lap(c, b=10.0).tolist())
    assert p == [[0, 1], [1, 0]], p


def test_enlace_recupera_pistas_separadas():
    ids, fr, pos, _ = _pistas()
    orden = np.random.default_rng(1).permutation(len(fr))
    p = enlazar(fr[orden], pos[orden], radio=200.0)
    for k in range(5):
        assert len(np.unique(p[ids[orden] == k])) == 1
    assert len(np.unique(p)) == 5


def test_cierre_de_huecos():
    ids, fr, pos, _ = _pistas(n=3)
    quitar = ((ids == 1) & ((fr == 10) | (fr == 11))) | ((ids == 2) & (fr == 20))
    p = enlazar(fr[~quitar], pos[~quitar], radio=200.0, max_hueco=2)
    assert len(np.unique(p)) == 3
    p0 = enlazar(fr[~quitar], pos[~quitar], radio=200.0, max_hueco=0)
    assert len(np.unique(p0)) == 5
    quitar3 = (ids == 1) & (fr >= 10) & (fr <= 12)       # hueco de 3 cuadros: no se cierra
    assert len(np.unique(enlazar(fr[~quitar3], pos[~quitar3], radio=200.0, max_hueco=2))) == 4


def test_isbi_perfecto_y_vacio():
    ids, fr, pos, r = _pistas()
    X, _ = a_arreglo(ids, fr, pos, 30)
    s = isbi(X, X.copy(), eps=150.0)
    m = de_sumas(s)
    assert np.isclose(m["alpha"], 1) and np.isclose(m["beta"], 1) and m["jsc"] == 1 and m["jsc_pistas"] == 1
    s0 = isbi(X, np.zeros((0, 30, 3)), eps=150.0)
    m0 = de_sumas(s0)
    assert m0["alpha"] == 0 and m0["beta"] == 0 and m0["jsc"] == 0


def test_isbi_espurias_bajan_beta_no_alpha():
    ids, fr, pos, _ = _pistas()
    X, _ = a_arreglo(ids, fr, pos, 30)
    Y = np.concatenate([X, X[:1] + 1e5])                   # una pista espuria lejos de todo
    m = de_sumas(isbi(X, Y, eps=150.0))
    assert np.isclose(m["alpha"], 1) and m["beta"] < 1 and np.isclose(m["jsc_pistas"], 5 / 6)
    assert np.isclose(m["beta"], 5 * 30 / (5 * 30 + 30))


def test_isbi_distancia_union():
    X = np.full((1, 4, 3), np.nan); X[0, :2] = 0.0
    Y = np.full((1, 4, 3), np.nan); Y[0, 1:] = 0.0; Y[0, 1, 0] = 30.0
    # t0: solo X (eps); t1: los dos a 30; t2, t3: solo Y (eps, eps)
    assert np.isclose(distancias_pistas(X, Y, 100.0)[0, 0], 100 + 30 + 100 + 100)


def test_identidades():
    T = 10
    X = np.zeros((1, T, 3))
    Y = np.full((2, T, 3), np.nan)
    Y[0, :4] = 0.0
    Y[1, 6:] = 0.0                                         # cuadros 4-5 sin emparejar, luego otro id
    idsw, frag, n_ids = identidades(X, Y, eps=50.0)
    assert idsw.tolist() == [1] and frag.tolist() == [1] and n_ids.tolist() == [2]
    *_, m = identidades(X, Y, eps=50.0, emparejamiento=True)
    k, c = particula_mayoritaria(m, 2)
    assert k.tolist() == [0, 0] and c.tolist() == [4, 4]
    Y2 = np.concatenate([Y, np.full((1, T, 3), 1e4)])       # pista espuria: sin particula
    k, c = particula_mayoritaria(identidades(X, Y2, eps=50.0, emparejamiento=True)[3], 3)
    assert k.tolist() == [0, 0, -1]


def test_d_msd():
    rng = np.random.default_rng(0)
    D, dt = 0.2e6, 0.1                                     # nm^2/s
    est = [d_msd(np.cumsum(rng.normal(0, np.sqrt(2 * D * dt), (2000, 3)), 0), dt) for _ in range(5)]
    assert abs(np.mean(est) / D - 1) < 0.05, np.mean(est) / D
    # el error de localizacion va a la ordenada, no a la pendiente
    r = np.cumsum(rng.normal(0, np.sqrt(2 * D * dt), (20000, 1)), 0) + rng.normal(0, 300, (20000, 1))
    assert abs(d_msd(r, dt) / D - 1) < 0.1
    r[5:8] = np.nan                                        # huecos
    assert np.isfinite(d_msd(r, dt))
    assert np.isnan(d_msd(r[:9], dt))


def test_costo_euclideo_igual_a_radio():
    ids, fr, pos, _ = _pistas(n=4, paso=60.0)
    quitar = (ids == 2) & (fr == 7)
    a = enlazar(fr[~quitar], pos[~quitar], radio=250.0, max_hueco=2)
    b = enlazar(fr[~quitar], pos[~quitar], costo=CostoEuclideo(250.0), max_hueco=2)
    assert np.array_equal(a, b)


def test_costo_normalizado():
    # chi^2 por eje: un desplazamiento de 3 sigma en z y 0 en x, y pesa 9
    c = CostoNormalizado(sigma=[10.0, 10.0, 100.0], D=0.0, dt=0.1, r_chi=4.0, c_loc=1.0)
    d = c(np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 300.0]]), 1.0)
    assert np.isclose(d[0, 0], 9.0) and c.b == 16.0
    # la varianza del paso crece con g (cierre de huecos), y c_loc multiplica sigma^2
    c2 = CostoNormalizado(sigma=[10.0, 10.0, 10.0], D=1e6, dt=0.1, r_chi=4.0, c_loc=2.0)
    a, b = np.zeros((1, 3)), np.ones((1, 3)) * 100.0
    g = np.array([[3.0]])
    assert np.isclose(c2(a, b, g)[0, 0], 3 * 100.0 ** 2 / (2 * 100.0 + 2e6 * 0.1 * 3))
    # z mucho peor que x, y: con el costo normalizado se enlaza un salto grande en z, no en x
    rng = np.random.default_rng(3)
    T = 40
    r = np.c_[np.zeros(T), np.zeros(T), np.zeros(T)] + rng.normal(0, [10, 10, 100], (T, 3))
    p = enlazar(np.arange(T), r, costo=CostoNormalizado([10, 10, 100], 0.0, 0.1, 4.59, c_loc=2.0))
    assert len(np.unique(p)) == 1
    assert len(np.unique(enlazar(np.arange(T), r, radio=4.59 * np.sqrt(2) * 10))) > 1


def test_filtrar_cortas():
    p = np.array([0, 0, 0, 1, 2, 2])
    assert filtrar_cortas(p, 2).tolist() == [True, True, True, False, True, True]
    assert filtrar_cortas(p, 1).all()


def test_triangulacion_con_geometria_espejo():
    from geometria_espejo import GeometriaEspejo
    rng = np.random.default_rng(0)
    px = 6500 / 66
    for beta, phi in ((45.0, 30.0), (41.0, 200.0), (49.5, 333.0)):
        p0_px = np.array([20.0, 50.0])
        g = GeometriaEspejo(beta, phi, np.r_[p0_px * px, 0.0], -3000, 3000)
        p = np.c_[rng.uniform(0, 96, 50) * px, rng.uniform(0, 96, 50) * px, rng.uniform(-3000, 3000, 50)]
        p = g.plegar_nm(p)
        q = g.reflejar_nm(p)
        s, t = coordenadas_espejo(p[:, 0] / px, p[:, 1] / px, px, phi, p0_px)
        s2, t2 = coordenadas_espejo(q[:, 0] / px, q[:, 1] / px, px, phi, p0_px)
        assert np.allclose(t, t2, atol=1e-6)                         # misma coordenada a lo largo del espejo
        assert (s2 <= s + 1e-6).all()                                  # reflejo del lado del espejo
        assert np.allclose(triangular_z(s, s2, beta), p[:, 2], atol=1e-6)
    # a 45 grados el error es el del reflejo a lo largo de m
    assert np.isclose(sigma_z_triangulada(30.0, 50.0, 45.0), 50.0)


def test_kalman_rts():
    rng = np.random.default_rng(0)
    T, q, r = 400, 20.0 ** 2, 80.0 ** 2
    x = np.cumsum(rng.normal(0, np.sqrt(q), T))
    z = x + rng.normal(0, np.sqrt(r), T)
    z[100:110] = np.nan
    mu, var = kalman_rts(z, r, q)
    e_obs = np.sqrt(np.nanmean((z - x) ** 2))
    e_rts = np.sqrt(np.mean((mu - x) ** 2))
    assert e_rts < 0.6 * e_obs, (e_rts, e_obs)
    # la varianza reportada es la del error (calibrada) y crece en el hueco
    assert 0.7 < np.mean((mu - x) ** 2) / np.mean(var) < 1.4
    assert var[105] > var[50]
    mu2, _ = kalman_rts(np.r_[np.nan, 5.0, np.nan], r, q)
    assert np.isnan(mu2[0]) and mu2[1] == 5.0 and np.isnan(mu2[2])


def test_emparejar_pistas():
    rng = np.random.default_rng(1)
    T, n = 50, 4
    t0 = np.array([0.0, 400.0, 800.0, 3000.0])
    tt = t0[:, None] + np.cumsum(rng.normal(0, 150, (n, T)), 1)
    s = 3000 + rng.normal(0, 30, (n, T))
    R = np.stack([s, tt + rng.normal(0, 30, (n, T))], -1)
    F = np.stack([s - 2000, tt + rng.normal(0, 60, (n, T))], -1)[::-1].copy()      # orden invertido
    F[1, 20:] = np.nan                                                              # el reflejo de 2 se corta
    G = F[1].copy()
    G[:20] = np.nan
    G[20:] = np.c_[s[2, 20:] - 2000, tt[2, 20:] + rng.normal(0, 60, T - 20)]
    F = np.concatenate([F, G[None]], 0)
    sig = np.hypot(30, 60)
    pares, cubierto = emparejar_pistas(R, F, sigma_dt=sig)
    m = {(p["i"], p["j"]) for p in pares}
    assert {(0, 3), (1, 2), (2, 1), (3, 0), (2, 4)} == m, m                         # 2 se empareja con dos fragmentos
    assert cubierto.mean() > 0.97
    # un reflejo del lado equivocado no se empareja
    F2 = F.copy()
    F2[:, :, 0] = 5000
    assert emparejar_pistas(R, F2, sigma_dt=sig)[0] == []
    # cambio de identidad: la pista real 0 sigue a la particula 0 hasta 25 y despues a la 3; se empareja
    # con los dos reflejos, cada uno solo en los cuadros que le corresponden
    R3 = R.copy()
    R3[0, 25:], R3[3, 25:] = R[3, 25:], R[0, 25:]
    pares = emparejar_pistas(R3, F, sigma_dt=sig)[0]
    for p in pares:
        if p["i"] == 0:
            esperado = np.arange(T) < 25 if p["j"] == 3 else np.arange(T) >= 25
            assert p["j"] in (3, 0) and (p["cuadros"] <= esperado).all() and p["cuadros"].sum() > 0.9 * esperado.sum()
    assert {p["j"] for p in pares if p["i"] == 0} == {0, 3}
    # mismo t con un desplazamiento constante de 1.5 sigma (otra particula): no se empareja
    F4 = R.copy()
    F4[..., 0] -= 2000
    F4[..., 1] += 1.5 * sig + rng.normal(0, 60, (n, T))
    assert emparejar_pistas(R, F4, sigma_dt=sig)[0] == []


def test_detectar_manchas():
    rng = np.random.default_rng(0)
    H, W = 40, 56
    Y, X = np.mgrid[:H, :W].astype(float)
    fuentes = [(10.3, 12.7, 1.3), (40.6, 25.2, 2.0)]
    img = 10.0 + sum(300 * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * s * s)) for x, y, s in fuentes)
    d = detectar_manchas(rng.poisson(img).astype(float))
    assert len(d["x_px"]) == 2, d
    o = np.argsort(d["x_px"])
    for k, (x, y, s) in enumerate(fuentes):
        assert abs(d["x_px"][o[k]] - x) < 0.15 and abs(d["y_px"][o[k]] - y) < 0.15     # convencion x = columna
        assert abs(d["sigma_px"][o[k]] - s) < 0.2


def test_isbi_por_pista():
    ids, fr, pos, r = _pistas()
    X, _ = a_arreglo(ids, fr, pos, 30)
    Y = X.copy()
    Y[0, 10:] = np.nan
    s, p = isbi(X, Y, 50.0, por_pista=True)
    assert np.isclose(p["d"].sum(), s["d_xy"]) and np.isclose(p["d0"].sum(), s["d_x0"])
    assert p["tp"].sum() == s["tp_det"] and p["tp"][0] == 10 and (p["fp"] == 0).all()
    assert s == isbi(X, Y, 50.0)


if __name__ == "__main__":
    filtros = sys.argv[1:]
    n = 0
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_") and (not filtros or any(s in nombre for s in filtros)):
            f()
            n += 1
            print(f"OK {nombre}")
    print(f"{n} tests OK")
