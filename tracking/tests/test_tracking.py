"""
Tests del enlace y de las metricas de tracking (assert, sin pytest).

    PY=~/miniforge3/envs/tesis/bin/python
    $PY tracking/tests/test_tracking.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enlace import enlazar, asignacion_lap  # noqa: E402
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


if __name__ == "__main__":
    filtros = sys.argv[1:]
    n = 0
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_") and (not filtros or any(s in nombre for s in filtros)):
            f()
            n += 1
            print(f"OK {nombre}")
    print(f"{n} tests OK")
