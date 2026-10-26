"""Tests de la rejilla de voxeles y del target gaussiano: convencion de centros
(la del simulador), ida y vuelta pixel <-> voxel, ejes x/y, posicion continua
recuperable, linealidad, relleno del batch y CPU = GPU."""
import numpy as np
import torch

from comun import ConfigTarget
from data.target import RejillaVoxeles

H, W = 24, 40                                                   # H != W a proposito
Z_RANGO = (-800.0, 800.0)


def rejilla(**kw):
    return RejillaVoxeles(ConfigTarget(**kw), H, W, Z_RANGO)


def test_numero_de_planos():
    assert rejilla(dz_nm=50.0, planos_margen=2).Nz == 33 + 4
    assert rejilla(dz_nm=100.0, planos_margen=0).Nz == 17
    r = rejilla(dz_nm=300.0, planos_margen=1)                   # 1600 / 300 no es entero: cubre hasta z_max
    assert r.z_planos[r.m] == -800.0 and r.z_planos[-1 - r.m] >= 800.0
    assert rejilla(upsampling=4).forma == (37, 4 * H, 4 * W)


def test_convencion_de_centros():
    for s in (1, 2, 3, 4):
        r = rejilla(upsampling=s)
        # el centro del pixel j (x = j) cae en el centro del bloque de s voxeles del pixel
        v, u, _ = r.a_voxel(np.array([0.0, 7.0]), np.array([0.0, 5.0]), 0.0)
        assert np.allclose(v, [(s - 1) / 2, 7 * s + (s - 1) / 2])
        assert np.allclose(u, [(s - 1) / 2, 5 * s + (s - 1) / 2])
        # los bordes del FOV son los bordes de la rejilla
        v, u, _ = r.a_voxel(np.array([-0.5, W - 0.5]), np.array([-0.5, H - 0.5]), 0.0)
        assert np.allclose(v, [-0.5, s * W - 0.5]) and np.allclose(u, [-0.5, s * H - 0.5])
    r = rejilla(dz_nm=50.0, planos_margen=2)
    assert np.allclose(r.a_voxel(0.0, 0.0, np.array(Z_RANGO))[2], [2, r.Nz - 3])
    assert np.allclose(r.z_planos[[2, -3]], Z_RANGO)


def test_ida_y_vuelta_pixel_voxel():
    rng = np.random.default_rng(0)
    x, y, z = rng.uniform(-0.5, W - 0.5, 100), rng.uniform(-0.5, H - 0.5, 100), rng.uniform(*Z_RANGO, 100)
    for s in (1, 2, 4):
        r = rejilla(upsampling=s, dz_nm=37.0)
        xx, yy, zz = r.desde_voxel(*r.a_voxel(x, y, z))
        assert np.allclose(xx, x) and np.allclose(yy, y) and np.allclose(zz, z)


def test_pico_en_el_voxel_correcto_sin_cambiar_ejes():
    r = rejilla()
    # emisor justo en el centro del voxel (k, u, v) = (10, 7, 130)
    x, y, z = r.desde_voxel(130.0, 7.0, 10.0)
    t = r.target(torch.tensor([[[x, y, z]]]), torch.tensor([[True]]))
    assert t.shape == (1, r.Nz, r.Hs, r.Ws)
    k, u, v = np.unravel_index(int(t[0].argmax()), t.shape[1:])
    assert (k, u, v) == (10, 7, 130)                            # columna = x, fila = y
    assert np.isclose(float(t.max()), r.escala, rtol=1e-6)


def _cdm(vol, k, u, v, w=4):
    """Centro de masa en una ventana (2w + 1)^3 alrededor de (k, u, v)."""
    sl = tuple(slice(c - w, c + w + 1) for c in (k, u, v))
    sub = vol[sl]
    ejes = np.meshgrid(*[np.arange(c - w, c + w + 1) for c in (k, u, v)], indexing="ij")
    return [float((sub * e).sum() / sub.sum()) for e in ejes]


def test_posicion_continua_recuperable():
    # el target esta centrado en la posicion continua: su centro de masa la devuelve
    # (margen de 4 planos para que la ventana de +-4 voxeles quepa en los extremos de z)
    r = rejilla(planos_margen=4)
    rng = np.random.default_rng(1)
    n = 6
    x = np.linspace(4, W - 5, n) + rng.uniform(-0.5, 0.5, n)    # separados y lejos del borde lateral
    y = rng.uniform(2.0, H - 3.0, n)
    z = np.concatenate([[-800.0, 800.0], rng.uniform(*Z_RANGO, n - 2)])   # incluye los extremos de z
    pos = torch.tensor(np.stack([x, y, z], axis=1)[None], dtype=torch.float64)
    t = r.target(pos, torch.ones(1, n, dtype=torch.bool))[0].numpy()
    for i in range(n):
        kk, uu, vv = np.round(r.a_voxel(x[i], y[i], z[i])[::-1]).astype(int)
        k, u, v = _cdm(t, kk, uu, vv)
        xe, ye, ze = r.desde_voxel(v, u, k)
        assert abs(xe - x[i]) < 1e-4 and abs(ye - y[i]) < 1e-4 and abs(ze - z[i]) < 0.01


def test_lineal_y_relleno_ignorado():
    r = rejilla(upsampling=2)
    a = [5.3, 7.9, -120.0]
    b = [30.1, 15.2, 410.0]
    basura = [1e3, -1e3, 5e4]
    v = lambda *p: torch.tensor([p])
    ta = r.target(v(a), torch.tensor([[True]]))
    tb = r.target(v(b), torch.tensor([[True]]))
    tab = r.target(v(a, b, basura), torch.tensor([[True, True, False]]))
    assert torch.allclose(tab, ta + tb, atol=1e-5)
    vacio = r.target(torch.zeros(2, 0, 3), torch.zeros(2, 0, dtype=torch.bool))
    assert vacio.shape == (2, *r.forma) and (vacio == 0).all()


def test_cpu_igual_gpu():
    if not torch.cuda.is_available():
        print("    (sin GPU: se omite)")
        return
    r = rejilla()
    pos = torch.tensor([[[5.3, 7.9, -120.0], [30.1, 15.2, 410.0]]])
    val = torch.ones(1, 2, dtype=torch.bool)
    assert torch.allclose(r.target(pos, val), r.target(pos.cuda(), val.cuda()).cpu(), atol=1e-4)
