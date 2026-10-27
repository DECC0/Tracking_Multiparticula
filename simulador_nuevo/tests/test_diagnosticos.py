"""Tests de las funciones de diagnostico: SNR del pixel pico y cota de
Cramer-Rao (CRLB) de localizacion, con z conocido (crlb) o desconocido (crlb_3d)."""
import numpy as np

from comun import sim, banco, espera_error


def test_snr_pico_vs_simulacion():
    # emisor centrado en un pixel + fondo uniforme, muchas realizaciones del ruido:
    # la media y la desviacion del pixel pico (en e-) deben dar el SNR de la formula
    b = banco("esferica")
    cfg = b.cfg
    N, fondo, z = 3000.0, 20.0, 0.0
    p_max, senal, snr = sim.snr_pico(b, N, fondo)
    i = int(np.argmin(np.abs(b.z_grid - z)))

    L = 15
    img = np.full((L, L), fondo)
    b.colocar(img, L // 2, L // 2, z, N)
    assert np.isclose(img.max() - fondo, senal[i])

    rng = np.random.default_rng(0)
    n = 20_000
    e = sim.adu_a_electrones(sim.aplicar_camara(np.broadcast_to(img, (n, L, L)), cfg, rng), cfg)
    pico = e[:, L // 2, L // 2]
    var_teo = senal[i] + fondo + (cfg.read_noise**2 + 1.0 / 12.0) * cfg.gain**2
    assert abs(pico.mean() - (senal[i] + fondo)) < 5 * np.sqrt(var_teo / n)
    assert abs(pico.var() / var_teo - 1) < 5 * np.sqrt(2.0 / n)
    assert np.isclose(snr[i], senal[i] / np.sqrt(var_teo))


def test_crlb_escala_con_fotones():
    # solo Poisson y sin fondo, la informacion de Fisher es proporcional a N
    b = banco("ninguna")
    s1 = sim.crlb(b, 1000.0, 0.0, var_lectura_e2=0.0)
    s4 = sim.crlb(b, 4000.0, 0.0, var_lectura_e2=0.0)
    assert np.isclose(s4["x"] / s1["x"], 0.5, rtol=1e-6)
    assert np.isclose(s4["y"] / s1["y"], 0.5, rtol=1e-6)
    # simetria x/y para una PSF radial centrada
    assert np.isclose(s1["x"], s1["y"], rtol=1e-6)
    # el fondo y el ruido de lectura solo pueden empeorar la precision
    assert sim.crlb(b, 1000.0, 20.0)["x"] > s1["x"]
    # orden de magnitud: para una PSF ~gaussiana de ancho sigma, CRLB ~ sigma / sqrt(N)
    sigma = np.sqrt((b.cfg.sigma0_nm / b.cfg.pixel_nm)**2 + b.cfg.blur_extra_px**2 + 1.0 / 12.0)
    assert 0.7 < s1["x"] / (sigma / np.sqrt(1000.0)) < 2.0, s1["x"]


def _crlb_3d_directa(cfg, N, fondo, z, L=15):
    # referencia independiente del banco: CRLB de (x, y, z, N, b) con la PSF exacta de psf_directa
    # (sin interpolar en z) recortada a una ROI L x L alrededor del emisor
    c = cfg.M_pupila // 2
    s = slice(c - L // 2, c - L // 2 + L)
    p = lambda dx, dy, zz: N * sim.psf_directa(cfg, zz, dx, dy)[s, s] + fondo
    mu, h, hz = p(0, 0, z), 1e-3, 0.5
    J = np.stack([(p(h, 0, z) - p(-h, 0, z)) / (2 * h), (p(0, h, z) - p(0, -h, z)) / (2 * h),
                  (p(0, 0, z + hz) - p(0, 0, z - hz)) / (2 * hz), (mu - fondo) / N,
                  np.ones_like(mu)]).reshape(5, -1)
    fisher = (J / (mu.ravel() + sim._var_lectura_e2(cfg))) @ J.T
    return dict(zip(("x", "y", "z"), np.sqrt(np.diag(np.linalg.inv(fisher)))[:3]))


def test_crlb_3d_vs_psf_directa():
    # la derivada en z sale del banco (interpolacion lineal entre planos cada 20 nm); debe dar la
    # misma cota que la PSF exacta, tambien en los extremos del rango (derivada de un solo lado)
    b = banco("esferica")
    for z in (-800.0, -410.0, -7.0, 0.0, 300.0, 610.0, 800.0):
        s = sim.crlb_3d(b, 3000.0, 20.0, z_nm=z)
        ref = _crlb_3d_directa(b.cfg, 3000.0, 20.0, z)
        for k in ("x", "y", "z"):
            assert np.isclose(s[k], ref[k], rtol=5e-3), (z, k, s[k], ref[k])


def test_crlb_3d_vs_2d_y_escala_con_fotones():
    # esferica (radial) con el emisor centrado en un pixel: dmu/dx es impar y dmu/dz par, asi que
    # z no se correlaciona con x, y y la cota lateral es la misma que con z conocido
    b = banco("esferica")
    s3, s2 = sim.crlb_3d(b, 3000.0, 20.0, z_nm=250.0), sim.crlb(b, 3000.0, 20.0, z_nm=250.0)
    assert np.isclose(s3["x"], s2["x"], rtol=1e-6) and np.isclose(s3["y"], s2["y"], rtol=1e-6)
    # fuera del centro, o con una PSF que rota con z (corkscrew), estimar z solo puede empeorarla
    s3, s2 = sim.crlb_3d(b, 3000.0, 20.0, z_nm=250.0, dx=0.3, dy=-0.2), sim.crlb(b, 3000.0, 20.0, z_nm=250.0, dx=0.3, dy=-0.2)
    assert s3["x"] >= s2["x"] * (1 - 1e-9) and s3["y"] >= s2["y"] * (1 - 1e-9)
    bc = banco("corkscrew")
    assert sim.crlb_3d(bc, 3000.0, 20.0, z_nm=250.0)["x"] > 1.01 * sim.crlb(bc, 3000.0, 20.0, z_nm=250.0)["x"]
    # solo Poisson y sin fondo, la informacion de Fisher es proporcional a N: x, y, z escalan como 1/sqrt(N)
    for bb in (b, bc):
        s1 = sim.crlb_3d(bb, 1000.0, 0.0, z_nm=250.0, var_lectura_e2=0.0)
        s4 = sim.crlb_3d(bb, 4000.0, 0.0, z_nm=250.0, var_lectura_e2=0.0)
        for k in ("x", "y", "z"):
            assert np.isclose(s4[k] / s1[k], 0.5, rtol=1e-6), (bb.cfg.mascara, k)
    # el fondo solo puede empeorar la cota en z
    assert sim.crlb_3d(b, 1000.0, 20.0, z_nm=250.0)["z"] > sim.crlb_3d(b, 1000.0, 0.0, z_nm=250.0)["z"]
    espera_error(ValueError, sim.crlb_3d, banco("esferica", z_rango_nm=(0.0, 0.0)), 1000.0, 20.0)
