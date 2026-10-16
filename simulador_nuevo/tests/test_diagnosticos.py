"""Tests de las funciones de diagnostico: SNR del pixel pico y cota de
Cramer-Rao (CRLB) de localizacion."""
import numpy as np

from comun import sim, banco


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
