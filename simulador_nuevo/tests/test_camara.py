"""Tests del modelo de camara sCMOS: Poisson -> /gain -> ruido de lectura ->
offset -> redondeo -> uint16. Momentos contra la teoria, curva de
transferencia de fotones, saturacion, conversion inversa y reproducibilidad."""
import numpy as np

from comun import sim


def test_media_y_varianza_en_adu():
    #   media    = F / g + offset
    #   varianza = F / g^2 + sigma_r^2 + 1/12   (Poisson + lectura + cuantizacion)
    cfg = sim.Config()
    rng = np.random.default_rng(1)
    n = 200_000
    for F in (0.0, 5.0, 50.0, 500.0, 5000.0):
        adu = sim.aplicar_camara(np.full(n, F), cfg, rng).astype(float)
        media_teo = F / cfg.gain + cfg.offset
        var_teo = F / cfg.gain**2 + cfg.read_noise**2 + 1.0 / 12.0
        assert abs(adu.mean() - media_teo) < 5 * np.sqrt(var_teo / n), (F, adu.mean(), media_teo)
        assert abs(adu.var() - var_teo) < 5 * var_teo * np.sqrt(2.0 / n), (F, adu.var(), var_teo)


def test_curva_de_transferencia_de_fotones():
    # calibracion clasica de camaras: var(ADU) = (media - offset) / g + sigma_r^2,
    # la pendiente da la ganancia y los frames oscuros el ruido de lectura
    cfg = sim.Config()
    rng = np.random.default_rng(2)
    medias, varianzas = [], []
    for F in np.geomspace(10, 20000, 8):
        adu = sim.aplicar_camara(np.full(50_000, F), cfg, rng).astype(float)
        medias.append(adu.mean())
        varianzas.append(adu.var())
    pendiente = np.polyfit(np.array(medias) - cfg.offset, varianzas, 1)[0]
    assert abs(1.0 / pendiente - cfg.gain) < 0.02 * cfg.gain, (1.0 / pendiente, cfg.gain)

    oscuro = sim.aplicar_camara(np.zeros(200_000), cfg, rng).astype(float)
    rn_est = np.sqrt(oscuro.var() - 1.0 / 12.0)
    assert abs(rn_est - cfg.read_noise) < 0.02 * cfg.read_noise, (rn_est, cfg.read_noise)


def test_tipo_saturacion_y_limites():
    cfg = sim.Config()
    rng = np.random.default_rng(3)
    img = sim.aplicar_camara(np.full((16, 16), 300.0), cfg, rng)
    assert img.dtype == np.uint16 and img.shape == (16, 16)
    # 1e6 fotones -> 2e6 ADU: satura en el maximo de uint16
    assert np.all(sim.aplicar_camara(np.full(1000, 1e6), cfg, rng) == 65535)
    # el offset evita recortar en 0 los frames oscuros
    assert sim.aplicar_camara(np.zeros(100_000), cfg, rng).min() > 0
    # entradas negativas (no fisicas) se tratan como 0 sin fallar
    neg = sim.aplicar_camara(np.full(10_000, -5.0), cfg, rng).astype(float)
    assert abs(neg.mean() - cfg.offset) < 0.5


def test_conversion_adu_a_electrones():
    cfg = sim.Config()
    rng = np.random.default_rng(4)
    n, F = 100_000, 250.0
    e = sim.adu_a_electrones(sim.aplicar_camara(np.full(n, F), cfg, rng), cfg)
    var_e = F + (cfg.read_noise**2 + 1.0 / 12.0) * cfg.gain**2
    assert abs(e.mean() - F) < 5 * np.sqrt(var_e / n), (e.mean(), F)


def test_reproducibilidad_de_la_camara():
    cfg = sim.Config()
    F = np.full((20, 20), 300.0)
    a = sim.aplicar_camara(F, cfg, np.random.default_rng(7))
    b = sim.aplicar_camara(F, cfg, np.random.default_rng(7))
    c = sim.aplicar_camara(F, cfg, np.random.default_rng(8))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
