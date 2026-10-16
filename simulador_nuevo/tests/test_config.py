"""Tests de Config: unidades derivadas, validacion de parametros y serializacion."""
import json

import numpy as np

from comun import sim, espera_error


def test_pixel_en_muestra():
    # pixel de camara 6.5 um / magnificacion 66 -> ~98.48 nm en la muestra
    cfg = sim.Config()
    assert np.isclose(cfg.pixel_nm, 6500.0 / 66.0)


def test_muestreo_nyquist_por_defecto():
    # la PSF de intensidad tiene frecuencia de corte 2 NA / lambda; para no
    # tener aliasing el pixel debe ser menor que lambda / (4 NA)
    cfg = sim.Config()
    assert np.isclose(cfg.nyquist_nm, cfg.lambda_nm / (4 * cfg.NA))
    assert cfg.pixel_nm < cfg.nyquist_nm


def test_tiempo_de_exposicion():
    assert sim.Config().t_exp == sim.Config().dt_s      # None -> exposicion = intervalo entre frames
    assert sim.Config(t_exp_s=0.03).t_exp == 0.03


def test_config_rechaza_valores_invalidos():
    malos = [
        dict(mascara="astigmatica"),
        dict(muestreo_fotones="normal"),
        dict(NA=1.4),                        # NA > n del medio (agua): imposible
        dict(NA=0.0),
        dict(magnificacion=20.0),            # pixel de 325 nm: la pupila no cabe en la rejilla FFT
        dict(gain=0.0),
        dict(read_noise=-1.0),
        dict(offset=-5.0),
        dict(dt_s=0.0),
        dict(t_exp_s=0.2),                   # exposicion mas larga que el intervalo entre frames
        dict(n_subpasos=0),
        dict(K_psf=47),                      # impar: la PSF no queda centrada en K // 2
        dict(K_psf=512),                     # mas grande que la rejilla de pupila
        dict(M_pupila=255),
        dict(energia_max_recortada=0.0),
        dict(z_paso_nm=0.0),
        dict(z_rango_nm=(800.0, -800.0)),
        dict(fotones_rango=(0.0, 100.0)),    # log(0) en el muestreo log-uniforme
        dict(fondo_rango=(-1.0, 5.0)),
        dict(D_rango_um2_s=(0.3, 0.1)),
        dict(n_emisores_rango=(5, 2)),
        dict(H=0),
        dict(n_frames=0),
        dict(blur_extra_px=-0.1),
    ]
    for kw in malos:
        espera_error(ValueError, sim.Config, **kw)


def test_config_valida_en_los_bordes():
    # valores limite pero validos no deben lanzar
    sim.Config(fotones_rango=(0.0, 100.0), muestreo_fotones="uniforme")
    sim.Config(D_rango_um2_s=(0.0, 0.0), n_emisores_rango=(0, 0), read_noise=0.0)
    sim.Config(t_exp_s=0.1, dt_s=0.1, n_subpasos=10)


def test_config_json_ida_y_vuelta():
    cfg = sim.Config(mascara="corkscrew", H=40, W=72, fotones_rango=(500.0, 2000.0),
                     n_emisores_rango=(2, 4), t_exp_s=0.05, n_subpasos=3, seed=123)
    texto = json.dumps(cfg.a_dict())
    assert sim.Config.desde_dict(json.loads(texto)) == cfg
