"""Tests de los datos: la muestra es la del simulador, reproducibilidad por
(seed, idx), randomizacion extra por muestra, SNR por emisor, normalizacion,
collate y pregeneracion a disco."""
import tempfile

import numpy as np
import pandas as pd
import torch

from comun import sim, cfg_datos, banco, espera_error, ConfigDatos
from config import cargar_config, a_dict
from data.simulacion import config_simulador, config_muestra, generar_muestra, snr_emisores, COLUMNAS_GT
from data.dataset import DatasetSimulado, DatasetDisco, collate, normalizar, pregenerar


def test_muestra_igual_a_generar_video():
    # sin randomizacion extra, la muestra idx es exactamente el video idx de 1 frame
    cd = cfg_datos()
    cs = config_simulador(cd, seed=5)
    for idx in range(3):
        img, gt, _ = generar_muestra(cd, cs, banco(), idx)
        video, gt_sim = sim.generar_video(cs, banco(), idx)
        assert img.shape == (1, cs.H, cs.W) and img.dtype == np.uint16
        assert np.array_equal(img, video)
        assert list(gt.columns) == COLUMNAS_GT
        pd.testing.assert_frame_equal(gt[sim.COLUMNAS_GT], gt_sim)


def test_reproducible_por_seed_e_idx():
    cd = cfg_datos(fotones_dispersion=4.0, prob_fondo_uniforme=0.5, read_noise_rango=(1.0, 5.0))
    a = generar_muestra(cd, config_simulador(cd, 1), banco(), 7)
    b = generar_muestra(cd, config_simulador(cd, 1), banco(), 7)
    assert np.array_equal(a[0], b[0]) and a[1].equals(b[1]) and a[2] == b[2]
    otro_idx = generar_muestra(cd, config_simulador(cd, 1), banco(), 8)
    otra_seed = generar_muestra(cd, config_simulador(cd, 2), banco(), 7)
    assert not np.array_equal(a[0], otro_idx[0])
    assert not np.array_equal(a[0], otra_seed[0])


def test_randomizacion_de_fotones():
    d = 4.0
    cd = cfg_datos(fotones_dispersion=d)
    cs = config_simulador(cd, 3)
    lo, hi = cs.fotones_rango
    rangos = []
    for idx in range(200):
        cm = config_muestra(cd, cs, idx)
        a, b = cm.fotones_rango
        assert lo <= a < b <= hi and b / a <= d * (1 + 1e-9)
        rangos.append((a, b))
    a, b = np.array(rangos).T
    # el centro es log-uniforme: hay imagenes en ambos extremos del rango global
    assert a.min() == lo and b.max() == hi
    assert np.mean(b / a > 0.99 * d) > 0.5                       # la mayoria con la dispersion completa
    for idx in range(20):
        _, gt, meta = generar_muestra(cd, cs, banco(), idx)
        assert gt.photons.between(meta["fotones_min"], meta["fotones_max"]).all()


def test_randomizacion_de_fondo_y_ruido_de_lectura():
    cd = cfg_datos(prob_fondo_uniforme=0.3, read_noise_rango=(1.0, 5.0))
    cs = config_simulador(cd, 4)
    cms = [config_muestra(cd, cs, idx) for idx in range(400)]
    frac = np.mean([not c.fondo_no_uniforme for c in cms])
    assert abs(frac - 0.3) < 0.08
    rn = np.array([c.read_noise for c in cms])
    assert rn.min() >= 1.0 and rn.max() <= 5.0 and rn.std() > 0.9   # U(1, 5): std 1.15
    # sin randomizacion extra la Config de la muestra es la del split
    cd0 = cfg_datos()
    assert config_muestra(cd0, config_simulador(cd0, 4), 9) == config_simulador(cd0, 4)


def test_randomizacion_no_cambia_posiciones():
    # la randomizacion tiene su propio flujo: las posiciones son las mismas con y sin ella
    cd0 = cfg_datos()
    cd1 = cfg_datos(fotones_dispersion=3.0, prob_fondo_uniforme=1.0, read_noise_rango=(0.0, 8.0))
    for idx in range(5):
        _, g0, _ = generar_muestra(cd0, config_simulador(cd0, 6), banco(), idx)
        _, g1, _ = generar_muestra(cd1, config_simulador(cd1, 6), banco(), idx)
        assert np.array_equal(g0[["x_px", "y_px", "z_nm"]].to_numpy(), g1[["x_px", "y_px", "z_nm"]].to_numpy())


def test_snr_igual_a_snr_pico_del_simulador():
    cd = cfg_datos()
    cs = config_simulador(cd, 0)
    B = banco()
    N, b = 3000.0, 17.0
    _, _, snr_ref = sim.snr_pico(B, N, b)
    iz = [0, 20, 40, 63, 80]                                    # planos del banco
    gt = pd.DataFrame({"x_px": [3.0, 10.2, 39.4, -2.0, 20.0], "y_px": [5.0, 0.0, 23.0, 12.0, 30.0],
                       "z_nm": B.z_grid[iz], "photons": N})
    snr = snr_emisores(cs, B, gt, np.full((cs.H, cs.W), b))
    assert np.allclose(snr, snr_ref[iz], rtol=1e-12)
    # el fondo se toma en el pixel del emisor (el mas cercano dentro de la imagen)
    fondo = np.zeros((cs.H, cs.W))
    fondo[5, 3] = b
    s = snr_emisores(cs, B, gt.iloc[:1], fondo)
    assert np.isclose(s[0], snr_ref[0])


def test_normalizacion():
    cd = cfg_datos(sim_kw=dict(n_emisores_rango=(0, 0), fondo_no_uniforme=False, fondo_rango=(30.0, 30.0)))
    cs = config_simulador(cd, 0)
    img, gt, _ = generar_muestra(cd, cs, banco(), 0)
    assert len(gt) == 0
    x = normalizar(img, cd, cs)
    assert x.dtype == np.float32 and x.shape == img.shape
    # solo ruido: media ~0 y desviacion ~1 (Poisson(30) + lectura, casi gaussiano)
    assert abs(x.mean()) < 0.1 and abs(x.std() - 1) < 0.1
    cdf = ConfigDatos(simulador=cd.simulador, normalizacion="fija", escala_entrada=10.0)
    assert np.allclose(normalizar(img, cdf, cs), sim.adu_a_electrones(img, cs) / 10.0, rtol=1e-6)


def test_items_y_collate():
    cd = cfg_datos(sim_kw=dict(margen_px=6.0))
    ds = DatasetSimulado(cd, 6, seed=3, banco=banco())
    items = [ds[i] for i in range(6)]
    for i, it in enumerate(items):
        _, gt, _ = ds.muestra(i)
        assert it["idx"] == i and it["imagen"].shape == (1, 24, 40) and it["imagen"].dtype == torch.float32
        esperado = gt.loc[gt.en_fov, ["x_px", "y_px", "z_nm"]].to_numpy(np.float32)
        assert np.array_equal(it["pos"].numpy(), esperado)        # solo los emisores dentro del FOV
    n_fuera = sum((~ds.muestra(i)[1].en_fov).sum() for i in range(6))
    assert n_fuera > 0                                          # el caso se ejercita
    lote = collate(items)
    E = max(len(it["pos"]) for it in items)
    assert lote["imagen"].shape == (6, 1, 24, 40) and lote["pos"].shape == (6, E, 3)
    assert lote["valido"].sum(1).tolist() == [len(it["pos"]) for it in items]
    assert (lote["pos"][~lote["valido"]] == 0).all()
    assert lote["idx"].tolist() == list(range(6))


def test_collate_sin_emisores():
    cd = cfg_datos(sim_kw=dict(n_emisores_rango=(0, 0)))
    ds = DatasetSimulado(cd, 3, seed=0, banco=banco())
    lote = collate([ds[i] for i in range(3)])
    assert lote["pos"].shape == (3, 0, 3) and lote["valido"].shape == (3, 0)


def test_pregenerar_igual_a_simulado():
    cd = cfg_datos(fotones_dispersion=3.0, prob_fondo_uniforme=0.5,
                   sim_kw=dict(n_emisores_rango=(0, 4)))
    with tempfile.TemporaryDirectory() as tmp:
        gt = pregenerar(cd, 8, 11, tmp, banco=banco())
        ds_d = DatasetDisco(tmp)
        ds_s = DatasetSimulado(cd, 8, 11, banco=banco())
        assert len(ds_d) == 8 and a_dict(ds_d.cfg_datos) == a_dict(cd)
        assert sorted(gt.video_id.unique()) == sorted(set(gt.video_id))   # video_id = idx de la muestra
        for i in range(8):
            a, b = ds_d[i], ds_s[i]
            assert torch.equal(a["imagen"], b["imagen"]) and torch.equal(a["pos"], b["pos"])
        assert len(ds_d.muestras) == 8 and (ds_d.muestras.video_id == range(8)).all()


def test_renovar_por_epoca():
    cd = cfg_datos(renovar_por_epoca=True)
    ds = DatasetSimulado(cd, 4, seed=0, banco=banco())
    a = ds[1]
    ds.fijar_epoca(2)
    b = ds[1]
    assert b["idx"] == 9 and not torch.equal(a["imagen"], b["imagen"])
    fija = DatasetSimulado(cfg_datos(), 4, seed=0, banco=banco())
    fija.fijar_epoca(2)
    assert fija[1]["idx"] == 1


def test_errores_de_configuracion():
    espera_error(NotImplementedError, generar_muestra, cfg_datos(canales=2),
                 config_simulador(cfg_datos(), 0), banco(), 0)
    espera_error(ValueError, config_simulador, cfg_datos(sim_kw=dict(n_frames=5)), 0)
    espera_error(TypeError, config_simulador, cfg_datos(sim_kw=dict(campo_que_no_existe=1)), 0)
    espera_error(ValueError, cfg_datos, fotones_dispersion=0.5)
    espera_error(ValueError, cfg_datos, normalizacion="otra")
    # un banco de otra optica no se acepta
    espera_error(ValueError, DatasetSimulado, cfg_datos("esferica"), 2, 0, banco("ninguna"))


def test_yaml_y_overrides():
    import os
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = cargar_config(os.path.join(raiz, "configs", "base.yaml"),
                        ["datos.simulador.H=32", "target.dz_nm=100", "datos.read_noise_rango=[1, 2]"])
    assert cfg.datos.simulador["H"] == 32 and cfg.datos.simulador["W"] == 64
    assert cfg.target.dz_nm == 100.0 and cfg.datos.read_noise_rango == (1.0, 2.0)
    espera_error(ValueError, cargar_config, None, ["datos.campo_inexistente=1"])
    espera_error(ValueError, cargar_config, None, ["seccion_inexistente.x=1"])
    espera_error(ValueError, cargar_config, None, ["sin_igual"])
