"""Tests de extremo a extremo: forma del video, contenido del ground truth,
consistencia imagen <-> ground truth, estadistica del ruido, reproducibilidad
(semilla, independencia entre videos y entre componentes) y escritura a disco."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import stats

from comun import sim, banco, espera_error


def cfg_pequena(**kw):
    # H != W a proposito: un intercambio de ejes x/y no pasa desapercibido
    base = dict(mascara="ninguna", H=32, W=40, n_frames=8, n_videos=3,
                n_emisores_rango=(3, 6), seed=11)
    base.update(kw)
    return sim.Config(**base)


B = lambda: banco("ninguna")


def test_forma_y_tipo_del_video():
    cfg = cfg_pequena()
    video, gt = sim.generar_video(cfg, B(), 0)
    assert video.shape == (cfg.n_frames, cfg.H, cfg.W) and video.dtype == np.uint16
    assert isinstance(gt, pd.DataFrame)


def test_ground_truth_columnas_y_filas():
    cfg = cfg_pequena()
    for v in range(3):
        _, gt = sim.generar_video(cfg, B(), v)
        assert list(gt.columns) == sim.COLUMNAS_GT
        n = gt.particle_id.nunique()
        assert cfg.n_emisores_rango[0] <= n <= cfg.n_emisores_rango[1]
        assert len(gt) == cfg.n_frames * n
        assert not gt.duplicated(["video_id", "frame", "particle_id"]).any()
        assert sorted(gt.particle_id.unique()) == list(range(n))
        assert sorted(gt.frame.unique()) == list(range(cfg.n_frames))
        assert (gt.video_id == v).all()
        # sin fotofisica: fotones y D constantes por particula durante el video
        assert (gt.groupby("particle_id")[["photons", "D_um2_s"]].nunique() == 1).all().all()


def test_en_fov_en_los_bordes():
    # FOV = [-0.5, W - 0.5) x [-0.5, H - 0.5): los centros de pixel estan en enteros,
    # asi que el pixel 0 cubre [-0.5, 0.5) y el ultimo [W - 1.5, W - 0.5)
    cfg = cfg_pequena()
    W, H = cfg.W, cfg.H
    x = np.array([-0.5, -0.51, -0.2, W - 0.51, W - 0.5, W - 0.2, 10.0, 10.0, 10.0, 10.0])
    y = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -0.5, -0.51, H - 0.51, H - 0.5])
    esperado = [True, False, True, True, False, False, True, False, True, False]
    assert list(sim.en_fov(cfg, x, y)) == esperado


def test_en_fov_consistente_con_las_coordenadas():
    cfg = cfg_pequena(margen_px=6.0, n_frames=20, D_rango_um2_s=(0.2, 0.3))
    _, gt = sim.generar_video(cfg, B(), 0)
    assert (gt.en_fov == sim.en_fov(cfg, gt.x_px.values, gt.y_px.values)).all()
    assert gt.en_fov.dtype == bool and gt.en_fov.any() and not gt.en_fov.all()


def test_rangos_del_ground_truth():
    cfg = cfg_pequena(n_frames=20)
    _, gt = sim.generar_video(cfg, B(), 1)
    assert gt.z_nm.between(*cfg.z_rango_nm).all()
    assert gt.photons.between(*cfg.fotones_rango).all()
    assert gt.D_um2_s.between(*cfg.D_rango_um2_s).all()


def test_muestreo_de_fotones():
    rng = np.random.default_rng(0)
    cfg = sim.Config()
    lo, hi = cfg.fotones_rango
    N = sim.muestrear_fotones(cfg, 20_000, rng)                # log-uniforme por defecto
    assert N.min() >= lo and N.max() <= hi
    assert stats.kstest(np.log(N), "uniform", args=(np.log(lo), np.log(hi / lo))).pvalue > 1e-3
    N = sim.muestrear_fotones(sim.Config(muestreo_fotones="uniforme"), 20_000, rng)
    assert stats.kstest(N, "uniform", args=(lo, hi - lo)).pvalue > 1e-3


def test_imagen_limpia_reconstruible_desde_el_gt():
    # con las filas del GT y el fondo se reconstruye exactamente la imagen sin ruido
    cfg = cfg_pequena(margen_px=4.0)
    _, gt, ex = sim.generar_video(cfg, B(), 2, extra=True)
    for t in range(cfg.n_frames):
        img = np.zeros((cfg.H, cfg.W))
        for r in gt[gt.frame == t].itertuples():
            B().colocar(img, r.x_px, r.y_px, r.z_nm, r.photons)
        assert np.allclose(img + ex["fondo"], ex["limpio"][t], rtol=1e-10, atol=1e-10), t


def test_ruido_consistente_con_la_imagen_limpia():
    # residuo normalizado ((ADU - offset) * g - limpio) / sigma ~ media 0, varianza 1
    cfg = cfg_pequena(n_frames=30)
    video, _, ex = sim.generar_video(cfg, B(), 0, extra=True)
    mu = ex["limpio"]
    sigma = np.sqrt(mu + (cfg.read_noise**2 + 1.0 / 12.0) * cfg.gain**2)
    r = ((sim.adu_a_electrones(video, cfg) - mu) / sigma).ravel()
    n = r.size
    assert abs(r.mean()) < 5 / np.sqrt(n), r.mean()
    assert abs(r.var() - 1) < 5 * np.sqrt(2.0 / n), r.var()


def test_reproducibilidad_con_la_semilla():
    cfg = cfg_pequena()
    v1, g1 = sim.generar_video(cfg, B(), 1)
    v2, g2 = sim.generar_video(cfg, B(), 1)
    v3, g3 = sim.generar_video(cfg_pequena(seed=12), B(), 1)
    assert np.array_equal(v1, v2) and g1.equals(g2)
    assert not np.array_equal(v1, v3)


def test_ruido_y_contraste_no_alteran_las_trayectorias():
    # cada componente usa su propio flujo aleatorio: cambiar camara, fondo o
    # fotones no cambia las trayectorias (sirve para barridos de SNR)
    a = cfg_pequena()
    b = cfg_pequena(read_noise=10.0, gain=1.0, offset=300.0, fondo_rango=(50.0, 100.0),
                    fotones_rango=(100.0, 200.0), fondo_no_uniforme=False)
    _, ga = sim.generar_video(a, B(), 0)
    _, gb = sim.generar_video(b, B(), 0)
    cols = ["frame", "particle_id", "x_px", "y_px", "z_nm", "D_um2_s"]
    assert ga[cols].equals(gb[cols])
    assert not np.allclose(ga.photons, gb.photons)


def test_componentes_sin_correlacion():
    # D, fotones y posicion inicial se sortean de forma independiente: si dos
    # componentes compartieran flujo aleatorio, por ejemplo, las particulas
    # brillantes serian siempre las rapidas
    cfg = cfg_pequena(H=16, W=16, n_frames=1, n_emisores_rango=(5, 5))
    gt = pd.concat([sim.generar_video(cfg, B(), v)[1] for v in range(200)])
    cols = ["x_px", "y_px", "z_nm", "photons", "D_um2_s"]
    rho = stats.spearmanr(gt[cols]).statistic
    fuera_diag = np.abs(rho[~np.eye(len(cols), dtype=bool)])
    assert fuera_diag.max() < 5 / np.sqrt(len(gt)), np.round(rho, 3)


def test_prefijo_temporal():
    # alargar el video no cambia los primeros frames (ni imagen ni GT)
    v10, g10 = sim.generar_video(cfg_pequena(n_frames=10), B(), 0)
    v20, g20 = sim.generar_video(cfg_pequena(n_frames=20), B(), 0)
    assert np.array_equal(v10, v20[:10])
    assert g10.reset_index(drop=True).equals(g20[g20.frame < 10].reset_index(drop=True))


def test_desenfoque_de_movimiento_sin_difusion_es_neutro():
    a = cfg_pequena(D_rango_um2_s=(0.0, 0.0))
    b = cfg_pequena(D_rango_um2_s=(0.0, 0.0), n_subpasos=4)
    _, ga, ea = sim.generar_video(a, B(), 0, extra=True)
    _, gb, eb = sim.generar_video(b, B(), 0, extra=True)
    assert np.allclose(ea["limpio"], eb["limpio"], rtol=1e-10)
    assert np.allclose(ga[["x_px", "y_px", "z_nm"]], gb[["x_px", "y_px", "z_nm"]])


def test_desenfoque_de_movimiento_conserva_fotones():
    # la senal de cada frame suma N * (suma de la PSF recortada), promediada
    # entre las subposiciones de la exposicion
    cfg = cfg_pequena(H=200, W=200, n_emisores_rango=(1, 1), margen_px=-60.0,
                      n_subpasos=6, D_rango_um2_s=(0.3, 0.3))
    _, gt, ex = sim.generar_video(cfg, B(), 0, extra=True)
    N = gt.photons.iloc[0]
    for t in range(cfg.n_frames):
        esperado = N * np.mean([B().psf_en_z(z).sum() for z in ex["sub"][0, t, :, 2]])
        assert np.isclose((ex["limpio"][t] - ex["fondo"]).sum(), esperado, rtol=1e-9), t


def test_banco_incompatible():
    cfg = cfg_pequena(z_rango_nm=(-1000.0, 1000.0))          # el banco solo cubre +-800 nm
    espera_error(ValueError, sim.generar_video, cfg, B(), 0)


def test_escritura_del_dataset():
    cfg = cfg_pequena()
    with tempfile.TemporaryDirectory() as d:
        gt = sim.generar_dataset(cfg, d, banco=B())
        d = Path(d)
        assert sorted(p.name for p in d.glob("*.tif")) == [f"video_{v:03d}.tif" for v in range(cfg.n_videos)]
        # cada video se puede regenerar solo, sin generar los anteriores
        video2, gt2 = sim.generar_video(cfg, B(), 2)
        with tifffile.TiffFile(d / "video_002.tif") as tif:
            leido = tif.asarray()
            assert np.isclose(tif.imagej_metadata["finterval"], cfg.dt_s)
        assert leido.dtype == np.uint16 and np.array_equal(leido, video2)
        assert gt[gt.video_id == 2].reset_index(drop=True).equals(gt2)
        # el CSV guardado es el mismo DataFrame
        csv = pd.read_csv(d / "ground_truth.csv")
        assert list(csv.columns) == sim.COLUMNAS_GT and len(csv) == len(gt)
        assert np.allclose(csv[["x_px", "y_px", "z_nm", "photons"]], gt[["x_px", "y_px", "z_nm", "photons"]])
        # la configuracion queda guardada y se puede recargar
        assert sim.cargar_config(d / "config.json") == cfg
        meta = json.loads((d / "config.json").read_text())
        assert meta["K_psf_usado"] == B().K
