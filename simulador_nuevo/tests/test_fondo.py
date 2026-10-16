"""Tests del fondo de fotones (fijo por video)."""
import numpy as np

from comun import sim


def test_fondo_uniforme():
    cfg = sim.Config(H=20, W=30, fondo_no_uniforme=False)
    f = sim.generar_fondo(cfg, np.random.default_rng(0))
    assert f.shape == (20, 30)
    assert np.all(f == f[0, 0])
    assert cfg.fondo_rango[0] <= f[0, 0] <= cfg.fondo_rango[1]


def test_fondo_no_uniforme_acotado_y_suave():
    # base + mancha gaussiana: base en fondo_rango y pico en fondo_rango
    cfg = sim.Config(H=48, W=64, fondo_no_uniforme=True)
    rng = np.random.default_rng(1)
    for _ in range(50):
        f = sim.generar_fondo(cfg, rng)
        assert f.shape == (48, 64)
        assert f.min() >= cfg.fondo_rango[0]
        assert f.max() <= 2 * cfg.fondo_rango[1]
        # suave: el ancho de la mancha es >= 0.3 del campo, sin saltos entre pixeles
        assert np.abs(np.diff(f, axis=0)).max() < 0.2 * f.max()
        assert np.abs(np.diff(f, axis=1)).max() < 0.2 * f.max()


def test_fondo_cambia_entre_videos():
    b = sim.BancoPSF(sim.Config(mascara="ninguna", z_rango_nm=(-20.0, 20.0)))
    cfg = sim.Config(H=16, W=16, mascara="ninguna", z_rango_nm=(-20.0, 20.0),
                     n_emisores_rango=(0, 0), n_frames=1)
    _, _, ex0 = sim.generar_video(cfg, b, 0, extra=True)
    _, _, ex1 = sim.generar_video(cfg, b, 1, extra=True)
    assert not np.allclose(ex0["fondo"], ex1["fondo"])
