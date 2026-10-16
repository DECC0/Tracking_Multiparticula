"""Tests del movimiento browniano 3D: paredes reflectantes en z, MSD = 2 D t
por eje (unidades), incrementos gaussianos independientes, distribucion
estacionaria en z y desenfoque de movimiento durante la exposicion."""
import numpy as np
from scipy import stats

from comun import sim


def test_reflejar():
    lo, hi = -800.0, 800.0
    L = hi - lo
    z = np.linspace(-700, 700, 11)
    assert np.allclose(sim.reflejar(z, lo, hi), z)                  # adentro: identidad
    assert np.isclose(sim.reflejar(hi + 50, lo, hi), hi - 50)        # espejo en la pared de arriba
    assert np.isclose(sim.reflejar(lo - 50, lo, hi), lo + 50)        # espejo en la pared de abajo
    assert np.isclose(sim.reflejar(lo + 2 * L + 30, lo, hi), lo + 30)  # dos cruces
    w = sim.reflejar(np.random.default_rng(0).normal(0, 1e5, 10_000), lo, hi)
    assert w.min() >= lo and w.max() <= hi


def test_msd_lineal_en_los_tres_ejes():
    # MSD por eje = 2 D t; x, y salen en px y z en nm (unidades)
    D = 0.2
    cfg = sim.Config(n_frames=11, D_rango_um2_s=(D, D), z_rango_nm=(-1e6, 1e6))
    n = 5000
    tray, _, Ds = sim.generar_trayectorias(cfg, n, np.random.default_rng(0))
    assert tray.shape == (n, cfg.n_frames, 3) and np.all(Ds == D)
    escala = np.array([cfg.pixel_nm, cfg.pixel_nm, 1.0])            # a nm
    for m in (1, 5, 10):
        d2 = ((tray[:, m] - tray[:, 0]) * escala)**2
        teo = 2 * D * m * cfg.dt_s * 1e6                             # um^2 -> nm^2
        for eje in range(3):
            assert abs(d2[:, eje].mean() / teo - 1) < 5 * np.sqrt(2.0 / n), (m, eje, d2[:, eje].mean(), teo)


def test_incrementos_gaussianos_e_independientes():
    cfg = sim.Config(n_frames=50, z_rango_nm=(-1e6, 1e6))
    tray, _, D = sim.generar_trayectorias(cfg, 2000, np.random.default_rng(1))
    sigma_px = np.sqrt(2 * D * cfg.dt_s) * 1000.0 / cfg.pixel_nm
    u = np.diff(tray[..., 0], axis=1) / sigma_px[:, None]             # incrementos normalizados
    v = u.ravel()
    n = v.size
    assert abs(v.mean()) < 5 / np.sqrt(n)
    assert abs(v.var() - 1) < 5 * np.sqrt(2.0 / n)
    assert stats.kstest(v, "norm").pvalue > 1e-3
    r = np.corrcoef(u[:, :-1].ravel(), u[:, 1:].ravel())[0, 1]        # correlacion a un paso
    assert abs(r) < 5 / np.sqrt(n), r


def test_z_dentro_de_las_paredes_y_uniforme():
    # si se parte de z uniforme, la difusion con paredes reflectantes la mantiene uniforme
    cfg = sim.Config(n_frames=200)
    tray, _, _ = sim.generar_trayectorias(cfg, 5000, np.random.default_rng(2))
    lo, hi = cfg.z_rango_nm
    z = tray[..., 2]
    assert z.min() >= lo and z.max() <= hi
    assert stats.kstest(z[:, -1], "uniform", args=(lo, hi - lo)).pvalue > 1e-3


def test_parametros_por_particula():
    cfg = sim.Config(H=40, W=60, margen_px=3.0)
    tray, sub, D = sim.generar_trayectorias(cfg, 20_000, np.random.default_rng(3))
    lo, hi = cfg.D_rango_um2_s
    assert D.min() >= lo and D.max() <= hi
    assert stats.kstest(D, "uniform", args=(lo, hi - lo)).pvalue > 1e-3
    x0, y0 = tray[:, 0, 0], tray[:, 0, 1]
    assert x0.min() >= -0.5 - cfg.margen_px and x0.max() <= cfg.W - 0.5 + cfg.margen_px
    assert y0.min() >= -0.5 - cfg.margen_px and y0.max() <= cfg.H - 0.5 + cfg.margen_px
    # con n_subpasos = 1 la posicion del GT es la unica subposicion
    assert sub.shape == (20_000, cfg.n_frames, 1, 3) and np.array_equal(sub[:, :, 0], tray)


def test_desenfoque_de_movimiento_msd():
    # con exposicion t_e muestreada en n subpasos, el GT es la posicion media
    # durante la exposicion y su MSD por eje es (Savin y Doyle, 2005):
    #   MSD(m) = 2 D [m dt - t_e (n^2 - 1) / (3 n^2)]
    D, n_p = 0.2, 20_000
    for t_e, n_sub in [(0.1, 8), (0.05, 4)]:
        cfg = sim.Config(n_frames=6, D_rango_um2_s=(D, D), t_exp_s=t_e, n_subpasos=n_sub,
                         z_rango_nm=(-1e6, 1e6))
        tray, sub, _ = sim.generar_trayectorias(cfg, n_p, np.random.default_rng(4))
        assert sub.shape == (n_p, cfg.n_frames, n_sub, 3)
        assert np.allclose(tray, sub.mean(axis=2))
        for m in (1, 5):
            d2 = ((tray[:, m, :2] - tray[:, 0, :2]) * cfg.pixel_nm)**2   # x e y juntos
            teo = 2 * D * (m * cfg.dt_s - t_e * (n_sub**2 - 1) / (3 * n_sub**2)) * 1e6
            assert abs(d2.mean() / teo - 1) < 5 * np.sqrt(2.0 / (2 * n_p)), (t_e, n_sub, m, d2.mean(), teo)


def test_sin_difusion_las_subposiciones_coinciden():
    cfg = sim.Config(D_rango_um2_s=(0.0, 0.0), n_subpasos=5)
    tray, sub, _ = sim.generar_trayectorias(cfg, 10, np.random.default_rng(5))
    assert np.allclose(sub, tray[:, :, None, :])
    assert np.allclose(tray, tray[:, :1])
