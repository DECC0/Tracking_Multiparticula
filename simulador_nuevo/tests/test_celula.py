"""Tests de la celula (usar_celula=True, geometria_celula.py): configuracion y
JSON, celula tangente al espejo, posiciones iniciales uniformes en el volumen
(mismos sorteos que sin celula), reflexion especular exacta, ningun emisor sale
de la celula ni cruza el espejo (tambien en los subpasos), MSD que se satura en
6 R^2 / 5, reflejos coherentes (dentro de la celula virtual), en_rango_z y
banco, y que con la celula apagada sus parametros no cambian nada."""
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

from comun import sim, espera_error
from geometria_celula import GeometriaCelula
from geometria_espejo import GeometriaEspejo

# H != W a proposito; PSF "ninguna" con recorte chico y banco grueso: barato
BASE = dict(mascara="ninguna", H=40, W=56, n_frames=6, n_videos=2, K_psf=32, z_paso_nm=100.0,
            n_emisores_rango=(5, 9), D_rango_um2_s=(0.1, 1.0), z_rango_nm=(-2500.0, 2500.0),
            espejo_punto_px=(8.0, 20.0), celula_centro_px=(30.0, 20.0), seed=31)
GEOMETRIAS = [(45.0, 0.0, 0.0), (40.0, 37.0, -500.0), (50.0, 200.0, 300.0), (0.0, 90.0, 0.0), (-30.0, 300.0, 0.0)]


def cfg_c(espejo=True, **kw):
    return sim.Config(**{**BASE, "usar_celula": True, "usar_espejo": espejo, **kw})


# --- configuracion ------------------------------------------------------------

def test_config_y_json():
    espera_error(ValueError, cfg_c, diametro_celula_um=0.0)
    espera_error(ValueError, cfg_c, celula_centro_px=(1.0, 2.0, 3.0))
    espera_error(ValueError, cfg_c, dist_max_espejo_um=2.0)          # con celula la region es la celula
    c = sim.Config()
    assert not c.usar_celula and c.diametro_celula_um == 5.0 and c.altura_celula_nm == 0.0
    off = sim.Config(**BASE)
    assert not set(sim.CAMPOS_CELULA) & set(off.a_dict())
    cfg = cfg_c(diametro_celula_um=4.2, altura_celula_nm=-350.0, celula_centro_px=(25.0, 12.0))
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "config.json"
        sim.guardar_config(cfg, ruta)
        assert sim.cargar_config(ruta) == cfg
        assert json.loads(ruta.read_text())["config"]["diametro_celula_um"] == 4.2
    assert "celula" in cfg.resumen() and "celula" not in off.resumen()


def test_celula_tangente_al_espejo():
    """El centro queda a distancia R del plano, del lado de la muestra, a la
    altura pedida; el contacto c - R n esta en el plano, y toda la celula (y
    nada de la celula virtual) del lado d >= 0."""
    rng = np.random.default_rng(0)
    for b, f, zr in GEOMETRIAS:
        e = GeometriaEspejo(b, f, [1000.0, -400.0, zr], -1e9, 1e9)
        for R, zc in ((2500.0, 0.0), (2000.0, 900.0), (3000.0, -1200.0)):
            cel = GeometriaCelula.tangente_al_espejo(e, R, zc, np.array([5000.0, 3000.0]))
            assert np.isclose(e.distancia_nm(cel.c), R) and np.isclose(cel.c[2], zc)
            assert np.isclose(e.distancia_nm(cel.contacto_nm(e)), 0.0, atol=1e-6)
            assert np.isclose((cel.c[:2] - [5000.0, 3000.0]) @ e.t[:2], 0.0, atol=1e-6)   # misma posicion en t
            p = cel.uniforme(*rng.uniform(size=(3, 2000)))
            assert (e.distancia_nm(p) >= -1e-6).all()
            virtual = cel.reflejada(e)
            assert (virtual.profundidad_nm(e.reflejar_nm(p)) >= -1e-6).all()


# --- muestreo y movimiento ------------------------------------------------------

def test_uniforme_en_el_volumen():
    cel = GeometriaCelula([100.0, -50.0, 300.0], 2500.0)
    p = cel.uniforme(*np.random.default_rng(1).uniform(size=(3, 20000)))
    rel = p - cel.c
    r = np.linalg.norm(rel, axis=1)
    assert (r <= cel.R + 1e-9).all()
    assert stats.kstest((r / cel.R) ** 3, "uniform").pvalue > 1e-3          # densidad uniforme en el volumen
    assert stats.kstest(rel[:, 2] / r, "uniform", args=(-1, 2)).pvalue > 1e-3  # cos(theta) uniforme
    assert stats.kstest(np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi), "uniform",
                        args=(0, 2 * np.pi)).pvalue > 1e-3


def test_posiciones_iniciales_y_mismos_sorteos():
    """Con celula las posiciones iniciales son uniformes en la celula y el rng
    consume exactamente lo mismo que sin ella (n, D, fotones y fondo no cambian)."""
    for espejo in (True, False):
        cfg = cfg_c(espejo=espejo, n_frames=1)
        cel = sim.Celula(cfg)
        r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
        _, sub, D1 = sim.generar_trayectorias(cfg, 5000, r1)
        _, _, D2 = sim.generar_trayectorias(sim.Config(**{**BASE, "n_frames": 1, "usar_espejo": espejo}), 5000, r2)
        assert r1.bit_generator.state == r2.bit_generator.state and np.array_equal(D1, D2)
        r = cel.R - cel.profundidad_nm(cel.a_nm(sub[:, 0, 0]))
        assert (r <= cel.R + 1e-6).all() and stats.kstest((r / cel.R) ** 3, "uniform").pvalue > 1e-3


def test_reflexion_especular_exacta():
    cel = GeometriaCelula([0.0, 0.0, 0.0], 1000.0)
    # radial: desde el centro, 1300 nm hacia +x -> rebota a 700 nm
    assert np.allclose(cel.mover_nm(np.zeros((1, 3)), np.array([[1300.0, 0.0, 0.0]])), [[700.0, 0.0, 0.0]])
    # oblicuo: la longitud del camino se conserva y el punto final es el reflejo especular
    p0, s = np.array([[0.0, 600.0, 0.0]]), np.array([[1200.0, 0.0, 0.0]])
    q = cel.mover_nm(p0, s)
    h = np.array([800.0, 600.0, 0.0])                                    # salida: |h| = 1000
    u = h / 1000.0
    resto = s[0] * (1 - 800.0 / 1200.0)
    assert np.allclose(q[0], h + resto - 2 * (resto @ u) * u)
    assert np.isclose(np.linalg.norm(h - p0[0]) + np.linalg.norm(q[0] - h), 1200.0)
    # pasos muy largos (varias reflexiones) siguen dentro
    rng = np.random.default_rng(2)
    p = cel.uniforme(*rng.uniform(size=(3, 500)))
    q = cel.mover_nm(p, rng.normal(size=(500, 3)) * 3000.0)
    assert (cel.profundidad_nm(q) >= -1e-9).all()


def test_ningun_emisor_sale_ni_cruza_el_espejo_en_los_subpasos():
    for b, f, zr in GEOMETRIAS:
        cfg = cfg_c(angulo_espejo_eje_deg=b, orientacion_espejo_deg=f, z_ref_espejo_nm=zr, n_frames=60,
                    n_subpasos=3, t_exp_s=0.06, D_rango_um2_s=(1.0, 3.0), diametro_celula_um=4.0)
        cel, e = sim.Celula(cfg), sim.Espejo(cfg)
        tray, sub, _ = sim.generar_trayectorias(cfg, 60, np.random.default_rng(int(abs(b)) + 1))
        assert (cel.profundidad_nm(cel.a_nm(sub)) >= -1e-6).all(), (b, f)
        assert (e.distancia_nm(e.a_nm(sub)) >= -1e-6).all(), (b, f)
        assert np.allclose(tray, sub.mean(axis=2))
    cfg = cfg_c(espejo=False, n_frames=60, n_subpasos=2, t_exp_s=0.05, D_rango_um2_s=(1.0, 3.0))
    cel = sim.Celula(cfg)
    _, sub, _ = sim.generar_trayectorias(cfg, 60, np.random.default_rng(3))
    assert (cel.profundidad_nm(cel.a_nm(sub)) >= -1e-6).all()


def test_msd_se_satura_al_valor_de_la_esfera():
    """Desde posiciones uniformes en la esfera, el MSD 3D a tiempo corto es
    6 D t y a tiempo largo se satura en 2 <r^2> = 6 R^2 / 5 (2 R^2 / 5 por eje)."""
    R_um, D = 1.0, 0.2
    cfg = cfg_c(espejo=False, diametro_celula_um=2 * R_um, D_rango_um2_s=(D, D), n_frames=400, dt_s=0.1)
    tray, _, _ = sim.generar_trayectorias(cfg, 300, np.random.default_rng(7))
    p = sim.Celula(cfg).a_nm(tray) / 1000.0                              # um
    corto = np.mean(np.sum((p[:, 1:] - p[:, :-1]) ** 2, -1))
    assert np.isclose(corto, 6 * D * cfg.dt_s, rtol=0.15), corto        # reflexiones: algo menos que libre
    largo = np.mean([np.mean(np.sum((p[:, k:] - p[:, :-k]) ** 2, -1)) for k in range(150, 250, 10)])
    assert np.isclose(largo, 6 * R_um ** 2 / 5, rtol=0.05), (largo, 6 * R_um ** 2 / 5)


# --- reflejos, GT y banco ---------------------------------------------------------

def test_reflejos_coherentes_y_gt():
    cfg = cfg_c(n_subpasos=3, t_exp_s=0.05, diametro_celula_um=4.0)
    banco = sim.BancoPSF(cfg)
    cel, e = sim.Celula(cfg), sim.Espejo(cfg)
    virtual = cel.reflejada(e)
    for vid in (0, 1):
        _, gt, ex = sim.generar_video(cfg, banco, vid, extra=True)
        assert list(gt.columns) == sim.COLUMNAS_GT_ESPEJO + sim.COLUMNAS_CELULA
        assert np.allclose(ex["sub_reflejo"], e.reflejar(ex["sub"]), atol=1e-9)
        assert (virtual.profundidad_nm(cel.a_nm(ex["sub_reflejo"])) >= -1e-6).all()
        real = gt[["x_px", "y_px", "z_nm"]].to_numpy()
        assert np.allclose(gt[["x_reflejo_px", "y_reflejo_px", "z_reflejo_nm"]].to_numpy(), e.reflejar(real), atol=1e-9)
        assert gt.reflejo_en_banco.all() and (gt.z_reflejo_nm < gt.z_nm).all()       # beta = 45
        dentro = (gt.z_nm >= cfg.z_rango_nm[0]) & (gt.z_nm <= cfg.z_rango_nm[1])
        assert (gt.en_rango_z == dentro).all()
    lo, hi = sim.rango_banco_nm(cfg)
    assert lo <= virtual.rango_z()[0] + 1e-6 and hi >= cel.rango_z()[1] - 1e-6
    # sin espejo: columnas de siempre + en_rango_z
    c2 = cfg_c(espejo=False, z_rango_nm=(-800.0, 800.0))
    _, g2 = sim.generar_video(c2, sim.BancoPSF(c2), 0)
    assert list(g2.columns) == sim.COLUMNAS_GT + sim.COLUMNAS_CELULA and not g2.en_rango_z.all()


def test_celula_apagada_ignora_sus_parametros_y_no_cambia_otros_flujos():
    a = sim.Config(**{k: v for k, v in BASE.items() if k != "celula_centro_px"})
    b = sim.Config(**{**BASE, "diametro_celula_um": 3.0, "altura_celula_nm": -700.0, "celula_centro_px": (1.0, 2.0)})
    ba = sim.BancoPSF(a)
    va, ga = sim.generar_video(a, ba, 0)
    vb, gb = sim.generar_video(b, ba, 0)
    assert np.array_equal(va, vb) and ga.equals(gb) and a.a_dict() == b.a_dict() and a.resumen() == b.resumen()
    on = cfg_c(espejo=False)
    _, g_on, e_on = sim.generar_video(on, sim.BancoPSF(on), 0, extra=True)
    _, g_off, e_off = sim.generar_video(a, ba, 0, extra=True)
    assert np.array_equal(g_on.photons, g_off.photons) and np.array_equal(g_on.D_um2_s, g_off.D_um2_s)
    assert np.array_equal(e_on["fondo"], e_off["fondo"])
