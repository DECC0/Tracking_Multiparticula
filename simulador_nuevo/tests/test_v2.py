"""Tests del simulador v2: aberraciones de la pupila (astigmatismo y coma),
PSF directa cuando la pupila de la muestra no es la del banco (el real y su
reflejo comparten la pupila), fondo log-uniforme, autofluorescencia de la
celula y densidad en z con peso en los polos. Con los valores por defecto
nada cambia (test_regresion y localizacion/tests/test_regresion_celula.py
lo verifican bit a bit; aqui, la Config, el JSON y la fase)."""
from dataclasses import replace

import numpy as np

from comun import sim, espera_error

CELULA = dict(usar_espejo=True, usar_celula=True, diametro_celula_um=3.0, K_psf=64, z_paso_nm=100.0,
              z_rango_nm=(-1500.0, 1500.0), H=48, W=56, celula_centro_px=(24.0, 30.0), espejo_punto_px=(8.0, 30.0))


def test_config_v2_valida_y_json_sin_cambios_por_defecto():
    d = sim.Config().a_dict()
    assert not set(sim.DEFECTO_V2) & set(d)                    # apagadas: el JSON de antes
    espera_error(ValueError, sim.Config, amp_astig=-0.1)
    espera_error(ValueError, sim.Config, amp_coma=-0.1)
    espera_error(ValueError, sim.Config, muestreo_fondo="gauss")
    espera_error(ValueError, sim.Config, muestreo_fondo="log", fondo_rango=(0.0, 10.0))
    espera_error(ValueError, sim.Config, celula_peso_polos=0.5)                       # sin celula
    espera_error(ValueError, sim.Config, fondo_celula_contraste_rango=(0.0, 1.0))     # sin celula
    espera_error(ValueError, sim.Config, **CELULA, celula_peso_polos=1.5)
    espera_error(ValueError, sim.Config, **CELULA, fondo_celula_contraste_rango=(1.0, 0.5))
    cfg = sim.Config(**CELULA, amp_astig=0.7, angulo_astig_deg=20.0, amp_coma=0.3, angulo_coma_deg=200.0,
                     muestreo_fondo="log", celula_peso_polos=0.5, fondo_celula_contraste_rango=[0.0, 2.0],
                     fondo_celula_sigma_px=3.0)
    assert cfg.fondo_celula_contraste_rango == (0.0, 2.0)
    assert sim.Config.desde_dict(cfg.a_dict()) == cfg
    assert "aberraciones" in cfg.resumen() and "autofluorescencia" in cfg.resumen()
    assert "aberraciones" not in sim.Config().resumen()


def test_fase_de_las_aberraciones():
    """amp cos(2 (t - ang)) = a22 cos 2t + a2m2 sin 2t (la convencion de
    calibracion_pupila.py); maximo amp en el borde del disco; con amplitud 0 la
    rejilla es la de antes bit a bit."""
    rng = np.random.default_rng(0)
    r, t = rng.uniform(0, 1, 500), rng.uniform(-np.pi, np.pi, 500)
    A, ta, C, tc = 1.1, 37.0, 0.6, 250.0
    cfg = sim.Config(amp_astig=A, angulo_astig_deg=ta, amp_coma=C, angulo_coma_deg=tc)
    a22, a2m2 = A * np.cos(2 * np.radians(ta)), A * np.sin(2 * np.radians(ta))
    a31, a3m1 = C * np.cos(np.radians(tc)), C * np.sin(np.radians(tc))
    esperada = (a22 * r ** 2 * np.cos(2 * t) + a2m2 * r ** 2 * np.sin(2 * t)
                + (3 * r ** 3 - 2 * r) * (a31 * np.cos(t) + a3m1 * np.sin(t)))
    assert np.allclose(sim.fase_aberraciones(r, t, cfg), esperada, atol=1e-12)
    borde = sim.fase_aberraciones(np.ones(3600), np.linspace(-np.pi, np.pi, 3600), replace(cfg, amp_coma=0.0))
    assert np.isclose(np.abs(borde).max(), A, rtol=1e-5)
    r0 = sim._rejilla_pupila(sim.Config())
    r1 = sim._rejilla_pupila(sim.Config(angulo_astig_deg=30.0, angulo_coma_deg=10.0))   # amplitudes 0
    assert np.array_equal(r0["P0"], r1["P0"])


def test_astigmatismo_alarga_la_psf_en_su_eje():
    """Fuera de foco, la PSF con astigmatismo se alarga; girar el angulo 90
    grados cambia el eje del alargamiento."""
    def momentos(p):
        yy, xx = np.mgrid[0:p.shape[0], 0:p.shape[1]]
        s = p.sum(); x0, y0 = (p * xx).sum() / s, (p * yy).sum() / s
        return ((p * (xx - x0) ** 2).sum() / s, (p * (yy - y0) ** 2).sum() / s)
    base = dict(mascara="ninguna", amp_astig=1.2, K_psf=48)
    c = sim.psf_directa(sim.Config(**base, angulo_astig_deg=0.0), 400.0)
    d = sim.psf_directa(sim.Config(**base, angulo_astig_deg=90.0), 400.0)
    n = sim.psf_directa(sim.Config(mascara="ninguna"), 400.0)
    cx, cy = momentos(c[96:160, 96:160]); dx, dy = momentos(d[96:160, 96:160]); nx, ny = momentos(n[96:160, 96:160])
    assert abs(nx / ny - 1) < 1e-6
    assert abs(cx / cy - 1) > 0.1 and np.isclose(cx / cy, dy / dx, rtol=1e-6)


def test_psf_directa_sin_recorte_es_la_del_banco():
    """PSFRecortada(recortar=False) es la PSF de la pupila completa: igual al
    plano del banco (fraccion 1)."""
    cfg = sim.Config(mascara="esferica", z_rango_nm=(-400.0, 400.0), z_paso_nm=200.0, K_psf=64, amp_astig=0.5)
    b = sim.BancoPSF(cfg)
    p = sim.PSFRecortada(cfg, b.K, recortar=False)
    assert p.fraccion == 1.0 and p.completa
    for i, z in enumerate(b.z_grid):
        assert np.allclose(p.psf(z), b.stack[i], atol=1e-14)


def test_pupila_de_la_muestra_distinta_del_banco_usa_la_psf_directa():
    """Con un banco sin aberraciones y una muestra con ellas, generar_video
    dibuja con la PSF directa de la pupila de la muestra: igual (redondeo) a
    usar un banco construido con esa pupila, emisores en un plano del banco y
    sin movimiento. K = M_pupila: el desplazamiento subpixel del banco (en el
    recorte K) y el de la PSF directa (en la pupila) son el mismo."""
    base = dict(mascara="esferica", z_rango_nm=(0.0, 0.0), K_psf=256, H=40, W=48, n_emisores_rango=(3, 3),
                D_rango_um2_s=(0.0, 0.0), fondo_no_uniforme=False, seed=2, n_frames=2)
    cfg_ab = sim.Config(**base, amp_esferica=5.0, amp_astig=0.9, angulo_astig_deg=60.0, amp_coma=0.5,
                        angulo_coma_deg=10.0)
    banco_sin, banco_ab = sim.BancoPSF(sim.Config(**base)), sim.BancoPSF(cfg_ab)
    assert not sim.pupila_del_banco(cfg_ab, banco_sin) and sim.pupila_del_banco(cfg_ab, banco_ab)
    _, gt1, e1 = sim.generar_video(cfg_ab, banco_sin, 0, extra=True)
    _, gt2, e2 = sim.generar_video(cfg_ab, banco_ab, 0, extra=True)
    assert gt1.equals(gt2)
    assert np.allclose(e1["limpio"], e2["limpio"], rtol=1e-9, atol=1e-9)
    _, _, e0 = sim.generar_video(sim.Config(**base), banco_sin, 0, extra=True)
    assert np.abs(e0["limpio"] - e1["limpio"]).max() > 1.0        # la pupila si cambia la imagen


def test_real_y_reflejo_comparten_la_pupila():
    """Con espejo y recorte del reflejo, la imagen es la suma del real con la
    PSF directa de la pupila de la muestra y del reflejo con PSFRecortada de
    esa misma pupila (reconstruida desde las subposiciones)."""
    cfg = sim.Config(**CELULA, n_frames=1, n_videos=1, recorte_pupila_reflejo=True, amp_esferica=5.5,
                     amp_astig=1.0, angulo_astig_deg=30.0, amp_coma=0.7, angulo_coma_deg=120.0,
                     anillos_hacia_objetivo=True, n_emisores_rango=(4, 4), seed=5)
    banco = sim.BancoPSF(replace(cfg, amp_esferica=4.0, amp_astig=0.0, amp_coma=0.0))
    _, gt, ex = sim.generar_video(cfg, banco, 0, extra=True)
    assert gt.reflejo_en_banco.all()
    real, refl = sim.PSFRecortada(cfg, banco.K, recortar=False), sim.PSFRecortada(cfg, banco.K)
    img = np.zeros((cfg.H, cfg.W))
    for p in range(len(gt)):
        real.colocar(img, *ex["sub"][p, 0, 0], gt.photons[p])
        refl.colocar(img, *ex["sub_reflejo"][p, 0, 0], gt.photons_reflejo[p])
    assert np.array_equal(img + ex["fondo"], ex["limpio"][0])
    sin_astig = sim.PSFRecortada(replace(cfg, amp_astig=0.0), banco.K)
    assert np.abs(sin_astig.psf(-2000.0) - refl.psf(-2000.0)).max() > 1e-4


def test_fondo_log_uniforme():
    cfg = sim.Config(H=4, W=4, fondo_rango=(1.0, 40.0), muestreo_fondo="log", fondo_no_uniforme=False)
    rng = np.random.default_rng(0)
    b = np.array([sim.generar_fondo(cfg, rng)[0, 0] for _ in range(4000)])
    assert b.min() >= 1.0 and b.max() <= 40.0
    assert abs(np.median(b) / np.sqrt(40.0) - 1) < 0.08          # mediana log-uniforme: sqrt(1 * 40)
    assert abs(np.mean(b < 6.32) - 0.5) < 0.03
    # mismos sorteos que con muestreo uniforme: el fondo no uniforme usa los mismos 6 numeros
    c_u, c_l = sim.Config(H=8, W=8), sim.Config(H=8, W=8, muestreo_fondo="log")
    r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
    sim.generar_fondo(c_u, r1); sim.generar_fondo(c_l, r2)
    assert r1.uniform() == r2.uniform()


def test_autofluorescencia_de_la_celula():
    """fondo con celula = fondo sin celula + amp perfil_celula, con amp / base
    en el rango; el perfil vale ~1 en el centro de la celula, ~ratio_reflejo en
    el de la celula virtual y 0 lejos de las dos."""
    cfg = sim.Config(**{**CELULA, "diametro_celula_um": 2.0}, fondo_celula_contraste_rango=(0.5, 2.0),
                     fondo_celula_sigma_px=2.0, ratio_reflejo=0.6)
    sin = replace(cfg, fondo_celula_contraste_rango=None)
    plano = replace(sin, fondo_no_uniforme=False)
    for s in range(30):
        f_con, amp = sim._generar_fondo(cfg, np.random.default_rng(s))
        f_sin = sim.generar_fondo(sin, np.random.default_rng(s))
        base = sim.generar_fondo(plano, np.random.default_rng(s))[0, 0]
        assert 0.5 <= amp / base <= 2.0
        assert np.allclose(f_con - f_sin, amp * sim.perfil_celula(cfg), atol=1e-9)
    p = sim.perfil_celula(cfg)
    px = cfg.pixel_nm
    cel = sim.Celula(cfg)
    cr = np.round(cel.c[:2] / px).astype(int)
    cv = np.round(cel.reflejada(sim.Espejo(cfg)).c[:2] / px).astype(int)
    R = cfg.diametro_celula_um * 500 / px
    assert abs(p[cr[1], cr[0]] - 1.0) < 0.05
    assert abs(p[cv[1], cv[0]] - 0.6) < 0.05
    yy, xx = np.mgrid[0:cfg.H, 0:cfg.W]
    lejos = (np.hypot(xx - cr[0], yy - cr[1]) > R + 12) & (np.hypot(xx - cv[0], yy - cv[1]) > R + 12)
    assert lejos.any() and p[lejos].max() < 1e-6
    _, _, ex = sim.generar_video(replace(cfg, n_frames=1, n_videos=1), sim.BancoPSF(cfg), 0, extra=True)
    assert "fondo_celula" in ex


def test_peso_polos_densidad_en_z_y_disco():
    """Con w = 1/3 z es uniforme en [-R, R]; con w = 0 sigue (1 - s^2); a cada
    z, uniforme en el disco; todo dentro de la esfera."""
    cel = sim.GeometriaCelula([10.0, -5.0, 300.0], 2000.0)
    rng = np.random.default_rng(0)
    n = 200000
    for w, dens in ((1 / 3, lambda s: np.full_like(s, 0.5)), (0.0, lambda s: 0.75 * (1 - s ** 2)),
                    (0.7, lambda s: 0.3 * 0.75 * (1 - s ** 2) + 0.7 * 1.5 * s ** 2)):
        p = cel.peso_polos(*rng.uniform(size=(3, n)), w)
        assert (np.linalg.norm(p - cel.c, axis=1) <= cel.R * (1 + 1e-12)).all()
        s = (p[:, 2] - cel.c[2]) / cel.R
        h, bordes = np.histogram(s, bins=20, range=(-1, 1), density=True)
        medio = 0.5 * (bordes[1:] + bordes[:-1])
        assert np.abs(h - dens(medio)).max() < 0.04, (w, h.round(3))
        rho2 = ((p[:, :2] - cel.c[:2]) ** 2).sum(1) / (cel.R ** 2 * (1 - s ** 2))
        assert abs(rho2.mean() - 0.5) < 0.01                    # uniforme en el disco: r^2 / r_max^2 ~ U(0, 1)


def test_peso_polos_en_generar_video():
    """El simulador usa los mismos sorteos (mismas D, mismo fondo) y pone mas
    emisores en |z - z_c| > R / 2 que el muestreo en el volumen."""
    base = sim.Config(**CELULA, n_frames=1, n_videos=1, n_emisores_rango=(400, 400))
    banco = sim.BancoPSF(base)
    _, g0, e0 = sim.generar_video(base, banco, 0, extra=True)
    _, g1, e1 = sim.generar_video(replace(base, celula_peso_polos=0.6), banco, 0, extra=True)
    assert g0.D_um2_s.equals(g1.D_um2_s) and np.array_equal(e0["fondo"], e1["fondo"])
    c = sim.Celula(base)
    f0 = np.mean(np.abs(g0.z_nm - c.c[2]) > c.R / 2)
    f1 = np.mean(np.abs(g1.z_nm - c.c[2]) > c.R / 2)
    assert f1 > f0 + 0.15                                       # teorico: 0.31 -> 0.57
