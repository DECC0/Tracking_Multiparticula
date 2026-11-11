"""Tests del recorte de pupila del reflejo (recorte_pupila_reflejo=True): la
mascara (fraccion que pasa, beta = 0 -> la mitad, beta >= arcsin(NA/n) ->
toda), la PSF directa sin renormalizar, rotacion exacta con phi, las dos
variantes de anillos_hacia_objetivo con la misma geometria, el lado al que
se corre el centroide (optica geometrica con la mascara "ninguna"), y en el
video: real, GT y otros flujos sin cambio, imagen reconstruible y, sin
recorte efectivo o apagado, bit a bit lo de antes."""
import json

import numpy as np

from comun import sim, espera_error
from test_espejo import BASE, cfg_e, banco_e, video

BETA_LIMITE = np.degrees(np.arcsin(1.1 / 1.333))       # 55.6: el cono de coleccion


def rec(**kw):
    return cfg_e(recorte_pupila_reflejo=True, **kw)


def fraccion_geometrica(beta_deg, n=4000):
    """Fraccion del disco de la pupila (amplitud uniforme en sin(theta)) con
    w . n > 0, por cuadratura en (rho, psi); phi no importa."""
    rho = (np.arange(n) + 0.5) / n
    psi = (np.arange(2 * n) + 0.5) / (2 * n) * 2 * np.pi
    R, P = np.meshgrid(rho, psi)
    st = 1.1 / 1.333 * R
    b = np.radians(beta_deg)
    ok = np.cos(b) * st * np.cos(P) + np.sin(b) * np.sqrt(1 - st ** 2) > 0
    return float((ok * R).sum() / R.sum())


def test_config_del_recorte():
    espera_error(ValueError, sim.Config, recorte_pupila_reflejo=True)           # sin espejo
    assert "recorte_pupila_reflejo" not in cfg_e().a_dict()                      # apagado: el JSON de antes
    c = rec()
    assert c.a_dict()["recorte_pupila_reflejo"] is True
    assert sim.Config.desde_dict(json.loads(json.dumps(c.a_dict()))) == c
    assert "recorte de pupila" in c.resumen() and "recorte" not in cfg_e().resumen()


def test_fraccion_de_la_pupila():
    """La fraccion es la integral de la mascara sobre la pupila (y coincide con
    la geometrica); beta = 0 pierde la mitad, beta >= 55.6 nada; fraccion(-beta)
    = 1 - fraccion(beta); no depende de phi."""
    for b in (0.0, 30.0, 45.0, 50.0):
        f = sim.PSFReflejo(rec(angulo_espejo_eje_deg=b), 32).fraccion
        assert abs(f - fraccion_geometrica(b)) < 2e-3, (b, f, fraccion_geometrica(b))
        f_neg = sim.PSFReflejo(rec(angulo_espejo_eje_deg=-b), 32).fraccion
        assert abs(f + f_neg - 1.0) < 1e-12
        f_phi = sim.PSFReflejo(rec(angulo_espejo_eje_deg=b, orientacion_espejo_deg=37.0), 32).fraccion
        assert abs(f_phi - f) < 2e-3
    assert abs(sim.PSFReflejo(rec(angulo_espejo_eje_deg=0.0), 32).fraccion - 0.5) < 1e-3
    for b in (BETA_LIMITE + 0.01, 60.0, 80.0):
        r = sim.PSFReflejo(rec(angulo_espejo_eje_deg=b), 32)
        assert r.completa and r.fraccion == 1.0
    assert not sim.PSFReflejo(rec(angulo_espejo_eje_deg=BETA_LIMITE - 0.5), 32).completa


def test_psf_sin_renormalizar():
    """La PSF recortada suma la fraccion (Parseval: la luz que pasa); el recorte
    K x K de PSFReflejo es el de psf_directa(recorte=True); a beta >= 55.6 es la
    PSF completa."""
    c = rec(mascara="esferica", K_psf=None, orientacion_espejo_deg=25.0)
    r = sim.PSFReflejo(c, 64)
    M = c.M_pupila
    for z in (-3000.0, 0.0, 700.0):
        full = sim.psf_directa(c, z, 0.3, 0.6, recorte=True)
        assert abs(full.sum() - r.fraccion) < 1e-12
        ref = full[M // 2 - 32: M // 2 + 32, M // 2 - 32: M // 2 + 32]
        assert np.allclose(r.psf(z, 0.3, 0.6), ref, rtol=0, atol=1e-14)
    c60 = rec(mascara="esferica", angulo_espejo_eje_deg=60.0)
    assert np.array_equal(sim.psf_directa(c60, -1500.0, recorte=True), sim.psf_directa(c60, -1500.0))


def test_phi_90_es_una_rotacion_exacta():
    """phi = 90 recorta el lado +y como phi = 0 el +x: la PSF es la de phi = 0
    rotada 90 grados alrededor del pixel central (psf90(x, y) = psf0(y, -x))."""
    M = BASE.get("M_pupila", 256)
    idx = (2 * (M // 2) - np.arange(M)) % M
    for masc in ("ninguna", "esferica"):
        p0 = sim.psf_directa(rec(mascara=masc, orientacion_espejo_deg=0.0), -2000.0, recorte=True)
        p90 = sim.psf_directa(rec(mascara=masc, orientacion_espejo_deg=90.0), -2000.0, recorte=True)
        assert np.allclose(p90, p0.T[:, idx], rtol=0, atol=1e-15 * p0.max() + 1e-18), masc
        assert not np.allclose(p90, p0, atol=1e-3 * p0.max())


def test_misma_geometria_con_las_dos_variantes_de_anillos():
    """anillos_hacia_objetivo invierte el signo de q; la mascara se toma en -f
    para que el espejo recorte el mismo lado: con la mascara "ninguna" (sin
    fase) las dos variantes dan la misma PSF recortada."""
    for z in (-2500.0, 0.0, 1200.0):
        a = sim.psf_directa(rec(orientacion_espejo_deg=30.0), z, 0.2, 0.4, recorte=True)
        b = sim.psf_directa(rec(orientacion_espejo_deg=30.0, anillos_hacia_objetivo=True), z, 0.2, 0.4, recorte=True)
        assert np.allclose(a, b, rtol=0, atol=1e-12 * a.max())


def test_centroide_se_corre_hacia_la_muestra_lejos_del_objetivo():
    """Optica geometrica (mascara "ninguna"): se pierden los rayos que van
    hacia -m (hacia el espejo); con el reflejo a z' < 0 (lejos del objetivo)
    esos rayos caen del lado -m en el plano focal, asi que el centroide se
    corre hacia +m, ~57 nm por um a 45 grados (uniforme), y hacia -m con z' > 0."""
    M = 256
    yy, xx = np.mgrid[0:M, 0:M] - M // 2
    for phi in (0.0, 120.0):
        c = rec(orientacion_espejo_deg=phi)
        m = sim.Espejo(c).m[:2]
        for z, esperado in ((-2000.0, 114.0), (-1000.0, 57.0), (1000.0, -57.0)):
            a, b = sim.psf_directa(c, z, recorte=True), sim.psf_directa(c, z)
            d = np.array([((a * xx).sum() / a.sum() - (b * xx).sum() / b.sum()),
                          ((a * yy).sum() / a.sum() - (b * yy).sum() / b.sum())]) * c.pixel_nm
            assert abs(d @ m - esperado) < 0.15 * abs(esperado), (phi, z, d @ m)
            assert abs(d @ np.array([-m[1], m[0]])) < 1.0                       # nada a lo largo de la linea


def test_video_real_gt_y_flujos_sin_cambio():
    """Con el recorte cambia solo la luz del reflejo: GT, fondo y subposiciones
    iguales; con ratio_reflejo = 0 el video es bit a bit el de sin recorte."""
    _, _, g0, e0 = video()
    _, _, g1, e1 = video(recorte_pupila_reflejo=True)
    assert g0.equals(g1) and np.array_equal(e0["fondo"], e1["fondo"]) and np.array_equal(e0["sub"], e1["sub"])
    assert not np.allclose(e0["limpio"], e1["limpio"])
    _, v0, _, _ = video(ratio_reflejo=0.0)
    _, v1, _, _ = video(ratio_reflejo=0.0, recorte_pupila_reflejo=True)
    assert np.array_equal(v0, v1)


def test_imagen_reconstruible_con_recorte():
    """limpio = fondo + reales con el banco + reflejos con PSFReflejo, con
    subpasos; la luz del reflejo es ratio * fraccion * fotones (sin la parte
    fuera del recorte K)."""
    kw = dict(recorte_pupila_reflejo=True, n_subpasos=3, n_frames=3)
    cfg, _, gt, ex = video(**kw)
    b, r = banco_e(**kw), sim.PSFReflejo(cfg, banco_e(**kw).K)
    assert not r.completa
    sub, sub_r = ex["sub"], ex["sub_reflejo"]
    for t in range(cfg.n_frames):
        img = np.zeros((cfg.H, cfg.W))
        g = gt[gt.frame == t]
        for p, f in zip(g.particle_id, g.itertuples()):
            for k in range(cfg.n_subpasos):
                b.colocar(img, *sub[p, t, k], f.photons / cfg.n_subpasos)
                if f.reflejo_en_banco:
                    r.colocar(img, *sub_r[p, t, k], f.photons_reflejo / cfg.n_subpasos)
        assert np.allclose(ex["limpio"][t], img + ex["fondo"], rtol=1e-12, atol=1e-9), t
    grande = sim.PSFReflejo(cfg, 256)
    lienzo = np.zeros((300, 300))
    grande.colocar(lienzo, 150.3, 149.6, -1800.0, 1000.0)
    assert abs(lienzo.sum() - 1000.0 * grande.fraccion) < 1e-8


def test_sin_recorte_efectivo_es_bit_a_bit():
    """beta >= 55.6: no se pierde ningun rayo y el video es el de sin recorte."""
    kw = dict(angulo_espejo_eje_deg=60.0)
    _, v0, g0, e0 = video(**kw)
    _, v1, g1, e1 = video(recorte_pupila_reflejo=True, **kw)
    assert np.array_equal(v0, v1) and g0.equals(g1) and np.array_equal(e0["limpio"], e1["limpio"])
