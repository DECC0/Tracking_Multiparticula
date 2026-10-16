"""Tests de la PSF: normalizacion, recorte, simetrias, escala optica contra la
teoria (Airy y desenfoque en el eje), interpolacion en z, integracion sobre el
pixel y comportamiento de cada mascara de fase."""
import numpy as np
from scipy.integrate import quad
from scipy.special import j1

from comun import sim, banco, MASCARAS


def _recorte_central(p, K):
    c = p.shape[0] // 2
    return p[c - K // 2: c + K // 2, c - K // 2: c + K // 2]


def test_psf_normalizada_y_no_negativa():
    for m in MASCARAS:
        b = banco(m)
        assert b.stack.min() >= 0.0, m
        # el recorte suma 1 - (energia que quedo fuera del recorte)
        suma = b.stack.sum(axis=(1, 2))
        assert np.allclose(suma, 1.0 - b.energia_recortada, atol=1e-12), m
        # la PSF completa (sin recorte) suma 1 a cualquier z
        assert np.isclose(sim.psf_directa(b.cfg, 333.0).sum(), 1.0), m


def test_recorte_automatico_respeta_tolerancia():
    for m in MASCARAS:
        b = banco(m)
        tol = b.cfg.energia_max_recortada
        assert b.K % 2 == 0 and b.K <= b.cfg.M_pupila, (m, b.K)
        assert b.energia_recortada.max() <= tol, (m, b.K, b.energia_recortada.max())
        # y no es mas grande de lo necesario: 16 px menos ya pasaria la tolerancia
        if b.K > 16:
            fuera = 1.0 - b.stack[:, 8:-8, 8:-8].sum(axis=(1, 2))
            assert fuera.max() > tol, (m, b.K)
    # un K explicito se respeta
    assert banco("ninguna", K_psf=32).K == 32


def test_rejilla_z():
    for zr, paso in [((-800.0, 800.0), 20.0), ((-500.0, 500.0), 30.0), ((0.0, 0.0), 20.0)]:
        cfg = sim.Config(z_rango_nm=zr, z_paso_nm=paso)
        z = sim.rejilla_z(cfg)
        assert z[0] == zr[0] and z[-1] == zr[1], (zr, paso)
        if len(z) > 1:
            d = np.diff(z)
            assert np.all(d > 0) and d.max() <= paso + 1e-9 and np.allclose(d, d[0])
    assert np.array_equal(banco("ninguna").z_grid, sim.rejilla_z(banco("ninguna").cfg))


def test_psf_centrada_y_simetrica():
    # sin mascara o con la esferica (radial) la PSF es simetrica respecto al
    # pixel K // 2: invariante ante x -> -x, y -> -y y x <-> y.
    # (la fila/columna 0 no tiene pareja simetrica en un recorte par)
    for m in ("ninguna", "esferica"):
        b = banco(m)
        for i in range(0, len(b.z_grid), 10):
            p = b.stack[i][1:, 1:]
            escala = p.max()
            assert np.abs(p - p[:, ::-1]).max() < 1e-9 * escala, (m, b.z_grid[i])
            assert np.abs(p - p[::-1, :]).max() < 1e-9 * escala, (m, b.z_grid[i])
            assert np.abs(p - p.T).max() < 1e-9 * escala, (m, b.z_grid[i])


def test_desenfoque_simetrico_sin_mascara():
    # modelo escalar sin aberraciones: PSF(z) = PSF(-z)
    b = banco("ninguna")
    assert np.allclose(b.z_grid, -b.z_grid[::-1])
    assert np.abs(b.stack - b.stack[::-1]).max() < 1e-9 * b.stack.max()


def test_pico_maximo_en_foco_sin_mascara():
    b = banco("ninguna")
    pico = b.stack.max(axis=(1, 2))
    i0 = int(np.argmin(np.abs(b.z_grid)))
    assert pico.argmax() == i0
    assert np.all(np.diff(pico[i0:]) < 0) and np.all(np.diff(pico[:i0 + 1]) > 0)


def test_airy_en_foco():
    # con pupila uniforme, sin blur ni integracion de pixel, la PSF en foco es
    # exactamente el patron de Airy [2 J1(v) / v]^2, v = 2 pi NA r / lambda.
    # Verifica la escala optica (NA, lambda, tamano de pixel) y las unidades.
    cfg = sim.Config(mascara="ninguna", blur_extra_px=0.0, integrar_pixel=False)
    p = sim.psf_directa(cfg, 0.0)
    c = cfg.M_pupila // 2
    R = 12
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    v = 2 * np.pi * cfg.NA * np.hypot(xx, yy) * cfg.pixel_nm / cfg.lambda_nm
    with np.errstate(invalid="ignore", divide="ignore"):
        airy = np.where(v == 0, 1.0, (2 * j1(v) / v)**2)
    sim_rel = p[c - R: c + R + 1, c - R: c + R + 1] / p[c, c]
    assert np.abs(sim_rel - airy).max() < 1e-2, np.abs(sim_rel - airy).max()


def test_desenfoque_en_el_eje_vs_teoria():
    # intensidad en el eje optico vs z para la pupila escalar:
    #   A(z) ~ int_0^{NA/lambda} exp(i k z cos(theta(f))) 2 pi f df,
    #   cos(theta) = sqrt(1 - (lambda f / n)^2), k = 2 pi n / lambda
    # Verifica el termino de desenfoque k z cos(theta) (incluido el indice n).
    cfg = sim.Config(mascara="ninguna", blur_extra_px=0.0, integrar_pixel=False)
    k = 2 * np.pi * cfg.n_medio / cfg.lambda_nm
    fmax = cfg.NA / cfg.lambda_nm

    def amplitud(z):
        cos_t = lambda f: np.sqrt(1 - (cfg.lambda_nm * f / cfg.n_medio)**2)
        re = quad(lambda f: np.cos(k * z * cos_t(f)) * f, 0, fmax, limit=200)[0]
        im = quad(lambda f: np.sin(k * z * cos_t(f)) * f, 0, fmax, limit=200)[0]
        return re**2 + im**2

    c = cfg.M_pupila // 2
    I0 = sim.psf_directa(cfg, 0.0)[c, c]
    for z in (200.0, 400.0, 600.0, 800.0):
        rel_sim = sim.psf_directa(cfg, z)[c, c] / I0
        rel_teo = amplitud(z) / amplitud(0.0)
        assert abs(rel_sim - rel_teo) < 1e-2, (z, rel_sim, rel_teo)


def test_interpolacion_en_z():
    b = banco("esferica")
    # en los nodos de la rejilla es exacta
    for i in (0, 7, len(b.z_grid) - 1):
        assert np.array_equal(b.psf_en_z(b.z_grid[i]), b.stack[i])
    # fuera del rango se satura al borde
    assert np.array_equal(b.psf_en_z(b.z_grid[0] - 500), b.stack[0])
    assert np.array_equal(b.psf_en_z(b.z_grid[-1] + 500), b.stack[-1])
    # entre nodos es lineal: combinacion convexa de los dos planos vecinos
    i, w = 30, 0.25
    z = b.z_grid[i] + w * (b.z_grid[i + 1] - b.z_grid[i])
    assert np.allclose(b.psf_en_z(z), (1 - w) * b.stack[i] + w * b.stack[i + 1], rtol=1e-12, atol=0)


def test_interpolacion_en_z_vs_psf_exacta():
    # en el punto medio entre nodos, el error contra la PSF exacta es < 1% del
    # pico para todas las mascaras (sin mascara la PSF cambia mas rapido con z)
    for m in MASCARAS:
        b = banco(m)
        for i in range(0, len(b.z_grid) - 1, 8):
            z = 0.5 * (b.z_grid[i] + b.z_grid[i + 1])
            exacta = _recorte_central(sim.psf_directa(b.cfg, z), b.K)
            err = np.abs(b.psf_en_z(z) - exacta).max() / exacta.max()
            assert err < 1e-2, (m, z, err)


def test_integracion_sobre_el_pixel():
    # el detector integra la intensidad sobre el area del pixel; el atajo en
    # Fourier (OTF * sinc) debe coincidir con integrar numericamente una PSF
    # sobremuestreada 5x5. Sin blur extra es el caso mas exigente.
    cfg = sim.Config(mascara="ninguna", blur_extra_px=0.0, integrar_pixel=True)
    cfg_punto = sim.Config(mascara="ninguna", blur_extra_px=0.0, integrar_pixel=False)
    for z in (0.0, 400.0):
        rapida = _recorte_central(sim.psf_directa(cfg, z), 32)
        ref = _recorte_central(sim.psf_directa(cfg, z, sobremuestreo=5), 32)
        puntual = _recorte_central(sim.psf_directa(cfg_punto, z), 32)
        err = np.abs(rapida - ref).max() / ref.max()
        err_puntual = np.abs(puntual - ref).max() / ref.max()
        assert err < 5e-3, (z, err)
        # y el efecto es real: muestrear en el centro del pixel se aleja mucho mas
        assert err_puntual > 10 * err, (z, err, err_puntual)


def test_dof_extendido_esferica():
    # proposito de la mascara esferica: mantener el pico a lo largo de z
    def razon(b):
        pico = b.stack.max(axis=(1, 2))
        return pico.min() / pico.max()
    r_esf, r_sin = razon(banco("esferica")), razon(banco("ninguna"))
    assert r_esf >= 0.5, r_esf
    assert r_esf >= 3 * r_sin, (r_esf, r_sin)


def test_corkscrew_rota_con_z():
    # el lobulo principal del corkscrew gira de forma monotona con z (codifica z)
    b = banco("corkscrew")
    zs = np.arange(-800.0, 801.0, 50.0)
    ang = np.unwrap(np.radians([sim.angulo_lobulo(b, z)[0] for z in zs]))
    ang = np.degrees(ang) * np.sign(ang[-1] - ang[0])        # sentido de giro positivo
    assert np.all(np.diff(ang) > -10.0), np.round(ang)
    assert ang[-1] - ang[0] >= 90.0, ang[-1] - ang[0]
