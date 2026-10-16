"""Tests de BancoPSF.colocar: convencion de ejes y de centro de pixel,
conservacion de fotones, exactitud del desplazamiento subpixel contra la PSF
calculada directamente, linealidad y bordes."""
import numpy as np

from comun import sim, banco, centroide, MASCARAS


def test_ejes_x_y_no_intercambiados():
    b = banco("ninguna")
    img = np.zeros((40, 80))                     # H = 40 filas (y), W = 80 columnas (x)
    b.colocar(img, x=60.0, y=10.0, z_nm=0.0, N=1000.0)
    assert np.unravel_index(img.argmax(), img.shape) == (10, 60)


def test_convencion_centro_de_pixel():
    # el centro del pixel (fila i, columna j) es la coordenada continua (x=j, y=i)
    for m in ("ninguna", "esferica"):
        b = banco(m)
        L = b.K + 40
        x, y, h = 70.0, 75.0, 15
        img = np.zeros((L, L))
        b.colocar(img, x, y, 0.0, 1000.0)
        iy, ix = int(y), int(x)
        assert np.unravel_index(img.argmax(), img.shape) == (iy, ix), m
        v = img[iy - h: iy + h + 1, ix - h: ix + h + 1]
        assert np.abs(v - v[::-1, ::-1]).max() < 1e-9 * v.max(), m
        # a medio pixel la PSF queda repartida igual entre las dos columnas
        img = np.zeros((L, L))
        b.colocar(img, x + 0.5, y, 0.0, 1000.0)
        assert abs(img[iy, ix] - img[iy, ix + 1]) < 1e-3 * img.max(), m


def test_conservacion_de_fotones_subpixel():
    # el desplazamiento en Fourier no cambia la suma (termino DC): la energia
    # colocada es N * (suma de la PSF recortada a ese z), sin importar el offset
    rng = np.random.default_rng(0)
    for m in MASCARAS:
        b = banco(m)
        L = b.K + 20
        for _ in range(10):
            x, y = L / 2 + rng.uniform(-1, 1, 2)
            z = rng.uniform(*b.cfg.z_rango_nm)
            img = np.zeros((L, L))
            b.colocar(img, x, y, z, 1234.0)
            esperado = 1234.0 * b.psf_en_z(z).sum()
            assert np.isclose(img.sum(), esperado, rtol=1e-10), (m, x, y, z)


def test_desplazamiento_subpixel_vs_psf_exacta():
    # colocar() desplaza el recorte con una fase lineal en Fourier; la
    # referencia desplaza el campo en la pupila antes de |.|^2 (exacto)
    for m in MASCARAS:
        b = banco(m)
        K, c = b.K, b.cfg.M_pupila // 2
        z = b.z_grid[50]
        for dx, dy in [(0.25, 0.5), (0.73, 0.1), (0.5, 0.5)]:
            img = np.zeros((K, K))
            b.colocar(img, K // 2 + dx, K // 2 + dy, z, 1.0)
            ref = sim.psf_directa(b.cfg, z, dx, dy)[c - K // 2: c + K // 2, c - K // 2: c + K // 2]
            err = np.abs(img - ref).max() / ref.max()
            assert err < 2e-3, (m, dx, dy, err)


def test_centroide_subpixel_igual_al_de_la_psf_exacta():
    # el centroide de toda la imagen lo dominan las colas truncadas (brazo de
    # palanca de hasta K/2 px): incluso la PSF exacta se corre ~0.01 px con el
    # halo de la esferica. Lo que se exige es que colocar() no agregue sesgo
    # de localizacion respecto a la PSF exacta desplazada.
    for m in ("ninguna", "esferica"):
        b = banco(m)
        K, c = b.K, b.cfg.M_pupila // 2
        for off in np.linspace(0.0, 0.9, 7):
            dx, dy = off, 0.3 * off
            img = np.zeros((K, K))
            b.colocar(img, K // 2 + dx, K // 2 + dy, 0.0, 1.0)
            ref = sim.psf_directa(b.cfg, 0.0, dx, dy)[c - K // 2: c + K // 2, c - K // 2: c + K // 2]
            (cx, cy), (rx, ry) = centroide(img), centroide(ref)
            assert np.hypot(cx - rx, cy - ry) < 2e-3, (m, off, cx - rx, cy - ry)


def test_equivariancia_ante_traslaciones_enteras():
    b = banco("esferica")
    a, c = np.zeros((300, 300)), np.zeros((300, 300))
    b.colocar(a, 150.3, 147.8, 120.0, 500.0)
    b.colocar(c, 153.3, 145.8, 120.0, 500.0)
    assert np.abs(np.roll(a, (-2, 3), axis=(0, 1)) - c).max() < 1e-12 * a.max()


def test_superposicion_lineal():
    b = banco("esferica")
    a, c, ambos = np.zeros((90, 70)), np.zeros((90, 70)), np.zeros((90, 70))
    b.colocar(a, 30.2, 40.7, -300.0, 800.0)
    b.colocar(c, 35.9, 44.1, 250.0, 1500.0)
    b.colocar(ambos, 30.2, 40.7, -300.0, 800.0)
    b.colocar(ambos, 35.9, 44.1, 250.0, 1500.0)
    assert np.allclose(ambos, a + c, rtol=0, atol=1e-12 * ambos.max())


def test_bordes_y_fuera_del_campo():
    b = banco("ninguna")
    H, W, N = 32, 48, 1000.0
    img = np.zeros((H, W))
    b.colocar(img, -0.4, 10.0, 0.0, N)           # en el borde: entra solo una parte
    assert 0.2 * N < img.sum() < 0.9 * N
    for x, y in [(-200.0, 10.0), (W + 200.0, 10.0), (10.0, -200.0), (10.0, H + 200.0)]:
        img = np.zeros((H, W))
        b.colocar(img, x, y, 0.0, N)             # muy lejos: no aporta nada y no falla
        assert img.sum() == 0.0, (x, y)


def test_cero_fotones_no_cambia_la_imagen():
    b = banco("ninguna")
    img = np.zeros((32, 32))
    b.colocar(img, 16.3, 15.2, 0.0, 0.0)
    assert not img.any()


def test_valores_negativos_despreciables():
    # el desplazamiento en Fourier de un recorte puede dejar oscilaciones
    # negativas minimas; deben ser despreciables frente al pico
    for m in MASCARAS:
        b = banco(m)
        img = np.zeros((b.K + 10, b.K + 10))
        b.colocar(img, b.K / 2 + 5.37, b.K / 2 + 5.81, 410.0, 1.0)
        assert img.min() > -1e-4 * img.max(), (m, img.min() / img.max())
