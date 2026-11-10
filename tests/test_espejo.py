"""
Tests con aserciones del modulo de espejo (simulador_psf/espejo.py).

Compatibles con pytest, pero se corren solos (pytest no esta en el entorno):
    $PY tests/test_espejo.py                 # todos
    $PY tests/test_espejo.py involucion 45   # solo los que contienen alguna de esas subcadenas

Convencion: +z hacia el objetivo y angulo del espejo desde el eje optico
(angulo_espejo_eje_deg, beta). Cubren: involucion, punto medio sobre el plano
y p' - p paralelo a n, distancias opuestas, casos analiticos (0 y 45 grados, y
45, 56 y 37 contra la formula general), dist_max_espejo_um, z_ref_espejo_nm,
dp' = R dp y MSD iguales, que ningun emisor real cruza el espejo, fotones del
reflejo = ratio * fotones del real (en el GT y en la imagen de fotones
esperados), que el espejo no consume el rng, la convencion fija (sin signos),
beta -> -beta, que anillos_hacia_objetivo solo cambia la PSF, y con el espejo
apagado la regresion bit a bit.
Las configuraciones usan H != W para que un cambio x <-> y falle.
"""
import functools
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import simulador_psf.generacion as generacion  # noqa: E402
from simulador_psf import (Config, BancoPSF, Espejo, generar_video, generar_trayectorias,  # noqa: E402
                           posiciones_iniciales, rango_z_reflejo, rango_banco_nm, rango_banco_q, a_q,
                           COLUMNAS_GT, COLUMNAS_GT_ESPEJO)

# configuracion pequena y barata (PSF "ninguna", recorte chico, paso de banco grueso)
BASE = dict(H=40, W=56, n_frames=30, mascara="ninguna", K_psf=32, z_paso_nm=50.0,
            n_emisores_rango=(6, 10), D_rango_um2_s=(0.1, 1.0), espejo_punto_px=(12.0, 20.0))
GEOMETRIAS = [(b, f, p) for b in (70.0, 45.0, 56.0, 37.0, 0.0, -45.0)        # beta: angulo desde el eje optico
              for f, p in ((0.0, (12.0, 20.0)), (37.0, (30.0, 5.0)), (200.0, (40.0, 33.0)))]


def cfg_espejo(**kw):
    return Config(**{**BASE, "usar_espejo": True, **kw})


@functools.lru_cache(maxsize=None)
def banco(**kw):
    return BancoPSF(cfg_espejo(**kw))


def puntos(n=500, seed=0):
    """Puntos [px, px, nm] repartidos en la caja (sin importar el lado del espejo)."""
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(-10, 70, n), r.uniform(-10, 50, n), r.uniform(-3000, 3000, n)], 1)


def espejo(b, f, p, **kw):
    return Espejo(cfg_espejo(angulo_espejo_eje_deg=b, orientacion_espejo_deg=f, espejo_punto_px=p, **kw))


# --- geometria pura ---------------------------------------------------------------

def test_base_ortonormal():
    for b, f, p in GEOMETRIAS:
        e = espejo(b, f, p)
        assert np.isclose(np.linalg.norm(e.n), 1.0)
        assert np.isclose(e.n @ e.t, 0.0)                         # la linea del espejo esta en el plano
        assert np.allclose(e.R @ e.R, np.eye(3)) and np.allclose(e.R, e.R.T)
        assert np.isclose(np.linalg.det(e.R), -1.0)              # reflexion, no rotacion
        assert np.isclose(e.distancia_nm(e.p0), 0.0)
        # el angulo entre el espejo y el eje optico es |beta|: |n . z| = sin |beta|
        assert np.isclose(np.degrees(np.arcsin(abs(e.n[2]))), abs(b))


def test_involucion():
    for b, f, p in GEOMETRIAS:
        e = espejo(b, f, p)
        q = puntos()
        assert np.allclose(e.reflejar(e.reflejar(q)), q, rtol=0, atol=1e-9)


def test_punto_medio_en_el_plano_y_paralelo_a_n():
    for b, f, p in GEOMETRIAS:
        e = espejo(b, f, p)
        q = e.a_nm(puntos())
        qr = e.reflejar_nm(q)
        assert np.allclose(e.distancia_nm((q + qr) / 2), 0.0, atol=1e-9)
        dif = qr - q
        assert np.allclose(np.cross(dif, e.n), 0.0, atol=1e-9)    # p' - p paralelo a n


def test_distancias_opuestas():
    for b, f, p in GEOMETRIAS:
        e = espejo(b, f, p)
        q = e.a_nm(puntos())
        assert np.allclose(e.distancia_nm(e.reflejar_nm(q)), -e.distancia_nm(q), atol=1e-9)


def test_analitico_0():
    """Espejo paralelo al eje optico (beta = 0, phi = 0): x' = 2 x0 - x, y' = y, z' = z."""
    cfg = cfg_espejo(angulo_espejo_eje_deg=0.0)
    e, x0 = Espejo(cfg), cfg.espejo_punto_px[0]
    q = puntos()
    qr = e.reflejar(q)
    assert np.allclose(qr[:, 0], 2 * x0 - q[:, 0], atol=1e-9)
    assert np.allclose(qr[:, 1], q[:, 1], atol=1e-9)
    assert np.allclose(qr[:, 2], q[:, 2], atol=1e-9)


def test_analitico_0_orientacion_90():
    """Con phi = 90 la muestra queda hacia +y: el papel de x lo hace y."""
    cfg = cfg_espejo(angulo_espejo_eje_deg=0.0, orientacion_espejo_deg=90.0)
    e, y0 = Espejo(cfg), cfg.espejo_punto_px[1]
    q = puntos()
    qr = e.reflejar(q)
    assert np.allclose(qr[:, 1], 2 * y0 - q[:, 1], atol=1e-9)
    assert np.allclose(qr[:, [0, 2]], q[:, [0, 2]], atol=1e-9)


def test_analitico_45():
    """beta = 45 (el montaje), phi = 0, +z hacia el objetivo:
    z' - z0 = -(x - x0) y x' - x0 = -(z - z0), en nm; y' = y."""
    cfg = cfg_espejo(angulo_espejo_eje_deg=45.0)
    e, px = Espejo(cfg), cfg.pixel_nm
    x0, z0 = cfg.espejo_punto_px[0], cfg.z_ref_espejo_nm
    q = puntos()
    qr = e.reflejar(q)
    assert np.allclose(qr[:, 2] - z0, -(q[:, 0] - x0) * px, atol=1e-6)
    assert np.allclose((qr[:, 0] - x0) * px, -(q[:, 2] - z0), atol=1e-6)
    assert np.allclose(qr[:, 1], q[:, 1], atol=1e-9)


def test_analitico_contra_formula_general():
    """beta = 45 (el montaje, desde el eje optico), 56 y 37, con phi, punto y
    z_ref arbitrarios, contra una construccion independiente: en la base local
    (m, t, z) la normal es (cos b, 0, sin b), y reflejar es aplicar
    [[-cos 2b, -sin 2b], [-sin 2b, cos 2b]] a (u, w); t no cambia. Ademas
    tan(beta) = -dif. de z / separacion lateral (el reflejo, mas lejos del objetivo)."""
    assert Config().angulo_espejo_eje_deg == 45.0
    for bb, f, p, zr in ((45.0, 0.0, (20.0, 32.0), 0.0), (45.0, 37.0, (30.0, 5.0), -700.0),
                         (56.0, 200.0, (40.0, 33.0), 450.0), (37.0, 37.0, (30.0, 5.0), 0.0)):
        cfg = cfg_espejo(angulo_espejo_eje_deg=bb, orientacion_espejo_deg=f, espejo_punto_px=p, z_ref_espejo_nm=zr)
        e = Espejo(cfg)
        be, fr = np.radians(bb), np.radians(f)
        m = np.array([np.cos(fr), np.sin(fr), 0.0]); t = np.array([-np.sin(fr), np.cos(fr), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        p0 = np.array([p[0] * cfg.pixel_nm, p[1] * cfg.pixel_nm, zr])
        q = e.a_nm(puntos())
        rel = q - p0
        u, s, w = rel @ m, rel @ t, rel @ up
        u2 = -np.cos(2 * be) * u - np.sin(2 * be) * w
        w2 = -np.sin(2 * be) * u + np.cos(2 * be) * w
        esperado = p0 + np.outer(u2, m) + np.outer(s, t) + np.outer(w2, up)
        assert np.allclose(e.reflejar_nm(q), esperado, atol=1e-6), bb
        # separacion lateral 2 d cos(beta) y diferencia de z -2 d sin(beta) < 0 con d > 0
        lado = e.distancia_nm(q) > 0
        d, dif = e.distancia_nm(q)[lado], (e.reflejar_nm(q) - q)[lado]
        assert np.allclose(np.hypot(dif[:, 0], dif[:, 1]), 2 * d * np.cos(be), atol=1e-6)
        assert np.allclose(dif[:, 2], -2 * d * np.sin(be), atol=1e-6) and (dif[:, 2] < 0).all()
        assert np.allclose(-dif[:, 2] / np.hypot(dif[:, 0], dif[:, 1]), np.tan(be))
    # numeros del montaje: un emisor a d = 1 um -> dlat = 2 cos(45) = 1.414 um, dz = -2 sin(45) = -1.414 um
    e = Espejo(cfg_espejo())
    q = e.punto_en_z_nm(0.0) + 1000.0 * e.n
    dif = e.reflejar_nm(q) - q
    assert np.isclose(np.hypot(*dif[:2]), 1414.2, atol=0.1) and np.isclose(dif[2], -1414.2, atol=0.1)


def test_plegar_lleva_al_dominio_y_respeta_puntos_dentro():
    for b, f, p in GEOMETRIAS:
        e = espejo(b, f, p)
        q = e.a_nm(puntos(2000, seed=3))
        r = e.plegar_nm(q)
        assert (e.distancia_nm(r) >= -1e-9).all()
        assert (r[:, 2] >= e.z_lo - 1e-9).all() and (r[:, 2] <= e.z_hi + 1e-9).all()
        dentro = (e.distancia_nm(q) >= 0) & (q[:, 2] >= e.z_lo) & (q[:, 2] <= e.z_hi)
        assert np.array_equal(r[dentro], q[dentro])               # lo que ya estaba dentro no se toca


def test_z_ref_espejo():
    """0 por defecto (la linea es la interseccion con el plano focal); un valor
    propio mueve p0 y el caso de 45 grados sigue valiendo con z0 = z_ref_espejo_nm.
    Sin cubreobjetos no hay restriccion: el plano de referencia puede quedar
    dentro o fuera de z_rango_nm, y los reales nunca cruzan el espejo."""
    assert Config().z_ref_espejo_nm == 0.0 and Espejo(cfg_espejo()).p0[2] == 0.0
    for zr in (0.0, -800.0, 300.0, -2500.0, 1500.0):
        cfg = cfg_espejo(angulo_espejo_eje_deg=45.0, z_ref_espejo_nm=zr)
        e, px, x0 = Espejo(cfg), cfg.pixel_nm, cfg.espejo_punto_px[0]
        assert e.p0[2] == zr == e.z_ref
        q = puntos()
        qr = e.reflejar(q)
        assert np.allclose(qr[:, 2] - zr, -(q[:, 0] - x0) * px, atol=1e-6)
        assert np.allclose((qr[:, 0] - x0) * px, -(q[:, 2] - zr), atol=1e-6)
        lo, hi = rango_banco_nm(cfg)
        assert lo <= rango_z_reflejo(cfg)[0] + 1e-6 and hi >= cfg.z_rango_nm[1]
        tray, _ = generar_trayectorias(cfg_espejo(z_ref_espejo_nm=zr, n_frames=100), 100,
                                       np.random.default_rng(0))
        e = Espejo(cfg_espejo(z_ref_espejo_nm=zr))
        assert (e.distancia_nm(e.a_nm(tray)) >= -1e-6).all()
        # punto_en_z_nm: en el plano del espejo, a esa z, y en z_ref es p0
        for z in (-1000.0, 0.0, zr, 2000.0):
            q = e.punto_en_z_nm(z)
            assert np.isclose(e.distancia_nm(q), 0.0, atol=1e-9) and q[2] == z
        assert np.allclose(e.punto_en_z_nm(zr), e.p0)
    # 45 grados: acercarse dz al objetivo desplaza la linea dz alejandose de la muestra
    e = Espejo(cfg_espejo(angulo_espejo_eje_deg=45.0, z_ref_espejo_nm=-300.0))
    assert np.allclose(e.punto_en_z_nm(700.0) - e.p0, [-1000.0, 0.0, 1000.0])


def test_dist_max_espejo():
    """dist_max_espejo_um es la distancia maxima al PLANO del espejo: reescala
    los mismos sorteos (el rng termina igual que sin el), z es la misma, todos
    los puntos quedan con d en [0, dist_max] sin plegar, y la posicion es la
    construccion esperada: s_t del sorteo de y, s_m del sorteo de x en la franja
    d en [0, dist_max] a esa z recortada al FOV ampliado proyectado sobre m."""
    from simulador_psf.config import MARGEN_PX
    n, dmax = 4000, 1.5
    for b, f, p in GEOMETRIAS:
        for zr in (0.0, -500.0):
            kw = dict(angulo_espejo_eje_deg=b, orientacion_espejo_deg=f, espejo_punto_px=p, z_ref_espejo_nm=zr)
            cfg = cfg_espejo(dist_max_espejo_um=dmax, **kw)
            e, px = Espejo(cfg), cfg.pixel_nm
            r1, r2, r3 = (np.random.default_rng(7) for _ in range(3))
            pos = posiciones_iniciales(cfg, n, r1)
            posiciones_iniciales(cfg_espejo(**kw), n, r2)
            crudo = posiciones_iniciales(Config(**BASE), n, r3)     # sin espejo: los sorteos sin plegar
            assert r1.bit_generator.state == r2.bit_generator.state == r3.bit_generator.state
            assert np.array_equal(pos[:, 2], crudo[:, 2])           # z: el mismo sorteo
            d = e.distancia_nm(pos)
            assert (d >= -1e-6).all() and (d <= dmax * 1000 + 1e-6).all(), (b, f, d.min(), d.max())
            # construccion esperada desde los sorteos uniformes de la version sin espejo
            U1 = (crudo[:, 0] / px + MARGEN_PX) / (cfg.W + 2 * MARGEN_PX)
            U2 = (crudo[:, 1] / px + MARGEN_PX) / (cfg.H + 2 * MARGEN_PX)
            esq = np.array([[x, y, 0.0] for x in (-MARGEN_PX, cfg.W + MARGEN_PX)
                            for y in (-MARGEN_PX, cfg.H + MARGEN_PX)]) * [px, px, 0] - [*e.p0[:2], 0]
            st, sm = esq @ e.t, esq @ e.m
            s_t = st.min() + U2 * (st.max() - st.min())
            # franja por la recta p0 + s m + s_t t a la altura z: d(s) = d(0) + s (n . m)
            base = np.column_stack([e.p0[:2] + np.outer(s_t, e.t[:2]), crudo[:, 2]])
            d0, pend = e.distancia_nm(base), e.n @ e.m
            f0, f1 = -d0 / pend, (dmax * 1000 - d0) / pend
            lo, hi = np.maximum(f0, sm.min()), np.minimum(f1, sm.max())
            vacia = lo > hi
            lo[vacia], hi[vacia] = f0[vacia], f1[vacia]
            esperado = base.copy()
            esperado[:, :2] += np.outer(lo + U1 * (hi - lo), e.m[:2])
            assert np.allclose(pos, esperado, rtol=0, atol=1e-6)
            # dentro del FOV ampliado salvo donde la franja no lo toca
            s_m = (pos[:, :2] - e.p0[:2]) @ e.m[:2]
            assert ((s_m >= sm.min() - 1e-6) & (s_m <= sm.max() + 1e-6))[~vacia].all()
            if b == 45.0 and f == 0.0 and zr == 0.0:                # el caso del montaje: la franja siempre toca el FOV
                assert not vacia.any()
    # con orientacion 0 el y es el de siempre
    cfg = cfg_espejo(dist_max_espejo_um=dmax)
    pos = posiciones_iniciales(cfg, n, np.random.default_rng(3))
    crudo = posiciones_iniciales(Config(**BASE), n, np.random.default_rng(3))
    assert np.allclose(pos[:, 1], crudo[:, 1], rtol=0, atol=1e-6)


def test_config_valida():
    for kw in (dict(angulo_espejo_eje_deg=90.0), dict(angulo_espejo_eje_deg=-90.0), dict(angulo_espejo_eje_deg=95.0),
               dict(ratio_reflejo=0.0), dict(ratio_reflejo=1.01), dict(espejo_punto_px=(1.0,)),
               dict(z_banco_nm=(100.0, -100.0)), dict(dist_max_espejo_um=0.0), dict(dist_max_espejo_um=-1.0)):
        try:
            Config(**kw)
        except ValueError:
            continue
        raise AssertionError(f"Config({kw}) no lanzo ValueError")
    Config(ratio_reflejo=1.0, angulo_espejo_eje_deg=89.9)        # bordes validos
    Config(angulo_espejo_eje_deg=-89.9)


# --- banco ---------------------------------------------------------------------------

def test_rango_banco():
    assert rango_banco_nm(Config()) == Config().z_rango_nm       # sin espejo: el de siempre
    assert rango_banco_nm(Config(z_banco_nm=(-1000.0, 2000.0))) == (-1000.0, 2000.0)
    for b, f, p in GEOMETRIAS:
        cfg = cfg_espejo(angulo_espejo_eje_deg=b, orientacion_espejo_deg=f, espejo_punto_px=p)
        lo, hi = rango_banco_nm(cfg)
        r1, r2 = rango_z_reflejo(cfg)
        assert lo <= min(r1, cfg.z_rango_nm[0]) + 1e-6 and hi >= max(r2, cfg.z_rango_nm[1]) - 1e-6
        assert hi - lo <= (max(r2, cfg.z_rango_nm[1]) - min(r1, cfg.z_rango_nm[0])) + 2 * cfg.z_paso_nm
        # rejilla (en q = -z) alineada con la del banco sin espejo
        ql, qbase = rango_banco_q(cfg)[0], a_q(cfg, tuple(cfg.z_rango_nm))[0]
        assert np.isclose(np.mod(ql - qbase, cfg.z_paso_nm), 0.0) or \
            np.isclose(np.mod(ql - qbase, cfg.z_paso_nm), cfg.z_paso_nm)
        # los reflejos de las posiciones iniciales caen en ese rango
        e = Espejo(cfg)
        q0 = posiciones_iniciales(cfg, 3000, np.random.default_rng(1))
        zr = e.reflejar_nm(q0)[:, 2]
        assert (zr >= r1 - 1e-6).all() and (zr <= r2 + 1e-6).all()


# --- trayectorias ------------------------------------------------------------------

def test_ningun_real_cruza_el_espejo():
    """Posiciones iniciales y todos los frames del lado de la muestra y dentro de
    las paredes en z, con D alto para forzar muchos choques."""
    for b, f, p in GEOMETRIAS:
        cfg = cfg_espejo(angulo_espejo_eje_deg=b, orientacion_espejo_deg=f, espejo_punto_px=p,
                         n_frames=200, D_rango_um2_s=(1.0, 3.0))
        e = Espejo(cfg)
        tray, _ = generar_trayectorias(cfg, 200, np.random.default_rng(int(b) + int(f) + 100))
        assert (e.distancia_nm(e.a_nm(tray)) >= -1e-6).all(), (b, f, p)
        assert (tray[..., 2] >= cfg.z_rango_nm[0] - 1e-9).all() and (tray[..., 2] <= cfg.z_rango_nm[1] + 1e-9).all()


def test_espejo_lejano_no_cambia_trayectorias():
    """Un espejo que nadie toca da las mismas trayectorias que sin espejo (salvo
    redondeo: con espejo se integra paso a paso), con el mismo rng. Las paredes
    en z se alejan para que nadie las toque: con espejo se refleja la posicion
    en cada paso y sin el se pliega la suma acumulada (que refleja los pasos
    siguientes), asi que tras un choque los caminos difieren aunque la
    distribucion sea la misma."""
    lejos = dict(z_rango_nm=(-1e7, 1e7))
    cfg_off = Config(**{**BASE, **lejos})
    cfg_on = cfg_espejo(angulo_espejo_eje_deg=0.0, espejo_punto_px=(-1e5, 0.0), **lejos)
    t_off, D_off = generar_trayectorias(cfg_off, 20, np.random.default_rng(5))
    t_on, D_on = generar_trayectorias(cfg_on, 20, np.random.default_rng(5))
    assert np.array_equal(D_off, D_on)
    assert np.allclose(t_off, t_on, rtol=0, atol=1e-6)


def _video(seed=2, **kw):
    cfg = cfg_espejo(**kw)
    v, filas = generar_video(cfg, banco(**kw), np.random.default_rng(seed), 0)
    return cfg, v, pd.DataFrame(filas)


def test_gt_columnas_y_banderas():
    cfg, video, gt = _video()
    assert tuple(gt.columns) == COLUMNAS_GT_ESPEJO
    assert video.shape == (cfg.n_frames, cfg.H, cfg.W) and video.dtype == np.uint16
    esp = (gt.x_reflejo_px >= 0) & (gt.x_reflejo_px < cfg.W) & (gt.y_reflejo_px >= 0) & (gt.y_reflejo_px < cfg.H)
    assert (gt.reflejo_en_fov == esp).all()
    z = banco().z_grid
    assert (gt.reflejo_en_banco == ((gt.z_reflejo_nm >= z[0]) & (gt.z_reflejo_nm <= z[-1]))).all()
    assert gt[gt.frame == 0].reflejo_en_banco.all()               # el banco cubre la caja inicial


def test_movimiento_ligado_y_msd():
    """dp' = R dp en todos los frames y MSD 3D (por retardo) igual en real y reflejo."""
    for kw in (dict(), dict(angulo_espejo_eje_deg=0.0), dict(angulo_espejo_eje_deg=-30.0, orientacion_espejo_deg=200.0,
                                                             espejo_punto_px=(40.0, 33.0))):
        cfg, _, gt = _video(**kw)
        e = Espejo(cfg)
        for _, g in gt.groupby("particle_id"):
            g = g.sort_values("frame")
            p = e.a_nm(g[["x_px", "y_px", "z_nm"]].to_numpy())
            pr = e.a_nm(g[["x_reflejo_px", "y_reflejo_px", "z_reflejo_nm"]].to_numpy())
            dp, dpr = np.diff(p, axis=0), np.diff(pr, axis=0)
            assert np.allclose(dpr, dp @ e.R.T, atol=1e-6)
            for lag in range(1, cfg.n_frames // 2):
                msd = np.mean(np.sum((p[lag:] - p[:-lag])**2, 1))
                msd_r = np.mean(np.sum((pr[lag:] - pr[:-lag])**2, 1))
                assert np.isclose(msd, msd_r, rtol=1e-9, atol=1e-6)


def test_fotones_del_reflejo():
    """photons_reflejo = ratio * photons en cada frame (0.8 y otros), y la imagen
    de fotones esperados que entra a la camara es exactamente la suma de reales,
    reflejos (con esos fotones) y un fondo fijo en el video."""
    for ratio in (0.8, 0.35, 1.0):
        capturas = []
        original = generacion.aplicar_camara

        def espia(fotones, cfg, rng):
            capturas.append(fotones.copy())
            return original(fotones, cfg, rng)
        generacion.aplicar_camara = espia
        try:
            cfg, _, gt = _video(ratio_reflejo=ratio)
        finally:
            generacion.aplicar_camara = original
        assert np.allclose(gt.photons_reflejo, ratio * gt.photons, rtol=1e-12)
        b = banco(ratio_reflejo=ratio)
        fondos = []
        for t, g in gt.groupby("frame"):
            img = np.zeros((cfg.H, cfg.W))
            for r in g.itertuples():
                b.colocar(img, r.x_px, r.y_px, r.z_nm, r.photons)
                if r.reflejo_en_banco:
                    b.colocar(img, r.x_reflejo_px, r.y_reflejo_px, r.z_reflejo_nm, r.photons_reflejo)
            fondos.append(capturas[t] - img)
        assert len(capturas) == cfg.n_frames                      # el ruido se aplica una vez por frame
        for f in fondos[1:]:
            assert np.allclose(f, fondos[0], atol=1e-9)           # lo que sobra es el fondo, fijo


def test_espejo_no_consume_rng():
    """El espejo no sortea nada: al llegar a la camara (paso 7 del orden de
    generacion.py) el rng esta en el mismo estado con y sin espejo, y n, D,
    fotones y fondo son los mismos. Despues ya no: rng.poisson consume una
    cantidad de numeros que depende de la imagen, y con reflejos la imagen es
    otra (igual que si cambian los fotones)."""
    estados = {}
    original = generacion.aplicar_camara

    def espia(clave):
        def f(fotones, cfg, rng):
            estados.setdefault(clave, rng.bit_generator.state)
            return original(fotones, cfg, rng)
        return f
    cfg_off = Config(**BASE)
    try:
        generacion.aplicar_camara = espia("off")
        _, f_off = generar_video(cfg_off, BancoPSF(cfg_off), np.random.default_rng(9), 0)
        generacion.aplicar_camara = espia("on")
        _, f_on = generar_video(cfg_espejo(), banco(), np.random.default_rng(9), 0)
    finally:
        generacion.aplicar_camara = original
    assert estados["off"] == estados["on"]
    g_off, g_on = pd.DataFrame(f_off), pd.DataFrame(f_on)
    assert np.array_equal(g_off.particle_id, g_on.particle_id)
    assert np.array_equal(g_off.photons, g_on.photons) and np.array_equal(g_off.D_um2_s, g_on.D_um2_s)


def test_espejo_apagado_ignora_sus_parametros():
    """Con usar_espejo=False los demas parametros del espejo no cambian nada."""
    cfg_a = Config(**BASE)
    cfg_b = Config(**{**BASE, "angulo_espejo_eje_deg": -30.0, "orientacion_espejo_deg": 33.0,
                      "espejo_punto_px": (1.0, 2.0), "ratio_reflejo": 0.1, "z_ref_espejo_nm": -3000.0,
                      "dist_max_espejo_um": 2.0})
    va, fa = generar_video(cfg_a, BancoPSF(cfg_a), np.random.default_rng(4), 0)
    vb, fb = generar_video(cfg_b, BancoPSF(cfg_b), np.random.default_rng(4), 0)
    assert np.array_equal(va, vb) and fa == fb
    assert tuple(pd.DataFrame(fa).columns) == COLUMNAS_GT
    assert cfg_a.resumen() == cfg_b.resumen() and "espejo" not in cfg_a.resumen()


# --- convencion: sin signos, beta -> -beta, anillos -----------------------------------

def test_convencion_fija_sin_signos():
    """+z hacia el objetivo y el angulo desde el eje optico son fijos: Config y
    la geometria no tienen campos de signo de la convencion anterior, ni hay
    constantes de modulo SIGNO*; la unica bandera es la de la PSF."""
    import inspect
    import simulador_psf.espejo as m_espejo
    import simulador_psf.posiciones as m_pos
    from geometria_espejo import GeometriaEspejo
    cfg = Config()
    viejos = ("signo" + "_z", "signo" + "_inclinacion", "angulo_espejo" + "_deg")
    assert not any(hasattr(cfg, c) for c in viejos)
    assert cfg.angulo_espejo_eje_deg == 45.0 and cfg.anillos_hacia_objetivo is False and cfg.signo_q == -1.0
    assert list(inspect.signature(GeometriaEspejo).parameters) == ["beta_deg", "phi_deg", "p0_nm", "z_lo", "z_hi"]
    for mod in (m_espejo, m_pos, generacion):
        assert not any(k.startswith("SIGNO") for k in vars(mod)), mod.__name__


def test_reflejo_del_lado_correcto_en_plegado_muestreo_y_gt():
    """Con beta > 0 (el montaje) el reflejo queda siempre mas lejos del objetivo
    (z' < z) y el banco se amplia hacia -z; con beta < 0, al reves. El muestreo
    con dist_max (d en [0, dist_max]), el plegado (d >= 0), el browniano
    (ningun real cruza) y el reflejo del GT (= Espejo(cfg).reflejar) usan todos
    el mismo espejo: si alguno usara otro, sus d saldrian negativas."""
    dmax = 2.0
    for b, f, p in GEOMETRIAS[::4]:
        kw = dict(angulo_espejo_eje_deg=b, orientacion_espejo_deg=f, espejo_punto_px=p)
        e = Espejo(cfg_espejo(**kw))
        assert np.isclose(e.n[2], np.sin(np.radians(b)))
        pos = posiciones_iniciales(cfg_espejo(dist_max_espejo_um=dmax, **kw), 400, np.random.default_rng(1))
        d = e.distancia_nm(pos)
        assert (d >= -1e-6).all() and (d <= dmax * 1000 + 1e-6).all(), (b, f)
        pos = posiciones_iniciales(cfg_espejo(**kw), 400, np.random.default_rng(2))
        assert (e.distancia_nm(pos) >= -1e-6).all(), (b, f)
        tray, _ = generar_trayectorias(cfg_espejo(n_frames=100, D_rango_um2_s=(1.0, 3.0), **kw), 100,
                                       np.random.default_rng(3))
        assert (e.distancia_nm(e.a_nm(tray)) >= -1e-6).all(), (b, f)
    for b in (45.0, -30.0):
        cfg, _, gt = _video(seed=4, angulo_espejo_eje_deg=b, dist_max_espejo_um=dmax)
        e = Espejo(cfg)
        real = gt[["x_px", "y_px", "z_nm"]].to_numpy()
        refl = gt[["x_reflejo_px", "y_reflejo_px", "z_reflejo_nm"]].to_numpy()
        assert np.allclose(refl, e.reflejar(real), rtol=0, atol=1e-9)
        dz = (gt.z_reflejo_nm - gt.z_nm).to_numpy()
        assert (np.sign(b) * dz <= 1e-6).all() and (np.abs(dz) > 1.0).any(), b
        lo, hi = rango_banco_nm(cfg)
        assert (lo < cfg.z_rango_nm[0]) == (b > 0) and (hi > cfg.z_rango_nm[1]) == (b < 0)
        assert gt[gt.frame == 0].reflejo_en_banco.all()


def test_angulo_negativo_es_el_espejo_en_z():
    """Espejar toda la configuracion en z respecto de z_ref (z -> 2 z_ref - z)
    lleva el espejo de beta al de -beta: reflejar con -beta es S R_beta S."""
    for b, f, p in GEOMETRIAS:
        for zr in (0.0, -400.0):
            e_mas, e_menos = espejo(b, f, p, z_ref_espejo_nm=zr), espejo(-b, f, p, z_ref_espejo_nm=zr)
            q = e_mas.a_nm(puntos())

            def S(r):
                return np.column_stack([r[:, 0], r[:, 1], 2 * zr - r[:, 2]])
            assert np.allclose(e_menos.reflejar_nm(q), S(e_mas.reflejar_nm(S(q))), atol=1e-6), (b, f)


def test_anillos_solo_cambian_la_psf():
    """anillos_hacia_objetivo no toca trayectorias, reflejos, fotones ni fondo
    (mismo GT), solo que PSF se dibuja: la de True en z es la de False en -z.
    Sin mascara la PSF escalar es simetrica en q y la bandera no cambia la
    imagen; con la esferica si."""
    for mascara, K in (("ninguna", 32), ("esferica", 64)):
        cfg_f = cfg_espejo(mascara=mascara, K_psf=K)
        cfg_t = cfg_espejo(mascara=mascara, K_psf=K, anillos_hacia_objetivo=True)
        bf, bt = BancoPSF(cfg_f), BancoPSF(cfg_t)
        assert cfg_t.signo_q == 1.0
        for z in (-777.7, -517.0, 0.0, 123.4, 700.0):             # dentro de z_rango: los dos bancos tienen esos planos
            assert np.array_equal(bt.psf_en_z(z), bf.psf_en_z(-z)), z
        vf, ff = generar_video(cfg_f, bf, np.random.default_rng(3), 0)
        vt, ft = generar_video(cfg_t, bt, np.random.default_rng(3), 0)
        gf, gt_ = pd.DataFrame(ff), pd.DataFrame(ft)
        assert gf.equals(gt_)
        assert np.array_equal(vf, vt) == (mascara == "ninguna"), mascara
        assert "anillos hacia el objetivo" in cfg_t.resumen() and "anillos" not in cfg_f.resumen()


def test_regresion_bit_a_bit():
    """Espejo apagado (por defecto) -> tests/regresion.py identico a la referencia."""
    proc = subprocess.run([sys.executable, str(RAIZ / "tests" / "regresion.py")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "IDENTICO" in proc.stdout


if __name__ == "__main__":
    import time
    import traceback
    filtros = sys.argv[1:]
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    tests = [(k, v) for k, v in tests if not filtros or any(f in k for f in filtros)]
    fallos = 0
    for nombre, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"  [OK  ] {nombre}  ({time.time() - t0:.1f}s)")
        except Exception:
            fallos += 1
            print(f"  [FAIL] {nombre}")
            traceback.print_exc()
    print(f"\n{len(tests) - fallos}/{len(tests)} tests OK")
    sys.exit(1 if fallos else 0)
