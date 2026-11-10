"""Tests del espejo en simulador_nuevo (usar_espejo=True): configuracion,
banco ampliado, trayectorias (el espejo es una pared), reflejo en el GT y en
la imagen (con desenfoque de movimiento ligado al del real), flujos
aleatorios, convencion de signos y que con el espejo apagado sus parametros
no cambian nada. La geometria es la de geometria_espejo.py (compartida con
simulador_psf); test_misma_geometria_que_simulador_psf lo comprueba."""
import json
from functools import lru_cache

import numpy as np

from comun import sim, espera_error

# H != W a proposito; PSF "ninguna" con recorte chico y banco grueso: barato
BASE = dict(mascara="ninguna", H=32, W=40, n_frames=6, n_videos=2, K_psf=32, z_paso_nm=50.0,
            n_emisores_rango=(5, 9), D_rango_um2_s=(0.1, 1.0), espejo_punto_px=(10.0, 16.0), seed=21)
SIGNOS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]


def cfg_e(**kw):
    return sim.Config(**{**BASE, "usar_espejo": True, **kw})


@lru_cache(maxsize=None)
def _banco(clave):
    return sim.BancoPSF(cfg_e(**dict(clave)))


def banco_e(**kw):
    """BancoPSF cacheado de cfg_e(**kw) (solo importan los campos que lo definen)."""
    return _banco(tuple(sorted(kw.items())))


def video(video_id=0, **kw):
    cfg = cfg_e(**kw)
    v, gt, ex = sim.generar_video(cfg, banco_e(**kw), video_id, extra=True)
    return cfg, v, gt, ex


# --- configuracion ----------------------------------------------------------

def test_config_valida_el_espejo():
    for kw in (dict(angulo_espejo_deg=0.0), dict(angulo_espejo_deg=180.0), dict(ratio_reflejo=0.0),
               dict(ratio_reflejo=1.5), dict(espejo_punto_px=(1.0, 2.0, 3.0)), dict(dist_max_espejo_um=0.0),
               dict(signo_z=0), dict(signo_inclinacion=2), dict(z_banco_nm=(100.0, -100.0))):
        espera_error(ValueError, cfg_e, **kw)
    c = sim.Config()
    assert not c.usar_espejo and c.angulo_espejo_deg == 34.0 and c.signo_z == 1 and c.signo_inclinacion == 1


def test_config_json_con_y_sin_espejo():
    """Con espejo todos sus campos van y vuelven (tuplas incluidas); sin espejo
    se omiten del JSON (queda el de antes) y vuelven con su valor por defecto."""
    cfg = cfg_e(angulo_espejo_deg=50.0, orientacion_espejo_deg=200.0, dist_max_espejo_um=3.0,
                signo_z=-1, z_banco_nm=(-3000.0, 3000.0), z_ref_espejo_nm=120.0)
    assert sim.Config.desde_dict(json.loads(json.dumps(cfg.a_dict()))) == cfg
    off = sim.Config(**{k: v for k, v in BASE.items() if k != "espejo_punto_px"})
    d = off.a_dict()
    assert not set(sim.CAMPOS_ESPEJO) & set(d) and "z_banco_nm" not in d
    assert sim.Config.desde_dict(json.loads(json.dumps(d))) == off
    # sin espejo sus parametros no se guardan: vuelven con el valor por defecto (no cambian nada)
    vuelta = sim.Config.desde_dict(sim.Config(**BASE).a_dict())
    assert vuelta.espejo_punto_px == sim.Config().espejo_punto_px
    assert "espejo" not in off.resumen() and "espejo" in cfg.resumen()


def test_misma_geometria_que_simulador_psf():
    """La clase de geometria es la misma: con los mismos parametros, simulador_psf
    y simulador_nuevo dan la misma normal, p0 y reflexion."""
    import simulador_psf as sp                                   # la raiz ya esta en sys.path
    from geometria_espejo import GeometriaEspejo
    for a, f, sz in ((34.0, 0.0, 1), (53.0, 37.0, -1), (120.0, 200.0, 1)):
        kw = dict(angulo_espejo_deg=a, orientacion_espejo_deg=f, espejo_punto_px=(12.0, 20.0),
                  z_ref_espejo_nm=-300.0, signo_z=sz)
        en, ep = sim.Espejo(cfg_e(**kw)), sp.Espejo(sp.Config(usar_espejo=True, **kw))
        assert isinstance(en, GeometriaEspejo) and type(en).__mro__[1] is type(ep).__mro__[1]
        assert np.array_equal(en.n, ep.n) and np.array_equal(en.p0, ep.p0)
        q = np.random.default_rng(0).uniform(-50, 50, (100, 3)) * [1, 1, 40]
        assert np.array_equal(en.reflejar(q), ep.reflejar(q))


# --- banco ------------------------------------------------------------------

def test_banco_ampliado_conserva_los_planos_de_z_rango():
    """El banco con espejo cubre rango_z_reflejo hacia el lado que toca, y sus
    planos dentro de z_rango_nm (y sus PSF) son los del banco sin espejo."""
    off = sim.BancoPSF(sim.Config(**BASE))
    for sz in (1, -1):
        cfg = cfg_e(signo_z=sz)
        b = banco_e(signo_z=sz)
        r1, r2 = sim.rango_z_reflejo(cfg)
        assert b.z_grid[0] <= r1 + 1e-6 and b.z_grid[-1] >= r2 - 1e-6
        assert (b.z_grid[0] < cfg.z_rango_nm[0]) == (sz < 0) and (b.z_grid[-1] > cfg.z_rango_nm[1]) == (sz > 0)
        dentro = (b.z_grid >= cfg.z_rango_nm[0]) & (b.z_grid <= cfg.z_rango_nm[1])
        assert np.array_equal(b.z_grid[dentro], off.z_grid) and np.array_equal(b.stack[dentro], off.stack)
        assert np.allclose(np.diff(b.z_grid), np.diff(off.z_grid)[0])
    # z_banco_nm explicito: cubre ese rango (y siempre z_rango_nm)
    zb = sim.rejilla_z(cfg_e(z_banco_nm=(-2000.0, 100.0)))
    assert zb[0] <= -2000.0 and zb[-1] >= BASE.get("z_rango_nm", (-800.0, 800.0))[1]


# --- trayectorias -----------------------------------------------------------

def test_ningun_real_cruza_el_espejo_ni_en_los_subpasos():
    for a, f in ((20.0, 0.0), (34.0, 37.0), (90.0, 200.0), (135.0, 300.0)):
        cfg = cfg_e(angulo_espejo_deg=a, orientacion_espejo_deg=f, n_frames=60, n_subpasos=3,
                    t_exp_s=0.06, D_rango_um2_s=(1.0, 3.0))
        e = sim.Espejo(cfg)
        tray, sub, _ = sim.generar_trayectorias(cfg, 80, np.random.default_rng(int(a)))
        assert (e.distancia_nm(e.a_nm(sub)) >= -1e-6).all(), (a, f)
        assert (sub[..., 2] >= cfg.z_rango_nm[0] - 1e-9).all() and (sub[..., 2] <= cfg.z_rango_nm[1] + 1e-9).all()
        assert np.allclose(tray, sub.mean(axis=2))


def test_dist_max_y_plegado_de_las_posiciones_iniciales():
    for dmax in (None, 1.0, 3.0):
        cfg = cfg_e(dist_max_espejo_um=dmax, n_frames=1)
        e = sim.Espejo(cfg)
        _, sub, _ = sim.generar_trayectorias(cfg, 500, np.random.default_rng(3))
        d = e.distancia_nm(e.a_nm(sub[:, 0, 0]))
        assert (d >= -1e-6).all()
        if dmax is not None:
            assert (d <= dmax * 1000 + 1e-6).all()
        if dmax == 1.0:                                          # con 3 um la franja se recorta al FOV
            assert d.max() > 0.8 * dmax * 1000


def test_espejo_lejano_no_cambia_trayectorias():
    """Un espejo que nadie toca da las mismas trayectorias que sin espejo (salvo
    redondeo: se integra paso a paso en nm), con el mismo rng: mismos sorteos."""
    lejos = dict(z_rango_nm=(-1e7, 1e7), n_subpasos=2, t_exp_s=0.05)
    off = sim.Config(**{**BASE, **lejos})
    on = cfg_e(angulo_espejo_deg=90.0, espejo_punto_px=(-1e5, 0.0), **lejos)
    t0, s0, D0 = sim.generar_trayectorias(off, 20, np.random.default_rng(5))
    t1, s1, D1 = sim.generar_trayectorias(on, 20, np.random.default_rng(5))
    assert np.array_equal(D0, D1) and np.allclose(s0, s1, rtol=0, atol=1e-6)


# --- reflejo en el GT y en la imagen ------------------------------------------

def test_gt_del_reflejo():
    cfg, v, gt, ex = video(n_subpasos=3, t_exp_s=0.05)
    assert list(gt.columns) == sim.COLUMNAS_GT_ESPEJO
    e = sim.Espejo(cfg)
    real = gt[["x_px", "y_px", "z_nm"]].to_numpy()
    refl = gt[["x_reflejo_px", "y_reflejo_px", "z_reflejo_nm"]].to_numpy()
    assert np.allclose(refl, e.reflejar(real), rtol=0, atol=1e-9)        # media de subposiciones = reflejo de la media
    assert np.allclose(gt.photons_reflejo, cfg.ratio_reflejo * gt.photons, rtol=1e-15, atol=0)
    assert (gt.reflejo_en_fov == sim.en_fov(cfg, gt.x_reflejo_px, gt.y_reflejo_px)).all()
    zg = banco_e(n_subpasos=3, t_exp_s=0.05).z_grid
    assert (gt.reflejo_en_banco <= ((gt.z_reflejo_nm >= zg[0]) & (gt.z_reflejo_nm <= zg[-1]))).all()
    assert gt[gt.frame == 0].reflejo_en_banco.all()                       # el banco cubre la caja inicial
    assert (gt.z_reflejo_nm > gt.z_nm).mean() > 0.9                       # alpha < 90 con signos +1: z' > z


def test_desenfoque_del_reflejo_ligado_al_del_real():
    """Cada subposicion del reflejo es el reflejo de la del real: mismos
    desplazamientos (R ortogonal) y GT = media de las subposiciones."""
    cfg, _, gt, ex = video(n_subpasos=4, t_exp_s=0.08, D_rango_um2_s=(1.0, 2.0))
    e = sim.Espejo(cfg)
    sub, sub_r = ex["sub"], ex["sub_reflejo"]
    assert np.allclose(sub_r, e.reflejar(sub), rtol=0, atol=1e-9)
    paso = np.linalg.norm(np.diff(e.a_nm(sub), axis=2), axis=-1)
    paso_r = np.linalg.norm(np.diff(e.a_nm(sub_r), axis=2), axis=-1)
    assert np.allclose(paso, paso_r, rtol=0, atol=1e-6)
    media = sub_r.mean(axis=2).transpose(1, 0, 2).reshape(-1, 3)
    assert np.allclose(gt[["x_reflejo_px", "y_reflejo_px", "z_reflejo_nm"]].to_numpy(), media, atol=1e-9)


def test_imagen_limpia_reconstruible_con_reflejos():
    """limpio = fondo + sum N/S PSF(real) + sum ratio N/S PSF(reflejo), en cada
    subpaso; los reflejos fuera del banco no se dibujan."""
    kw = dict(n_subpasos=2, t_exp_s=0.05)
    cfg, _, gt, ex = video(**kw)
    b = banco_e(**kw)
    sub, sub_r = ex["sub"], ex["sub_reflejo"]
    g0 = gt[gt.frame == 0].sort_values("particle_id")
    N = g0.photons.to_numpy()
    for t in range(cfg.n_frames):
        img = np.zeros((cfg.H, cfg.W))
        for p in range(len(N)):
            for k in range(cfg.n_subpasos):
                b.colocar(img, *sub[p, t, k], N[p] / cfg.n_subpasos)
                if b.z_grid[0] - 1e-6 <= sub_r[p, t, k, 2] <= b.z_grid[-1] + 1e-6:
                    b.colocar(img, *sub_r[p, t, k], cfg.ratio_reflejo * N[p] / cfg.n_subpasos)
        assert np.allclose(ex["limpio"][t], img + ex["fondo"], rtol=1e-12, atol=1e-9), t


def test_imagen_limpia_desde_el_gt():
    """Sin desenfoque, el GT basta para rehacer la imagen: cada reflejo es un
    emisor mas en (x_reflejo_px, y_reflejo_px, z_reflejo_nm) con
    photons_reflejo, dibujado solo si reflejo_en_banco."""
    cfg, _, gt, ex = video(ratio_reflejo=0.5)
    b = banco_e(ratio_reflejo=0.5)
    assert (~gt.reflejo_en_banco).sum() < len(gt)
    for t in range(cfg.n_frames):
        img = np.zeros((cfg.H, cfg.W))
        for _, f in gt[gt.frame == t].iterrows():
            b.colocar(img, f.x_px, f.y_px, f.z_nm, f.photons)
            if f.reflejo_en_banco:
                b.colocar(img, f.x_reflejo_px, f.y_reflejo_px, f.z_reflejo_nm, f.photons_reflejo)
        assert np.allclose(ex["limpio"][t], img + ex["fondo"], rtol=1e-12, atol=1e-9), t
    # y la luz de los reflejos se nota: el espejo agrega fotones a la imagen
    off = sim.Config(**BASE)
    _, _, ex_off = sim.generar_video(off, sim.BancoPSF(off), 0, extra=True)
    assert (ex["limpio"] - ex["fondo"]).sum() > (ex_off["limpio"] - ex_off["fondo"]).sum() * 0.9 + \
        0.25 * gt.photons_reflejo[gt.reflejo_en_fov].sum()


# --- flujos aleatorios ------------------------------------------------------

def test_espejo_no_cambia_los_otros_flujos():
    """Con y sin espejo (mismo seed, video_id): mismos n, D, fotones y fondo."""
    off = sim.Config(**BASE)
    b_off = sim.BancoPSF(off)
    for vid in (0, 1, 7):
        _, g0, e0 = sim.generar_video(off, b_off, vid, extra=True)
        _, _, g1, e1 = video(vid)
        assert np.array_equal(g0.particle_id, g1.particle_id)
        assert np.array_equal(g0.photons, g1.photons) and np.array_equal(g0.D_um2_s, g1.D_um2_s)
        assert np.array_equal(e0["fondo"], e1["fondo"])


def test_reproducible_y_regenerable_por_video():
    _, v1, g1, _ = video(1)
    cfg = cfg_e()
    b = banco_e()
    for vid in (3, 0):                                           # otros videos antes no cambian el 1
        sim.generar_video(cfg, b, vid)
    v1b, g1b = sim.generar_video(cfg, b, 1)
    assert np.array_equal(v1, v1b) and g1.equals(g1b)


# --- convencion de signos -----------------------------------------------------

def test_signos_consistentes():
    """Con cada combinacion de signos, el muestreo con dist_max, el plegado, el
    browniano y el reflejo del GT usan el mismo espejo de cfg."""
    for sz, si in SIGNOS:
        kw = dict(signo_z=sz, signo_inclinacion=si, dist_max_espejo_um=2.0, D_rango_um2_s=(1.0, 2.0))
        cfg, _, gt, ex = video(**kw)
        e = sim.Espejo(cfg)
        d = e.distancia_nm(e.a_nm(ex["sub"]))
        assert (d >= -1e-6).all() and (d[:, 0, 0] <= 2000.0 + 1e-6).all(), (sz, si)
        dz = (gt.z_reflejo_nm - gt.z_nm).to_numpy()
        assert (si * sz * dz >= -1e-6).all(), (sz, si)
        assert gt[gt.frame == 0].reflejo_en_banco.all()


def test_signos_solo_entran_como_producto():
    kw = dict(dist_max_espejo_um=2.5, orientacion_espejo_deg=37.0)
    ref = {1: video(**kw)[2], -1: video(angulo_espejo_deg=180.0 - sim.Config().angulo_espejo_deg, **kw)[2]}
    cols = ["x_px", "y_px", "z_nm", "x_reflejo_px", "y_reflejo_px", "z_reflejo_nm"]
    for sz, si in SIGNOS:
        gt = video(signo_z=sz, signo_inclinacion=si, **kw)[2]
        assert np.allclose(gt[cols], ref[sz * si][cols], rtol=0, atol=1e-6), (sz, si)


# --- espejo apagado -----------------------------------------------------------

def test_espejo_apagado_ignora_sus_parametros():
    a = sim.Config(**BASE)
    b = sim.Config(**{**BASE, "angulo_espejo_deg": 120.0, "orientacion_espejo_deg": 33.0,
                      "espejo_punto_px": (1.0, 2.0), "ratio_reflejo": 0.1, "z_ref_espejo_nm": -3000.0,
                      "dist_max_espejo_um": 2.0, "signo_z": -1})
    ba = sim.BancoPSF(a)
    assert np.array_equal(ba.z_grid, sim.BancoPSF(b).z_grid)
    va, ga = sim.generar_video(a, ba, 0)
    vb, gb = sim.generar_video(b, ba, 0)
    assert np.array_equal(va, vb) and ga.equals(gb) and list(ga.columns) == sim.COLUMNAS_GT
    assert a.resumen() == b.resumen() and a.a_dict() == b.a_dict()
