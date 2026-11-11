"""Tests del espejo en localizacion: opciones de la config, geometria aleatoria
por muestra con su propio flujo, banco que cubre las z' de cualquier geometria
(ningun reflejo sin dibujar), regeneracion de muestras, mapa de distancia al
espejo (canal extra), DatasetDisco = DatasetSimulado con el canal, metricas
del espejo y, de punta a punta en CPU, entrenar, inferir y evaluar."""
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from comun import cfg_datos, espera_error, SMOKE
from config import ConfigDatos, cargar_config
from data.dataset import DatasetSimulado, DatasetDisco, pregenerar
from data.simulacion import (sim, config_simulador, generar_muestra, geometria_muestra, config_geometria,
                             config_muestra, mapa_distancia_espejo, z_banco_espejo, COLUMNAS_GEOMETRIA)
import metricas

RAIZ = Path(__file__).resolve().parents[1]
ESPEJO_SIM = dict(usar_espejo=True, dist_max_espejo_um=1.0, K_psf=32, z_paso_nm=100.0)
GEO = dict(espejo_angulo_eje_rango_deg=(40.0, 50.0), espejo_orientacion_aleatoria=True, espejo_punto_aleatorio=True)
SMOKE_ESPEJO = ["datos.simulador.usar_espejo=true", "datos.simulador.dist_max_espejo_um=1.0",
                "datos.espejo_angulo_eje_rango_deg=[40.0, 50.0]", "datos.espejo_orientacion_aleatoria=true",
                "datos.espejo_punto_aleatorio=true", "datos.canal_distancia_espejo=true",
                "modelo.dilataciones=[1, 2, 4, 8, 16, 32]"]
mudo = lambda *a, **k: None


def cfg_esp(sim_kw=None, **kw):
    return cfg_datos(sim_kw={**ESPEJO_SIM, **(sim_kw or {})}, **{**GEO, "canal_distancia_espejo": True, **kw})


@lru_cache(maxsize=None)
def banco_esp(anillos=False):
    return sim.BancoPSF(config_simulador(cfg_esp(sim_kw=dict(anillos_hacia_objetivo=anillos)), 0))


def test_config_del_espejo():
    espera_error(ValueError, ConfigDatos, canal_distancia_espejo=True)            # sin usar_espejo
    espera_error(ValueError, ConfigDatos, espejo_orientacion_aleatoria=True)
    espera_error(ValueError, cfg_esp, espejo_angulo_eje_rango_deg=(50.0, 40.0))
    espera_error(ValueError, cfg_esp, espejo_angulo_eje_rango_deg=(-95.0, 0.0))
    espera_error(ValueError, cfg_esp, escala_distancia_px=0.0)
    assert cfg_esp().canales_entrada == 2 and cfg_esp(canal_distancia_espejo=False).canales_entrada == 1
    for nombre in ("base", "mezcla", "mezcla_renovar_largo", "smoke"):            # las configs de antes no cambian
        c = cargar_config(RAIZ / "configs" / f"{nombre}.yaml")
        assert c.datos.canales_entrada == 1 and not c.datos.usa_espejo
        assert nombre == "smoke" or c.modelo.dilataciones == (1, 2, 4, 8, 16)
    e = cargar_config(RAIZ / "configs" / "espejo.yaml")
    assert e.datos.canales_entrada == 2 and e.modelo.dilataciones == (1, 2, 4, 8, 16, 32)
    assert e.datos.simulador["anillos_hacia_objetivo"] is False and e.datos.simulador["ratio_reflejo"] == 0.8
    assert e.datos.simulador["angulo_espejo_eje_deg"] == 45.0 and e.datos.espejo_angulo_eje_rango_deg == (40.0, 50.0)
    assert e.datos.simulador["dist_max_espejo_um"] == 4.0


def test_geometria_por_muestra():
    cd = cfg_esp()
    cs = config_simulador(cd, 7)
    gs = [geometria_muestra(cd, cs, i) for i in range(200)]
    assert gs[3] == geometria_muestra(cd, cs, 3) and gs[3] != geometria_muestra(cd, cs, 3, intento=1)
    a = np.array([g["angulo_espejo_eje_deg"] for g in gs])
    f = np.array([g["orientacion_espejo_deg"] for g in gs])
    p = np.array([g["espejo_punto_px"] for g in gs])
    assert (a >= 40.0).all() and (a <= 50.0).all() and a.std() > 2
    assert (f >= 0).all() and (f < 360).all() and f.std() > 80
    assert (p[:, 0] >= -0.5).all() and (p[:, 0] < cs.W - 0.5).all() and (p[:, 1] < cs.H - 0.5).all()
    # flujo propio: activar una opcion no cambia lo que sortean las otras
    solo_f = cfg_esp(espejo_angulo_eje_rango_deg=None, espejo_punto_aleatorio=False)
    for i in (0, 5):
        g = geometria_muestra(solo_f, config_simulador(solo_f, 7), i)
        assert g["orientacion_espejo_deg"] == gs[i]["orientacion_espejo_deg"]
        assert g["angulo_espejo_eje_deg"] == cs.angulo_espejo_eje_deg and g["espejo_punto_px"] == cs.espejo_punto_px


def test_linea_del_espejo_en_el_borde_del_fov():
    """Con espejo_linea_rango_fov la interseccion del espejo con el plano focal
    queda a u h del centro (u en el rango, h el semiancho del FOV sobre m) y el
    centro del lado de la muestra, tambien con z_ref != 0 y con las dos PSF."""
    espera_error(ValueError, cfg_esp, espejo_punto_aleatorio=False, espejo_linea_rango_fov=(0.3, 1.0))
    espera_error(ValueError, cfg_esp, espejo_linea_rango_fov=(1.0, 0.5))
    for an, zr in ((False, 0.0), (True, -300.0), (False, 450.0)):
        cd = cfg_esp(sim_kw=dict(anillos_hacia_objetivo=an, z_ref_espejo_nm=zr), espejo_linea_rango_fov=(1 / 3, 1.1))
        cs = config_simulador(cd, 5)
        c = np.array([(cs.W - 1) / 2, (cs.H - 1) / 2])
        us = []
        for i in range(150):
            e = sim.Espejo(config_muestra(cd, cs, i))
            q = e.punto_en_z_nm(0.0)[:2] / e.pixel_nm
            h = cs.W / 2 * abs(e.m[0]) + cs.H / 2 * abs(e.m[1])
            us.append(((c - q) @ e.m[:2]) / h)
            mapa = mapa_distancia_espejo(config_muestra(cd, cs, i), cs.H, cs.W, 1.0)
            assert mapa[cs.H // 2, cs.W // 2] > 0                              # el centro, del lado de la muestra
        us = np.array(us)
        assert (us >= 1 / 3 - 1e-9).all() and (us <= 1.1 + 1e-9).all() and us.std() > 0.15, (an, zr)


def test_la_geometria_no_cambia_los_otros_flujos():
    """Con y sin espejo, y con o sin geometria aleatoria: mismos n, fotones y
    fondo en cada muestra (el espejo solo pliega las posiciones)."""
    cd_on, cd_off = cfg_esp(), cfg_datos(sim_kw=dict(K_psf=32, z_paso_nm=100.0))
    cs_on, cs_off = config_simulador(cd_on, 4), config_simulador(cd_off, 4)
    b_off = sim.BancoPSF(cs_off)
    for i in range(6):
        _, g_on, m_on = generar_muestra(cd_on, cs_on, banco_esp(), i)
        _, g_off, m_off = generar_muestra(cd_off, cs_off, b_off, i)
        assert m_on["reintentos"] == 0
        assert np.array_equal(g_on.photons, g_off.photons) and m_on["fondo_medio"] == m_off["fondo_medio"]


def test_banco_cubre_cualquier_geometria():
    """Ningun reflejo queda sin dibujar ni hace falta regenerar, con las dos
    PSF, y el banco es el de z_banco_espejo: ampliado hacia -z (con beta > 0 el
    reflejo queda mas lejos del objetivo), en z fisica con cualquier bandera."""
    for an in (False, True):
        cd = cfg_esp(sim_kw=dict(anillos_hacia_objetivo=an))
        cs = config_simulador(cd, 9)
        b = banco_esp(an)
        lo, hi = z_banco_espejo(cd, cs)
        assert b.z_grid[0] <= lo + 1e-6 and b.z_grid[-1] >= hi - 1e-6
        assert lo < cs.z_rango_nm[0] and hi == cs.z_rango_nm[1]
        for i in range(150):
            _, gt = sim.generar_video(config_muestra(cd, cs, i), b, i)    # primer intento, sin regenerar
            assert gt.reflejo_en_banco.all(), (an, i)


def test_regenera_las_muestras_con_reflejos_sin_dibujar():
    """Con un banco demasiado corto la muestra se regenera (otro video_id y otra
    geometria) hasta que todos los reflejos se dibujan; se cuenta en meta."""
    cd = cfg_esp(sim_kw=dict(z_banco_nm=(-1100.0, 800.0)))
    cs = config_simulador(cd, 3)
    b = sim.BancoPSF(cs)
    reint = []
    for i in range(25):
        _, gt, meta = generar_muestra(cd, cs, b, i)
        assert gt.reflejo_en_banco.all() and (gt.video_id == i).all()
        reint.append(meta["reintentos"])
    assert max(reint) > 0 and min(reint) == 0
    i = int(np.argmax(reint))
    v1, g1, m1 = generar_muestra(cd, cs, b, i)
    v2, g2, m2 = generar_muestra(cd, cs, b, i)
    assert np.array_equal(v1, v2) and g1.equals(g2) and m1 == m2              # regenerable sola
    sin_reintentos = cfg_esp(sim_kw=dict(z_banco_nm=(-1100.0, 800.0)), max_reintentos_espejo=0)
    espera_error(RuntimeError, generar_muestra, sin_reintentos, config_simulador(sin_reintentos, 3), b, i)


def test_mapa_de_distancia_analitico():
    cs = config_simulador(cfg_esp(), 0)
    H, W, px = cs.H, cs.W, cs.pixel_nm
    yy, xx = np.mgrid[0:H, 0:W]
    fila = dict(espejo_angulo_eje_deg=0.0, espejo_orientacion_deg=0.0, espejo_punto_x_px=10.3, espejo_punto_y_px=5.0)
    assert np.allclose(mapa_distancia_espejo(config_geometria(cs, fila), H, W, 40.0), (xx - 10.3) / 40.0, atol=1e-6)
    fila.update(espejo_orientacion_deg=90.0)                                    # muestra hacia +y
    assert np.allclose(mapa_distancia_espejo(config_geometria(cs, fila), H, W, 40.0), (yy - 5.0) / 40.0, atol=1e-6)
    # beta = 45 con el plano de referencia 500 nm mas lejos del objetivo que el foco (z_ref = -500): en
    # z = 0 la linea se corre 500 nm alejandose de la muestra
    fila.update(espejo_angulo_eje_deg=45.0, espejo_orientacion_deg=0.0)
    from dataclasses import replace
    m = mapa_distancia_espejo(replace(config_geometria(cs, fila), z_ref_espejo_nm=-500.0), H, W, 1.0)
    assert np.allclose(m, xx - 10.3 + 500.0 / px, atol=1e-5)


def test_disco_igual_a_simulado_con_el_canal():
    cd = cfg_esp()
    with tempfile.TemporaryDirectory() as tmp:
        pregenerar(cd, 5, 11, tmp, banco=banco_esp())
        disco, simu = DatasetDisco(tmp), DatasetSimulado(cd, 5, 11, banco=banco_esp())
        m = pd.read_csv(Path(tmp) / "muestras.csv")
        assert set(COLUMNAS_GEOMETRIA + ["reintentos", "n_reflejos_en_fov"]) <= set(m.columns)
        gt = pd.read_csv(Path(tmp) / "ground_truth.csv")
        assert gt.reflejo_en_banco.all() and "z_reflejo_nm" in gt
        for i in range(5):
            a, b = disco[i], simu[i]
            assert a["imagen"].shape == (2, cd.simulador["H"], cd.simulador["W"])
            assert torch.equal(a["imagen"], b["imagen"]) and torch.equal(a["pos"], b["pos"])


def test_distancia_al_espejo_y_fp_sobre_reflejos():
    cd = cfg_esp()
    cs = config_simulador(cd, 2)
    _, gt, meta = generar_muestra(cd, cs, banco_esp(), 4)
    e = sim.Espejo(config_geometria(cs, meta))
    d = metricas.distancia_al_espejo_nm(gt, cs.pixel_nm)
    assert np.allclose(d, e.distancia_nm(e.a_nm(gt[["x_px", "y_px", "z_nm"]].to_numpy())), atol=1e-6)
    # FP sinteticos: uno sobre un reflejo, uno lejos de todo, uno TP
    g = pd.DataFrame(dict(video_id=[0, 0], frame=[0, 0], x_px=[5.0, 20.0], y_px=[5.0, 5.0], z_nm=[0.0, 0.0],
                          en_fov=[True, True], x_reflejo_px=[9.0, 30.0], y_reflejo_px=[5.0, 12.0],
                          z_reflejo_nm=[900.0, 2000.0], reflejo_en_fov=[True, True]))
    pred = pd.DataFrame(dict(video_id=[0, 0, 0], frame=[0, 0, 0], x_px=[5.0, 9.5, 15.0], y_px=[5.0, 5.0, 15.0],
                             z_nm=[0.0, 700.0, 0.0], conf=[1.0, 1.0, 1.0]))
    ge, pe = metricas.evaluar(pred, g, cs.pixel_nm)
    pe, ge = metricas.fp_sobre_reflejos(pe, ge, cs.pixel_nm, 150.0)
    assert list(pe.tp) == [True, False, False] and list(pe.sobre_reflejo) == [False, True, False]
    assert list(ge.reflejo_con_fp) == [True, False]
    r = metricas.resumen_espejo(pe, ge)
    assert (r["fp"], r["fp_sobre_reflejo"], r["reflejos_en_fov"], r["reflejos_con_fp"]) == (2, 1, 2, 1)


def test_entrena_infiere_y_evalua_con_espejo():
    """smoke.yaml con espejo y canal de distancia: entrena 1 epoca en CPU, el
    checkpoint tiene 2 canales de entrada, infer usa la geometria de
    muestras.csv (y falla sin ella) y evaluate escribe las tablas del espejo."""
    from train import entrenar
    from utils import cargar_modelo
    import evaluate
    import infer
    cfg = cargar_config(SMOKE, SMOKE_ESPEJO + ["entrenamiento.epocas=1"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "corrida").mkdir()
        log = entrenar(cfg, tmp / "corrida", log=mudo)
        assert len(log) == 1 and np.isfinite(log.val_total).all()
        modelo, cfg_c, rej = cargar_modelo(tmp / "corrida" / "mejor.pt")
        assert modelo.canales_entrada == 2 and cfg_c.datos == cfg.datos

        pregenerar(cfg.datos, 4, 5, tmp / "test")
        e = infer.leer_entradas([tmp / "test"])[0]
        assert e["geometria"] is not None and len(e["geometria"]) == 4
        cs = infer.camara_de(e, cfg_c)
        y = infer.predecir(modelo, e["imagenes"], cfg_c, cs, torch.device("cpu"), amp="no", geometria=e["geometria"])
        assert y.shape == (4, *rej.forma)
        espera_error(ValueError, infer.predecir, modelo, e["imagenes"], cfg_c, cs, torch.device("cpu"), "no", None)

        ev = tmp / "ev"
        r = evaluate.main([str(tmp / "corrida" / "mejor.pt"), str(tmp / "test"), "--salida", str(ev), "--figuras", "0",
                           "--dispositivo", "cpu", "--set", "postproc.umbral=0.0"])
        for nombre in ("por_dist_espejo.csv", "por_dist_espejo_fotones.csv", "resumen.yaml"):
            assert (ev / nombre).exists(), nombre
        resumen = __import__("yaml").safe_load((ev / "resumen.yaml").read_text())
        esp = resumen["espejo"]
        assert esp["fp"] == r["fp"] and 0 <= esp["fp_sobre_reflejo"] <= esp["fp"]
        assert pd.read_csv(ev / "por_dist_espejo_fotones.csv").n.sum() == r["n_gt"]


def test_ratio_reflejo_por_muestra():
    """datos.ratio_reflejo_rango sortea ratio_reflejo por muestra con su propio
    flujo: en el rango, regenerable, y sin cambiar posiciones, fotones, fondo
    ni geometria; photons_reflejo = ratio * photons y muestras.csv lo guarda."""
    espera_error(ValueError, cfg_esp, ratio_reflejo_rango=(0.9, 0.5))
    espera_error(ValueError, cfg_datos, ratio_reflejo_rango=(0.3, 0.8))                # sin espejo
    cd, cd_fijo = cfg_esp(ratio_reflejo_rango=(0.35, 0.8)), cfg_esp()
    cs = config_simulador(cd, 5)
    rs = []
    for i in range(30):
        m, m_fijo = config_muestra(cd, cs, i), config_muestra(cd_fijo, cs, i)
        assert 0.35 <= m.ratio_reflejo <= 0.8 and m == config_muestra(cd, cs, i)
        from dataclasses import replace
        assert replace(m, ratio_reflejo=m_fijo.ratio_reflejo) == m_fijo
        rs.append(m.ratio_reflejo)
    assert np.std(rs) > 0.08
    _, gt, meta = generar_muestra(cd, cs, banco_esp(), 3)
    _, gt_f, _ = generar_muestra(cd_fijo, cs, banco_esp(), 3)
    assert np.allclose(gt.photons_reflejo, meta["ratio_reflejo"] * gt.photons)
    assert gt[["x_px", "y_px", "z_nm", "photons"]].equals(gt_f[["x_px", "y_px", "z_nm", "photons"]])
    with tempfile.TemporaryDirectory() as tmp:
        pregenerar(cd, 4, 5, Path(tmp) / "t")
        assert "ratio_reflejo" in pd.read_csv(Path(tmp) / "t" / "muestras.csv")


def test_recorte_de_pupila_por_muestra():
    """datos.simulador.recorte_pupila_reflejo pasa al simulador: la geometria de
    cada muestra fija la mascara; mismas posiciones y GT que sin recorte, y la
    imagen cambia (solo la luz del reflejo)."""
    cd, cd_r = cfg_esp(), cfg_esp(sim_kw=dict(recorte_pupila_reflejo=True))
    cs, cs_r = config_simulador(cd, 5), config_simulador(cd_r, 5)
    assert cs_r.recorte_pupila_reflejo and not cs.recorte_pupila_reflejo
    for i in range(3):
        img, gt, meta = generar_muestra(cd, cs, banco_esp(), i)
        img_r, gt_r, meta_r = generar_muestra(cd_r, cs_r, banco_esp(), i)
        assert gt.equals(gt_r) and meta == meta_r
        assert not np.array_equal(img, img_r)
