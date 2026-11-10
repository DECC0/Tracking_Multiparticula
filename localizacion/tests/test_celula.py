"""Tests de la celula en localizacion: opciones de la config, celula sorteada
por muestra con su propio flujo, contacto celula-espejo centrado en el FOV,
banco que cubre la celula y la celula virtual (sin reintentos), emisores
fuera del rango axial (en_rango_z = False) fuera de los objetivos y de las
metricas, configs/celula.yaml y, de punta a punta en CPU, entrenar y evaluar."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from comun import cfg_datos, espera_error, SMOKE
from config import ConfigDatos, cargar_config
from data.dataset import DatasetSimulado, pregenerar, _pos_en_fov
from data.simulacion import (sim, config_simulador, config_muestra, generar_muestra, z_banco_celula, evaluable,
                             COLUMNAS_CELULA)

RAIZ = Path(__file__).resolve().parents[1]
CELULA_SIM = dict(usar_espejo=True, usar_celula=True, diametro_celula_um=2.0, K_psf=32, z_paso_nm=100.0,
                  z_rango_nm=(-800.0, 800.0))
GEO = dict(espejo_angulo_eje_rango_deg=(40.0, 50.0), espejo_orientacion_aleatoria=True,
           celula_diametro_rango_um=(1.5, 2.5), celula_altura_rango_nm=(-300.0, 300.0),
           celula_contacto_centrado=True, celula_desplazamiento_px=3.0)
SMOKE_CELULA = ["datos.simulador.usar_espejo=true", "datos.simulador.usar_celula=true",
                "datos.simulador.diametro_celula_um=2.0", "datos.simulador.z_rango_nm=[-1200.0, 1200.0]",
                "datos.espejo_orientacion_aleatoria=true", "datos.celula_contacto_centrado=true",
                "datos.celula_desplazamiento_px=3.0", "datos.celula_diametro_rango_um=[1.5, 2.5]"]
mudo = lambda *a, **k: None


def cfg_cel(sim_kw=None, **kw):
    return cfg_datos(sim_kw={**CELULA_SIM, **(sim_kw or {})}, **{**GEO, **kw})


def test_config_de_la_celula():
    espera_error(ValueError, ConfigDatos, celula_contacto_centrado=True)          # sin usar_celula
    espera_error(ValueError, cfg_cel, celula_diametro_rango_um=(3.0, 2.0))
    espera_error(ValueError, cfg_cel, celula_desplazamiento_px=-1.0)
    espera_error(ValueError, cfg_cel, espejo_punto_aleatorio=True)                # el contacto fija la linea
    assert cfg_cel().celula_aleatoria and not cfg_datos().celula_aleatoria
    c = cargar_config(RAIZ / "configs" / "celula.yaml")
    s = c.datos.simulador
    assert s["usar_celula"] and s["H"] == s["W"] == 96 and tuple(s["z_rango_nm"]) == (-2500.0, 2500.0)
    assert c.datos.celula_contacto_centrado and c.datos.celula_desplazamiento_px == 10.0
    assert c.datos.celula_diametro_rango_um == (4.0, 6.0) and "dist_max_espejo_um" not in s


def test_celula_por_muestra_y_contacto_centrado():
    """Diametro y altura en su rango, distintos entre muestras, regenerables;
    el contacto esta en el plano del espejo y en el centro del FOV +- el
    desplazamiento, y la celula entera del lado de la muestra."""
    cd = cfg_cel(sim_kw=dict(z_ref_espejo_nm=-250.0))
    cs = config_simulador(cd, 7)
    centro = np.array([(cs.W - 1) / 2, (cs.H - 1) / 2])
    diam = []
    for i in range(60):
        m = config_muestra(cd, cs, i)
        assert m == config_muestra(cd, cs, i)
        e, cel = sim.Espejo(m), sim.Celula(m)
        q = cel.contacto_nm(e)
        assert np.isclose(e.distancia_nm(q), 0.0, atol=1e-6) and np.isclose(e.distancia_nm(cel.c), cel.R)
        assert (np.abs(q[:2] / m.pixel_nm - centro) <= 3.0 + 1e-9).all()
        assert 1.5 <= m.diametro_celula_um <= 2.5 and -300.0 <= m.altura_celula_nm <= 300.0
        diam.append(m.diametro_celula_um)
    assert np.std(diam) > 0.2


def test_banco_cubre_la_celula_y_la_celula_virtual():
    cd = cfg_cel()
    cs = config_simulador(cd, 9)
    lo, hi = z_banco_celula(cd, cs)
    assert cs.z_banco_nm == (lo, hi) and lo < cs.z_rango_nm[0]
    b = sim.BancoPSF(cs)
    for i in range(40):
        _, gt, meta = generar_muestra(cd, cs, b, i)
        assert gt.reflejo_en_banco.all() and meta["reintentos"] == 0
        assert set(COLUMNAS_CELULA) <= set(meta)


def test_fuera_del_rango_axial_no_es_objetivo():
    """Con un rango axial chico, parte de los emisores queda con en_rango_z =
    False: no entran al target ni a las metricas (evaluate los marca como
    fuera del FOV)."""
    cd = cfg_cel(sim_kw=dict(z_rango_nm=(-300.0, 300.0), n_emisores_rango=(8, 12)))
    ds = DatasetSimulado(cd, 12, 4)
    hubo = False
    for i in range(12):
        _, gt, meta = ds.muestra(i)
        pos = _pos_en_fov(gt)
        assert len(pos) == int(evaluable(gt).sum()) == meta["n_emisores"]
        assert (np.abs(pos[:, 2]) <= 300.0 + 1e-6).all()
        hubo |= bool((gt.en_fov & ~gt.en_rango_z).any())
    assert hubo
    import evaluate
    with tempfile.TemporaryDirectory() as tmp:
        pregenerar(cd, 6, 4, Path(tmp) / "t")
        g = evaluate.ground_truth(dict(config_sim=Path(tmp) / "t" / "config_simulador.json", nombre="t",
                                       video_id=np.arange(6), frame=np.zeros(6, int)))
        crudo = pd.read_csv(Path(tmp) / "t" / "ground_truth.csv")
        assert g.en_fov.sum() == (crudo.en_fov & crudo.en_rango_z).sum() < crudo.en_fov.sum()


def test_entrena_y_evalua_con_celula():
    from train import entrenar
    import evaluate
    cfg = cargar_config(SMOKE, SMOKE_CELULA + ["entrenamiento.epocas=1"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "corrida").mkdir()
        log = entrenar(cfg, tmp / "corrida", log=mudo)
        assert len(log) == 1 and np.isfinite(log.val_total).all()
        pregenerar(cfg.datos, 4, 5, tmp / "test")
        r = evaluate.main([str(tmp / "corrida" / "mejor.pt"), str(tmp / "test"), "--salida", str(tmp / "ev"),
                           "--figuras", "0", "--dispositivo", "cpu", "--set", "postproc.umbral=0.0"])
        crudo = pd.read_csv(tmp / "test" / "ground_truth.csv")
        assert r["n_gt"] == int((crudo.en_fov & crudo.en_rango_z).sum())
