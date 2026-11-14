"""Tests del simulador v2 en localizacion: PSF aleatorizada por muestra
(amp_esferica, astigmatismo y coma con su propio flujo), rangos degenerados
que reproducen las muestras de hoy bit a bit, meta y SNR con la pupila de la
muestra, y configs/celula_v2.yaml."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from comun import cfg_datos, espera_error
from config import ConfigDatos, cargar_config
from data.simulacion import sim, config_simulador, config_muestra, generar_muestra, pupila_muestra, snr_emisores

RAIZ = Path(__file__).resolve().parents[1]
CELULA_SIM = dict(usar_espejo=True, usar_celula=True, diametro_celula_um=2.0, K_psf=64, z_paso_nm=100.0,
                  z_rango_nm=(-1200.0, 1200.0), recorte_pupila_reflejo=True, recorte_pupila_real=True,
                  mascara="esferica", anillos_hacia_objetivo=True)
GEO = dict(espejo_angulo_eje_rango_deg=(40.0, 50.0), espejo_orientacion_aleatoria=True,
           celula_diametro_rango_um=(1.5, 2.5), celula_contacto_centrado=True, celula_desplazamiento_px=3.0,
           ratio_reflejo_rango=(0.45, 1.0))
PSF = dict(psf_amp_esferica_rango=(4.0, 6.5), psf_astig_rango=(0.0, 1.3), psf_coma_rango=(0.0, 0.9))


def cfg_cel(sim_kw=None, **kw):
    return cfg_datos(sim_kw={**CELULA_SIM, **(sim_kw or {})}, **{**GEO, **kw})


def test_config_de_la_psf_aleatoria():
    espera_error(ValueError, cfg_cel, psf_astig_rango=(1.0, 0.5))
    espera_error(ValueError, cfg_cel, psf_coma_rango=(-0.1, 0.5))
    espera_error(ValueError, cfg_datos, mascara="ninguna", psf_amp_esferica_rango=(4.0, 6.5))
    assert cfg_cel(**PSF).psf_aleatoria and not cfg_cel().psf_aleatoria
    assert ConfigDatos(**{**PSF, "psf_astig_rango": [0, 1.3]}).psf_astig_rango == (0.0, 1.3)


def test_pupila_por_muestra_en_los_rangos():
    """Esferica, magnitudes y angulos en sus rangos, cubriendolos; regenerable;
    distinta entre muestras y entre splits."""
    cd = cfg_cel(**PSF)
    cs = config_simulador(cd, 4)
    p = [pupila_muestra(cd, cs, i) for i in range(400)]
    assert p[7] == pupila_muestra(cd, cs, 7)
    a = {c: np.array([q[c] for q in p]) for c in p[0]}
    for c, (lo, hi) in (("amp_esferica", (4.0, 6.5)), ("amp_astig", (0.0, 1.3)), ("amp_coma", (0.0, 0.9)),
                        ("angulo_astig_deg", (0.0, 180.0)), ("angulo_coma_deg", (0.0, 360.0))):
        assert a[c].min() >= lo and a[c].max() <= hi, c
        assert a[c].min() < lo + 0.05 * (hi - lo) and a[c].max() > hi - 0.05 * (hi - lo), c   # cubre el rango
        assert abs(a[c].mean() - (lo + hi) / 2) < 0.06 * (hi - lo), c                      # uniforme
    assert pupila_muestra(cd, config_simulador(cd, 5), 7) != p[7]
    solo_esf = cfg_cel(psf_amp_esferica_rango=(4.0, 6.5))
    assert set(pupila_muestra(solo_esf, cs, 0)) == {"amp_esferica"}
    assert pupila_muestra(solo_esf, cs, 0)["amp_esferica"] == p[0]["amp_esferica"]     # mismo flujo


def test_la_pupila_no_cambia_los_demas_sorteos():
    cd0, cd1 = cfg_cel(), cfg_cel(**PSF)
    cs0, cs1 = config_simulador(cd0, 3), config_simulador(cd1, 3)
    for i in range(20):
        m0, m1 = config_muestra(cd0, cs0, i), config_muestra(cd1, cs1, i)
        assert replace(m1, **{c: getattr(m0, c) for c in ("amp_esferica", "amp_astig", "angulo_astig_deg",
                                                             "amp_coma", "angulo_coma_deg")}) == m0
        assert m1.amp_esferica != m0.amp_esferica


def test_rangos_degenerados_reproducen_las_muestras_de_hoy():
    """Esferica fija en el valor de la config y aberraciones [0, 0] (angulos
    sorteados pero sin efecto): imagen y GT bit a bit iguales a sin PSF aleatoria."""
    amp = float(sim.Config().amp_esferica)
    cd0 = cfg_cel()
    cd1 = cfg_cel(psf_amp_esferica_rango=(amp, amp), psf_astig_rango=(0.0, 0.0), psf_coma_rango=(0.0, 0.0))
    cs = config_simulador(cd0, 1)
    banco = sim.BancoPSF(cs)
    for i in range(4):
        i0, g0, _ = generar_muestra(cd0, cs, banco, i)
        i1, g1, m1 = generar_muestra(cd1, cs, banco, i)
        assert np.array_equal(i0, i1) and g0.equals(g1)
        assert m1["amp_astig"] == 0.0 and m1["amp_esferica"] == amp


def test_muestra_con_la_pupila_sorteada():
    """La imagen usa la pupila de la muestra (cambia respecto de la del banco),
    la meta la guarda y la SNR usa su PSF sin recorte."""
    cd0, cd1 = cfg_cel(), cfg_cel(**PSF)
    cs = config_simulador(cd0, 1)
    banco = sim.BancoPSF(cs)
    for i in range(3):
        i0, g0, _ = generar_muestra(cd0, cs, banco, i)
        i1, g1, m1 = generar_muestra(cd1, cs, banco, i)
        assert g0[["x_px", "y_px", "z_nm", "photons"]].equals(g1[["x_px", "y_px", "z_nm", "photons"]])
        if len(g0):
            assert not np.array_equal(i0, i1)
        cm = config_muestra(cd1, cs, i)
        assert all(m1[c] == getattr(cm, c) for c in ("amp_esferica", "amp_astig", "angulo_coma_deg"))
        if len(g1):
            psf = sim.PSFRecortada(cm, banco.K, recortar=False)
            p_max = np.array([psf.psf(z).max() for z in g1.z_nm])
            fondo = np.zeros((cs.H, cs.W)) + 0.0
            assert np.allclose(snr_emisores(cm, banco, g1, fondo + 10.0),
                               g1.photons * p_max / np.sqrt(g1.photons * p_max + 10.0 + sim._var_lectura_e2(cm)))


def test_config_celula_v2():
    """celula_v2.yaml: camara medida, fondo log con autofluorescencia, z +-3000 con
    peso en los polos, PSF aleatoria de cerca y el entrenamiento con checkpoints
    y metricas de val; una muestra se genera con todo eso."""
    c = cargar_config(RAIZ / "configs" / "celula_v2.yaml", ["datos.simulador.z_paso_nm=200.0"])
    s, d = c.datos.simulador, c.datos
    assert (s["gain"], s["offset"]) == (0.45, 97.75) and d.read_noise_rango == (2.18, 2.90)
    assert tuple(s["fondo_rango"]) == (1.0, 40.0) and s["muestreo_fondo"] == "log"
    assert tuple(s["fondo_celula_contraste_rango"]) == (0.0, 2.5)
    assert tuple(s["fotones_rango"]) == (1000.0, 50000.0) and tuple(s["z_rango_nm"]) == (-3000.0, 3000.0)
    assert s["celula_peso_polos"] == 0.5 and s["anillos_hacia_objetivo"] is True
    assert (d.psf_amp_esferica_rango, d.psf_astig_rango, d.psf_coma_rango) == ((4.0, 6.5), (0.0, 1.3), (0.0, 0.9))
    assert c.entrenamiento.checkpoint_cada == 10 and c.entrenamiento.metricas_val
    assert c.target.dz_nm == 100.0
    cs = config_simulador(d, 1)
    img, gt, meta = generar_muestra(d, cs, sim.BancoPSF(cs), 0)
    assert img.shape == (1, 96, 96) and 0 <= meta["fondo_celula"] and 4.0 <= meta["amp_esferica"] <= 6.5
    assert gt.en_rango_z.all() and meta["reintentos"] == 0
