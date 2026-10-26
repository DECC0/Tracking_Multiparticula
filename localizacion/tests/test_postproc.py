"""Post-procesado (postproc.py) sobre targets exactos: recupera las posiciones
continuas (tambien en los bordes de z), una deteccion por meseta saturada,
umbral y las dos confianzas, frames de un batch, ejes x/y sin intercambiar y
salida vacia."""
import copy
from dataclasses import replace

import numpy as np
import torch

import comun  # noqa: F401  (sys.path)
from config import ConfigTarget, ConfigPostproc
from data.target import RejillaVoxeles
from postproc import detectar

H, W = 24, 40
REJ = RejillaVoxeles(ConfigTarget(), H, W, (-800.0, 800.0))
PP = ConfigPostproc(umbral=0.5, confianza="pico")


def _target(pos_por_frame, amp=1.0):
    B, E = len(pos_por_frame), max(1, max(len(p) for p in pos_por_frame))
    pos = torch.zeros(B, E, 3, dtype=torch.float64)
    valido = torch.zeros(B, E, dtype=torch.bool)
    for b, p in enumerate(pos_por_frame):
        if len(p):
            pos[b, :len(p)] = torch.tensor(p, dtype=torch.float64)
            valido[b, :len(p)] = True
    return amp * REJ.target(pos, valido)


def _errores(det, pos):
    """Errores (dx, dy [px], dz [nm]) de cada emisor contra su deteccion mas cercana."""
    d = det[["x_px", "y_px", "z_nm"]].to_numpy()
    p = np.asarray(pos)
    i = np.argmin(np.hypot(d[None, :, 0] - p[:, None, 0], d[None, :, 1] - p[:, None, 1]), axis=1)
    return d[i] - p


def test_recupera_posiciones_continuas():
    pos = [(5.3, 4.7, -120.0), (20.1, 15.8, 333.0), (33.6, 7.2, 611.0), (12.45, 19.9, -800.0), (27.0, 11.0, 800.0)]
    det = detectar(_target([pos]), REJ, PP)
    assert len(det) == len(pos) and (det.b == 0).all()
    e = _errores(det, pos)
    # el centro de masa en una ventana truncada sesga < 0.05 voxeles (1.2 nm lateral, 2.5 nm axial)
    assert np.abs(e[:, :2]).max() < 0.02 and np.abs(e[:, 2]).max() < 3.0, e
    assert det.conf.min() > 0.6                                   # pico de la gaussiana muestreada fuera del centro


def test_meseta_saturada_una_deteccion():
    pos = [(8.2, 6.3, 0.0), (30.7, 17.1, -400.0)]
    sat = torch.clamp(4 * _target([pos]), max=REJ.escala)        # como la hardtanh: mesetas de valor escala
    assert int((sat == REJ.escala).sum()) > 2 * len(pos)
    det = detectar(sat, REJ, PP)
    assert len(det) == len(pos)
    assert np.abs(_errores(det, pos)[:, :2]).max() < 0.1


def test_umbral_y_confianzas():
    v, u, k = 40, 30, 20                                          # emisor en el centro de un voxel
    x, y, z = REJ.desde_voxel(v, u, k)
    pred = _target([[(x, y, z)]], amp=0.3)
    for conf in ("pico", "masa"):
        pp = replace(PP, confianza=conf)
        assert len(detectar(pred, REJ, replace(pp, umbral=0.5))) == 0
        det = detectar(pred, REJ, replace(pp, umbral=0.25))
        assert len(det) == 1 and abs(det.conf[0] - 0.3) < 1e-4, (conf, det)
        assert np.allclose(det[["x_px", "y_px", "z_nm"]].to_numpy()[0], (x, y, z), atol=1e-4)


def test_masa_tolera_emisor_repartido_en_z():
    """Si la red reparte un emisor en varios planos el pico baja, pero la masa no."""
    x, y, z = REJ.desde_voxel(40, 36, 18)                        # centro de un voxel: pico = escala
    ancho = copy.copy(REJ)
    ancho.sigma_z = 3.0
    pred = ancho.target(torch.tensor([[[x, y, z]]]), torch.ones(1, 1, dtype=torch.bool))
    pico = detectar(pred, REJ, replace(PP, umbral=0.0)).conf[0]
    masa = detectar(pred, REJ, replace(PP, umbral=0.0, confianza="masa")).conf[0]
    assert pico > 0.99 and masa > 1.5


def test_batch_y_ejes():
    frames = [[(30.0, 5.0, 0.0)], [], [(5.0, 18.0, -400.0), (12.0, 3.0, 200.0)]]
    det = detectar(_target(frames), REJ, PP)
    assert list(det.b) == [0, 2, 2]
    # H = 24 < 30: si x e y se intercambiaran el emisor del frame 0 no cabria
    x, y, z = det[det.b == 0][["x_px", "y_px", "z_nm"]].to_numpy()[0]
    assert abs(x - 30.0) < 0.02 and abs(y - 5.0) < 0.02 and abs(z) < 3.0
    assert np.allclose(sorted(det[det.b == 2].x_px), [5.0, 12.0], atol=0.02)


def test_resolucion_axial_de_la_ventana():
    """Dos emisores en el mismo (x, y): separados mas que radio_z_planos se
    detectan los dos; dentro de la ventana quedan en una sola deteccion."""
    lejos = detectar(_target([[(15.0, 10.0, -200.0), (15.0, 10.0, 200.0)]]), REJ, PP)   # 8 planos
    cerca = detectar(_target([[(15.0, 10.0, -50.0), (15.0, 10.0, 50.0)]]), REJ, PP)     # 2 planos
    assert len(lejos) == 2 and len(cerca) == 1


def test_salida_vacia():
    det = detectar(torch.zeros(2, *REJ.forma), REJ, PP)
    assert len(det) == 0 and list(det.columns) == ["b", "x_px", "y_px", "z_nm", "conf"]
