"""Metricas (metricas.py): emparejamiento perfecto, errores y sesgos conocidos,
tolerancias lateral y axial, emparejamiento hungaro (no codicioso), emisores
fuera del FOV que no cuentan, frames que no se mezclan y sin detecciones."""
import numpy as np
import pandas as pd

import comun  # noqa: F401  (sys.path)
from metricas import evaluar, resumen

PX = 100.0                              # nm por pixel


def _gt(filas, en_fov=None):
    gt = pd.DataFrame(filas, columns=["video_id", "frame", "x_px", "y_px", "z_nm"])
    gt["en_fov"] = True if en_fov is None else en_fov
    return gt


def _pred(filas):
    pred = pd.DataFrame(filas, columns=["video_id", "frame", "x_px", "y_px", "z_nm"])
    pred["conf"] = 1.0
    return pred


GT = _gt([(0, 0, 5.0, 5.0, 0.0), (0, 0, 15.0, 8.0, 300.0), (0, 1, 7.0, 2.0, -500.0), (3, 0, 1.0, 1.0, 100.0)])


def _resumen(pred, gt=GT, **kw):
    return resumen(*evaluar(pred, gt, PX, **kw))


def test_perfecto():
    r = _resumen(_pred(GT[["video_id", "frame", "x_px", "y_px", "z_nm"]].to_numpy()))
    assert (r["tp"], r["fp"], r["fn"]) == (4, 0, 0)
    assert r["jaccard"] == r["recall"] == r["precision"] == 1.0
    assert r["rmse_lat_nm"] == 0.0 and r["rmse_z_nm"] == 0.0


def test_errores_y_sesgos():
    p = GT.copy()
    p["x_px"] += 0.2
    p["y_px"] -= 0.1
    p["z_nm"] += 50.0
    r = _resumen(_pred(p[["video_id", "frame", "x_px", "y_px", "z_nm"]].to_numpy()))
    assert r["tp"] == 4
    assert np.isclose(r["rmse_x_nm"], 20) and np.isclose(r["rmse_y_nm"], 10) and np.isclose(r["rmse_z_nm"], 50)
    assert np.isclose(r["rmse_lat_nm"], np.hypot(20, 10))
    assert np.isclose(r["sesgo_x_nm"], 20) and np.isclose(r["sesgo_y_nm"], -10) and np.isclose(r["sesgo_z_nm"], 50)


def test_tolerancias():
    gt = _gt([(0, 0, 5.0, 5.0, 0.0)])
    dentro = _resumen(_pred([(0, 0, 6.49, 5.0, 299.0)]), gt)                 # 149 nm, 299 nm
    lateral = _resumen(_pred([(0, 0, 6.51, 5.0, 0.0)]), gt)                  # 151 nm
    axial = _resumen(_pred([(0, 0, 5.0, 5.0, 301.0)]), gt)
    assert dentro["tp"] == 1
    for r in (lateral, axial):
        assert (r["tp"], r["fp"], r["fn"], r["jaccard"]) == (0, 1, 1, 0.0)
    assert _resumen(_pred([(0, 0, 6.51, 5.0, 0.0)]), gt, r_xy_nm=200.0)["tp"] == 1


def test_hungaro_maximiza_pares():
    """Codicioso (par mas cercano primero) daria 1 par: p1-g2 a 20 nm deja a
    p2 sin pareja valida. El hungaro empareja p1-g1 y p2-g2."""
    gt = _gt([(0, 0, 0.0, 0.0, 0.0), (0, 0, 1.2, 0.0, 0.0)])
    pred = _pred([(0, 0, 1.0, 0.0, 0.0), (0, 0, 2.2, 0.0, 0.0)])
    r = _resumen(pred, gt)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
    assert np.isclose(r["rmse_lat_nm"], 100.0)


def test_fuera_del_fov_no_cuenta():
    gt = _gt([(0, 0, -0.7, 3.0, 0.0), (0, 0, 10.0, 3.0, 0.0), (0, 0, 20.0, -0.8, 0.0)], en_fov=[False, True, False])
    gt_e, pred_e = evaluar(_pred([(0, 0, -0.4, 3.0, 0.0), (0, 0, 10.0, 3.0, 0.0)]), gt, PX)
    r = resumen(gt_e, pred_e)
    assert (r["tp"], r["fp"], r["fn"], r["n_gt"], r["n_pred"]) == (1, 0, 0, 1, 1)
    assert list(pred_e.fuera_fov) == [True, False]


def test_frames_y_videos_no_se_mezclan():
    gt = _gt([(0, 0, 5.0, 5.0, 0.0)])
    for clave in ((0, 1), (1, 0)):
        r = _resumen(_pred([(*clave, 5.0, 5.0, 0.0)]), gt)
        assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)


def test_sin_detecciones():
    gt_e, pred_e = evaluar(_pred([]), GT, PX)
    r = resumen(gt_e, pred_e)
    assert (r["tp"], r["fp"], r["fn"], r["recall"], r["jaccard"]) == (0, 0, 4, 0.0, 0.0)
    assert np.isnan(r["precision"]) and np.isnan(r["rmse_lat_nm"])
    assert not gt_e.detectado.any() and gt_e.dx_nm.isna().all()
