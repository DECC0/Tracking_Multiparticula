"""
Detecciones vs ground truth, frame por frame (video_id, frame).

Emparejamiento hungaro (scipy.optimize.linear_sum_assignment) con tolerancia
cilindrica: distancia lateral <= r_xy_nm y |dz| <= r_z_nm. Un par valido
cuesta (d_xy / r_xy)^2 + (dz / r_z)^2 y uno invalido mas que cualquier
emparejamiento de pares validos, asi que se maximiza el numero de pares y,
entre esos, se minimiza el error.

Los emisores del GT fuera del FOV (en_fov = False) entran al emparejamiento
pero no cuentan: no son falsos negativos, y la deteccion que se les empareja
no es falso positivo (la red no se entrena con ellos, pero uno justo afuera
del borde se ve en la imagen).

Metricas (resumen): Jaccard = TP / (TP + FP + FN), recall = TP / (TP + FN),
precision = TP / (TP + FP); sobre los pares emparejados, RMSE lateral
sqrt(<dx^2 + dy^2>), por eje y axial, y sesgo <d> (d = deteccion - GT), en nm.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

CLAVES = ["video_id", "frame"]
COSTO_INVALIDO = 1e6                    # un par valido cuesta <= 2


def emparejar(pred_nm, gt_nm, r_xy_nm, r_z_nm):
    """Pares (i_pred, i_gt) del emparejamiento hungaro entre posiciones
    (n, 3) y (m, 3) [x, y, z] en nm, solo los que caen dentro de la tolerancia."""
    if len(pred_nm) == 0 or len(gt_nm) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = pred_nm[:, None, :] - gt_nm[None, :, :]
    dxy2 = d[..., 0]**2 + d[..., 1]**2
    dz = np.abs(d[..., 2])
    valido = (dxy2 <= r_xy_nm**2) & (dz <= r_z_nm)
    costo = np.where(valido, dxy2 / r_xy_nm**2 + (dz / r_z_nm)**2, COSTO_INVALIDO)
    ip, ig = linear_sum_assignment(costo)
    ok = valido[ip, ig]
    return ip[ok], ig[ok]


def _nm(df, px_nm):
    return np.c_[df.x_px.to_numpy(float) * px_nm, df.y_px.to_numpy(float) * px_nm, df.z_nm.to_numpy(float)]


def evaluar(pred, gt, px_nm, r_xy_nm=150.0, r_z_nm=300.0):
    """Empareja pred (columnas de postproc.COLUMNAS_PRED) con gt (columnas del
    simulador; en_fov opcional) en cada (video_id, frame).

    Devuelve (gt_eval, pred_eval):
        gt_eval   : gt + detectado, conf, dx_nm, dy_nm, dz_nm (error de su deteccion, NaN si no la tiene)
        pred_eval : pred + tp, fuera_fov (emparejada con un emisor fuera del FOV: no cuenta)"""
    gt = gt.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    en_fov = gt.en_fov.to_numpy(bool) if "en_fov" in gt else np.ones(len(gt), bool)
    g_nm, p_nm = _nm(gt, px_nm), _nm(pred, px_nm)
    detectado = np.zeros(len(gt), bool)
    conf = np.full(len(gt), np.nan)
    err = np.full((len(gt), 3), np.nan)
    tp = np.zeros(len(pred), bool)
    fuera = np.zeros(len(pred), bool)

    grupos_gt = gt.groupby(CLAVES).indices if len(gt) else {}
    grupos_pred = pred.groupby(CLAVES).indices if len(pred) else {}
    for clave, ip_todos in grupos_pred.items():
        ig_todos = grupos_gt.get(clave)
        if ig_todos is None:
            continue
        ip, ig = emparejar(p_nm[ip_todos], g_nm[ig_todos], r_xy_nm, r_z_nm)
        ip, ig = ip_todos[ip], ig_todos[ig]
        tp[ip] = en_fov[ig]
        fuera[ip] = ~en_fov[ig]
        detectado[ig] = True
        conf[ig] = pred.conf.to_numpy(float)[ip]
        err[ig] = p_nm[ip] - g_nm[ig]

    gt_eval = gt.assign(detectado=detectado, conf=conf, dx_nm=err[:, 0], dy_nm=err[:, 1], dz_nm=err[:, 2])
    pred_eval = pred.assign(tp=tp, fuera_fov=fuera)
    return gt_eval, pred_eval


def _cociente(a, b):
    return a / b if b else float("nan")


def resumen(gt_eval, pred_eval):
    """Metricas globales de evaluar(): dict con conteos, Jaccard, recall,
    precision, RMSE y sesgo (nm)."""
    g = gt_eval[gt_eval.en_fov] if "en_fov" in gt_eval else gt_eval
    tp = int(g.detectado.sum())
    fn = len(g) - tp
    fp = int((~pred_eval.tp & ~pred_eval.fuera_fov).sum())
    pares = g[g.detectado]
    dx, dy, dz = (pares[c].to_numpy(float) for c in ("dx_nm", "dy_nm", "dz_nm"))
    rms = lambda a: float(np.sqrt(np.mean(a**2))) if len(a) else float("nan")
    media = lambda a: float(np.mean(a)) if len(a) else float("nan")
    return dict(n_gt=len(g), n_pred=tp + fp, tp=tp, fp=fp, fn=fn,
                jaccard=_cociente(tp, tp + fp + fn), recall=_cociente(tp, tp + fn),
                precision=_cociente(tp, tp + fp),
                rmse_lat_nm=rms(np.hypot(dx, dy)), rmse_x_nm=rms(dx), rmse_y_nm=rms(dy), rmse_z_nm=rms(dz),
                sesgo_x_nm=media(dx), sesgo_y_nm=media(dy), sesgo_z_nm=media(dz))


# ----------------------------------------------------------------------------
# Espejo: distancia al espejo y falsos positivos sobre reflejos
# ----------------------------------------------------------------------------

COLUMNAS_REFLEJO_XYZ = ["x_reflejo_px", "y_reflejo_px", "z_reflejo_nm"]


def tiene_reflejos(gt):
    return set(COLUMNAS_REFLEJO_XYZ) <= set(gt.columns)


def distancia_al_espejo_nm(gt, px_nm):
    """Distancia (perpendicular) de cada emisor real al plano del espejo [nm]:
    p' - p = -2 d n, asi que d = |p' - p| / 2 (no hace falta la geometria)."""
    d = np.c_[(gt.x_reflejo_px - gt.x_px) * px_nm, (gt.y_reflejo_px - gt.y_px) * px_nm, gt.z_reflejo_nm - gt.z_nm]
    return 0.5 * np.linalg.norm(d.astype(float), axis=1)


def fp_sobre_reflejos(pred_eval, gt, px_nm, r_xy_nm=150.0):
    """Marca los falsos positivos (ni tp ni fuera_fov) que caen a distancia
    lateral <= r_xy_nm de algun reflejo del mismo frame (la z del reflejo no se
    usa: suele quedar fuera del z_rango de la red). Devuelve (pred_eval +
    sobre_reflejo, gt + reflejo_con_fp: algun FP sobre ese reflejo)."""
    pred_eval, gt = pred_eval.reset_index(drop=True), gt.reset_index(drop=True)
    sobre = np.zeros(len(pred_eval), bool)
    con_fp = np.zeros(len(gt), bool)
    fp = (~pred_eval.tp & ~pred_eval.fuera_fov).to_numpy()
    grupos_gt = gt.groupby(CLAVES).indices if len(gt) else {}
    p_xy = pred_eval[["x_px", "y_px"]].to_numpy(float) * px_nm
    r_xy = gt[COLUMNAS_REFLEJO_XYZ[:2]].to_numpy(float) * px_nm
    for clave, ip in (pred_eval[fp].groupby(CLAVES).indices.items() if fp.any() else []):
        ip = np.flatnonzero(fp)[ip]
        ig = grupos_gt.get(clave)
        if ig is None:
            continue
        d2 = ((p_xy[ip, None, :] - r_xy[None, ig, :]) ** 2).sum(-1)
        cerca = d2 <= r_xy_nm ** 2
        sobre[ip] = cerca.any(axis=1)
        con_fp[ig] = cerca.any(axis=0)
    return pred_eval.assign(sobre_reflejo=sobre), gt.assign(reflejo_con_fp=con_fp)


def resumen_espejo(pred_eval, gt_eval):
    """Conteos de fp_sobre_reflejos: FP sobre reflejos, su fraccion de los FP, y
    fraccion de los reflejos dentro del FOV con algun FP encima."""
    fp = ~pred_eval.tp & ~pred_eval.fuera_fov
    n_fp, n_sobre = int(fp.sum()), int((fp & pred_eval.sobre_reflejo).sum())
    refl = gt_eval[gt_eval.reflejo_en_fov] if "reflejo_en_fov" in gt_eval else gt_eval
    return dict(fp=n_fp, fp_sobre_reflejo=n_sobre, fraccion_fp_sobre_reflejo=_cociente(n_sobre, n_fp),
                reflejos_en_fov=len(refl), reflejos_con_fp=int(refl.reflejo_con_fp.sum()),
                fraccion_reflejos_con_fp=_cociente(int(refl.reflejo_con_fp.sum()), len(refl)))
