"""
Escritura de las salidas: videos .tif y ground truth en CSV.

Formato (sin cambios respecto de sim_imagenesPSF.py):
    video_XXX.tif     (T, H, W) uint16, ImageJ con ejes TYX y finterval = dt_s
    ground_truth.csv  una fila por (video_id, frame, particle_id) con las
                      columnas COLUMNAS_GT; x_px, y_px continuos (el pixel
                      (fila i, columna j) esta centrado en x = j, y = i), z_nm
                      en nm, photons por frame. Las particulas fuera de
                      0 <= x < W, 0 <= y < H se conservan con en_fov=False.
                      Con espejo se agregan al final COLUMNAS_REFLEJO.
"""

from pathlib import Path
import tifffile

COLUMNAS_GT = ("video_id", "frame", "particle_id", "x_px", "y_px", "z_nm",
               "photons", "D_um2_s", "en_fov")

# Solo con cfg.usar_espejo, en la misma fila del emisor real (particle_id es el
# vinculo real <-> reflejo): posicion del reflejo en las mismas unidades,
# photons_reflejo = ratio_reflejo * photons, y dos banderas: reflejo_en_fov
# (mismo criterio que en_fov) y reflejo_en_banco (z_reflejo_nm dentro del banco
# de PSF; si es False el reflejo NO se renderizo).
COLUMNAS_REFLEJO = ("x_reflejo_px", "y_reflejo_px", "z_reflejo_nm", "photons_reflejo",
                    "reflejo_en_fov", "reflejo_en_banco")
COLUMNAS_GT_ESPEJO = COLUMNAS_GT + COLUMNAS_REFLEJO


def nombre_video(video_id):
    """Nombre del archivo del video video_id: video_XXX.tif."""
    return f"video_{video_id:03d}.tif"


def guardar_video(ruta, video, dt_s):
    """Escribe video (T, H, W) uint16 en ruta como TIFF de ImageJ, con el
    intervalo entre frames dt_s [s] en los metadatos."""
    tifffile.imwrite(Path(ruta), video, imagej=True,
                     metadata={"axes": "TYX", "finterval": dt_s})


def guardar_gt(gt, ruta="ground_truth.csv"):
    """Escribe el DataFrame de ground truth (columnas COLUMNAS_GT) en ruta, sin indice."""
    gt.to_csv(ruta, index=False)
