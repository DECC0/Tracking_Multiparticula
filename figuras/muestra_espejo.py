"""
Carpeta de muestra para revisar a ojo el espejo de simulador_psf: un video con
el espejo activado, semilla fija y todos los demas parametros por defecto de
Config (el video 0 de generar_dataset con esa semilla).

Escribe en --salida (por defecto muestra_espejo/ en el directorio actual):
    frame_000_camara.png        frame 0 tal como lo ve la camara (con ruido), sin marcas
    frame_000_marcadores.png    el mismo frame con reales (circulos cian) y reflejos
                                (cuadrados naranja) unidos por una linea fina, y la
                                linea del espejo en el cubreobjetos
    video.gif / video_marcadores.gif   toda la secuencia, sin y con marcas
    video.tif                   el video en ADU (uint16, ImageJ TYX)
    ground_truth.csv            GT con las columnas del reflejo
    parametros.txt              configuracion completa para reproducirla

El contraste es fijo en todos los frames y archivos (percentiles sobre todo el
video, como figuras/gif_video.py), para que no haya parpadeo por autoescala.
Los PNG y GIF se agrandan --zoom veces (vecino mas cercano: cada pixel de la
camara queda como un bloque).

Uso:
    ~/miniforge3/envs/tesis/bin/python figuras/muestra_espejo.py [--seed 0] [--salida muestra_espejo]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from simulador_psf import Config, BancoPSF, Espejo, generar_video, guardar_video, guardar_gt
from simulador_psf.espejo import SIGNO_Z, SIGNO_INCLINACION

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seed", type=int, default=Config.seed)
ap.add_argument("--salida", default="muestra_espejo")
ap.add_argument("--zoom", type=int, default=8)
ap.add_argument("--percentiles", type=float, nargs=2, default=(1.0, 99.9), metavar=("NEGRO", "BLANCO"))
ap.add_argument("--ms", type=int, default=100, help="duracion de cada cuadro del GIF [ms]")
args = ap.parse_args()

cfg = Config(usar_espejo=True, seed=args.seed)
print(cfg.resumen())
banco = BancoPSF(cfg)
espejo = Espejo(cfg)
video, filas = generar_video(cfg, banco, np.random.default_rng(cfg.seed), video_id=0)
gt = pd.DataFrame(filas)

sal = Path(args.salida)
sal.mkdir(parents=True, exist_ok=True)
guardar_video(sal / "video.tif", video, cfg.dt_s)
guardar_gt(gt, sal / "ground_truth.csv")

# --- contraste fijo para todo el video ---
lo, hi = np.percentile(video, args.percentiles)
Z = args.zoom


def a_rgb(frame):
    g = (np.clip((frame - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(g).resize((cfg.W * Z, cfg.H * Z), Image.NEAREST).convert("RGB")


def img_px(x, y):
    """[px continuo] -> coordenada de la imagen agrandada: el pixel j cubre [j-0.5, j+0.5)."""
    return ((x + 0.5) * Z, (y + 0.5) * Z)


# linea del espejo en el cubreobjetos, recortada a la imagen
L = 4 * max(cfg.H, cfg.W) * cfg.pixel_nm
seg = espejo.p0 + np.outer([-L, L], espejo.t)
linea = [img_px(*(seg[k, :2] / cfg.pixel_nm)) for k in range(2)]


def con_marcas(frame, t):
    im = a_rgb(frame)
    dr = ImageDraw.Draw(im)
    dr.line(linea, fill=(255, 230, 0), width=1)
    r = 0.45 * Z
    for f in gt[gt.frame == t].itertuples():
        a, b = img_px(f.x_px, f.y_px), img_px(f.x_reflejo_px, f.y_reflejo_px)
        dr.line([a, b], fill=(200, 200, 200), width=1)
        dr.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r], outline=(0, 255, 255), width=2)
        color = (255, 150, 0) if f.reflejo_en_banco else (255, 0, 0)   # rojo: reflejo no renderizado
        dr.rectangle([b[0] - r, b[1] - r, b[0] + r, b[1] + r], outline=color, width=2)
    return im


a_rgb(video[0]).save(sal / "frame_000_camara.png")
con_marcas(video[0], 0).save(sal / "frame_000_marcadores.png")
for nombre, fn in (("video.gif", lambda f, t: a_rgb(f)), ("video_marcadores.gif", con_marcas)):
    cuadros = [fn(f, t).quantize(colors=256) for t, f in enumerate(video)]
    cuadros[0].save(sal / nombre, save_all=True, append_images=cuadros[1:], duration=args.ms, loop=0)

# --- parametros para reproducir ---
D = gt.groupby("particle_id").D_um2_s.first()
N = gt.groupby("particle_id").photons.first()
lineas = [
    "Muestra del espejo de simulador_psf (figuras/muestra_espejo.py)",
    f"reproducir: ~/miniforge3/envs/tesis/bin/python figuras/muestra_espejo.py --seed {args.seed} "
    f"--salida {args.salida} --zoom {Z}",
    "    = generar_video(Config(usar_espejo=True, seed=SEED), BancoPSF(cfg), np.random.default_rng(SEED), 0)",
    "",
    "espejo:",
    f"  angulo_espejo_deg      {cfg.angulo_espejo_deg} (espejo-cubreobjetos, dentro de la cuna de la muestra)",
    f"  orientacion_espejo_deg {cfg.orientacion_espejo_deg} (azimut de la direccion linea -> muestra, de +x hacia +y)",
    f"  espejo_punto_px        {cfg.espejo_punto_px} (punto de la linea del espejo en el cubreobjetos)",
    f"  p0 [nm]                {np.round(espejo.p0, 2).tolist()}",
    f"  normal n               {np.round(espejo.n, 6).tolist()}",
    f"  ratio_reflejo          {cfg.ratio_reflejo}",
    f"  SIGNO_Z {SIGNO_Z}, SIGNO_INCLINACION {SIGNO_INCLINACION}",
    f"  banco de PSF: z en [{banco.z_grid[0]:.0f}, {banco.z_grid[-1]:.0f}] nm, {len(banco.z_grid)} planos",
    "",
    f"video: {cfg.n_frames} frames de {cfg.H}x{cfg.W} px, dt {cfg.dt_s} s, semilla {cfg.seed}, video_id 0",
    f"emisores: {len(D)}",
    "  particle_id  D [um^2/s]  fotones/frame  fotones reflejo",
    *[f"  {p:11d}  {D[p]:10.4f}  {N[p]:13.0f}  {cfg.ratio_reflejo * N[p]:15.0f}" for p in D.index],
    f"reflejo_en_fov {100 * gt.reflejo_en_fov.mean():.1f}% de las filas, "
    f"reflejo_en_banco {100 * gt.reflejo_en_banco.mean():.1f}%",
    "",
    f"contraste fijo (todos los PNG y GIF): negro = {lo:.0f} ADU, blanco = {hi:.0f} ADU "
    f"(percentiles {args.percentiles[0]}, {args.percentiles[1]} del video); zoom x{Z}",
    "marcas: circulo cian = real, cuadrado naranja = reflejo (rojo si no se renderizo: fuera del banco),",
    "        linea gris = vinculo real-reflejo, linea amarilla = espejo en el cubreobjetos",
    "",
    "Config completa:",
    *[f"  {k} = {v}" for k, v in asdict(cfg).items()],
    "",
    "resumen():",
    *[f"  {s}" for s in cfg.resumen().splitlines()],
]
(sal / "parametros.txt").write_text("\n".join(lineas) + "\n")
print(f"\nescrito en {sal.resolve()}:")
for f in sorted(sal.iterdir()):
    print(f"  {f.name:28s} {f.stat().st_size/1024:8.1f} KB")
