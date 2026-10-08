"""
GIF (+ tif y ground truth) de un video de sim_imagenesPSF.py.

Por defecto reproduce el video de la seed 89 de fig_cuadros_gt a 256x256
(video_fig_cuadros_gt_b_256). Cambiar --fondo, --fotones o --read-noise no
altera el numero de llamadas al rng, asi que las trayectorias y el brillo
relativo de cada particula se mantienen; solo cambian el ruido y el contraste.

Escribe en el directorio actual:
  video_fig_cuadros_gt_b_<W><sufijo>.gif / .tif
  ground_truth_fig_cuadros_gt_b_<W><sufijo>.csv

Uso:
    ~/miniforge3/envs/tesis/bin/python figuras/gif_video.py
    ~/miniforge3/envs/tesis/bin/python figuras/gif_video.py --fondo 20 160 --read-noise 9.78 --sufijo _ruido
    ~/miniforge3/envs/tesis/bin/python figuras/gif_video.py --fondo 10 80 --fotones 3000 90000 --sufijo _contraste
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from sim_imagenesPSF import Config, BancoPSF, generar_video

base = Config()
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=89)
ap.add_argument("--tam", type=int, default=256, help="H = W [px]")
ap.add_argument("--frames", type=int, default=11)
ap.add_argument("--fondo", type=float, nargs=2, default=base.fondo_rango,
                metavar=("MIN", "MAX"), help="fotones/px")
ap.add_argument("--fotones", type=float, nargs=2, default=base.fotones_rango,
                metavar=("MIN", "MAX"), help="fotones por emisor")
ap.add_argument("--read-noise", type=float, default=base.read_noise, help="ADU rms")
ap.add_argument("--percentiles", type=float, nargs=2, default=(1.0, 99.9),
                metavar=("NEGRO", "BLANCO"), help="escala de grises del GIF")
ap.add_argument("--ms", type=int, default=100, help="duracion de cada cuadro [ms]")
ap.add_argument("--sufijo", default="")
args = ap.parse_args()

# n_emisores_rango = (4, 12) como en fig_cuadros_gt de figuras_documento.py
cfg = Config(mascara="esferica", n_frames=args.frames, n_emisores_rango=(4, 12),
             seed=args.seed, H=args.tam, W=args.tam,
             fondo_rango=tuple(args.fondo), fotones_rango=tuple(args.fotones),
             read_noise=args.read_noise)
print(cfg.resumen())

banco = BancoPSF(cfg)
rng = np.random.default_rng(cfg.seed)
video, filas = generar_video(cfg, banco, rng, video_id=cfg.seed)

nombre = f"fig_cuadros_gt_b_{cfg.W}{args.sufijo}"
tifffile.imwrite(f"video_{nombre}.tif", video, imagej=True,
                 metadata={"axes": "TYX", "finterval": cfg.dt_s})
pd.DataFrame(filas).to_csv(f"ground_truth_{nombre}.csv", index=False)

# escala de grises fija en todo el video (percentiles sobre todos los cuadros)
lo, hi = np.percentile(video, args.percentiles)
cuadros = [Image.fromarray((np.clip((f - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))
           for f in video]
cuadros[0].save(f"video_{nombre}.gif", save_all=True, append_images=cuadros[1:],
                duration=args.ms, loop=0)
print(f"video_{nombre}.gif : {len(cuadros)} cuadros, negro = {lo:.0f} ADU, "
      f"blanco = {hi:.0f} ADU")
