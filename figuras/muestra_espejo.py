"""
Carpeta de muestra para revisar a ojo el espejo de simulador_psf: un video con
el espejo activado, semilla fija y todos los demas parametros por defecto de
Config (el video 0 de generar_dataset con esa semilla).

Escribe en --salida (por defecto muestra_espejo/ en el directorio actual):
    frame_000_camara.png        frame 0 tal como lo ve la camara (con ruido), sin marcas
    frame_000_marcadores.png    el mismo frame con reales (circulos cian, con su particle_id) y reflejos
                                (cuadrados naranja) unidos por una linea fina, la
                                interseccion del espejo con el plano focal en linea
                                discontinua, y una leyenda en una franja debajo de la imagen
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
    # celula de levadura pegada al espejo:
    ~/miniforge3/envs/tesis/bin/python figuras/muestra_espejo.py --dist-max-um 4 [--z-ref-espejo-nm 0] [--z-rango-nm MIN MAX]
    # pares separados visibles (emisores brillantes, fondo bajo):
    ~/miniforge3/envs/tesis/bin/python figuras/muestra_espejo.py --dist-max-um 4 --fotones-rango 20000 30000 --fondo-rango 5 10
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from simulador_psf import Config, BancoPSF, Espejo, generar_video, guardar_video, guardar_gt

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seed", type=int, default=Config.seed)
ap.add_argument("--salida", default="muestra_espejo")
ap.add_argument("--zoom", type=int, default=8)
ap.add_argument("--percentiles", type=float, nargs=2, default=(1.0, 99.9), metavar=("NEGRO", "BLANCO"))
ap.add_argument("--ms", type=int, default=100, help="duracion de cada cuadro del GIF [ms]")
ap.add_argument("--dist-max-um", type=float, default=None, help="Config.dist_max_espejo_um [um]")
ap.add_argument("--z-ref-espejo-nm", type=float, default=None, help="Config.z_ref_espejo_nm [nm]")
ap.add_argument("--z-rango-nm", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                help="Config.z_rango_nm [nm]")
ap.add_argument("--fotones-rango", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                help=f"Config.fotones_rango [fotones por emisor y frame] (por defecto {Config.fotones_rango})")
ap.add_argument("--fondo-rango", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                help=f"Config.fondo_rango [fotones/px] (por defecto {Config.fondo_rango})")
args = ap.parse_args()

extra = {}                                                      # solo lo que se pide; el resto, por defecto
if args.dist_max_um is not None:
    extra["dist_max_espejo_um"] = args.dist_max_um
if args.z_ref_espejo_nm is not None:
    extra["z_ref_espejo_nm"] = args.z_ref_espejo_nm
if args.z_rango_nm is not None:
    extra["z_rango_nm"] = tuple(args.z_rango_nm)
if args.fotones_rango is not None:
    extra["fotones_rango"] = tuple(args.fotones_rango)
if args.fondo_rango is not None:
    extra["fondo_rango"] = tuple(args.fondo_rango)
cfg = Config(usar_espejo=True, seed=args.seed, **extra)
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


# interseccion del espejo con el plano focal (z = 0), recortada a la imagen
L = 4 * max(cfg.H, cfg.W) * cfg.pixel_nm
seg = espejo.punto_en_z_nm(0.0) + np.outer([-L, L], espejo.t)
linea = [img_px(*(seg[k, :2] / cfg.pixel_nm)) for k in range(2)]


def discontinua(dr, a, b, fill, largo=8, hueco=6, width=1):
    """Linea discontinua de a a b (PIL no las tiene)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.hypot(*(b - a))
    u = (b - a) / L
    for s in np.arange(0.0, L, largo + hueco):
        dr.line([tuple(a + s * u), tuple(a + min(s + largo, L) * u)], fill=fill, width=width)


AMARILLO, CIAN, NARANJA, ROJO, GRIS = (255, 230, 0), (0, 255, 255), (255, 150, 0), (255, 0, 0), (200, 200, 200)
FUENTE = ImageFont.load_default(size=13)
FRANJA = 44                                                     # alto de la leyenda bajo la imagen [px]


def leyenda(im):
    """Agrega debajo de im una franja negra con la leyenda (no tapa la imagen)."""
    out = Image.new("RGB", (im.width, im.height + FRANJA), (0, 0, 0))
    out.paste(im, (0, 0))
    dr = ImageDraw.Draw(out)
    y1, y2 = im.height + 6, im.height + 25
    r = 5
    dr.ellipse([8 - r, y1 + 7 - r, 8 + r, y1 + 7 + r], outline=CIAN, width=2)
    dr.text((18, y1), "real", fill=(255, 255, 255), font=FUENTE)
    dr.rectangle([68 - r, y1 + 7 - r, 68 + r, y1 + 7 + r], outline=NARANJA, width=2)
    dr.text((78, y1), "reflejo", fill=(255, 255, 255), font=FUENTE)
    dr.rectangle([148 - r, y1 + 7 - r, 148 + r, y1 + 7 + r], outline=ROJO, width=2)
    dr.text((158, y1), "reflejo no renderizado", fill=(255, 255, 255), font=FUENTE)
    dr.line([(4, y2 + 7), (24, y2 + 7)], fill=GRIS, width=1)
    dr.text((30, y2), "vinculo", fill=(255, 255, 255), font=FUENTE)
    discontinua(dr, (98, y2 + 7), (128, y2 + 7), AMARILLO, largo=6, hueco=4)
    dr.text((134, y2), "interseccion del espejo con el plano focal", fill=(255, 255, 255), font=FUENTE)
    return out


def con_marcas(frame, t):
    im = a_rgb(frame)
    dr = ImageDraw.Draw(im)
    discontinua(dr, linea[0], linea[1], AMARILLO)
    r = 0.45 * Z
    for f in gt[gt.frame == t].itertuples():
        a, b = img_px(f.x_px, f.y_px), img_px(f.x_reflejo_px, f.y_reflejo_px)
        dr.line([a, b], fill=GRIS, width=1)
        dr.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r], outline=CIAN, width=2)
        dr.text((a[0] + r + 2, a[1] - r - 12), str(f.particle_id), fill=CIAN, font=FUENTE,
                stroke_width=2, stroke_fill=(0, 0, 0))                          # para cruzar con el CSV
        color = NARANJA if f.reflejo_en_banco else ROJO               # rojo: reflejo no renderizado
        dr.rectangle([b[0] - r, b[1] - r, b[0] + r, b[1] + r], outline=color, width=2)
    return leyenda(im)


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
    "reproducir: ~/miniforge3/envs/tesis/bin/python " + " ".join(["figuras/muestra_espejo.py"] + sys.argv[1:]),
    f"    = generar_video(Config(usar_espejo=True, seed={cfg.seed}"
    + "".join(f", {k}={v!r}" for k, v in extra.items())
    + f"), BancoPSF(cfg), np.random.default_rng({cfg.seed}), 0)",
    "",
    "espejo:",
    f"  angulo_espejo_deg      {cfg.angulo_espejo_deg} (espejo-plano focal, dentro de la cuna de la muestra)",
    f"  orientacion_espejo_deg {cfg.orientacion_espejo_deg} (azimut de la direccion linea -> muestra, de +x hacia +y)",
    f"  espejo_punto_px        {cfg.espejo_punto_px} (punto de la linea del espejo en el plano de referencia)",
    f"  p0 [nm]                {np.round(espejo.p0, 2).tolist()}",
    f"  normal n               {np.round(espejo.n, 6).tolist()}",
    f"  ratio_reflejo          {cfg.ratio_reflejo}",
    f"  z_ref_espejo_nm        {espejo.z_ref} (z del plano de referencia respecto del foco)",
    f"  dist_max_espejo_um     {cfg.dist_max_espejo_um}",
    f"  z_rango_nm (reales)    {cfg.z_rango_nm}",
    f"  signo_z {cfg.signo_z}, signo_inclinacion {cfg.signo_inclinacion}",
    f"  banco de PSF: z en [{banco.z_grid[0]:.0f}, {banco.z_grid[-1]:.0f}] nm, {len(banco.z_grid)} planos",
    "",
    f"video: {cfg.n_frames} frames de {cfg.H}x{cfg.W} px, dt {cfg.dt_s} s, semilla {cfg.seed}, video_id 0",
    f"fotones_rango {cfg.fotones_rango} fotones/emisor/frame ({cfg.muestreo_fotones}), "
    f"fondo_rango {cfg.fondo_rango} fotones/px (fondo_no_uniforme {cfg.fondo_no_uniforme})"
    + ("" if "fotones_rango" in extra or "fondo_rango" in extra else ", los de Config"),
    f"emisores: {len(D)}",
    "  particle_id  D [um^2/s]  fotones/frame  fotones reflejo",
    *[f"  {p:11d}  {D[p]:10.4f}  {N[p]:13.0f}  {cfg.ratio_reflejo * N[p]:15.0f}" for p in D.index],
    f"reflejo_en_fov {100 * gt.reflejo_en_fov.mean():.1f}% de las filas, "
    f"reflejo_en_banco {100 * gt.reflejo_en_banco.mean():.1f}%",
    "",
    f"contraste fijo (todos los PNG y GIF): negro = {lo:.0f} ADU, blanco = {hi:.0f} ADU "
    f"(percentiles {args.percentiles[0]}, {args.percentiles[1]} del video); zoom x{Z}",
    "marcas: circulo cian = real con su particle_id, cuadrado naranja = reflejo (rojo si no se renderizo:",
    "        fuera del banco),",
    "        linea gris = vinculo real-reflejo, linea amarilla discontinua = interseccion del",
    "        espejo con el plano focal; leyenda en una franja bajo la imagen",
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
