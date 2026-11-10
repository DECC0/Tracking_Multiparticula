"""
Genera un dataset con el espejo activo (formato de pregenerar.py) a partir de
configs/espejo.yaml: angulo 45 +- 5 grados desde el eje optico, ratio_reflejo
= 0.8, dist_max_espejo_um = 4, geometria aleatoria por muestra, y el lado de
los anillos de la PSF como parametro (--anillos-hacia-objetivo; por defecto
False, el valor del simulador). Convencion: +z hacia el objetivo.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY localizacion/generar_espejo.py --split val --salida localizacion/datos/val_espejo_b45_lejos
    $PY localizacion/generar_espejo.py --anillos-hacia-objetivo --split test --n 2000 --salida localizacion/datos/test_espejo_b45_cerca
    $PY localizacion/generar_espejo.py --sin-espejo --split test --n 2000 --salida localizacion/datos/test_sin_espejo

--split test usa la semilla --seed (por defecto 3, distinta de seed_train = 1
y seed_val = 2). --sin-espejo genera las mismas muestras (seed, idx) con el
espejo apagado: mismos n, fotones y fondo; las posiciones difieren, porque con
espejo se pliegan al lado de la muestra y, con dist_max_espejo_um, se sortean
en la franja junto al espejo (que puede caer en parte fuera del FOV: hay menos
reales dentro de la imagen). Al terminar imprime los reintentos (muestras
regeneradas porque algun reflejo quedaba fuera del banco) y los reflejos.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from config import cargar_config
from data.dataset import pregenerar

CONFIG = Path(__file__).resolve().parent / "configs" / "espejo.yaml"
SIN_ESPEJO = ["datos.simulador.usar_espejo=false", "datos.espejo_angulo_eje_rango_deg=null",
              "datos.espejo_orientacion_aleatoria=false", "datos.espejo_punto_aleatorio=false",
              "datos.canal_distancia_espejo=false"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--anillos-hacia-objetivo", action="store_true",
                   help="Config.anillos_hacia_objetivo = True (la PSF esferica gana anillos hacia +z)")
    p.add_argument("--sin-espejo", action="store_true", help="las mismas muestras con el espejo apagado")
    p.add_argument("--split", choices=("train", "val", "test"), required=True)
    p.add_argument("--n", type=int, default=None, help="numero de muestras (por defecto datos.n_<split>; test: n_val)")
    p.add_argument("--seed", type=int, default=3, help="semilla del split test")
    p.add_argument("--salida", required=True, help="carpeta de salida (relativa al directorio de trabajo)")
    p.add_argument("--config", default=str(CONFIG), help="config base (por defecto configs/espejo.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="CLAVE=VALOR", help="overrides extra del YAML")
    args = p.parse_args()

    anillos = "true" if args.anillos_hacia_objetivo else "false"
    overrides = [f"datos.simulador.anillos_hacia_objetivo={anillos}"] + (SIN_ESPEJO if args.sin_espejo else []) + args.set
    cfg = cargar_config(args.config, overrides).datos
    split = "val" if args.split == "test" else args.split
    n = args.n or getattr(cfg, f"n_{split}")
    seed = args.seed if args.split == "test" else getattr(cfg, f"seed_{split}")
    t0 = time.time()
    gt = pregenerar(cfg, n, seed, args.salida, cada=max(1, n // 10))
    print(f"{n} muestras ({args.split}, seed {seed}, anillos {'hacia' if args.anillos_hacia_objetivo else 'lejos del'} objetivo, "
          f"{'sin' if args.sin_espejo else 'con'} espejo) -> {args.salida}/  "
          f"[{int(gt.en_fov.sum())} emisores en el FOV, {time.time() - t0:.1f} s]")
    if not args.sin_espejo:
        m = pd.read_csv(Path(args.salida) / "muestras.csv")
        print(f"  reintentos: {int(m.reintentos.sum())} en {int((m.reintentos > 0).sum())} muestras; "
              f"reflejos sin dibujar: {int((~gt.reflejo_en_banco).sum())}; "
              f"reflejos en el FOV: {int(gt.reflejo_en_fov.sum())} de {len(gt)}")


if __name__ == "__main__":
    main()
