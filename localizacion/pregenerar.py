"""
Pregenera un split (train o val) a disco, para no simular durante el
entrenamiento (p. ej. en el cluster) o para fijar el set de validacion.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY localizacion/pregenerar.py --config localizacion/configs/base.yaml --split val --salida datos/val
    $PY localizacion/pregenerar.py --split train --n 2000 --salida datos/train --set datos.simulador.H=32
"""
import argparse
import time

from config import agregar_argumentos, config_desde_args
from data.dataset import pregenerar


def main():
    p = agregar_argumentos(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--salida", required=True, help="carpeta de salida (relativa al directorio de trabajo)")
    p.add_argument("--n", type=int, default=None, help="numero de muestras (por defecto datos.n_<split>)")
    args = p.parse_args()

    cfg = config_desde_args(args).datos
    n = args.n or getattr(cfg, f"n_{args.split}")
    seed = getattr(cfg, f"seed_{args.split}")
    t0 = time.time()
    gt = pregenerar(cfg, n, seed, args.salida, cada=max(1, n // 10))
    print(f"{n} muestras ({args.split}, seed {seed}) -> {args.salida}/  "
          f"[{gt.en_fov.sum()} emisores en el FOV, {time.time() - t0:.1f} s]")


if __name__ == "__main__":
    main()
