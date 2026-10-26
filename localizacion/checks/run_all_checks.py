"""
Corre todos los checks (diagnosticos con figuras) de la red de localizacion y
da un resumen. Un check "falla" solo si lanza una excepcion; la validacion con
asserts esta en tests/.

Uso:
    ~/miniforge3/envs/tesis/bin/python localizacion/checks/run_all_checks.py
"""
import subprocess
import sys
import time
from pathlib import Path

CHECKS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "check_datos.py",
    "check_modelo.py",
    "check_postproc.py",
]


def main():
    resultados = []
    for nombre in SCRIPTS:
        print(f"\n{'='*70}\n>>> {nombre}\n{'='*70}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(CHECKS_DIR / nombre)], cwd=CHECKS_DIR,
                              capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
        resultados.append((nombre, proc.returncode == 0, time.time() - t0))

    print(f"\n{'='*70}\nRESUMEN\n{'='*70}")
    for nombre, ok, dt in resultados:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {nombre:<22} ({dt:.1f}s)")
    if not all(ok for _, ok, _ in resultados):
        print("\nAlgun check fallo -- revisa el detalle arriba.")
        sys.exit(1)
    print("\nTodos los checks corrieron sin errores.")


if __name__ == "__main__":
    main()
