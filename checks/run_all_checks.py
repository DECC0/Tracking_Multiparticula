"""
Corre todos los checks de sim_imagenesPSF.py en orden y da un resumen final.

Uso:
    ~/miniforge3/envs/tesis/bin/python checks/run_all_checks.py

(usa el interprete del entorno conda "tesis", que tiene numpy/pandas/
tifffile/matplotlib instalados; con el python del sistema fallara por
falta de dependencias)
"""
import subprocess
import sys
import time
from pathlib import Path

CHECKS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "check_psf.py",
    "check_psf_full_sweep.py",
    "check_colocar.py",
    "check_camara.py",
    "check_difusion.py",
    "check_e2e.py",
]


def main():
    resultados = []
    for nombre in SCRIPTS:
        ruta = CHECKS_DIR / nombre
        print(f"\n{'='*70}\n>>> {nombre}\n{'='*70}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(ruta)], cwd=CHECKS_DIR,
                               capture_output=True, text=True)
        dt = time.time() - t0
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
        resultados.append((nombre, proc.returncode == 0, dt))

    print(f"\n{'='*70}\nRESUMEN\n{'='*70}")
    ok_total = True
    for nombre, ok, dt in resultados:
        estado = "OK  " if ok else "FAIL"
        print(f"  [{estado}] {nombre:<28} ({dt:.1f}s)")
        ok_total &= ok

    pngs = sorted(CHECKS_DIR.glob("*.png"))
    print(f"\nfiguras generadas ({len(pngs)}):")
    for p in pngs:
        print(f"  {p.name}")

    if not ok_total:
        print("\nAlgun check fallo -- revisa el detalle arriba.")
        sys.exit(1)
    print("\nTodos los checks corrieron sin errores.")


if __name__ == "__main__":
    main()
