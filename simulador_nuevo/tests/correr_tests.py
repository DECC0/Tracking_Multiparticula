"""
Corre todos los tests del simulador nuevo (sin depender de pytest) y da un
resumen OK/FAIL. Los archivos test_*.py tambien son compatibles con pytest.

Uso:
    ~/miniforge3/envs/tesis/bin/python simulador_nuevo/tests/correr_tests.py            # todos
    ~/miniforge3/envs/tesis/bin/python simulador_nuevo/tests/correr_tests.py psf camara # filtra por nombre
"""
import importlib
import inspect
import os
import sys
import time
import traceback

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

MODULOS = [
    "test_config",
    "test_psf",
    "test_colocar",
    "test_camara",
    "test_fondo",
    "test_difusion",
    "test_video",
    "test_diagnosticos",
]


def main():
    filtros = sys.argv[1:]
    resultados = []
    for nombre in MODULOS:
        try:
            mod = importlib.import_module(nombre)
        except Exception:
            print(f"[FAIL] {nombre}: no se pudo importar")
            traceback.print_exc(limit=1)
            resultados.append((nombre, False, 0.0))
            continue
        tests = sorted((f for n, f in inspect.getmembers(mod, inspect.isfunction)
                        if n.startswith("test_") and f.__module__ == nombre),
                       key=lambda f: f.__code__.co_firstlineno)
        for f in tests:
            ident = f"{nombre}.{f.__name__}"
            if filtros and not any(s in ident for s in filtros):
                continue
            t0 = time.time()
            try:
                f()
                ok = True
            except Exception as e:
                ok = False
                tb = traceback.extract_tb(e.__traceback__)[-1]
                print(f"[FAIL] {ident}  ({type(e).__name__} en linea {tb.lineno}: {e})")
            dt = time.time() - t0
            if ok:
                print(f"[OK  ] {ident}  ({dt:.1f}s)")
            resultados.append((ident, ok, dt))

    n_ok = sum(ok for _, ok, _ in resultados)
    print(f"\n{n_ok}/{len(resultados)} tests OK  ({sum(dt for *_, dt in resultados):.1f}s)")
    sys.exit(0 if n_ok == len(resultados) else 1)


if __name__ == "__main__":
    main()
