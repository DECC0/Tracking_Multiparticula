"""Regresion bit a bit de las muestras de configs/celula.yaml (espejo, celula,
recorte de pupila del real y del reflejo, geometria, celula y ratio sorteados
por muestra) y de configs/espejo.yaml: hashes de la imagen, el ground truth y
la meta de unas pocas muestras de generar_muestra, contra
referencia_regresion_celula.json, escrito con el codigo de antes del
simulador v2 (camara medida, fondo con autofluorescencia de la celula,
muestreo de z con peso en los polos y PSF aleatorizada). Con los valores por
defecto de las opciones nuevas, las muestras de hoy no cambian.

Para que el banco se construya rapido, z_paso_nm = 100 (el banco solo se usa
para la SNR: con el recorte de pupila el real y el reflejo se dibujan con
PSFRecortada). Los hashes dependen de la version de numpy: con otra version el
test se salta con un aviso. Regenerar solo si se cambia el comportamiento a
proposito:
    $PY localizacion/tests/test_regresion_celula.py --generar
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import comun  # noqa: F401  (agrega localizacion/ a sys.path)
from config import cargar_config
from data.simulacion import sim, config_simulador, generar_muestra

RAIZ = Path(__file__).resolve().parents[1]
REFERENCIA = Path(__file__).resolve().parent / "referencia_regresion_celula.json"
RAPIDO = ["datos.simulador.z_paso_nm=100.0"]
CASOS = {
    "celula_cerca": ("celula.yaml", ["datos.simulador.anillos_hacia_objetivo=true"], 1, (0, 1, 2)),
    "celula_lejos": ("celula.yaml", [], 2, (0, 5)),
    "espejo_cerca": ("espejo.yaml", ["datos.simulador.anillos_hacia_objetivo=true"], 3, (0, 1)),
}


def hash_caso(yaml, overrides, seed, indices, extra=()):
    cd = cargar_config(RAIZ / "configs" / yaml, RAPIDO + list(overrides) + list(extra)).datos
    cs = config_simulador(cd, seed)
    banco = sim.BancoPSF(cs)
    h = hashlib.sha256(banco.stack.tobytes())
    for i in indices:
        img, gt, meta = generar_muestra(cd, cs, banco, i)
        h.update(np.ascontiguousarray(img).tobytes())
        h.update(gt.to_csv(index=False).encode())
        h.update(json.dumps(meta, sort_keys=True, default=float).encode())
    return h.hexdigest()


def hashes(extra=()):
    out = {"numpy": np.__version__}
    for nombre, (yaml, ov, seed, idx) in CASOS.items():
        out[nombre] = hash_caso(yaml, ov, seed, idx, extra)
    return out


def _referencia():
    ref = json.loads(REFERENCIA.read_text())
    if ref["numpy"] != np.__version__:
        print(f"    [aviso] referencia escrita con numpy {ref['numpy']}, este entorno tiene "
              f"{np.__version__}: regresion saltada")
        return None
    return ref


def test_regresion_bit_a_bit_celula():
    ref = _referencia()
    if ref is None:
        return
    actual = hashes()
    distintos = [c for c in CASOS if actual[c] != ref[c]]
    assert not distintos, f"muestras distintas de la referencia en: {distintos}"


if __name__ == "__main__":
    if "--generar" in sys.argv:
        REFERENCIA.write_text(json.dumps(hashes(), indent=1) + "\n")
        print(f"escrita {REFERENCIA}")
    else:
        test_regresion_bit_a_bit_celula()
        print("regresion OK")
