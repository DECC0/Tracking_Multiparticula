"""Regresion bit a bit del simulador con el espejo apagado: hashes de videos,
imagenes limpias, fondo, subposiciones, ground truth, resumen() y a_dict()
de cuatro casos pequenos, contra referencia_regresion.json (escrito con el
codigo de antes de agregar el espejo).

Los hashes dependen de la version de numpy (FFT y redondeos): con otra version
el test se salta con un aviso en vez de fallar. Regenerar la referencia solo
si se cambia el comportamiento a proposito:
    $PY simulador_nuevo/tests/test_regresion.py --generar
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from comun import sim

REFERENCIA = Path(__file__).resolve().parent / "referencia_regresion.json"
CASOS = {
    "esferica": dict(seed=3),
    "ninguna_blur": dict(seed=4, mascara="ninguna", t_exp_s=0.05, n_subpasos=4, D_rango_um2_s=(0.5, 2.0)),
    "corkscrew_plano": dict(seed=5, mascara="corkscrew", fondo_no_uniforme=False, muestreo_fotones="uniforme"),
    "k64_pocos": dict(seed=6, K_psf=64, z_paso_nm=50.0, z_rango_nm=(-500.0, 500.0), n_emisores_rango=(0, 3)),
}


def hashes():
    out = {"numpy": np.__version__}
    for nombre, extra in CASOS.items():
        cfg = sim.Config(H=24, W=32, n_frames=5, n_videos=2, **extra)
        banco = sim.BancoPSF(cfg)
        h = hashlib.sha256(banco.stack.tobytes())
        for v in range(cfg.n_videos):
            video, gt, ex = sim.generar_video(cfg, banco, v, extra=True)
            for a in (video, ex["limpio"], ex["fondo"], ex["sub"]):
                h.update(np.ascontiguousarray(a).tobytes())
            h.update(gt.to_csv(index=False).encode())
        h.update(cfg.resumen().encode())
        h.update(json.dumps(cfg.a_dict(), sort_keys=True).encode())
        out[nombre] = h.hexdigest()
    return out


def test_regresion_bit_a_bit_sin_espejo():
    ref = json.loads(REFERENCIA.read_text())
    if ref["numpy"] != np.__version__:
        print(f"    [aviso] referencia escrita con numpy {ref['numpy']}, este entorno tiene "
              f"{np.__version__}: regresion saltada")
        return
    actual = hashes()
    distintos = [c for c in CASOS if actual[c] != ref[c]]
    assert not distintos, f"salidas distintas de la referencia en: {distintos}"


if __name__ == "__main__":
    if "--generar" in sys.argv:
        REFERENCIA.write_text(json.dumps(hashes(), indent=1) + "\n")
        print(f"escrita {REFERENCIA}")
    else:
        test_regresion_bit_a_bit_sin_espejo()
        print("regresion OK")
