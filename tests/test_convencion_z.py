"""
Verificacion del cambio de convencion de z contra el codigo anterior.

Antes: +z lejos del objetivo, angulo del espejo desde el plano focal
(angulo_espejo_deg) y signo_z / signo_inclinacion. Ahora: +z HACIA el objetivo
(fijo), angulo desde el eje optico (angulo_espejo_eje_deg) y la bandera de la
PSF anillos_hacia_objetivo (signo_z viejo +1 <-> False, -1 <-> True).

El codigo anterior se extrae del commit COMMIT_ANTERIOR con git archive y se
corre en un proceso aparte (mismos nombres de modulo); el nuevo, en otro.

Espejo apagado, BIT A BIT (anillos_hacia_objetivo = False y el z_rango
espejado, (lo, hi) <-> (-hi, -lo)):
    - simulador_psf: los cuatro casos de tests/regresion.py y uno con z_rango
      asimetrico (generar_video con el rng en secuencia): videos identicos,
      z_nm negada exactamente, el resto del GT identico.
    - simulador_nuevo: los cuatro casos de su test_regresion (con desenfoque
      de movimiento y corkscrew) y uno asimetrico: video, imagen limpia y
      fondo identicos, subposiciones con z negada, GT con z_nm negada.
    - localizacion: muestras de DatasetSimulado (configs/smoke.yaml):
      imagenes identicas, z_nm negada, snr identica.
Espejo encendido, con tolerancia (la geometria se calcula en otros ejes y
difiere en el ultimo bit): viejo (alpha, signo_z = +1) == nuevo
(beta = 90 - alpha, anillos False) con z y z' negadas.

Uso (~1 min; necesita git y COMMIT_ANTERIOR en el historial):
    ~/miniforge3/envs/tesis/bin/python tests/test_convencion_z.py
"""
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
COMMIT_ANTERIOR = "02d8be8"      # ultimo commit con la convencion anterior

# (nombre, simulador, kwargs comunes, kwargs solo viejo, kwargs solo nuevo, bit a bit)
B_PSF = dict(H=24, W=32, n_videos=2, n_frames=4)
B_NUEVO = dict(H=24, W=32, n_frames=5, n_videos=2)
ESPEJO_VIEJO = dict(usar_espejo=True, angulo_espejo_deg=34.0, signo_z=1)
ESPEJO_NUEVO = dict(usar_espejo=True, angulo_espejo_eje_deg=56.0)
CASOS = [
    ("psf_esferica", "psf", dict(B_PSF, seed=3), {}, {}, True),
    ("psf_uniforme_fondo_plano", "psf", dict(B_PSF, seed=4, muestreo_fotones="uniforme", fondo_no_uniforme=False), {}, {}, True),
    ("psf_ninguna", "psf", dict(B_PSF, seed=5, mascara="ninguna"), {}, {}, True),
    ("psf_k64_pocos", "psf", dict(B_PSF, seed=6, K_psf=64, z_paso_nm=50.0, z_rango_nm=(-500.0, 500.0),
                                  n_emisores_rango=(0, 3), D_rango_um2_s=(0.1, 1.0)), {}, {}, True),
    ("psf_asimetrico", "psf", dict(B_PSF, seed=7, D_rango_um2_s=(0.5, 2.0)),
     dict(z_rango_nm=(-700.0, 300.0)), dict(z_rango_nm=(-300.0, 700.0)), True),
    ("nuevo_esferica", "nuevo", dict(B_NUEVO, seed=3), {}, {}, True),
    ("nuevo_ninguna_blur", "nuevo", dict(B_NUEVO, seed=4, mascara="ninguna", t_exp_s=0.05, n_subpasos=4,
                                         D_rango_um2_s=(0.5, 2.0)), {}, {}, True),
    ("nuevo_corkscrew_plano", "nuevo", dict(B_NUEVO, seed=5, mascara="corkscrew", fondo_no_uniforme=False,
                                            muestreo_fotones="uniforme"), {}, {}, True),
    ("nuevo_k64_pocos", "nuevo", dict(B_NUEVO, seed=6, K_psf=64, z_paso_nm=50.0, z_rango_nm=(-500.0, 500.0),
                                      n_emisores_rango=(0, 3)), {}, {}, True),
    ("nuevo_asimetrico", "nuevo", dict(B_NUEVO, seed=7, n_subpasos=3, D_rango_um2_s=(0.5, 2.0)),
     dict(z_rango_nm=(-700.0, 300.0)), dict(z_rango_nm=(-300.0, 700.0)), True),
    ("localizacion_smoke", "localizacion", {}, {}, {}, True),
    ("psf_espejo", "psf", dict(B_PSF, seed=8, n_frames=6, dist_max_espejo_um=3.0, orientacion_espejo_deg=37.0),
     ESPEJO_VIEJO, ESPEJO_NUEVO, False),
    ("nuevo_espejo", "nuevo", dict(H=40, W=48, n_frames=6, n_videos=2, seed=8, n_subpasos=2, dist_max_espejo_um=3.0,
                                   orientacion_espejo_deg=37.0, espejo_punto_px=(10.0, 20.0)),
     ESPEJO_VIEJO, ESPEJO_NUEVO, False),
]

TRABAJADOR = r'''
import json, sys
from pathlib import Path
import numpy as np
raiz, modo, salida = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
casos = json.loads(Path(sys.argv[4]).read_text())
sys.path.insert(0, str(raiz))
sys.path.insert(0, str(raiz / "simulador_nuevo"))
out = {}
for nombre, simu, comun, viejo, nuevo in casos:
    kw = dict(comun, **(viejo if modo == "viejo" else nuevo))
    for k in ("z_rango_nm", "D_rango_um2_s", "n_emisores_rango"):
        if k in kw:
            kw[k] = tuple(kw[k])
    if simu == "psf":
        import simulador_psf as sp
        cfg = sp.Config(**kw)
        banco = sp.BancoPSF(cfg)
        rng = np.random.default_rng(cfg.seed)
        for v in range(cfg.n_videos):
            video, filas = sp.generar_video(cfg, banco, rng, v)
            out[f"{nombre}/v{v}/video"] = video
            for c in filas[0] if filas else []:
                out[f"{nombre}/v{v}/gt_{c}"] = np.array([f[c] for f in filas])
    elif simu == "nuevo":
        import simulador as sim
        cfg = sim.Config(**kw)
        banco = sim.BancoPSF(cfg)
        for v in range(cfg.n_videos):
            video, gt, ex = sim.generar_video(cfg, banco, v, extra=True)
            out[f"{nombre}/v{v}/video"] = video
            out[f"{nombre}/v{v}/limpio"] = ex["limpio"]
            out[f"{nombre}/v{v}/fondo"] = ex["fondo"]
            out[f"{nombre}/v{v}/sub"] = ex["sub"]
            for c in gt.columns:
                out[f"{nombre}/v{v}/gt_{c}"] = gt[c].to_numpy()
    else:
        sys.path.insert(0, str(raiz / "localizacion"))
        from config import cargar_config
        from data.dataset import DatasetSimulado
        cfg = cargar_config(raiz / "localizacion" / "configs" / "smoke.yaml")
        ds = DatasetSimulado(cfg.datos, 6, cfg.datos.seed_val)
        for i in range(6):
            imagen, gt, meta = ds.muestra(i)
            out[f"{nombre}/m{i}/imagen"] = imagen
            for c in gt.columns:
                out[f"{nombre}/m{i}/gt_{c}"] = gt[c].to_numpy()
np.savez(salida, **out)
'''


def es_z(clave):
    c = clave.rsplit("/", 1)[-1]
    return c in ("gt_z_nm", "gt_z_reflejo_nm")


def comparar(viejo, nuevo, bit_a_bit):
    """Lista de errores (vacia si todo coincide)."""
    errores = []
    if set(viejo.files) != set(nuevo.files):
        return [f"claves distintas: {sorted(set(viejo.files) ^ set(nuevo.files))[:6]}"]
    for k in viejo.files:
        a, b = viejo[k], nuevo[k]
        if a.shape != b.shape:
            errores.append(f"{k}: forma {a.shape} != {b.shape}")
            continue
        if k.endswith("/sub"):                                   # subposiciones (n, T, S, 3): z negada
            a = a.copy()
            a[..., 2] *= -1
        elif es_z(k):
            a = -a
        if bit_a_bit:
            if not np.array_equal(a, b):
                d = np.abs(a.astype(float) - b.astype(float)).max() if a.dtype.kind in "fiu" else "?"
                errores.append(f"{k}: distinto bit a bit (max |diferencia| {d})")
        elif k.endswith("/video") or k.endswith("/imagen"):
            iguales = np.mean(a == b)
            if iguales < 0.999:
                errores.append(f"{k}: solo {iguales:.4%} de los pixeles iguales")
        elif a.dtype.kind == "f":
            tol = 1e-6 * max(1.0, np.abs(a).max()) if k.endswith("/limpio") else 1e-6
            if not np.allclose(a, b, rtol=0, atol=tol):
                errores.append(f"{k}: max |diferencia| {np.abs(a - b).max():.3g}")
        elif not np.array_equal(a, b):
            errores.append(f"{k}: distinto")
    return errores


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        viejo_dir = tmp / "viejo"
        viejo_dir.mkdir()
        tar = tmp / "viejo.tar"
        r = subprocess.run(["git", "-C", str(RAIZ), "archive", "-o", str(tar), COMMIT_ANTERIOR],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[salta] no se pudo extraer {COMMIT_ANTERIOR} con git archive: {r.stderr.strip()}")
            return 0
        with tarfile.open(tar) as t:
            t.extractall(viejo_dir, filter="data")
        trabajador = tmp / "trabajador.py"
        trabajador.write_text(TRABAJADOR)
        fallos = 0
        for bit in (True, False):
            casos = [(n, s, c, v, w) for n, s, c, v, w, b in CASOS if b == bit]
            cj = tmp / f"casos_{bit}.json"
            cj.write_text(json.dumps(casos))
            procs = {m: subprocess.Popen([sys.executable, str(trabajador), str(r_), m, str(tmp / f"{m}_{bit}.npz"),
                                          str(cj)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for m, r_ in (("viejo", viejo_dir), ("nuevo", RAIZ))}
            for m, p in procs.items():
                _, err = p.communicate()
                if p.returncode != 0:
                    print(f"[FAIL] el codigo {m} fallo:\n{err[-3000:]}")
                    return 1
            viejo, nuevo = np.load(tmp / f"viejo_{bit}.npz"), np.load(tmp / f"nuevo_{bit}.npz")
            for n, *_ in casos:
                sub_v = {k for k in viejo.files if k.startswith(n + "/")}
                errores = comparar(_Filtro(viejo, sub_v), _Filtro(nuevo, sub_v), bit)
                estado = "OK  " if not errores else "FAIL"
                tipo = "bit a bit" if bit else "tolerancia"
                print(f"[{estado}] {n} ({tipo}, {len(sub_v)} arreglos)")
                for e in errores[:8]:
                    print(f"         {e}")
                fallos += bool(errores)
    print(f"\n{len(CASOS) - fallos}/{len(CASOS)} casos OK")
    return 1 if fallos else 0


class _Filtro:
    """Vista de un npz restringida a un conjunto de claves."""
    def __init__(self, npz, claves):
        self.npz, self.files = npz, sorted(claves)

    def __getitem__(self, k):
        return self.npz[k]


if __name__ == "__main__":
    sys.exit(main())
