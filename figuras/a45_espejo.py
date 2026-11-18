"""
Referencia A45: la CNN existente (celula.yaml) entrenada con beta = 45 grados
fijo, con reflejo (A45, configs/celula_b45fijo.yaml) y sin luz de reflejo
(A45_sin, configs/celula_b45fijo_sin_reflejo.yaml), para medir si aprende a
usar el reflejo. Ver resultados/resultados_a45_espejo.md.

Tests (semilla 4, independientes de train = 1, val = 2 y del test anterior = 3):
    localizacion/datos/test_celula45fijo_b45_cerca          con reflejo
    localizacion/datos/test_celula45fijo_sinrefl_b45_cerca  las mismas escenas, ratio_reflejo = 0
    (nombres test_<PREFIJO>_b45_cerca de hypatia/espejo.sbatch con PREFIJO = celula45fijo y
    celula45fijo_sinrefl: el job encuentra el test ya generado y no lo regenera)

Partes:
    verificar_tests   antes de entrenar: configs (solo difieren en lo previsto), banco de PSF
                      identico al de la referencia A (rango, planos, K y el stack bit a bit),
                      beta = 45 en todas las muestras, escenas equivalentes salvo la luz del
                      reflejo (GT, muestras.csv, imagen sin ruido por linealidad en
                      ratio_reflejo) y lo mismo en muestras de entrenamiento; sale con error si
                      algo falla
    verificar_lanzamiento  antes de lanzar en Hypatia: las configs efectivas de las cuatro
                      corridas (las que arma hypatia/espejo.sbatch: anillos cerca, epocas,
                      workers, nombre y semilla) difieren solo en lo previsto, nombres de corrida
                      unicos, el job encuentra los tests de semilla 4 (no regenera con la 3) y
                      manifiesto sha256 de los tests (a45_espejo/a45_tests_sha256.txt, para
                      comprobarlos en Hypatia con sha256sum -c)

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/a45_espejo.py verificar_tests [--n-imagenes 40]   # ~3 min (construye dos bancos)
    $PY figuras/a45_espejo.py verificar_lanzamiento               # ~5 s
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "localizacion"))
from config import cargar_config, a_dict  # noqa: E402
from data.simulacion import (sim, config_simulador, config_muestra, generar_muestra, banco_compatible,  # noqa: E402
                             CAMPOS_PSF, DESFASE_REINTENTO)

CONFIGS = {"con": RAIZ / "localizacion" / "configs" / "celula_b45fijo.yaml",
           "sin": RAIZ / "localizacion" / "configs" / "celula_b45fijo_sin_reflejo.yaml"}
CELULA = RAIZ / "localizacion" / "configs" / "celula.yaml"
CORRIDA_A = RAIZ / "localizacion" / "corridas" / "hypatia_celula_b45_cerca_20261110-234627"
TEST_A = RAIZ / "localizacion" / "datos" / "test_celula_b45_cerca"
TESTS = {"con": RAIZ / "localizacion" / "datos" / "test_celula45fijo_b45_cerca",
         "sin": RAIZ / "localizacion" / "datos" / "test_celula45fijo_sinrefl_b45_cerca"}
SEMILLA_TEST = 4
CERCA = ["datos.simulador.anillos_hacia_objetivo=true"]      # lo que fija hypatia/espejo.sbatch para cerca
SALIDA = AQUI / "a45_espejo"


def aplanar(d, pref=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(aplanar(v, f"{pref}{k}."))
        else:
            out[f"{pref}{k}"] = tuple(v) if isinstance(v, list) else v
    return out


def diferencias(a, b):
    a, b = aplanar(a), aplanar(b)
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def verificar_tests(n_imagenes):
    SALIDA.mkdir(parents=True, exist_ok=True)
    res = []

    def chequeo(nombre, ok, detalle):
        res.append(dict(chequeo=nombre, pasa=bool(ok), detalle=detalle))
        print(f"[{'PASA' if ok else 'FALLA'}] {nombre}: {detalle}", flush=True)

    t0 = time.time()
    # ---- 1. configs ----
    cfg = {k: cargar_config(v, CERCA) for k, v in CONFIGS.items()}
    base = cargar_config(CELULA, CERCA)
    corrida_a = cargar_config(CORRIDA_A / "config.yaml")
    d_base = diferencias(a_dict(base), a_dict(cfg["con"]))
    esperado = {"datos.espejo_angulo_eje_rango_deg", "datos.simulador.z_banco_nm", "entrenamiento.nombre"}
    chequeo("C1 celula.yaml -> celula_b45fijo.yaml", set(d_base) == esperado,
            "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in d_base.items()))
    d_sin = diferencias(a_dict(cfg["con"]), a_dict(cfg["sin"]))
    chequeo("C2 celula_b45fijo -> celula_b45fijo_sin_reflejo", set(d_sin) == {"datos.ratio_reflejo_rango", "entrenamiento.nombre"},
            "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in d_sin.items()))
    # la corrida A ademas tiene lo que fija el job (epocas, workers, nombre)
    d_a = diferencias(a_dict(corrida_a), a_dict(cfg["con"]))
    previstas = esperado | {"entrenamiento.workers", "entrenamiento.epocas"}
    chequeo("C3 config de la corrida A -> celula_b45fijo (+ anillos cerca)", set(d_a) <= previstas,
            "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in d_a.items()))

    # ---- 2. banco de PSF ----
    cs_a = config_simulador(corrida_a.datos, 3)
    cs = {k: config_simulador(c.datos, SEMILLA_TEST) for k, c in cfg.items()}
    for k in ("con", "sin"):
        dif = [c for c in CAMPOS_PSF + ("z_banco_nm",) if getattr(cs_a, c) != getattr(cs[k], c)]
        za, zn = sim.rejilla_z(cs_a), sim.rejilla_z(cs[k])
        chequeo(f"B1 campos de PSF y rejilla z ({k})", not dif and np.array_equal(za, zn),
                f"campos distintos {dif or 'ninguno'}; z_banco {cs[k].z_banco_nm}; {len(zn)} planos de "
                f"{zn[0]:.1f} a {zn[-1]:.1f} nm, paso {np.diff(zn).max():.1f} nm (A: {len(za)} planos)")
    ks = {}
    for nombre, carpeta in (("A", TEST_A), ("con", TESTS["con"]), ("sin", TESTS["sin"])):
        ks[nombre] = json.loads((carpeta / "config_simulador.json").read_text()).get("K_psf_usado")
    chequeo("B2 K_psf usado", len(set(ks.values())) == 1, f"{ks}")
    print("  construyendo los bancos de A y de A45...", flush=True)
    banco_a = sim.BancoPSF(cs_a)
    banco = sim.BancoPSF(cs["con"])
    chequeo("B3 stack del banco bit a bit", banco_a.K == banco.K and np.array_equal(banco_a.z_grid, banco.z_grid)
            and np.array_equal(banco_a.stack, banco.stack),
            f"K {banco_a.K} / {banco.K}, stack {banco.stack.shape}, max |diferencia| "
            f"{np.abs(banco_a.stack - banco.stack).max() if banco_a.stack.shape == banco.stack.shape else 'formas distintas'}")
    chequeo("B4 el banco sirve a las dos configs", banco_compatible(banco, cs["con"]) and banco_compatible(banco, cs["sin"]), "")

    # ---- 3. tests: beta, GT y muestras ----
    gt = {k: pd.read_csv(TESTS[k] / "ground_truth.csv") for k in TESTS}
    mu = {k: pd.read_csv(TESTS[k] / "muestras.csv") for k in TESTS}
    for k in TESTS:
        b = mu[k].espejo_angulo_eje_deg.to_numpy()
        chequeo(f"T1 beta = 45 en todas las muestras ({k})", np.all(b == 45.0), f"min {b.min()}, max {b.max()}, n {len(b)}")
    import yaml
    for k in TESTS:
        meta = yaml.safe_load((TESTS[k] / "datos.yaml").read_text())
        chequeo(f"T2 semilla y n ({k})", meta["seed"] == SEMILLA_TEST and meta["n"] == 2000, f"seed {meta['seed']}, n {meta['n']}")
    g1, g0 = gt["con"], gt["sin"]
    mismas_cols = list(g1.columns) == list(g0.columns) and len(g1) == len(g0)
    otras = [c for c in g1.columns if c != "photons_reflejo"]
    dif_cols = [c for c in otras if not g1[c].equals(g0[c])]
    chequeo("T3 GT identico salvo photons_reflejo", mismas_cols and not dif_cols,
            f"{len(g1)} filas; columnas distintas: {dif_cols or 'ninguna'}")
    r = g1.video_id.map(mu["con"].set_index("video_id").ratio_reflejo)
    chequeo("T4 photons_reflejo: con = ratio x photons, sin = 0",
            np.allclose(g1.photons_reflejo, r * g1.photons, rtol=1e-12, atol=0) and (g0.photons_reflejo == 0).all(),
            f"ratio en [{r.min():.3f}, {r.max():.3f}]")
    m1, m0 = mu["con"], mu["sin"]
    dif_m = [c for c in m1.columns if c != "ratio_reflejo" and not m1[c].equals(m0[c])]
    chequeo("T5 muestras.csv identico salvo ratio_reflejo", list(m1.columns) == list(m0.columns) and not dif_m
            and (m0.ratio_reflejo == 0).all(), f"columnas distintas: {dif_m or 'ninguna'}; reintentos con/sin "
            f"{int(m1.reintentos.sum())}/{int(m0.reintentos.sum())}")

    # ---- 4. imagenes: regeneradas, sin ruido y lineales en ratio_reflejo ----
    img = {k: np.load(TESTS[k] / "imagenes.npy", mmap_mode="r") for k in TESTS}
    ids = np.linspace(0, 1999, n_imagenes).astype(int)
    filas = []
    ok_rep = ok_fondo = ok_sub = ok_lin = ok_pos = True
    for idx in ids:
        ex, vid = {}, {}
        for k in TESTS:
            cm = config_muestra(cfg[k].datos, cs[k], int(idx), 0)
            vid[k], _, ex[k] = sim.generar_video(cm, banco, int(idx), extra=True)
            ok_rep &= np.array_equal(vid[k][0], img[k][idx, 0])
        cm1 = replace(config_muestra(cfg["con"].datos, cs["con"], int(idx), 0), ratio_reflejo=1.0)
        _, _, ex1 = sim.generar_video(cm1, banco, int(idx), extra=True)
        ok_fondo &= np.array_equal(ex["con"]["fondo"], ex["sin"]["fondo"])
        ok_sub &= np.array_equal(ex["con"]["sub"], ex["sin"]["sub"]) and np.array_equal(ex["con"]["sub_reflejo"], ex["sin"]["sub_reflejo"])
        R1 = ex1["limpio"][0] - ex["sin"]["limpio"][0]              # luz del reflejo con ratio 1
        dif = ex["con"]["limpio"][0] - ex["sin"]["limpio"][0]
        ratio = float(m1.ratio_reflejo.iloc[idx])
        err = np.abs(dif - ratio * R1).max() / max(np.abs(dif).max(), 1e-12)
        ok_lin &= err < 1e-9
        ok_pos &= R1.min() > -1e-9
        filas.append(dict(video_id=int(idx), ratio=ratio, fotones_reflejo_imagen=float(dif.sum()),
                          error_linealidad_rel=float(err), min_R1=float(R1.min()),
                          pixeles_ruido_distintos=float(np.mean(vid["con"][0] != vid["sin"][0]))))
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / "a45_verificacion_imagenes.csv", index=False)
    chequeo("I1 imagenes del test = regeneradas (bit a bit)", ok_rep, f"{len(ids)} imagenes por test")
    chequeo("I2 fondo y subposiciones (real y reflejo) identicos", ok_fondo and ok_sub, "")
    chequeo("I3 imagen sin ruido: con - sin = ratio x (luz del reflejo con ratio 1)", ok_lin and ok_pos,
            f"error relativo max {t.error_linealidad_rel.max():.1e}; min de la luz del reflejo {t.min_R1.min():.1e}; "
            f"fotones del reflejo en la imagen: mediana {t.fotones_reflejo_imagen.median():.0f}")
    chequeo("I4 ruido: realizacion distinta (informativo)", True,
            f"pixeles con distinto valor ADU: mediana {t.pixeles_ruido_distintos.median():.2f}")

    # ---- 5. muestras de entrenamiento y validacion emparejadas ----
    for split, semilla in (("train", cfg["con"].datos.seed_train), ("val", cfg["con"].datos.seed_val)):
        cst = {k: config_simulador(c.datos, semilla) for k, c in cfg.items()}
        dif_tr = 0
        for idx in range(0, 200, 10):
            gts = {k: generar_muestra(cfg[k].datos, cst[k], banco, idx)[1] for k in cfg}
            a, b = gts["con"].drop(columns="photons_reflejo"), gts["sin"].drop(columns="photons_reflejo")
            dif_tr += int(not a.equals(b))
        chequeo(f"E1 {split}: escenas con/sin reflejo identicas (20 muestras)", dif_tr == 0, f"distintas: {dif_tr}")

    pd.DataFrame(res).to_csv(SALIDA / "a45_verificacion.csv", index=False)
    fallas = [r_["chequeo"] for r_ in res if not r_["pasa"]]
    print(f"\n{len(res) - len(fallas)} de {len(res)} comprobaciones pasan ({time.time() - t0:.0f} s)"
          + (f"; fallan: {fallas}" if fallas else ""))
    if fallas:
        sys.exit(1)


# corridas aprobadas: (condicion, PREFIJO de lanzar_espejo.sh, SEMILLA)
CORRIDAS = [("con", "celula45fijo", 0), ("con", "celula45fijo", 1),
            ("sin", "celula45fijo_sinrefl", 0), ("sin", "celula45fijo_sinrefl", 1)]
EPOCAS, WORKERS, ANILLOS = 100, 8, "cerca"


def overrides_job(nombre, semilla):
    """Los --set que agrega hypatia/espejo.sbatch (con EXTRA_SET vacio)."""
    return CERCA + [f"entrenamiento.epocas={EPOCAS}", f"entrenamiento.workers={WORKERS}",
                    f"entrenamiento.nombre={nombre}", f"entrenamiento.semilla={semilla}"]


def verificar_lanzamiento():
    import hashlib
    SALIDA.mkdir(parents=True, exist_ok=True)
    res = []

    def chequeo(nombre, ok, detalle):
        res.append(dict(chequeo=nombre, pasa=bool(ok), detalle=detalle))
        print(f"[{'PASA' if ok else 'FALLA'}] {nombre}: {detalle}", flush=True)

    sbatch = (RAIZ / "hypatia" / "espejo.sbatch").read_text()
    lanzar = (RAIZ / "hypatia" / "lanzar_espejo.sh").read_text()
    patrones = ["NOMBRE=hypatia_${PREFIJO}_b45_${ANILLOS}", "TEST=$DATOS/test_${PREFIJO}_b45_${ANILLOS}",
                "NOMBRE=${NOMBRE}_s$SEMILLA", "SET_SEMILLA=entrenamiento.semilla=$SEMILLA",
                'if [ ! -f "$TEST/imagenes.npy" ]', "entrenamiento.workers=$WORKERS",
                "entrenamiento.epocas=$EPOCAS", "datos.simulador.anillos_hacia_objetivo=$BANDERA"]
    faltan = [x for x in patrones if x not in sbatch]
    chequeo("L0 espejo.sbatch arma nombre, test y overrides como se supone aqui", not faltan,
            f"faltan {faltan}" if faltan else "patrones presentes; DATOS por defecto "
            + ("localizacion/datos" if "DATOS=${DATOS:-localizacion/datos}" in lanzar else "DISTINTO"))

    # ---- configs efectivas ----
    nombres = {(c, s): f"hypatia_{pref}_b45_{ANILLOS}_s{s}" for c, pref, s in CORRIDAS}
    ef = {(c, s): a_dict(cargar_config(CONFIGS[c], overrides_job(nombres[(c, s)], s))) for c, _, s in CORRIDAS}
    for c in ("con", "sin"):
        d = diferencias(ef[(c, 0)], ef[(c, 1)])
        chequeo(f"L1 {c}: semilla 0 -> 1", set(d) == {"entrenamiento.semilla", "entrenamiento.nombre"},
                "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in d.items()))
    for s in (0, 1):
        d = diferencias(ef[("con", s)], ef[("sin", s)])
        chequeo(f"L2 semilla {s}: A45 -> A45_sin", set(d) == {"datos.ratio_reflejo_rango", "entrenamiento.nombre"},
                "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in d.items()))
    d = diferencias(a_dict(cargar_config(CORRIDA_A / "config.yaml")), ef[("con", 0)])
    esperado = {"datos.espejo_angulo_eje_rango_deg", "datos.simulador.z_banco_nm", "entrenamiento.nombre"}
    chequeo("L3 corrida A -> A45 s0 (efectiva)", set(d) == esperado,
            "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in d.items()))
    sem = {k: (v["entrenamiento"]["semilla"], v["datos"]["seed_train"], v["datos"]["seed_val"]) for k, v in ef.items()}
    chequeo("L4 semillas (entrenamiento, seed_train, seed_val)",
            all(v == (k[1], 1, 2) for k, v in sem.items()),
            "; ".join(f"{c} s{s}: {v}" for (c, s), v in sem.items()))

    # ---- nombres y carpetas ----
    ns = list(nombres.values())
    choque = [(a, b) for a in ns for b in ns if a != b and b.startswith(a + "_")]
    previas = sorted(p.name for n in ns for p in (RAIZ / "localizacion" / "corridas").glob(n + "_*"))
    chequeo("L5 nombres de corrida unicos (el glob ${NOMBRE}_* del job no cruza corridas)",
            len(set(ns)) == 4 and not choque and not previas,
            f"{ns}; cruces {choque or 'ninguno'}; corridas locales previas con esos nombres {previas or 'ninguna'}")

    # ---- tests ----
    lineas = []
    for c, pref, _ in CORRIDAS[::2]:
        t = RAIZ / "localizacion" / "datos" / f"test_{pref}_b45_{ANILLOS}"
        dy = (t / "datos.yaml").read_text()
        ok = t == TESTS[c] and (t / "imagenes.npy").is_file() and "seed: 4" in dy and "n: 2000" in dy
        chequeo(f"L6 test del job para {pref}", ok, f"{t.relative_to(RAIZ)}: imagenes.npy "
                f"{'existe' if (t / 'imagenes.npy').is_file() else 'NO existe'}, datos.yaml seed 4 y n 2000: "
                f"{'si' if ok else 'no'}")
        for f in sorted(t.iterdir()):
            lineas.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(RAIZ)}")
    (SALIDA / "a45_tests_sha256.txt").write_text("\n".join(lineas) + "\n")
    print(f"  manifiesto: {(SALIDA / 'a45_tests_sha256.txt').relative_to(RAIZ)} ({len(lineas)} archivos)")

    pd.DataFrame(res).to_csv(SALIDA / "a45_verificacion_lanzamiento.csv", index=False)
    fallas = [r_["chequeo"] for r_ in res if not r_["pasa"]]
    print(f"\n{len(res) - len(fallas)} de {len(res)} comprobaciones pasan" + (f"; fallan: {fallas}" if fallas else ""))
    if fallas:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("partes", nargs="+", choices=["verificar_tests", "verificar_lanzamiento"])
    p.add_argument("--n-imagenes", type=int, default=40)
    a = p.parse_args()
    for parte in a.partes:
        if parte == "verificar_tests":
            verificar_tests(a.n_imagenes)
        elif parte == "verificar_lanzamiento":
            verificar_lanzamiento()


if __name__ == "__main__":
    main()
