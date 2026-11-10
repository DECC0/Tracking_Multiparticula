"""
Regresion bit a bit del simulador (paquete simulador_psf, antes sim_imagenesPSF.py).

Con semillas fijas genera un conjunto pequeno de salidas y lo compara con la
referencia guardada en tests/referencia/. La referencia se escribio con el
codigo anterior a la reorganizacion en el paquete simulador_psf y se regenero
al cambiar la convencion de z (+z hacia el objetivo): los videos no cambiaron
y el GT solo cambio de signo en z, lo que verifica tests/test_convencion_z.py
contra el codigo anterior. Todo debe coincidir exactamente:

    - datasets pequenos (H != W): videos .tif y ground_truth.csv, byte a byte
    - banco de PSF (stack, z_grid, energia_recortada) e imagen sin ruido
    - componentes con rng fijo: trayectorias, fondo, fotones, camara, reflejar
    - diagnosticos: energia_encerrada, snr_pico, imprimir_snr_pico, resumen()
    - la corrida completa del punto de entrada (sha256 de videos, CSV y stdout)

Si algo difiere imprime el error maximo y media/varianza de ambos lados, para
poder compararlo con tolerancia, y termina con codigo 1.

Uso:
    $PY tests/regresion.py                              # compara con la referencia
    $PY tests/regresion.py --generar                    # (re)escribe la referencia
    $PY tests/regresion.py --modulo sim_imagenesPSF     # la capa de compatibilidad
"""
import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

REF = Path(__file__).resolve().parent / "referencia"

# datasets pequenos; ninguno usa corkscrew (eliminada de simulador_psf)
BASE = dict(H=24, W=32, n_videos=2, n_frames=4)
CASOS = {
    "esferica": dict(seed=3),
    "uniforme_fondo_plano": dict(seed=4, muestreo_fotones="uniforme", fondo_no_uniforme=False),
    "ninguna": dict(seed=5, mascara="ninguna"),
    "k64_pocos": dict(seed=6, K_psf=64, z_paso_nm=50.0, z_rango_nm=(-500.0, 500.0),
                      n_emisores_rango=(0, 3), D_rango_um2_s=(0.1, 1.0)),
}


def sha(datos):
    if isinstance(datos, np.ndarray):
        datos = f"{datos.dtype}{datos.shape}".encode() + np.ascontiguousarray(datos).tobytes()
    elif isinstance(datos, str):
        datos = datos.encode()
    return hashlib.sha256(datos).hexdigest()


def salidas(sim, script):
    """Calcula todo lo que se compara. Devuelve (arrays, hashes, textos, archivos):
    archivos es {ruta relativa en referencia/: bytes}."""
    arrays, hashes, textos, archivos = {}, {}, {}, {}

    # --- datasets pequenos ---
    bancos = {}
    for caso, extra in CASOS.items():
        cfg = sim.Config(**BASE, **extra)
        banco = sim.BancoPSF(cfg)
        bancos[caso] = banco
        hashes[f"{caso}/K_psf"] = str(cfg.K_psf)
        hashes[f"{caso}/stack"] = sha(banco.stack)
        arrays[f"{caso}/z_grid"] = banco.z_grid
        arrays[f"{caso}/energia_recortada"] = banco.energia_recortada
        for z in (-333.3, 0.0, 517.0):                           # fuera de la rejilla: interpola
            hashes[f"{caso}/psf_en_z({z})"] = sha(banco.psf_en_z(z))

        with tempfile.TemporaryDirectory() as tmp:
            gt = sim.generar_dataset(cfg, banco, carpeta=tmp)
            gt.to_csv(Path(tmp) / "ground_truth.csv", index=False)
            for f in sorted(Path(tmp).iterdir()):
                archivos[f"{caso}/{f.name}"] = f.read_bytes()

        # imagen sin ruido del frame 0 del video 0, reconstruida desde el GT
        img = np.zeros((cfg.H, cfg.W))
        for r in gt[(gt.video_id == 0) & (gt.frame == 0)].itertuples():
            banco.colocar(img, r.x_px, r.y_px, r.z_nm, r.photons)
        arrays[f"{caso}/sin_ruido"] = img

    # --- componentes con rng fijo ---
    cfg = sim.Config(H=24, W=32, n_frames=6)
    rng = np.random.default_rng(11)
    tray, D = sim.generar_trayectorias(cfg, 5, rng)
    arrays["comp/trayectorias"], arrays["comp/D"] = tray, D
    arrays["comp/fondo_no_uniforme"] = sim.generar_fondo(cfg, rng)
    cfg_plano = sim.Config(H=24, W=32, fondo_no_uniforme=False)
    arrays["comp/fondo_plano"] = sim.generar_fondo(cfg_plano, rng)
    arrays["comp/fotones_log"] = sim.muestrear_fotones(cfg, 50, rng)
    arrays["comp/fotones_uniforme"] = sim.muestrear_fotones(sim.Config(muestreo_fotones="uniforme"), 50, rng)
    fotones = np.linspace(-5.0, 3000.0, 24 * 32).reshape(24, 32)  # incluye negativos (se recortan a 0)
    arrays["comp/camara"] = sim.aplicar_camara(fotones, cfg, rng)
    arrays["comp/camara_saturada"] = sim.aplicar_camara(np.full(10, 1e6), cfg, rng)
    arrays["comp/reflejar"] = sim.reflejar(np.linspace(-5000.0, 5000.0, 101), -800.0, 800.0)
    arrays["comp/fase_esferica"] = sim.fase_esferica(np.linspace(0.0, 1.0, 51), 1.5 * np.pi)
    for caso in ("esferica", "ninguna"):
        textos[f"{caso}/resumen"] = sim.Config(**BASE, **CASOS[caso]).resumen()

    # --- diagnosticos ---
    for caso in ("esferica", "ninguna"):
        banco = bancos[caso]
        arrays[f"diag/{caso}/energia_encerrada"] = np.array(
            [[sim.energia_encerrada(banco, z, r) for r in (1.0, 2.0, 4.0)]
             for z in (-800.0, -250.0, 0.0, 410.0, 800.0)])
        for nombre, a in zip(("p_max", "senal", "snr"), sim.snr_pico(banco, 1000.0, 20.0)):
            arrays[f"diag/{caso}/snr_pico/{nombre}"] = a
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sim.imprimir_snr_pico(banco)
        textos[f"diag/{caso}/imprimir_snr_pico"] = buf.getvalue()

    # --- error con mascara desconocida (comportamiento que se conserva) ---
    try:
        sim.BancoPSF(sim.Config(mascara="no_existe"))
        textos["error_mascara_desconocida"] = "no lanzo"
    except ValueError as e:
        textos["error_mascara_desconocida"] = f"ValueError: {e}"

    # --- corrida completa del punto de entrada, en un directorio temporal ---
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([sys.executable, str(RAIZ / script)], cwd=tmp,
                              capture_output=True, text=True, check=True)
        textos["main/stdout"] = proc.stdout
        for f in sorted(Path(tmp).rglob("*")):
            if f.is_file():
                hashes[f"main/{f.relative_to(tmp).as_posix()}"] = sha(f.read_bytes())

    return arrays, hashes, textos, archivos


def comparar_arrays(nombre, a, b):
    if a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True):
        return True
    print(f"  [DIF ] {nombre}: forma {a.shape}/{b.shape}, dtype {a.dtype}/{b.dtype}")
    if a.shape == b.shape:
        af, bf = a.astype(float), b.astype(float)
        print(f"         max |dif| = {np.abs(af - bf).max():.3e}; media {af.mean():.6g} vs {bf.mean():.6g}; "
              f"var {af.var():.6g} vs {bf.var():.6g}; suma {af.sum():.6g} vs {bf.sum():.6g}")
    return False


def comparar_tif(nombre, nuevo, ref):
    """Si los bytes difieren, dice si es por los pixeles o por los metadatos."""
    import tifffile
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "n.tif").write_bytes(nuevo)
        (Path(tmp) / "r.tif").write_bytes(ref)
        a, b = tifffile.imread(Path(tmp) / "n.tif"), tifffile.imread(Path(tmp) / "r.tif")
    if comparar_arrays(nombre + " (pixeles)", a, b):
        print(f"  [DIF ] {nombre}: pixeles iguales, difieren los metadatos/bytes del archivo")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generar", action="store_true", help="reescribe tests/referencia/")
    p.add_argument("--modulo", default="simulador_psf", help="modulo que se prueba")
    p.add_argument("--script", default="generar_dataset.py", help="punto de entrada (relativo a la raiz)")
    args = p.parse_args()

    sim = importlib.import_module(args.modulo)
    arrays, hashes, textos, archivos = salidas(sim, args.script)

    if args.generar:
        REF.mkdir(exist_ok=True)
        np.savez_compressed(REF / "arrays.npz", **arrays)
        (REF / "hashes.json").write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n")
        (REF / "textos.json").write_text(json.dumps(textos, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        for ruta, datos in archivos.items():
            (REF / ruta).parent.mkdir(exist_ok=True)
            (REF / ruta).write_bytes(datos)
        print(f"referencia escrita en {REF} ({len(arrays)} arrays, {len(hashes)} hashes, "
              f"{len(textos)} textos, {len(archivos)} archivos)")
        return

    ref_arrays = dict(np.load(REF / "arrays.npz"))
    ref_hashes = json.loads((REF / "hashes.json").read_text())
    ref_textos = json.loads((REF / "textos.json").read_text())
    ref_archivos = {f.relative_to(REF).as_posix(): f.read_bytes() for f in sorted(REF.glob("*/*"))}

    ok, n = True, 0
    for tipo, nuevo, ref in (("arrays", arrays, ref_arrays), ("hashes", hashes, ref_hashes),
                             ("textos", textos, ref_textos), ("archivos", archivos, ref_archivos)):
        for clave in sorted(set(nuevo) - set(ref)) + sorted(set(ref) - set(nuevo)):
            print(f"  [DIF ] {tipo}/{clave}: solo en {'el codigo nuevo' if clave in nuevo else 'la referencia'}")
            ok = False
        for clave in sorted(set(nuevo) & set(ref)):
            n += 1
            if tipo == "arrays":
                ok &= comparar_arrays(clave, nuevo[clave], ref[clave])
            elif nuevo[clave] != ref[clave]:
                ok = False
                if tipo == "archivos" and clave.endswith(".tif"):
                    comparar_tif(clave, nuevo[clave], ref[clave])
                else:
                    print(f"  [DIF ] {tipo}/{clave}")
                    if tipo == "textos":
                        print(f"         nuevo: {nuevo[clave]!r}\n         ref  : {ref[clave]!r}")

    print(f"\nregresion ({args.modulo}, {args.script}): {n} comparaciones -> "
          f"{'IDENTICO a la referencia' if ok else 'HAY DIFERENCIAS'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
