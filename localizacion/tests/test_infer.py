"""infer.py de punta a punta en CPU con un checkpoint sin entrenar
(configs/smoke.yaml): lee carpetas de .tif del simulador (otro tamano de
imagen que el de entrenamiento) y de pregenerar.py, escribe el CSV con las
columnas pedidas, su yaml y las figuras, y solo acepta overrides de postproc."""
import tempfile
from pathlib import Path

import pandas as pd

from comun import banco, checkpoint_smoke, espera_error
from data.dataset import pregenerar
from data.simulacion import sim
from postproc import COLUMNAS_PRED
import infer


def test_infer_tif_pregenerados_y_figuras():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt, cfg = checkpoint_smoke(tmp)
        cs = sim.Config(mascara="esferica", H=20, W=28, n_frames=3, n_videos=2, seed=5, n_emisores_rango=(1, 3))
        sim.generar_dataset(cs, tmp / "videos", banco("esferica"))
        comunes = ["--dispositivo", "cpu", "--set", "postproc.umbral=0.0"]       # red sin entrenar: todo cuenta

        pred = infer.main([str(ckpt), str(tmp / "videos"), "--salida", str(tmp / "out" / "pred.csv"), *comunes,
                           "--figuras", "2", "--gt", str(tmp / "videos" / "ground_truth.csv")])
        assert list(pred.columns) == COLUMNAS_PRED and len(pred) > 0
        assert set(pred.video_id) <= {0, 1} and set(pred.frame) <= {0, 1, 2}
        assert pred.x_px.between(-0.5, cs.W - 0.5).all() and pred.y_px.between(-0.5, cs.H - 0.5).all()
        leido = pd.read_csv(tmp / "out" / "pred.csv")
        assert len(leido) == len(pred) and list(leido.columns) == COLUMNAS_PRED
        assert (tmp / "out" / "pred.yaml").exists()
        assert sorted(f.name for f in (tmp / "out" / "pred_figuras").iterdir()) == ["v000_f0000.png", "v000_f0001.png"]

        pregenerar(cfg.datos, 3, cfg.datos.seed_val, tmp / "pre")
        pred = infer.main([str(ckpt), str(tmp / "pre"), "--salida", str(tmp / "pre.csv"), *comunes])
        assert set(pred.video_id) <= {0, 1, 2} and (pred.frame == 0).all()

        # el checkpoint fija la red: solo se puede cambiar el post-procesado
        espera_error(ValueError, infer.main, [str(ckpt), str(tmp / "videos"), "--salida", str(tmp / "x.csv"),
                                              "--dispositivo", "cpu", "--set", "target.dz_nm=10"])
