"""evaluate.py: metricas por rango con conteos y errores conocidos y, de punta a
punta en CPU con un checkpoint sin entrenar, los archivos que escribe, el GT
restringido a los frames evaluados (un video suelto con el GT de todo el
dataset) y la tabla por SNR con datos pregenerados."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from comun import banco, checkpoint_smoke
from data.dataset import pregenerar
from data.simulacion import sim
import evaluate

NAN = float("nan")
GT_EVAL = pd.DataFrame({"photons": [1.0, 2.0, 3.0, 4.0], "en_fov": [True, True, True, False],
                        "detectado": [True, False, True, True], "dx_nm": [3.0, NAN, -3.0, 0.0],
                        "dy_nm": [4.0, NAN, 4.0, 0.0], "dz_nm": [10.0, NAN, -10.0, 0.0]})


def test_por_rangos():
    t = evaluate.por_rangos(GT_EVAL, "photons", np.array([1.0, 2.5, 4.0]))
    assert list(t.n) == [2, 1] and list(t.detectados) == [1, 1]           # el emisor fuera del FOV no cuenta
    assert list(t.recall) == [0.5, 1.0]
    assert np.allclose(t.rmse_lat_nm, 5.0) and np.allclose(t.rmse_z_nm, 10.0)
    assert list(t.sesgo_z_nm) == [10.0, -10.0]

    t = evaluate.por_rangos(GT_EVAL, "photons", np.array([0.0, 0.5, 2.5, 4.0]))   # un rango vacio
    assert list(t.n) == [0, 2, 1] and np.isnan(t.recall[0]) and np.isnan(t.rmse_z_nm[0])


def test_evaluate_videos_video_suelto_y_pregenerados():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt, cfg = checkpoint_smoke(tmp)
        cs = sim.Config(mascara="esferica", H=20, W=28, n_frames=3, n_videos=2, seed=5, n_emisores_rango=(1, 3))
        sim.generar_dataset(cs, tmp / "videos", banco("esferica"))
        gt = pd.read_csv(tmp / "videos" / "ground_truth.csv")
        comunes = ["--dispositivo", "cpu", "--set", "postproc.umbral=0.0"]       # red sin entrenar: todo cuenta

        ev = tmp / "ev"
        r = evaluate.main([str(ckpt), str(tmp / "videos"), "--salida", str(ev), "--figuras", "2", *comunes])
        assert r["n_gt"] == gt.en_fov.sum() > 0 and r["tp"] + r["fn"] == r["n_gt"]
        for nombre in ("resumen.yaml", "por_entrada.csv", "por_fotones.csv", "por_z.csv", "metricas.png",
                       "predicciones.csv", "gt_eval.csv"):
            assert (ev / nombre).exists(), nombre
        assert not (ev / "por_snr.csv").exists()                                  # el GT del simulador no trae snr
        assert pd.read_csv(ev / "por_fotones.csv").n.sum() == r["n_gt"] == pd.read_csv(ev / "por_z.csv").n.sum()
        assert len(pd.read_csv(ev / "por_entrada.csv")) == 2
        assert sorted(f.name for f in (ev / "figuras").iterdir()) == ["v000_f0000.png", "v000_f0001.png"]

        # un video suelto: solo cuenta el GT de sus frames
        r1 = evaluate.main([str(ckpt), str(tmp / "videos" / "video_001.tif"), "--salida", str(tmp / "ev1"),
                            "--figuras", "0", *comunes])
        assert r1["n_gt"] == (gt.en_fov & (gt.video_id == 1)).sum()

        pregenerar(cfg.datos, 3, cfg.datos.seed_val, tmp / "pre")
        evaluate.main([str(ckpt), str(tmp / "pre"), "--salida", str(tmp / "ev_pre"), "--figuras", "0", *comunes])
        n_fov = pd.read_csv(tmp / "pre" / "ground_truth.csv").en_fov.sum()
        assert n_fov > 0 and pd.read_csv(tmp / "ev_pre" / "por_snr.csv").n.sum() == n_fov
