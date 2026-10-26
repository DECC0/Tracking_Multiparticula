"""
Datasets de torch:
    DatasetSimulado : muestras generadas on-the-fly con el simulador
    DatasetDisco    : muestras pregeneradas con pregenerar()
Para el mismo (seed, idx) los dos devuelven lo mismo:
    {"imagen": (C, H, W) float32 normalizada,
     "pos":    (n, 3) float32 [x_px, y_px, z_nm] de los emisores dentro del FOV,
     "idx":    int}
collate() rellena pos a (B, E_max, 3) con una mascara "valido" (B, E_max); el
target se construye despues en el dispositivo (data/target.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from config import ConfigDatos, a_dict, guardar_yaml
from data.simulacion import sim, config_simulador, banco_compatible, generar_muestra


def normalizar(imagen, cfg_datos, cfg_sim):
    """ADU (C, H, W) -> entrada de la red (float32). Pasa a e- con la ganancia y
    el offset de la camara; "robusta" ademas centra cada canal con su mediana y
    lo divide por su desviacion robusta (1.4826 MAD), dominada por el ruido y
    el fondo cuando hay pocos emisores."""
    e = sim.adu_a_electrones(imagen, cfg_sim)
    if cfg_datos.normalizacion == "fija":
        return (e / cfg_datos.escala_entrada).astype(np.float32)
    med = np.median(e, axis=(-2, -1), keepdims=True)
    mad = np.median(np.abs(e - med), axis=(-2, -1), keepdims=True)
    return ((e - med) / np.maximum(1.4826 * mad, 1.0)).astype(np.float32)


def _item(imagen, pos, idx, cfg_datos, cfg_sim):
    return {"imagen": torch.from_numpy(normalizar(imagen, cfg_datos, cfg_sim)),
            "pos": torch.from_numpy(np.array(pos, dtype=np.float32)),   # copia: pandas 3 devuelve vistas de solo lectura
            "idx": int(idx)}


def _pos_en_fov(gt):
    return gt.loc[gt.en_fov, ["x_px", "y_px", "z_nm"]].to_numpy(np.float32)


class DatasetSimulado(Dataset):
    """n muestras del split de semilla seed, generadas al pedirlas."""

    def __init__(self, cfg_datos, n, seed, banco=None):
        self.cfg_datos = cfg_datos
        self.cfg_sim = config_simulador(cfg_datos, seed)
        if banco is None:
            banco = sim.BancoPSF(self.cfg_sim)
        elif not banco_compatible(banco, self.cfg_sim):
            raise ValueError("el banco de PSF no corresponde a la optica de datos.simulador")
        self.banco = banco
        self.n = int(n)
        self.desfase = 0

    def fijar_epoca(self, epoca):
        """Con renovar_por_epoca la epoca e usa las muestras e n ... (e + 1) n - 1.
        Llamarla antes de crear el iterador del DataLoader (sin persistent_workers)."""
        if self.cfg_datos.renovar_por_epoca:
            self.desfase = epoca * self.n

    def __len__(self):
        return self.n

    def muestra(self, i):
        """(imagen en ADU, gt, meta) sin normalizar de la muestra i."""
        return generar_muestra(self.cfg_datos, self.cfg_sim, self.banco, self.desfase + i)

    def __getitem__(self, i):
        imagen, gt, _ = self.muestra(i)
        return _item(imagen, _pos_en_fov(gt), self.desfase + i, self.cfg_datos, self.cfg_sim)


def collate(items):
    """Lista de items -> batch con pos rellenado a (B, E_max, 3) y mascara valido."""
    B = len(items)
    E = max(len(it["pos"]) for it in items)
    pos = torch.zeros(B, E, 3)
    valido = torch.zeros(B, E, dtype=torch.bool)
    for b, it in enumerate(items):
        n = len(it["pos"])
        pos[b, :n] = it["pos"]
        valido[b, :n] = True
    return {"imagen": torch.stack([it["imagen"] for it in items]),
            "pos": pos, "valido": valido,
            "idx": torch.tensor([it["idx"] for it in items])}


# ----------------------------------------------------------------------------
# Pregeneracion a disco
# ----------------------------------------------------------------------------

def pregenerar(cfg_datos, n, seed, carpeta, banco=None, cada=0):
    """Escribe en carpeta:
        imagenes.npy            (n, C, H, W) uint16 [ADU]
        ground_truth.csv        COLUMNAS_GT (+ snr), video_id = idx de la muestra
        muestras.csv            randomizacion de cada muestra
        config_simulador.json   simulador.Config del split + K_psf usado
        datos.yaml              ConfigDatos, n y seed
    y devuelve el ground truth. cada > 0 imprime el avance cada `cada` muestras."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ds = DatasetSimulado(cfg_datos, n, seed, banco)
    forma = (n, cfg_datos.canales, ds.cfg_sim.H, ds.cfg_sim.W)
    imagenes = np.lib.format.open_memmap(carpeta / "imagenes.npy", mode="w+", dtype=np.uint16, shape=forma)
    tablas, metas = [], []
    for i in range(n):
        imagen, gt, meta = ds.muestra(i)
        imagenes[i] = imagen
        tablas.append(gt)
        metas.append(meta)
        if cada and (i + 1) % cada == 0:
            print(f"  {i + 1}/{n} muestras")
    imagenes.flush()
    del imagenes

    no_vacias = [t for t in tablas if len(t)] or tablas[:1]
    gt = pd.concat(no_vacias, ignore_index=True)
    gt.to_csv(carpeta / "ground_truth.csv", index=False)
    pd.DataFrame(metas).to_csv(carpeta / "muestras.csv", index=False)
    sim.guardar_config(ds.cfg_sim, carpeta / "config_simulador.json", ds.banco)
    guardar_yaml({"datos": a_dict(cfg_datos), "n": n, "seed": seed}, carpeta / "datos.yaml")
    return gt


class DatasetDisco(Dataset):
    """Muestras escritas por pregenerar(); las imagenes se leen con mmap."""

    def __init__(self, carpeta):
        carpeta = Path(carpeta)
        meta = yaml.safe_load((carpeta / "datos.yaml").read_text())
        self.cfg_datos = ConfigDatos(**meta["datos"])
        self.cfg_sim = sim.cargar_config(carpeta / "config_simulador.json")
        self.imagenes = np.load(carpeta / "imagenes.npy", mmap_mode="r")
        self.gt = pd.read_csv(carpeta / "ground_truth.csv")
        self.muestras = pd.read_csv(carpeta / "muestras.csv")
        dentro = self.gt[self.gt.en_fov]
        self._pos = {int(i): g[["x_px", "y_px", "z_nm"]].to_numpy(np.float32)
                     for i, g in dentro.groupby("video_id")}

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, i):
        pos = self._pos.get(int(i), np.zeros((0, 3), np.float32))
        return _item(np.asarray(self.imagenes[i]), pos, i, self.cfg_datos, self.cfg_sim)
