"""
Perdidas sobre la salida volumetrica (B, Nz, s H, s W). Cada termino es una
funcion f(pred, target, escala, cfg_perdida) -> escalar, registrada en
TERMINOS con @termino; la perdida total es la suma pesada de los terminos de
cfg_perdida.terminos. Agregar o cambiar un termino = registrarlo y nombrarlo
en el YAML.

    mse  : MSE entre la salida y el target gaussiano
    dice : 1 - Dice suave entre salida y target normalizados por escala (el
           termino de solapamiento de DeepSTORM3D), por muestra y promediado:
           una imagen sin emisores predicha vacia da 0
"""
import torch.nn as nn
import torch.nn.functional as F

TERMINOS = {}


def termino(nombre):
    def registrar(f):
        TERMINOS[nombre] = f
        return f
    return registrar


@termino("mse")
def mse(pred, target, escala, cfg):
    return F.mse_loss(pred, target)


@termino("dice")
def dice(pred, target, escala, cfg):
    p, t = (pred / escala).flatten(1), (target / escala).flatten(1)
    inter = (p * t).sum(1)
    suma = (p * p).sum(1) + (t * t).sum(1)
    return (1 - (2 * inter + cfg.suavizado_dice) / (suma + cfg.suavizado_dice)).mean()


class Perdida(nn.Module):
    """Suma pesada de terminos; devuelve (total, {termino: valor sin gradiente})."""

    def __init__(self, cfg_perdida, escala):
        super().__init__()
        desconocidos = set(cfg_perdida.terminos) - set(TERMINOS)
        if desconocidos:
            raise ValueError(f"terminos de perdida desconocidos: {sorted(desconocidos)} "
                             f"(disponibles: {sorted(TERMINOS)})")
        self.cfg, self.escala = cfg_perdida, float(escala)

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()             # en float32 aunque la red corra con AMP
        valores = {n: TERMINOS[n](pred, target, self.escala, self.cfg) for n in self.cfg.terminos}
        total = sum(w * valores[n] for n, w in self.cfg.terminos.items())
        return total, {n: v.detach() for n, v in valores.items()}
