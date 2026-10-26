"""Modelo de localizacion 3D (estilo DeepSTORM3D) y perdidas."""
from models.deepstorm3d import LocalizadorCNN, n_parametros
from models.perdidas import Perdida, TERMINOS


def crear_modelo(cfg, rejilla):
    """LocalizadorCNN para ConfigCorrida cfg: C canales de entrada, Nz planos y upsampling de la rejilla."""
    return LocalizadorCNN(cfg.datos.canales, rejilla.Nz, rejilla.s, rejilla.escala, cfg.modelo)


def crear_perdida(cfg, rejilla):
    return Perdida(cfg.perdida, rejilla.escala)
