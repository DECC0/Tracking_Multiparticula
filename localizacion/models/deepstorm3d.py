"""
Red de localizacion 3D inspirada en DeepSTORM3D (Nehme et al., 2020).

    entrada (B, C, H, W)
      -> encoder a resolucion completa: conv 3x3 y convs dilatadas (d = 1, 2, 4,
         8, 16). Cada capa recibe tambien la entrada (concatenada) y su salida
         se suma a la anterior (residual), como en DeepSTORM3D
      -> decoder: log2(s) etapas de upsampling x2 (interpolacion + conv 3x3)
      -> cabeza: conv 3x3 a Nz canales, capas residuales y conv 1x1
      -> activacion acotada a [0, escala]
    salida (B, Nz, s H, s W): un canal por plano z

Sin pooling: a 98 nm/px la imagen ya es gruesa frente a la PSF, asi que el
campo receptivo (65 px con las dilataciones por defecto) se consigue con
dilatacion, sin reducir la resolucion.

Con interpolacion x2 (bilinear con align_corners=False, o nearest) el voxel
(u, v) de la salida queda centrado donde lo pone data/target.py:
x = (v + 0.5) / s - 0.5.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(tipo, c):
    if tipo == "batch":
        return nn.BatchNorm2d(c)
    return nn.GroupNorm(math.gcd(8, c), c)                      # grupos que dividen c (Nz puede ser primo)


class BloqueConv(nn.Sequential):
    """conv 3x3 (dilatada) -> normalizacion -> LeakyReLU."""

    def __init__(self, c_in, c_out, dilatacion=1, norm="batch", pendiente=0.2):
        super().__init__(nn.Conv2d(c_in, c_out, 3, padding=dilatacion, dilation=dilatacion, bias=False),
                         _norm(norm, c_out),
                         nn.LeakyReLU(pendiente, inplace=True))


class LocalizadorCNN(nn.Module):
    def __init__(self, canales_entrada, n_planos, upsampling, escala, cfg_modelo):
        super().__init__()
        s = int(upsampling)
        if s < 1 or s & (s - 1):
            raise ValueError(f"upsampling debe ser una potencia de 2 (1, 2, 4, 8...), no {upsampling}")
        c, F_ = canales_entrada, cfg_modelo.canales_base
        kw = dict(norm=cfg_modelo.norm, pendiente=cfg_modelo.pendiente_leaky)
        self.canales_entrada, self.n_planos, self.upsampling, self.escala = c, n_planos, s, float(escala)
        self.modo = cfg_modelo.modo_upsampling
        self._alin = dict(align_corners=False) if self.modo == "bilinear" else {}
        self.activacion = cfg_modelo.activacion_salida

        self.entrada = BloqueConv(c, F_, 1, **kw)
        self.encoder = nn.ModuleList([BloqueConv(F_ + c, F_, d, **kw) for d in cfg_modelo.dilataciones])
        self.decoder = nn.ModuleList([BloqueConv(F_, F_, 1, **kw) for _ in range(int(math.log2(s)))])
        self.cabeza_entrada = BloqueConv(F_, n_planos, 1, **kw)
        self.cabeza = nn.ModuleList([BloqueConv(n_planos, n_planos, 1, **kw)
                                     for _ in range(cfg_modelo.capas_salida)])
        self.final = nn.Conv2d(n_planos, n_planos, 1)
        if self.activacion == "sigmoide":
            nn.init.constant_(self.final.bias, -4.0)             # salida inicial ~ 2% de escala, no 50%
        self.campo_receptivo_px = 1 + 2 * (1 + sum(cfg_modelo.dilataciones))

    def forward(self, x):
        out = self.entrada(x)
        for capa in self.encoder:
            out = capa(torch.cat([out, x], dim=1)) + out
        for etapa in self.decoder:
            out = etapa(F.interpolate(out, scale_factor=2, mode=self.modo, **self._alin))
        out = self.cabeza_entrada(out)
        for capa in self.cabeza:
            out = capa(out) + out
        out = self.final(out)
        if self.activacion == "sigmoide":
            return self.escala * torch.sigmoid(out)
        return F.hardtanh(out, 0.0, self.escala)


def n_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
