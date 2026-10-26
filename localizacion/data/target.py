"""
Rejilla de voxeles de la salida de la red y target gaussiano 3D.

Geometria, con la convencion del simulador (el centro del pixel (fila i,
columna j) es (x = j, y = i)):
    - lateral: s = upsampling voxeles por pixel. El voxel (fila u, columna v)
      tiene centro x = (v + 0.5) / s - 0.5, y = (u + 0.5) / s - 0.5, asi que la
      rejilla cubre exactamente el FOV [-0.5, W - 0.5) x [-0.5, H - 0.5)
    - axial: el plano k esta en z = z_min + (k - m) dz, con m = planos_margen
      planos extra bajo z_min y sobre z_max, para que las gaussianas de los
      emisores en los extremos de z_rango_nm no queden truncadas

Target: cada emisor (dentro del FOV) aporta una gaussiana 3D separable de pico
`escala`, centrada en su posicion continua (no en el voxel mas cercano), con
sigma en voxeles. Se construye en torch, en el dispositivo del batch.
"""
import numpy as np
import torch


class RejillaVoxeles:
    def __init__(self, cfg_target, H, W, z_rango_nm):
        self.s = int(cfg_target.upsampling)
        self.H, self.W = int(H), int(W)
        self.Hs, self.Ws = self.H * self.s, self.W * self.s
        self.dz = float(cfg_target.dz_nm)
        self.m = int(cfg_target.planos_margen)
        self.z_min, self.z_max = map(float, z_rango_nm)
        n = int(np.ceil((self.z_max - self.z_min) / self.dz - 1e-9)) + 1   # planos que cubren z_rango
        self.Nz = n + 2 * self.m
        self.z_planos = self.z_min + (np.arange(self.Nz) - self.m) * self.dz
        self.sigma_xy, self.sigma_z = float(cfg_target.sigma_xy_vox), float(cfg_target.sigma_z_vox)
        self.escala = float(cfg_target.escala)

    @classmethod
    def desde_config(cls, cfg_target, cfg_sim):
        return cls(cfg_target, cfg_sim.H, cfg_sim.W, cfg_sim.z_rango_nm)

    @property
    def forma(self):
        """(Nz, s H, s W): canales y tamano de la salida de la red."""
        return self.Nz, self.Hs, self.Ws

    def a_voxel(self, x, y, z):
        """(x_px, y_px, z_nm) -> (v, u, k) continuos: columna, fila y plano."""
        return ((x + 0.5) * self.s - 0.5,
                (y + 0.5) * self.s - 0.5,
                (z - self.z_min) / self.dz + self.m)

    def desde_voxel(self, v, u, k):
        """(v, u, k) continuos -> (x_px, y_px, z_nm)."""
        return ((v + 0.5) / self.s - 0.5,
                (u + 0.5) / self.s - 0.5,
                self.z_min + (k - self.m) * self.dz)

    def target(self, pos, valido):
        """Target (B, Nz, s H, s W) desde pos (B, E, 3) [x_px, y_px, z_nm] y
        valido (B, E) bool (False = relleno del batch)."""
        v, u, k = self.a_voxel(pos[..., 0], pos[..., 1], pos[..., 2])

        def gauss(c, n, sigma):                                   # (B, E, n)
            eje = torch.arange(n, device=pos.device, dtype=pos.dtype)
            return torch.exp(-(eje - c[..., None])**2 / (2 * sigma**2))

        gz = gauss(k, self.Nz, self.sigma_z) * valido[..., None]
        gy = gauss(u, self.Hs, self.sigma_xy)
        gx = gauss(v, self.Ws, self.sigma_xy)
        return self.escala * torch.einsum("bek,beu,bev->bkuv", gz, gy, gx)
