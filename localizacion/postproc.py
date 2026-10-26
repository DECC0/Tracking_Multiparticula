"""
Post-procesado: salida de la red (B, Nz, s H, s W) -> emisores con posicion
continua, en torch y en el dispositivo de la salida.

    1. maximos locales 3D: voxeles iguales al maximo de su ventana
       (2 radio_z + 1) x (2 radio_xy + 1)^2 (max_pool3d) y mayores que 0. Los
       maximos empatados (mesetas, p. ej. la salida saturada en escala por la
       hardtanh) que caen en la ventana de otro se cuentan una vez
    2. centro de masa en una ventana (radio_cm) alrededor de cada maximo: da la
       posicion continua en voxeles, que la rejilla pasa a (x_px, y_px, z_nm)
    3. confianza >= umbral, con confianza
           "pico": valor del maximo / escala (1 en el centro de una gaussiana del target)
           "masa": suma en la ventana del centro de masa / la misma suma para
                   una gaussiana del target centrada en un voxel. No se cae
                   cuando la red reparte un emisor en varios planos de z
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

COLUMNAS_PRED = ["video_id", "frame", "x_px", "y_px", "z_nm", "conf"]


def masa_gaussiana(rejilla, cfg_pp):
    """Suma de una gaussiana del target centrada en un voxel, dentro de la
    ventana del centro de masa (normaliza la confianza "masa")."""
    def suma(r, sigma):
        d = np.arange(-r, r + 1)
        return np.exp(-d**2 / (2 * sigma**2)).sum()
    return rejilla.escala * suma(cfg_pp.radio_cm_z_planos, rejilla.sigma_z) * suma(cfg_pp.radio_cm_xy_vox, rejilla.sigma_xy)**2


def _empates(b, k, u, v, pico, rz, rxy):
    """Mascara de los maximos que sobran: empatados (mismo frame y valor) con
    otro anterior dentro de su ventana. Solo dos maximos de igual valor pueden
    caer uno en la ventana del otro, asi que basta revisar los empatados."""
    sobra = torch.zeros_like(b, dtype=torch.bool)
    if len(b) < 2:
        return sobra
    clave = torch.stack([b.to(pico.dtype), pico], 1)
    _, inv, cuenta = torch.unique(clave, dim=0, return_inverse=True, return_counts=True)
    i = (cuenta[inv] > 1).nonzero(as_tuple=True)[0]
    if len(i) == 0:
        return sobra
    cerca = ((inv[i, None] == inv[None, i])
             & ((k[i, None] - k[None, i]).abs() <= rz)
             & ((u[i, None] - u[None, i]).abs() <= rxy)
             & ((v[i, None] - v[None, i]).abs() <= rxy))
    sobra[i] = torch.triu(cerca, diagonal=1).any(0)              # hay uno anterior (indice menor) en su ventana
    return sobra


@torch.no_grad()
def detectar(pred, rejilla, cfg_pp):
    """Detecciones de un batch de salidas pred (B, Nz, s H, s W).

    Devuelve un DataFrame con columnas b (indice en el batch), x_px, y_px,
    z_nm y conf, ordenado por b y confianza decreciente."""
    p = pred.detach().float()
    B = p.shape[0]
    rz, rxy = int(cfg_pp.radio_z_planos), int(cfg_pp.radio_xy_vox)
    cz, cxy = int(cfg_pp.radio_cm_z_planos), int(cfg_pp.radio_cm_xy_vox)
    n_ventana = (2 * cz + 1) * (2 * cxy + 1)**2
    masa_ref = masa_gaussiana(rejilla, cfg_pp)

    # cota inferior exacta del pico para pasar el umbral: masa <= pico * n_ventana
    pico_min = cfg_pp.umbral * (rejilla.escala if cfg_pp.confianza == "pico" else masa_ref / n_ventana)
    maximo = F.max_pool3d(p[:, None], (2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1), stride=1,
                          padding=(rz, rxy, rxy))[:, 0]
    cand = (p == maximo) & (p > 0) & (p >= pico_min)
    b, k, u, v = cand.nonzero(as_tuple=True)
    pico = p[b, k, u, v]
    queda = ~_empates(b, k, u, v, pico, rz, rxy)
    b, k, u, v, pico = b[queda], k[queda], u[queda], v[queda], pico[queda]

    # centro de masa en la ventana (relleno con ceros en los bordes)
    pp = F.pad(p, (cxy, cxy, cxy, cxy, cz, cz))
    dk, du, dv = torch.meshgrid(torch.arange(-cz, cz + 1, device=p.device),
                                torch.arange(-cxy, cxy + 1, device=p.device),
                                torch.arange(-cxy, cxy + 1, device=p.device), indexing="ij")
    dk, du, dv = dk.reshape(-1), du.reshape(-1), dv.reshape(-1)
    w = pp[b[:, None], k[:, None] + cz + dk, u[:, None] + cxy + du, v[:, None] + cxy + dv]   # (n, n_ventana)
    masa = w.sum(1)                                              # > 0: incluye el pico
    kc = k + (w * dk).sum(1) / masa
    uc = u + (w * du).sum(1) / masa
    vc = v + (w * dv).sum(1) / masa
    conf = pico / rejilla.escala if cfg_pp.confianza == "pico" else masa / masa_ref
    queda = conf >= cfg_pp.umbral

    x, y, z = rejilla.desde_voxel(vc[queda].double(), uc[queda].double(), kc[queda].double())
    det = pd.DataFrame({"b": b[queda].cpu().numpy(), "x_px": x.cpu().numpy(), "y_px": y.cpu().numpy(),
                        "z_nm": z.cpu().numpy(), "conf": conf[queda].double().cpu().numpy()})
    if B and len(det):
        det = det.sort_values(["b", "conf"], ascending=[True, False], kind="stable", ignore_index=True)
    return det
