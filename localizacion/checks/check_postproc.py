"""Check de la etapa 4 (post-procesado), con configs/base.yaml.

A. Sin red: sobre targets exactos (posiciones aleatorias), el post-procesado
   recupera las posiciones (sesgo del centro de masa en ventana truncada) y su
   tiempo en GPU.
B. Con un checkpoint (argumento, o el mejor.pt mas reciente de corridas/): barre
   confianza, radios y umbral sobre imagenes de validacion y reporta Jaccard,
   recall, precision y RMSE (tolerancia 150 nm lateral, 300 nm axial); recall y
   RMSE por SNR del emisor con la mejor combinacion. Genera fig_postproc.png y
   fig_postproc_ejemplos.png. Sin checkpoint se omite.

Uso: check_postproc.py [CHECKPOINT] [N_IMAGENES]"""
import itertools
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import cargar_config
from data.dataset import DatasetSimulado
from infer import predecir, dibujar_detecciones
from metricas import evaluar, resumen
from postproc import detectar
from utils import rejilla_de, cargar_modelo, DIR_LOCALIZACION

OUT = os.path.dirname(os.path.abspath(__file__))
COLORES = ["#2a78d6", "#eb6834", "#1baf7a"]                      # paleta categorica (orden fijo)
GPU = torch.cuda.is_available()
dev = torch.device("cuda" if GPU else "cpu")
R_XY, R_Z = 150.0, 300.0
UMBRALES = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
BORDES_SNR = [0, 2, 3, 5, 8, 12, np.inf]

cfg = cargar_config(os.path.join(RAIZ, "configs", "base.yaml"))
rej = rejilla_de(cfg)
ds = DatasetSimulado(cfg.datos, 8, cfg.datos.seed_val)
px = ds.cfg_sim.pixel_nm

# --- A. targets exactos ---
rng = np.random.default_rng(0)
B, E = 64, 8
pos = np.zeros((B, E, 3))
for b in range(B):
    n = 0
    while n < E:                                                 # separacion lateral >= 3 px
        p = rng.uniform([0, 0, rej.z_min], [rej.W - 1, rej.H - 1, rej.z_max])
        if n == 0 or np.hypot(*(pos[b, :n, :2] - p[:2]).T).min() >= 3:
            pos[b, n], n = p, n + 1
tgt = rej.target(torch.from_numpy(pos).to(dev), torch.ones(B, E, dtype=torch.bool, device=dev))
gt = pd.DataFrame({"video_id": np.repeat(np.arange(B), E), "frame": 0,
                   "x_px": pos[..., 0].ravel(), "y_px": pos[..., 1].ravel(), "z_nm": pos[..., 2].ravel()})
print(f"A. targets exactos: {B} frames x {E} emisores, rejilla {rej.forma}, tolerancia {R_XY:.0f}/{R_Z:.0f} nm")
for cz in (1, 2, 4):
    pp = replace(cfg.postproc, umbral=0.5, radio_cm_z_planos=cz)
    det = detectar(tgt, rej, pp).rename(columns={"b": "video_id"}).assign(frame=0)
    r = resumen(*evaluar(det, gt, px, R_XY, R_Z))
    print(f"   radio_cm_z {cz}: Jaccard {r['jaccard']:.3f}; RMSE lateral {r['rmse_lat_nm']:.2f} nm, "
          f"axial {r['rmse_z_nm']:.2f} nm; sesgo z {r['sesgo_z_nm']:+.2f} nm")
lote = tgt[:8]
detectar(lote, rej, cfg.postproc)
if GPU:
    torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    detectar(lote, rej, cfg.postproc)
if GPU:
    torch.cuda.synchronize()
print(f"   tiempo de detectar(): {(time.time() - t0) / 10 * 1e3:.1f} ms por batch de 8 ({dev.type})")

# --- B. red entrenada ---
args = sys.argv[1:]
if args:
    ckpt = Path(args[0])
else:
    candidatos = sorted((DIR_LOCALIZACION / "corridas").glob("*/mejor.pt"), key=lambda p: p.stat().st_mtime)
    ckpt = candidatos[-1] if candidatos else None
if ckpt is None:
    print("\nB. sin checkpoint en corridas/: se omite el barrido con la red")
    sys.exit(0)
N = int(args[1]) if len(args) > 1 else 500
modelo, cfg_r, rej_r = cargar_modelo(ckpt, dev)
ds = DatasetSimulado(cfg_r.datos, N, cfg_r.datos.seed_val)
print(f"\nB. {ckpt} sobre {N} imagenes de validacion (seed {cfg_r.datos.seed_val})")

variantes = [dict(confianza=c, radio_z_planos=rz, radio_cm_z_planos=cz, radio_cm_xy_vox=cxy)
            for c, rz, cz, cxy in itertools.product(("pico", "masa"), (2, 3, 5), (1, 2, 4), (1, 2))]
dets = [[] for _ in variantes]
gts, imagenes, preds_ej = [], [], []
t0 = time.time()
for i0 in range(0, N, 8):
    muestras = [ds.muestra(i) for i in range(i0, min(i0 + 8, N))]
    imgs = np.stack([m[0][0] for m in muestras])
    gts += [m[1] for m in muestras]
    pred = predecir(modelo, imgs, cfg_r, ds.cfg_sim, dev)
    for j, kw in enumerate(variantes):
        d = detectar(pred, rej_r, replace(cfg_r.postproc, umbral=UMBRALES[0], **kw))
        d["b"] += i0
        dets[j].append(d)
    if i0 == 0:
        imagenes, preds_ej = imgs[:4], pred[:4].cpu()
gt_val = pd.concat(gts, ignore_index=True)
dets = [pd.concat(d, ignore_index=True).rename(columns={"b": "video_id"}).assign(frame=0) for d in dets]
print(f"   red + {len(variantes)} post-procesados por batch: {time.time() - t0:.0f} s; "
      f"{int(gt_val.en_fov.sum())} emisores en el FOV")

filas = []
for kw, d in zip(variantes, dets):
    for u in UMBRALES:
        filas.append({**kw, "umbral": u, **resumen(*evaluar(d[d.conf >= u], gt_val, px, R_XY, R_Z))})
tabla = pd.DataFrame(filas)
cols = ["confianza", "radio_z_planos", "radio_cm_z_planos", "radio_cm_xy_vox", "umbral",
        "jaccard", "recall", "precision", "rmse_lat_nm", "rmse_z_nm", "sesgo_z_nm"]
pd.set_option("display.width", 200)
print("\n   mejores 8 combinaciones por Jaccard:")
print(tabla.sort_values("jaccard", ascending=False)[cols].head(8).to_string(index=False, float_format="%.3f"))
for c in ("pico", "masa"):
    t = tabla[tabla.confianza == c]
    print(f"\n   mejor con confianza {c}:")
    print(t.loc[[t.jaccard.idxmax()], cols].to_string(index=False, float_format="%.3f"))
print("\n   efecto de radio_cm_z con la mejor combinacion de lo demas:")
mejor = tabla.loc[tabla.jaccard.idxmax()]
igual = ((tabla.confianza == mejor.confianza) & (tabla.radio_z_planos == mejor.radio_z_planos)
         & (tabla.radio_cm_xy_vox == mejor.radio_cm_xy_vox) & (tabla.umbral == mejor.umbral))
print(tabla[igual][cols].to_string(index=False, float_format="%.3f"))

# por SNR con la mejor combinacion
kw_mejor = {k: mejor[k] for k in variantes[0]}
d = dets[variantes.index({k: (int(v) if k != "confianza" else v) for k, v in kw_mejor.items()})]
gt_e, pred_e = evaluar(d[d.conf >= mejor.umbral], gt_val, px, R_XY, R_Z)
g = gt_e[gt_e.en_fov].copy()
g["bin"] = pd.cut(g.snr, BORDES_SNR, right=False)
por_snr = g.groupby("bin", observed=True).apply(lambda x: pd.Series({
    "n": len(x), "recall": x.detectado.mean(),
    "rmse_lat_nm": np.sqrt(np.nanmean(x.dx_nm**2 + x.dy_nm**2)) if x.detectado.any() else np.nan,
    "rmse_z_nm": np.sqrt(np.nanmean(x.dz_nm**2)) if x.detectado.any() else np.nan,
    "sesgo_z_nm": np.nanmean(x.dz_nm) if x.detectado.any() else np.nan}), include_groups=False)
print(f"\n   por SNR del emisor (umbral {mejor.umbral}, confianza {mejor.confianza}):")
print(por_snr.to_string(float_format="%.2f"))

# --- figuras ---
fig, axes = plt.subplots(1, 4, figsize=(19, 4))
base = (tabla.radio_z_planos == mejor.radio_z_planos) & (tabla.radio_cm_z_planos == mejor.radio_cm_z_planos) \
    & (tabla.radio_cm_xy_vox == mejor.radio_cm_xy_vox)
for c, color in zip(("pico", "masa"), COLORES):
    t = tabla[base & (tabla.confianza == c)]
    axes[0].plot(t.umbral, t.jaccard, "o-", color=color, lw=2, ms=5, label=f"confianza {c}")
    axes[1].plot(t.recall, t.precision, "o-", color=color, lw=2, ms=5, label=f"confianza {c}")
axes[0].set_xlabel("umbral"); axes[0].set_ylabel("Jaccard"); axes[0].legend(frameon=False)
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
centros = [f"[{b.left:g}, {b.right:g})" for b in por_snr.index]
axes[2].bar(centros, por_snr.recall, color=COLORES[0], width=0.6)
axes[2].set_ylim(0, 1); axes[2].set_xlabel("SNR del emisor"); axes[2].set_ylabel("recall")
axes[3].plot(centros, por_snr.rmse_lat_nm, "o-", color=COLORES[0], lw=2, ms=6, label="lateral")
axes[3].plot(centros, por_snr.rmse_z_nm, "o-", color=COLORES[1], lw=2, ms=6, label="axial")
axes[3].set_xlabel("SNR del emisor"); axes[3].set_ylabel("RMSE [nm]"); axes[3].legend(frameon=False)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
fig.suptitle(f"{ckpt.parent.name}: {N} imagenes de val, tolerancia {R_XY:.0f}/{R_Z:.0f} nm; "
             f"radio_z {mejor.radio_z_planos}, cm z {mejor.radio_cm_z_planos}, cm xy {mejor.radio_cm_xy_vox}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_postproc.png"), dpi=120)

fig, axes = plt.subplots(len(imagenes), 3, figsize=(16, 4.2 * len(imagenes)),
                         gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
d_ej = d[(d.conf >= mejor.umbral) & (d.video_id < len(imagenes))]
for i, fila in enumerate(axes):
    g = gt_val[(gt_val.video_id == i) & gt_val.en_fov]
    fila[0].imshow(preds_ej[i].max(0).values, cmap="magma", vmin=0,
                   extent=(-0.5, rej_r.W - 0.5, rej_r.H - 0.5, -0.5))
    fila[0].set_title("prediccion (max en z)" if i == 0 else "")
    sc = dibujar_detecciones(fila[1], fila[2], imagenes[i], d_ej[d_ej.video_id == i], ds.cfg_sim.z_rango_nm, g)
    fila[1].set_title("SNR " + " ".join(f"{s:.0f}" for s in sorted(g.snr)), fontsize=9)
    fig.colorbar(sc, ax=fila[2], label="z detectado [nm]", pad=0.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_postproc_ejemplos.png"), dpi=110)
print("\nfiguras: fig_postproc.png, fig_postproc_ejemplos.png")
