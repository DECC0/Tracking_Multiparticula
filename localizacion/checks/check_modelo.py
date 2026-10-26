"""Check de la etapa 2 (modelo + perdida) con configs/base.yaml: parametros
para C canales y upsampling s, memoria de GPU y tiempo por iteracion con AMP
segun el batch, valores iniciales de la perdida y sobreajuste de un batch fijo
de muestras reales (el modelo puede representar el target). Genera fig_modelo.png."""
import os
import sys
import time
from dataclasses import replace
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import cargar_config
from data.dataset import DatasetSimulado, collate
from data.target import RejillaVoxeles
from models import crear_modelo, crear_perdida, n_parametros

OUT = os.path.dirname(os.path.abspath(__file__))
COLORES = ["#2a78d6", "#eb6834", "#1baf7a"]                      # paleta categorica (orden fijo)
GPU = torch.cuda.is_available()
dev = "cuda" if GPU else "cpu"

cfg = cargar_config(os.path.join(RAIZ, "configs", "base.yaml"))
ds = DatasetSimulado(cfg.datos, 64, cfg.datos.seed_train)
cs = ds.cfg_sim
rej = RejillaVoxeles.desde_config(cfg.target, cs)

# --- parametros ---
print(f"salida (Nz, sH, sW) = {rej.forma}; parametros segun C y s:")
for C in (1, 2):
    fila = []
    for s in (1, 2, 4):
        r = RejillaVoxeles(replace(cfg.target, upsampling=s), cs.H, cs.W, cs.z_rango_nm)
        m = crear_modelo(replace(cfg, datos=replace(cfg.datos, canales=C)), r)
        fila.append(f"s={s}: {n_parametros(m) / 1e3:6.1f} k")
    print(f"  C={C}:  " + "   ".join(fila))
modelo = crear_modelo(cfg, rej).to(dev)
perdida = crear_perdida(cfg, rej)
print(f"modelo base: {n_parametros(modelo) / 1e3:.1f} k parametros, campo receptivo {modelo.campo_receptivo_px} px")


def iteracion(modelo, opt, escalador, x, pos, val, amp=True):
    with torch.autocast(dev, dtype=torch.float16, enabled=amp and GPU):
        pred = modelo(x)
    total, v = perdida(pred, rej.target(pos, val))
    opt.zero_grad(set_to_none=True)
    escalador.scale(total).backward()
    escalador.step(opt)
    escalador.update()
    return total, v


# --- memoria y tiempo por iteracion segun el batch ---
batches, mem, tiempo = [2, 4, 8, 16], [], []
datos = [ds[i] for i in range(16)]
for B in batches:
    lote = collate(datos[:B])
    x, pos, val = (lote[k].to(dev) for k in ("imagen", "pos", "valido"))
    m = crear_modelo(cfg, rej).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    esc = torch.amp.GradScaler(dev, enabled=GPU)
    try:
        if GPU:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        iteracion(m, opt, esc, x, pos, val)                      # calentamiento
        if GPU:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            iteracion(m, opt, esc, x, pos, val)
        if GPU:
            torch.cuda.synchronize()
        tiempo.append((time.time() - t0) / 5)
        mem.append(torch.cuda.max_memory_allocated() / 2**30 if GPU else np.nan)
    except torch.OutOfMemoryError:
        tiempo.append(np.nan)
        mem.append(np.nan)
    del m, opt
    print(f"  batch {B:2d} ({dev}, AMP fp16): {1000 * tiempo[-1]:6.0f} ms/iter, "
          f"memoria pico {mem[-1]:.2f} GB, {B / tiempo[-1]:5.0f} muestras/s")
if GPU:
    lote = collate(datos[:8])
    x, pos, val = (lote[k].to(dev) for k in ("imagen", "pos", "valido"))
    m = crear_modelo(cfg, rej).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    esc = torch.amp.GradScaler(dev, enabled=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    iteracion(m, opt, esc, x, pos, val, amp=False)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        iteracion(m, opt, esc, x, pos, val, amp=False)
    torch.cuda.synchronize()
    print(f"  batch  8 (cuda, sin AMP):  {1000 * (time.time() - t0) / 5:6.0f} ms/iter, "
          f"memoria pico {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    del m, opt
lote = collate(datos[:2])
m = crear_modelo(cfg, rej).train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x, pos, val = lote["imagen"], lote["pos"], lote["valido"]
t0 = time.time()
with torch.autocast("cpu", enabled=False):
    pred = m(x)
total, _ = perdida(pred, rej.target(pos, val))
total.backward()
print(f"  batch  2 (cpu, float32):   {1000 * (time.time() - t0):6.0f} ms/iter")
del m, opt

# --- perdida inicial y sobreajuste de un batch fijo de 8 muestras reales ---
torch.manual_seed(0)
modelo = crear_modelo(cfg, rej).to(dev).train()
opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3)
esc = torch.amp.GradScaler(dev, enabled=GPU)
lote = collate(datos[:8])
x, pos, val = (lote[k].to(dev) for k in ("imagen", "pos", "valido"))
target = rej.target(pos, val)
N_IT = 400
hist = {"total": [], "mse": [], "dice": []}
t0 = time.time()
for it in range(N_IT):
    total, v = iteracion(modelo, opt, esc, x, pos, val)
    hist["total"].append(float(total.detach()))
    for k in v:
        hist[k].append(float(v[k]))
    if it == 0:
        print(f"\nperdida inicial: total {hist['total'][0]:.3f} (mse {hist['mse'][0]:.4f}, dice {hist['dice'][0]:.3f})")
print(f"sobreajuste de 8 muestras, {N_IT} iteraciones ({time.time() - t0:.0f} s): total {hist['total'][-1]:.4f} "
      f"(mse {hist['mse'][-1]:.4f}, dice {hist['dice'][-1]:.3f})")
modelo.eval()
with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=GPU):
    pred = modelo(x).float()
print(f"en eval: pico de la prediccion {pred.amax(dim=(1, 2, 3)).mean():.1f} (target {rej.escala:.0f}), "
      f"mse {float(((pred - target)**2).mean()):.4f}")

# --- figura ---
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])
ax = fig.add_subplot(gs[0, 0:2])
for (k, h), c in zip(hist.items(), COLORES):
    ax.plot(np.array(h) / h[0], color=c, lw=2, label=k)
ax.set_yscale("log"); ax.set_xlabel("iteracion"); ax.set_ylabel("perdida / perdida inicial")
ax.set_title("Sobreajuste de un batch fijo (8 muestras reales)")
ax.legend(frameon=False)
ax = fig.add_subplot(gs[0, 2])
ax.plot(batches, mem, "o-", color=COLORES[0], lw=2, ms=8)
ax.axhline(torch.cuda.get_device_properties(0).total_memory / 2**30 if GPU else np.nan,
           color="0.5", ls="--", lw=1)
ax.set_xlabel("batch"); ax.set_ylabel("memoria pico [GB]"); ax.set_xticks(batches)
ax.set_title("Memoria de GPU por iteracion (AMP)")
ax = fig.add_subplot(gs[0, 3])
ax.plot(batches, 1000 * np.array(tiempo), "o-", color=COLORES[0], lw=2, ms=8)
ax.set_xlabel("batch"); ax.set_ylabel("ms por iteracion"); ax.set_xticks(batches)
ax.set_title("Tiempo por iteracion (AMP)")
for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)

b = int(val.sum(1).argmax())                                     # la muestra con mas emisores
g = lote["pos"][b][lote["valido"][b]].numpy()
ext_xy = [-0.5, cs.W - 0.5, cs.H - 0.5, -0.5]
ext_xz = [-0.5, cs.W - 0.5, rej.z_planos[0] - rej.dz / 2, rej.z_planos[-1] + rej.dz / 2]
T, P = target[b].cpu().numpy(), pred[b].cpu().numpy()
vmax = rej.escala
paneles = [(T.max(0), ext_xy, "target: proyeccion en z", False), (P.max(0), ext_xy, "prediccion: proyeccion en z", False),
           (T.max(1), ext_xz, "target: x-z", True), (P.max(1), ext_xz, "prediccion: x-z", True)]
for c, (img, ext, titulo, xz) in enumerate(paneles):
    ax = fig.add_subplot(gs[1, c])
    ax.imshow(img, cmap="inferno", extent=ext, vmin=0, vmax=vmax, interpolation="nearest",
              origin="lower" if xz else "upper", aspect="auto" if xz else "equal")
    ax.scatter(g[:, 0], g[:, 2] if xz else g[:, 1], s=60, facecolors="none", edgecolors="cyan", lw=0.8)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("x [px]"); ax.set_ylabel("z [nm]" if xz else "y [px]")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_modelo.png", dpi=130)
plt.close(fig)
print("\nOK -> fig_modelo.png")
