"""Check del espejo (vista reflejada) de simulador_psf. Genera dos figuras:

  fig_espejo_frame.png        un video con espejo (semilla fija, parametros por
                              defecto): frame 0 con reales y reflejos marcados y
                              unidos, y la trayectoria de un par real <-> reflejo
                              (desde la camara, vista lateral y z vs frame).
  fig_espejo_comparacion.png  para comparar con datos reales (particulas puntuales):
                              - un emisor y su reflejo con alpha = 53 grados a
                                d = 0.5, 1, 2 y 3 um del espejo y a dos alturas sobre
                                el cubreobjetos, SNR alta (20000 fotones, fondo
                                5 fot/px) y contraste fijo, con la separacion lateral
                                2 d sin(alpha) y la diferencia de z 2 d cos(alpha);
                              - una galeria de la PSF esferica sola en z = -2.0 ...
                                +3.0 um (respecto del plano focal), mismo contraste,
                                para estimar z de la particula y de su reflejo y de
                                ahi alpha = atan(dlat / dz) y los signos de
                                SIGNO_Z / SIGNO_INCLINACION (espejo.py);
                              - opcional: un recorte real al lado, con el mismo
                                tamano fisico en um.

Uso:
    $PY checks/check_espejo.py
    $PY checks/check_espejo.py --real recorte.tif --pixel-nm 98.48 [--frame 0]

Las aserciones de verdad estan en tests/test_espejo.py; aqui solo se
comprueba que lo dibujado cumple dlat / dz = tan(alpha).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulador_psf import Config, BancoPSF, Espejo, generar_video, aplicar_camara
from simulador_psf.espejo import SIGNO_Z, SIGNO_INCLINACION

OUT = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--real", help="TIFF con un recorte real (2D, o 3D y se toma --frame)")
ap.add_argument("--pixel-nm", type=float, help="tamano de pixel del recorte real en la muestra [nm]")
ap.add_argument("--frame", type=int, default=0, help="frame del TIFF real si es un stack")
args = ap.parse_args()
if args.real and not args.pixel_nm:
    ap.error("--real necesita --pixel-nm")


def linea_espejo(ax, e, cfg, h_nm=0.0, **kw):
    """Dibuja en coordenadas de imagen [px] la recta donde el espejo esta a altura
    h_nm sobre el cubreobjetos (h = 0: la linea en el cubreobjetos)."""
    a = np.radians(cfg.angulo_espejo_deg)
    # punto del plano a altura h: p0 + s m + h arriba con d = 0 -> s = SIGNO_INCLINACION h cos/sin
    s = SIGNO_INCLINACION * h_nm * np.cos(a) / np.sin(a)
    q = e.p0 + s * e.m + h_nm * e.arriba
    L = 10 * max(cfg.H, cfg.W) * cfg.pixel_nm
    seg = q + np.outer([-L, L], e.t)
    ax.plot(seg[:, 0] / cfg.pixel_nm, seg[:, 1] / cfg.pixel_nm, **kw)


# =============================== figura 1: video con espejo ===============================
cfg = Config(usar_espejo=True, n_frames=40, n_emisores_rango=(6, 10), seed=3)
print(cfg.resumen())
banco = BancoPSF(cfg)
e = Espejo(cfg)
print(f"banco: z en [{banco.z_grid[0]:.0f}, {banco.z_grid[-1]:.0f}] nm, {len(banco.z_grid)} planos")
video, filas = generar_video(cfg, banco, np.random.default_rng(cfg.seed), 0)
gt = pd.DataFrame(filas)
print(f"{gt.particle_id.nunique()} emisores; reflejo_en_fov {100*gt.reflejo_en_fov.mean():.1f}%, "
      f"reflejo_en_banco {100*gt.reflejo_en_banco.mean():.1f}%")
d = e.distancia_nm(e.a_nm(gt[["x_px", "y_px", "z_nm"]].to_numpy()))
print(f"distancia minima de un real al espejo: {d.min():.1f} nm (debe ser >= 0)")
assert d.min() >= -1e-6

vmin, vmax = np.percentile(video, [0.5, 99.9])                  # contraste fijo para todo el video
fig, ax = plt.subplots(1, 4, figsize=(21, 5.4), gridspec_kw=dict(width_ratios=[1.15, 1, 1, 1]))
A = ax[0]
im = A.imshow(video[0], cmap="gray", vmin=vmin, vmax=vmax)
g0 = gt[gt.frame == 0]
for r in g0.itertuples():
    A.plot([r.x_px, r.x_reflejo_px], [r.y_px, r.y_reflejo_px], color="w", lw=0.5, alpha=0.7)
A.scatter(g0.x_px, g0.y_px, s=60, facecolors="none", edgecolors="cyan", lw=1.2, label="real")
A.scatter(g0.x_reflejo_px, g0.y_reflejo_px, s=60, marker="s", facecolors="none", edgecolors="orange",
          lw=1.2, label="reflejo")
BASE_ESPEJO = "base del espejo (contacto con el cubreobjetos)"
linea_espejo(A, e, cfg, 0.0, color="yellow", lw=1.0, ls="--", label=BASE_ESPEJO)
linea_espejo(A, e, cfg, SIGNO_Z * (cfg.z_rango_nm[1] - e.z_cubre), color="yellow", lw=0.8, ls=":",
             label="espejo a la altura de la pared superior de los reales")
A.set_xlim(-0.5, cfg.W - 0.5); A.set_ylim(cfg.H - 0.5, -0.5)
A.set_title(f"frame 0 (alpha {cfg.angulo_espejo_deg:.0f} deg, ratio {cfg.ratio_reflejo}), contraste fijo", fontsize=10)
A.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)   # fuera de la imagen
plt.colorbar(im, ax=A, fraction=0.046, label="ADU")

# el par con mas frames con real y reflejo dentro del FOV
cuenta = gt[gt.en_fov & gt.reflejo_en_fov].groupby("particle_id").size()
pid = int(cuenta.idxmax()) if len(cuenta) else 0
g = gt[gt.particle_id == pid].sort_values("frame")
B = ax[1]
B.plot(g.x_px, g.y_px, "-", color="tab:blue", lw=1, label="real")
B.plot(g.x_reflejo_px, g.y_reflejo_px, "-", color="tab:orange", lw=1, label="reflejo")
B.plot(g.x_px.iloc[0], g.y_px.iloc[0], "o", color="tab:blue"); B.plot(g.x_reflejo_px.iloc[0], g.y_reflejo_px.iloc[0], "s", color="tab:orange")
for k in range(0, len(g), 5):
    B.plot([g.x_px.iloc[k], g.x_reflejo_px.iloc[k]], [g.y_px.iloc[k], g.y_reflejo_px.iloc[k]], color="0.7", lw=0.5)
linea_espejo(B, e, cfg, 0.0, color="k", lw=1.0, ls="--", label=BASE_ESPEJO)
B.set_aspect("equal"); B.invert_yaxis(); B.set_xlabel("x [px]"); B.set_ylabel("y [px]")
xs = np.concatenate([g.x_px, g.x_reflejo_px]); ys = np.concatenate([g.y_px, g.y_reflejo_px])
B.set_xlim(xs.min() - 3, xs.max() + 3); B.set_ylim(ys.max() + 3, ys.min() - 3)
B.set_title(f"particula {pid}: desde la camara (o = frame 0, gris cada 5 frames)", fontsize=10)
B.legend(fontsize=8)

C = ax[2]
lat = lambda q: ((q - e.p0) @ e.m / 1000, (q - e.p0) @ e.arriba / 1000)
pr = e.a_nm(g[["x_px", "y_px", "z_nm"]].to_numpy()); prr = e.a_nm(g[["x_reflejo_px", "y_reflejo_px", "z_reflejo_nm"]].to_numpy())
C.plot(*lat(pr), "-", color="tab:blue", lw=1, label="real")
C.plot(*lat(prr), "-", color="tab:orange", lw=1, label="reflejo")
a = np.radians(cfg.angulo_espejo_deg)
hs = np.array([0.0, 6.0])
C.plot(SIGNO_INCLINACION * hs * np.cos(a) / np.sin(a), hs, "k-", lw=2, label="espejo")
C.axhspan((cfg.z_rango_nm[0] - e.p0[2]) * SIGNO_Z / 1000, (cfg.z_rango_nm[1] - e.p0[2]) * SIGNO_Z / 1000,
          color="tab:green", alpha=0.08, label="paredes en z de los reales")
C.axhline((0 - e.p0[2]) * SIGNO_Z / 1000, color="tab:green", ls="-.", lw=0.8, label="plano focal")
C.set_aspect("equal"); C.set_xlabel("distancia a la linea del espejo, hacia la muestra [um]")
C.set_ylabel("altura sobre el cubreobjetos [um]")
C.set_title("vista lateral (proyeccion sobre el plano m-z)", fontsize=10); C.legend(fontsize=7)

D_ = ax[3]
D_.plot(g.frame, g.z_nm, "-o", ms=3, color="tab:blue", label="z real")
D_.plot(g.frame, g.z_reflejo_nm, "-s", ms=3, color="tab:orange", label="z reflejo")
D_.axhspan(*cfg.z_rango_nm, color="tab:green", alpha=0.08)
D_.set_xlabel("frame"); D_.set_ylabel("z [nm] (respecto del plano focal)")
D_.set_title("z del par vs frame", fontsize=10); D_.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_espejo_frame.png"), dpi=120)
plt.close(fig)
print("guardada fig_espejo_frame.png")

# =============================== figura 2: comparacion con datos reales ===============================
ALPHA = 53.0
DS_UM = (0.5, 1.0, 2.0, 3.0)                 # distancia del emisor al plano del espejo
HS_UM = (0.4, 1.2)                           # altura del emisor sobre el cubreobjetos
N_FOT, FONDO, RATIO = 20000.0, 5.0, 0.8
GALERIA_UM = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
S = 112                                      # lado de cada panel del par [px]
SG = 64                                      # lado de cada panel de la galeria [px]

base = Config(z_banco_nm=(-2100.0, 4200.0))
bc = BancoPSF(base)
px = base.pixel_nm
z_cubre = Espejo(Config(usar_espejo=True, z_banco_nm=base.z_banco_nm)).z_cubre
rng = np.random.default_rng(0)
al = np.radians(ALPHA)

paneles = []                                 # (imagen, textos, x real, x reflejo) por (altura, d)
for h in HS_UM:
    fila = []
    for dd in DS_UM:
        # emisor a altura h y distancia dd del plano: p = p0 + L m + h arriba con L sin a - h cos a = dd
        L = (dd * 1000 + SIGNO_INCLINACION * h * 1000 * np.cos(al)) / np.sin(al)
        m_medio = L - dd * 1000 * np.sin(al)        # el punto medio real-reflejo, en m (queda en el centro)
        c = (S - 1) / 2
        cfg_p = Config(usar_espejo=True, angulo_espejo_deg=ALPHA, espejo_punto_px=(c - m_medio / px, c),
                       z_banco_nm=base.z_banco_nm)
        ep = Espejo(cfg_p)
        p = ep.p0 + L * ep.m + h * 1000 * ep.arriba
        q = ep.reflejar_nm(p)
        assert np.isclose(ep.distancia_nm(p), dd * 1000)
        dlat, dz = np.hypot(*(q - p)[:2]), (q - p)[2]
        assert np.isclose(dlat, 2 * dd * 1000 * np.sin(al)) and np.isclose(dz, 2 * dd * 1000 * np.cos(al))
        assert np.isclose(np.degrees(np.arctan2(dlat, dz)), ALPHA)
        assert bc.z_grid[0] <= q[2] <= bc.z_grid[-1] and bc.z_grid[0] <= p[2] <= bc.z_grid[-1]
        img = np.zeros((S, S))
        bc.colocar(img, p[0] / px, p[1] / px, p[2], N_FOT)
        bc.colocar(img, q[0] / px, q[1] / px, q[2], RATIO * N_FOT)
        adu = aplicar_camara(img + FONDO, base, rng)
        txt = (f"h = {h:.1f} um, d = {dd:.1f} um\n"
               f"dlat = 2d sin a = {dlat/1000:.2f} um\n"
               f"dz = 2d cos a = {dz/1000:+.2f} um\n"
               f"z real {p[2]/1000:+.2f}, reflejo {q[2]/1000:+.2f} um")
        fila.append((adu, txt, (p[0] / px, p[1] / px), (q[0] / px, q[1] / px), ep.p0[0] / px))
    paneles.append(fila)

galeria = []
for z in GALERIA_UM:
    img = np.zeros((SG, SG))
    cg = (SG - 1) / 2
    bc.colocar(img, cg, cg, z * 1000, N_FOT)
    galeria.append(aplicar_camara(img + FONDO, base, rng))

todas = np.concatenate([a.ravel() for f in paneles for a, *_ in f] + [g.ravel() for g in galeria])
gmin, gmax = np.percentile(todas, [0.5, 99.95])                  # mismo contraste en toda la figura

n_col = len(DS_UM) + (1 if args.real else 0)
fig = plt.figure(figsize=(3.6 * max(n_col, 4) + 1, 3.6 * len(HS_UM) + 2.2))
sub_top, sub_bot = fig.subfigures(2, 1, height_ratios=[3.6 * len(HS_UM), 2.2])
axs = sub_top.subplots(len(HS_UM), n_col, squeeze=False)
medio_um = S * px / 2000                                         # semiancho fisico de cada panel [um]
ext = lambda n: (-n * px / 2000, n * px / 2000, n * px / 2000, -n * px / 2000)
for i, fila in enumerate(paneles):
    for j, (adu, txt, pr_, prf, xlin) in enumerate(fila):
        A = axs[i, j]
        A.imshow(adu, cmap="gray", vmin=gmin, vmax=gmax, extent=ext(S))
        cpx = (S - 1) / 2
        um = lambda v: (v - cpx) * px / 1000
        primero = (i, j) == (0, 0)
        A.axvline(um(xlin), color="yellow", lw=0.8, ls="--", label=BASE_ESPEJO if primero else None)
        A.plot(um(pr_[0]), -medio_um * 0.93, "v", color="cyan", ms=7,           # marcas fuera de la mancha
               label="x del real" if primero else None)
        A.plot(um(prf[0]), -medio_um * 0.93, "v", color="orange", ms=7, label="x del reflejo" if primero else None)
        if primero:
            A.legend(loc="upper right", fontsize=6, framealpha=0.7)
        A.text(0.02, 0.02, txt, transform=A.transAxes, va="bottom", fontsize=7, color="w",
               bbox=dict(facecolor="k", alpha=0.45, lw=0))
        A.set_xlim(-medio_um, medio_um); A.set_ylim(medio_um, -medio_um)
        if i == len(HS_UM) - 1:
            A.set_xlabel("x [um]")
        if j == 0:
            A.set_ylabel("y [um]")
if args.real:
    import tifffile
    real = tifffile.imread(args.real).astype(float)
    if real.ndim == 3:
        real = real[args.frame]
    hy, hx = real.shape[0] * args.pixel_nm / 2000, real.shape[1] * args.pixel_nm / 2000
    rmin, rmax = np.percentile(real, [0.5, 99.9])
    for i in range(len(HS_UM)):
        A = axs[i, -1]
        A.imshow(real, cmap="gray", vmin=rmin, vmax=rmax, extent=(-hx, hx, hy, -hy))
        A.set_xlim(-medio_um, medio_um); A.set_ylim(medio_um, -medio_um)
        A.set_title(f"real: {os.path.basename(args.real)}\n{args.pixel_nm:.1f} nm/px, contraste propio", fontsize=8)
sub_top.suptitle(f"emisor (cian) y reflejo (naranja), alpha = {ALPHA:.0f} deg, {N_FOT:.0f} fotones, "
                 f"reflejo x{RATIO}, fondo {FONDO:.0f} fot/px, contraste fijo; plano focal a {(0 - z_cubre) * SIGNO_Z / 1000:.1f} um sobre el cubreobjetos",
                 fontsize=10)

axg = sub_bot.subplots(1, len(GALERIA_UM))
for A, z, gimg in zip(axg, GALERIA_UM, galeria):
    A.imshow(gimg, cmap="gray", vmin=gmin, vmax=gmax, extent=ext(SG))
    A.set_title(f"z = {z:+.1f} um", fontsize=9)
    A.set_xticks([]); A.set_yticks([])
lado = "lejos del objetivo" if SIGNO_Z > 0 else "hacia el objetivo"
sub_bot.suptitle(f"PSF esferica sola (mismo contraste, {N_FOT:.0f} fotones, panel de "
                 f"{SG * px / 1000:.1f} um), z respecto del plano focal:   "
                 f"<-  -z  hacia el objetivo   |   +z  {lado}  ->", fontsize=10)
fig.savefig(os.path.join(OUT, "fig_espejo_comparacion.png"), dpi=120)
plt.close(fig)
print("guardada fig_espejo_comparacion.png")
