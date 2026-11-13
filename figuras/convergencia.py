"""
Convergencia del entrenamiento de las corridas con celula, solo desde sus logs
(log.csv por epoca y log_iter.csv cada log_cada iteraciones; no corre la red).

1. Perdidas de entrenamiento y validacion por epoca (total = MSE + Dice, y cada
   termino) y la epoca del minimo de validacion. log.csv no guarda metricas de
   deteccion (recall, precision, Jaccard, RMSE): solo perdidas.
2. Tendencia de las ultimas N epocas (por defecto 20): pendiente por minimos
   cuadrados contra la epoca, con su error estandar OLS y el de Newey-West (lag
   3; las epocas consecutivas no son independientes), el cambio esperado en 10
   epocas y si el IC 95% excluye 0.
3. Tasa de aprendizaje usada en cada epoca (de log_iter.csv: la columna lr de
   log.csv se escribe despues de ReduceLROnPlateau.step, asi que es la de la
   epoca siguiente) y las epocas en que decae.

Uso:
    PY=~/miniforge3/envs/tesis/bin/python
    $PY figuras/convergencia.py [--ultimas 20] [--corridas CARPETA ...]
Escribe en --salida (por defecto figuras/): convergencia_tendencia.csv,
convergencia_ventanas.csv (pendiente en ventanas consecutivas de N epocas),
convergencia_lr.csv y fig_convergencia.png (300 dpi).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CORRIDAS = RAIZ / "localizacion" / "corridas"
POR_DEFECTO = ["hypatia_celula_b45_cerca_20261110-234627", "hypatia_celula_b45_lejos_20261110-234617"]
COLOR = {"cerca": "#2a78d6", "lejos": "#eb6834"}
TERMINOS = ("total", "mse", "dice")
LAG_NW = 3


def nombre(carpeta):
    return "cerca" if "cerca" in Path(carpeta).name else "lejos" if "lejos" in Path(carpeta).name else Path(carpeta).name


def pendiente(x, y, lag=LAG_NW):
    """(pendiente, ee OLS, ee Newey-West, ordenada) de y ~ a + b x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.c_[np.ones_like(x), x]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    n = len(x)
    s2 = e @ e / (n - 2)
    ee_ols = np.sqrt(s2 * XtX_inv[1, 1])
    # Newey-West: S = sum_l w_l sum_t e_t e_{t-l} (x_t x_{t-l}^T + x_{t-l} x_t^T), w_l = 1 - l / (lag + 1)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * n / (n - 2)
    return beta[1], ee_ols, np.sqrt(V[1, 1]), beta[0]


def lr_por_epoca(it):
    """lr usada en cada epoca (constante dentro de la epoca: se verifica)."""
    g = it.groupby("epoca").lr
    assert (g.min() == g.max()).all(), "la lr cambia dentro de una epoca"
    return g.first()


def analizar(carpeta, ultimas):
    carpeta = Path(carpeta)
    log = pd.read_csv(carpeta / "log.csv")
    it = pd.read_csv(carpeta / "log_iter.csv")
    lr = lr_por_epoca(it)
    var = nombre(carpeta)
    filas, ventanas = [], []
    sel = log[log.epoca >= log.epoca.max() - ultimas + 1]
    for parte in ("train", "val"):
        for t in TERMINOS:
            col = f"{parte}_{t}"
            b, ee, ee_nw, a = pendiente(sel.epoca, sel[col])
            ee_max = max(ee, ee_nw)
            filas.append(dict(variante=var, serie=col, epocas=f"{sel.epoca.min()}-{sel.epoca.max()}",
                              pendiente_por_epoca=b, ee_ols=ee, ee_newey_west=ee_nw,
                              ic95_lo=b - 1.96 * ee_max, ic95_hi=b + 1.96 * ee_max,
                              baja=(b + 1.96 * ee_max) < 0, cambio_10_epocas=10 * b,
                              cambio_relativo_10_epocas=10 * b / sel[col].mean(),
                              valor_final=log[col].iloc[-1], valor_inicial=log[col].iloc[0],
                              minimo=log[col].min(), epoca_minimo=int(log.epoca[log[col].idxmin()]),
                              std_residuo=float(np.std(sel[col] - (a + b * sel.epoca), ddof=2))))
    t = pd.DataFrame(filas)
    # desaceleracion: pendiente de val_total y val_mse en ventanas consecutivas de `ultimas` epocas
    for ini in range(0, int(log.epoca.max()) + 1 - ultimas + 1, ultimas):
        w = log[(log.epoca >= ini) & (log.epoca < ini + ultimas)]
        for col in ("val_total", "val_mse"):
            b, ee, ee_nw, _ = pendiente(w.epoca, w[col])
            ventanas.append(dict(variante=var, serie=col, epocas=f"{ini}-{ini + ultimas - 1}", pendiente_por_epoca=b,
                                 ee=max(ee, ee_nw), cambio_relativo_10_epocas=10 * b / w[col].mean(),
                                 lr_media=float(lr.loc[w.epoca].mean())))
    cambios = lr[lr.diff().fillna(1) != 0]
    t_lr = pd.DataFrame(dict(variante=var, epoca=cambios.index, lr=cambios.values))
    return var, log, it, lr, t, t_lr, pd.DataFrame(ventanas)


def figura(datos, ultimas, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    for var, log, it, lr, t, _, _ in datos:
        c = COLOR.get(var, "#52514e")
        ax = axes[0, 0]
        suave = it.set_index("epoca_continua").total.rolling(20, center=True, min_periods=5).mean()
        ax.plot(suave.index, suave.values, "-", color=c, lw=0.6, alpha=0.35)
        ax.plot(log.epoca, log.train_total, "-", color=c, lw=1.4, label=f"{var}: train (muestras nuevas por epoca)")
        ax.plot(log.epoca, log.val_total, "--", color=c, lw=1.6, label=f"{var}: val (1000 fijas)")
        k = log.val_total.idxmin()
        ax.plot(log.epoca[k], log.val_total[k], "*", color=c, ms=11, mec="white", mew=0.8)
        for j, term in enumerate(("mse", "dice")):
            ax = axes[0, 1 + j]
            ax.plot(log.epoca, log[f"train_{term}"], "-", color=c, lw=1.4, label=f"{var}: train")
            ax.plot(log.epoca, log[f"val_{term}"], "--", color=c, lw=1.6, label=f"{var}: val")
        ax = axes[1, 0]
        ax.step(lr.index, lr.values, where="post", color=c, lw=1.6, label=var)
        # ultimas epocas con el ajuste
        for j, parte in enumerate(("val", "train")):
            ax = axes[1, 1 + j]
            col = f"{parte}_total"
            sel = log[log.epoca >= log.epoca.max() - ultimas + 1]
            fila = t[t.serie == col].iloc[0]
            ref = sel[col].iloc[-1]                          # cada corrida respecto de su ultima epoca
            ax.plot(sel.epoca, 1e3 * (sel[col] - ref), "o", color=c, ms=3.5)
            xx = np.array([sel.epoca.min(), sel.epoca.max()])
            a = sel[col].mean() - fila.pendiente_por_epoca * sel.epoca.mean()
            ax.plot(xx, 1e3 * (a + fila.pendiente_por_epoca * xx - ref), "-", color=c, lw=1.6,
                    label=f"{var}: {fila.pendiente_por_epoca * 10:+.4f} / 10 epocas "
                          f"[{fila.ic95_lo * 10:+.4f}, {fila.ic95_hi * 10:+.4f}]")
            for e in lr.index[(lr.diff() != 0) & (lr.index > 0)]:
                if e >= sel.epoca.min():
                    ax.axvline(e, color=c, lw=0.7, ls=":")
            ax.set_title(f"{parte}: ultimas {ultimas} epocas (punteado: la lr decae)", fontsize=9)
            ax.set_xlabel("epoca")
            ax.set_ylabel(f"perdida total ({parte}) - la de la epoca {int(log.epoca.max())}  [x 1e-3]")
            ax.legend(frameon=False, fontsize=7)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("perdida total (MSE + Dice); estrella: minimo de val", fontsize=9)
    axes[0, 1].set_title("termino MSE", fontsize=9)
    axes[0, 2].set_title("termino Dice", fontsize=9)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("tasa de aprendizaje usada en cada epoca (ReduceLROnPlateau)", fontsize=9)
    axes[1, 0].axhline(1e-5, color="#8a8984", lw=0.8, ls=":")
    axes[1, 0].text(2, 1.15e-5, "lr_min", fontsize=7, color="#8a8984")
    for ax in axes.flat:
        ax.grid(True, color="#e6e5e0", lw=0.6)
    for ax in list(axes[0]) + [axes[1, 0]]:
        ax.set_xlabel("epoca (0 = la primera)")
        ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Convergencia de las corridas con celula (log.csv y log_iter.csv; sin metricas de deteccion por epoca)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--corridas", nargs="+", default=[str(CORRIDAS / c) for c in POR_DEFECTO])
    p.add_argument("--ultimas", type=int, default=20)
    p.add_argument("--salida", default=str(Path(__file__).resolve().parent))
    a = p.parse_args()
    salida = Path(a.salida)
    datos = [analizar(c, a.ultimas) for c in a.corridas]
    t = pd.concat([d[4] for d in datos], ignore_index=True)
    t_lr = pd.concat([d[5] for d in datos], ignore_index=True)
    t.to_csv(salida / "convergencia_tendencia.csv", index=False)
    t_lr.to_csv(salida / "convergencia_lr.csv", index=False)
    tv = pd.concat([d[6] for d in datos], ignore_index=True)
    tv.to_csv(salida / "convergencia_ventanas.csv", index=False)
    figura(datos, a.ultimas, salida / "fig_convergencia.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for var, log, it, lr, _, _, _ in datos:
            print(f"{var}: {len(log)} epocas, columnas de log.csv {list(log.columns)}")
            print(f"  minimo de val_total {log.val_total.min():.4f} en la epoca {int(log.epoca[log.val_total.idxmin()])} "
                  f"(0 = la primera); mejor.pt guardado en las epocas {log.epoca[log.mejor].tail(10).tolist()}")
        print("\nlr usada (epoca desde la que rige):")
        print(t_lr.to_string(index=False))
        print(f"\npendiente por ventanas de {a.ultimas} epocas (desaceleracion):")
        print(tv.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        print(f"\ntendencia de las ultimas {a.ultimas} epocas (pendiente por epoca, IC 95% con el mayor de los ee):")
        print(t.drop(columns=["valor_inicial"]).to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
