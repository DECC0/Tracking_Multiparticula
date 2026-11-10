"""
Geometria del espejo plano junto a la muestra, compartida por simulador_psf
(simulador_psf/espejo.py) y simulador_nuevo (simulador_nuevo/simulador.py).

Solo numpy: no conoce la Config de ningun simulador ni consume numeros
aleatorios. Cada simulador la envuelve en su propia clase Espejo, que lee
los parametros de su Config y agrega la conversion [px, px, nm] <-> nm. Todo
aqui trabaja en nm en los tres ejes.

Convencion (diagramas en el docstring de simulador_psf/espejo.py): objetivo
arriba, +z HACIA el objetivo (fijo), y el angulo del espejo beta medido desde
el EJE OPTICO, en (-90, 90). beta > 0: el espejo queda detras de la muestra
vista desde el objetivo (el montaje; el reflejo siempre mas lejos del
objetivo que el real); beta = 0: pared paralela al eje optico; beta < 0:
inclinado hacia el otro lado. Los extremos +-90 (espejo paralelo al plano
focal) quedan excluidos.
    m = (cos phi, sin phi, 0)          direccion lateral linea del espejo -> muestra
    t = (-sin phi, cos phi, 0)         direccion de la linea del espejo
    n = cos(beta) m + sin(beta) z_hat  normal unitaria hacia la muestra
    d = (p - p0) . n                   distancia con signo (> 0: lado de la muestra)
    p' = p - 2 d n = p0 + R (p - p0),  R = I - 2 n n^T
    p' - p: lateral -2 d cos(beta) m, axial -2 d sin(beta) (z' < z con beta > 0)
Internamente alpha = 90 - beta es el angulo con el plano focal (no se expone).
"""

import numpy as np

MAX_PLIEGUES = 1000     # reflexiones maximas por punto en plegar_nm (en la practica, unas pocas)


def matriz_R(n):
    """Matriz de reflexion (3, 3) R = I - 2 n n^T del plano de normal unitaria n."""
    n = np.asarray(n, dtype=float)
    return np.eye(3) - 2.0 * np.outer(n, n)


class GeometriaEspejo:
    """Plano del espejo y paredes en z de los emisores reales.

    beta_deg           angulo espejo-eje optico [grados], en (-90, 90) (ver el docstring del modulo)
    phi_deg            azimut de m, de +x hacia +y
    p0_nm              (3,) punto del espejo en el plano de referencia [nm]; z_ref = p0[2]
    z_lo, z_hi         paredes en z de los emisores reales [nm] (+z hacia el objetivo)"""

    def __init__(self, beta_deg, phi_deg, p0_nm, z_lo, z_hi):
        if not -90.0 < beta_deg < 90.0:
            raise ValueError(f"el angulo desde el eje optico debe estar en (-90, 90): {beta_deg}")
        self.beta = b = np.radians(beta_deg)                     # [rad]
        self.alpha = np.pi / 2 - b                               # angulo con el plano focal (interno)
        f = np.radians(phi_deg)
        self.m = np.array([np.cos(f), np.sin(f), 0.0])
        self.t = np.array([-np.sin(f), np.cos(f), 0.0])
        self.n = np.cos(b) * self.m + np.sin(b) * np.array([0.0, 0.0, 1.0])
        self.R = matriz_R(self.n)
        self.z_lo, self.z_hi = z_lo, z_hi
        self.p0 = np.asarray(p0_nm, dtype=float)
        self.z_ref = float(self.p0[2])

    def distancia_nm(self, p_nm):
        """Distancia con signo al plano [nm] (...,): > 0 del lado de la muestra."""
        return (np.asarray(p_nm, dtype=float) - self.p0) @ self.n

    def punto_en_z_nm(self, z_nm):
        """Punto [nm] (3,) de la linea del espejo a la altura z_nm: p0 + s m en
        el plano z = z_nm, con d = 0 -> s cos(beta) = -(z_nm - z_ref) sin(beta).
        punto_en_z_nm(0) es la interseccion del espejo con el plano focal; la
        linea sigue en la direccion t."""
        s = -(z_nm - self.z_ref) * np.tan(self.beta)
        return self.p0 + s * self.m + np.array([0.0, 0.0, z_nm - self.z_ref])

    def reflejar_nm(self, p_nm):
        """p' = p - 2 [(p - p0) . n] n, todo en nm; p_nm (..., 3)."""
        p = np.asarray(p_nm, dtype=float)
        return p - 2.0 * self.distancia_nm(p)[..., None] * self.n

    def plegar_nm(self, p_nm):
        """Lleva cada punto (..., 3) [nm] al dominio de los emisores reales:
        z_lo <= z <= z_hi y d >= 0 (lado de la muestra), reflejando en la
        pared o en el espejo que viola, hasta que no viole ninguna.

        Es el mismo criterio que las paredes en z (reflexion especular) con el
        espejo como pared extra. Termina: el dominio es convexo y reflejar un
        punto en un semiespacio que viola acerca el punto a todo punto del
        semiespacio, en particular a todo punto del dominio."""
        p = np.array(p_nm, dtype=float)
        lo, hi = self.z_lo, self.z_hi
        for _ in range(MAX_PLIEGUES):
            z = p[..., 2]
            mal_z = (z < lo) | (z > hi)
            d = self.distancia_nm(p)
            mal_e = d < 0
            if not (mal_z.any() or mal_e.any()):
                return p
            # paredes en z: z -> 2 hi - z o 2 lo - z (una reflexion por vuelta)
            p[..., 2] = np.where(z > hi, 2 * hi - z, np.where(z < lo, 2 * lo - z, z))
            # espejo: p -> p - 2 d n solo donde d < 0 (d recalculado tras plegar z)
            d = self.distancia_nm(p)
            p -= 2.0 * np.minimum(d, 0.0)[..., None] * self.n
        raise RuntimeError(f"plegar_nm no convergio en {MAX_PLIEGUES} reflexiones")

    def rango_z_reflejo_caja(self, lo, hi):
        """Rango (z_min, z_max) [nm] de z' de los reflejos de los puntos de la
        caja [lo, hi] (dos vectores (3,) en nm) que estan del lado de la muestra.

        z' es lineal en p, asi que sus extremos sobre ese politopo convexo
        (caja ∩ {d >= 0}) estan en sus vertices: las esquinas de la caja del lado
        de la muestra y los cortes de las aristas de la caja con el plano.
        Si ningun punto de la caja queda del lado de la muestra lanza ValueError."""
        esq = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        d = self.distancia_nm(esq)
        vert = list(esq[d >= 0])
        for i in range(8):
            for j in range(i + 1, 8):
                if np.count_nonzero(esq[i] != esq[j]) == 1 and d[i] * d[j] < 0:   # arista que cruza el plano
                    s = d[i] / (d[i] - d[j])
                    vert.append(esq[i] + s * (esq[j] - esq[i]))
        if not vert:
            raise ValueError("el espejo deja toda la caja de posiciones iniciales fuera del lado de la muestra")
        zp = self.reflejar_nm(np.array(vert))[:, 2]
        return float(zp.min()), float(zp.max())

    def lateral_cerca(self, u_m, u_t, z, esq_xy_nm, dist_max_nm):
        """Posicion lateral (n, 2) [nm] a distancia d en [0, dist_max_nm] del
        PLANO del espejo, escrita como p0 + s_m m + s_t t, a partir de dos
        uniformes en [0, 1) (u_m, u_t) y de z [nm] (n,):
            s_t uniforme en la proyeccion sobre t de la region lateral (esquinas
                esq_xy_nm (4, 2) en nm);
            s_m uniforme en la franja d en [0, dist_max] a esa z. A altura z,
                d = s_m cos(beta) + sin(beta) (z - z_ref),
                asi que la franja es s_m en [s(0), s(dist_max)], recortada a la
                proyeccion sobre m de la region. Si a esa z la franja no toca la
                region se usa la franja entera (el punto queda fuera del FOV,
                pero respeta dist_max)."""
        esq = np.asarray(esq_xy_nm, dtype=float) - self.p0[:2]
        st, sm = esq @ self.t[:2], esq @ self.m[:2]              # region proyectada sobre t y m [nm]
        s_t = st.min() + u_t * (st.max() - st.min())
        corr = -np.sin(self.beta) * (z - self.z_ref)             # [nm]
        f0 = corr / np.cos(self.beta)                            # s_m con d = 0 a esa z
        f1 = (dist_max_nm + corr) / np.cos(self.beta)            # s_m con d = dist_max
        lo, hi = np.maximum(f0, sm.min()), np.minimum(f1, sm.max())
        vacia = lo > hi                                          # la franja no toca la region
        lo, hi = np.where(vacia, f0, lo), np.where(vacia, f1, hi)
        s_m = lo + u_m * (hi - lo)
        return self.p0[:2] + s_m[:, None] * self.m[:2] + s_t[:, None] * self.t[:2]
