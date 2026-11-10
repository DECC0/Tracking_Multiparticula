"""
Geometria de una celula esferica (levadura) que contiene a los emisores
reales. Modulo compartido, como geometria_espejo.py; por ahora solo lo usa
simulador_nuevo (Config.usar_celula).

Solo numpy: no conoce la Config de ningun simulador ni consume numeros
aleatorios (el muestreo recibe los uniformes). Todo en nm, con la convencion
del repo: +z hacia el objetivo, z = 0 en el plano focal.

Modelo:
    centro c (3,), radio R: los emisores cumplen |p - c| <= R.
    Posiciones iniciales uniformes en el volumen: r = R u^(1/3), direccion
    uniforme (cos(theta) = 1 - 2 u, phi = 2 pi u), con 3 uniformes por emisor.
    Browniano con frontera reflectante: un paso que sale de la esfera se
    refleja especularmente en el plano tangente del punto de salida (se
    resuelve la interseccion exacta del segmento con la esfera, y el resto del
    paso se refleja; se repite si vuelve a salir).
Pegada al espejo (tangente_al_espejo): el centro esta a distancia R del plano
del espejo, del lado de la muestra, asi que toda la celula cumple d >= 0 y
ningun emisor puede cruzar el espejo. El punto de contacto es c - R n; el
reflejo de la celula es la esfera de centro c' = c - 2 R n y el mismo radio.
"""

import numpy as np

MAX_REFLEXIONES = 100000   # reflexiones por paso en mover_nm (en la practica 0 o 1; un rayo casi rasante con
                           # un paso mucho mayor que R rebota muchas veces con cuerdas cortas)
TOL_NM = 1e-9              # un punto a menos de TOL_NM fuera de la superficie cuenta como dentro (redondeo)


class GeometriaCelula:
    """Esfera de centro centro_nm (3,) y radio radio_nm [nm]."""

    def __init__(self, centro_nm, radio_nm):
        if not radio_nm > 0:
            raise ValueError(f"el radio de la celula debe ser positivo: {radio_nm}")
        self.c = np.asarray(centro_nm, dtype=float).copy()
        self.R = float(radio_nm)

    @classmethod
    def tangente_al_espejo(cls, espejo, radio_nm, z_centro_nm, punto_xy_nm):
        """Celula tangente al plano de espejo (GeometriaEspejo), del lado de la
        muestra, con el centro a altura z_centro_nm. punto_xy_nm (2,) fija solo
        la posicion a lo largo de la linea del espejo (su proyeccion sobre t);
        la distancia a la linea la fija el contacto, d(c) = R:
            s_m = (R - sin(beta) (z_c - z_ref)) / cos(beta)."""
        e = espejo
        s_t = (np.asarray(punto_xy_nm, dtype=float) - e.p0[:2]) @ e.t[:2]
        s_m = (radio_nm - np.sin(e.beta) * (z_centro_nm - e.z_ref)) / np.cos(e.beta)
        c = e.p0 + s_m * e.m + s_t * e.t + np.array([0.0, 0.0, z_centro_nm - e.z_ref])
        return cls(c, radio_nm)

    def contacto_nm(self, espejo):
        """Punto de la celula que toca el plano del espejo: c - R n."""
        return self.c - self.R * espejo.n

    def reflejada(self, espejo):
        """La celula virtual (reflejo de la celula en el espejo): centro reflejado, mismo radio."""
        return GeometriaCelula(espejo.reflejar_nm(self.c), self.R)

    def rango_z(self):
        return float(self.c[2] - self.R), float(self.c[2] + self.R)

    def profundidad_nm(self, p_nm):
        """R - |p - c| (...,): >= 0 dentro de la celula."""
        return self.R - np.linalg.norm(np.asarray(p_nm, dtype=float) - self.c, axis=-1)

    def uniforme(self, u_r, u_cos, u_phi):
        """Puntos (n, 3) [nm] uniformes en el volumen, desde tres uniformes en [0, 1) (n,)."""
        r = self.R * np.asarray(u_r, dtype=float) ** (1.0 / 3.0)
        cos_t = 1.0 - 2.0 * np.asarray(u_cos, dtype=float)
        sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, None))
        phi = 2.0 * np.pi * np.asarray(u_phi, dtype=float)
        d = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)
        return self.c + r[:, None] * d

    def mover_nm(self, p_nm, paso_nm):
        """p + paso (n, 3) [nm] con reflexion especular en la superficie. p debe
        estar dentro (|p - c| <= R): el segmento p -> p + paso sale de la esfera a
        lo sumo por un punto h; el resto del paso se refleja en el plano tangente
        en h (r - 2 (r . u) u, u = (h - c) / R) y se sigue desde h."""
        p = np.array(p_nm, dtype=float)
        s = np.array(paso_nm, dtype=float)
        q = p + s
        fuera = np.linalg.norm(q - self.c, axis=-1) > self.R + TOL_NM
        for _ in range(MAX_REFLEXIONES):
            if not fuera.any():
                break
            a, d = p[fuera] - self.c, s[fuera]
            aa, ad, dd = (a * a).sum(-1), (a * d).sum(-1), (d * d).sum(-1)
            disc = np.clip(ad ** 2 - dd * (aa - self.R ** 2), 0.0, None)
            t = np.clip((-ad + np.sqrt(disc)) / np.maximum(dd, 1e-300), 0.0, 1.0)   # salida del segmento
            h = a + t[:, None] * d                               # relativo al centro, |h| = R
            u = h / np.linalg.norm(h, axis=-1, keepdims=True)
            resto = (1.0 - t)[:, None] * d
            resto = resto - 2.0 * (resto * u).sum(-1, keepdims=True) * u
            p[fuera] = self.c + h
            s[fuera] = resto
            q[fuera] = p[fuera] + resto
            fuera = np.linalg.norm(q - self.c, axis=-1) > self.R + TOL_NM
        else:
            raise RuntimeError(f"mover_nm no convergio en {MAX_REFLEXIONES} reflexiones")
        # el redondeo puede dejar un punto a ~1e-12 nm fuera: se lleva a la superficie
        r = np.linalg.norm(q - self.c, axis=-1)
        afuera = r > self.R
        q[afuera] = self.c + (q[afuera] - self.c) * (self.R / r[afuera])[:, None]
        return q
