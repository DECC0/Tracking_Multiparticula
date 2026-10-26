"""
Configuracion de una corrida: una dataclass por seccion, cargada desde YAML y
modificable desde la linea de comandos con --set seccion.campo=valor (el valor
se lee con sintaxis YAML, p. ej. --set datos.simulador.H=32).

Las rutas relativas se interpretan desde el directorio de trabajo, asi que la
misma configuracion corre en el portatil y en el cluster.
"""
import argparse
import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

NORMALIZACIONES = ("robusta", "fija")


def _exige(cond, msg):
    if not cond:
        raise ValueError(msg)


@dataclass
class ConfigDatos:
    simulador: dict = field(default_factory=dict)   # campos de simulador.Config; lo que falte usa su valor por defecto
    canales: int = 1                    # C canales por imagen (2 = biplano, pendiente del simulador biplano)

    # randomizacion extra por muestra, encima de la del simulador (n emisores, fotones, fondo)
    fotones_dispersion: float = None    # razon max/min de fotones dentro de una imagen; None -> todo fotones_rango
    prob_rango_completo: float = 0.0    # probabilidad de ignorar fotones_dispersion (tenues junto a brillantes)
    prob_fondo_uniforme: float = 0.0    # probabilidad de fondo uniforme en vez del gaussiano del simulador
    read_noise_rango: tuple = None      # ADU rms, uniforme por muestra; None -> el del simulador

    # entrada de la red
    normalizacion: str = "robusta"      # "robusta": (e- - mediana) / (1.4826 MAD); "fija": e- / escala_entrada
    escala_entrada: float = 100.0

    # splits: semillas distintas -> muestras distintas
    n_train: int = 20000
    n_val: int = 1000
    seed_train: int = 1
    seed_val: int = 2
    renovar_por_epoca: bool = False     # True: muestras de train nuevas en cada epoca

    def __post_init__(self):
        if self.read_noise_rango is not None:
            self.read_noise_rango = tuple(float(v) for v in self.read_noise_rango)
        self.validar()

    def validar(self):
        _exige(isinstance(self.simulador, dict), "datos.simulador debe ser un diccionario")
        _exige(self.canales >= 1, "datos.canales debe ser >= 1")
        _exige(self.fotones_dispersion is None or self.fotones_dispersion >= 1,
               "datos.fotones_dispersion debe ser >= 1 (o null)")
        _exige(0 <= self.prob_rango_completo <= 1, "datos.prob_rango_completo debe estar en [0, 1]")
        _exige(0 <= self.prob_fondo_uniforme <= 1, "datos.prob_fondo_uniforme debe estar en [0, 1]")
        if self.read_noise_rango is not None:
            lo, hi = self.read_noise_rango
            _exige(0 <= lo <= hi, "datos.read_noise_rango: se requiere 0 <= min <= max")
        _exige(self.normalizacion in NORMALIZACIONES, f"normalizacion desconocida: {self.normalizacion}")
        _exige(self.escala_entrada > 0, "datos.escala_entrada debe ser positiva")
        _exige(self.n_train >= 1 and self.n_val >= 1, "datos.n_train y datos.n_val deben ser >= 1")
        _exige(self.seed_train != self.seed_val, "seed_train y seed_val deben ser distintas (si no, train y val se solapan)")


@dataclass
class ConfigTarget:
    upsampling: int = 4                 # voxeles por pixel en x e y
    dz_nm: float = 50.0                 # tamano del voxel en z
    planos_margen: int = 2              # planos extra bajo y sobre z_rango_nm: gaussianas sin truncar en los bordes
    sigma_xy_vox: float = 1.0           # sigma de la gaussiana del target, en voxeles
    sigma_z_vox: float = 1.0
    escala: float = 100.0               # valor pico de cada gaussiana (factor de escala de DeepSTORM3D)

    def __post_init__(self):
        self.validar()

    def validar(self):
        _exige(int(self.upsampling) == self.upsampling and self.upsampling >= 1,
               "target.upsampling debe ser un entero >= 1")
        _exige(self.dz_nm > 0, "target.dz_nm debe ser positivo")
        _exige(self.planos_margen >= 0, "target.planos_margen debe ser >= 0")
        _exige(self.sigma_xy_vox > 0 and self.sigma_z_vox > 0, "las sigmas del target deben ser positivas")
        _exige(self.escala > 0, "target.escala debe ser positiva")


MODOS_UPSAMPLING = ("bilinear", "nearest")
NORMS = ("batch", "grupo")
ACTIVACIONES = ("hardtanh", "sigmoide")


@dataclass
class ConfigModelo:
    canales_base: int = 64              # filtros por capa (DeepSTORM3D: 64)
    dilataciones: tuple = (1, 2, 4, 8, 16)   # capas del encoder tras la primera; campo receptivo 1 + 2 (1 + suma)
    modo_upsampling: str = "bilinear"   # interpolacion x2 del decoder
    capas_salida: int = 2               # capas residuales con Nz canales antes de la conv 1x1
    norm: str = "batch"                 # "batch" o "grupo" (GroupNorm, para batches muy pequenos)
    pendiente_leaky: float = 0.2
    activacion_salida: str = "hardtanh" # "hardtanh": recorta a [0, escala] (DeepSTORM3D); "sigmoide": escala * sigmoid

    def __post_init__(self):
        self.dilataciones = tuple(int(d) for d in self.dilataciones)
        self.validar()

    def validar(self):
        _exige(self.canales_base >= 1, "modelo.canales_base debe ser >= 1")
        _exige(all(d >= 1 for d in self.dilataciones), "modelo.dilataciones deben ser >= 1")
        _exige(self.modo_upsampling in MODOS_UPSAMPLING, f"modo_upsampling desconocido: {self.modo_upsampling}")
        _exige(self.capas_salida >= 0, "modelo.capas_salida debe ser >= 0")
        _exige(self.norm in NORMS, f"norm desconocida: {self.norm}")
        _exige(self.activacion_salida in ACTIVACIONES, f"activacion_salida desconocida: {self.activacion_salida}")


@dataclass
class ConfigPerdida:
    terminos: dict = field(default_factory=lambda: {"mse": 1.0, "dice": 1.0})   # termino -> peso
    suavizado_dice: float = 1.0

    def __post_init__(self):
        self.validar()

    def validar(self):
        _exige(isinstance(self.terminos, dict) and self.terminos, "perdida.terminos debe ser un dict no vacio")
        _exige(all(w >= 0 for w in self.terminos.values()), "los pesos de la perdida deben ser >= 0")
        _exige(self.suavizado_dice > 0, "perdida.suavizado_dice debe ser positivo")


MODOS_AMP = ("fp16", "bf16", "no")


@dataclass
class ConfigEntrenamiento:
    nombre: str = "corrida"             # prefijo de la carpeta de la corrida
    epocas: int = 30
    batch: int = 8
    acumulacion: int = 1                # batches por paso del optimizador (batch efectivo = batch * acumulacion)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: str = "fp16"                   # "fp16" (con GradScaler), "bf16" o "no"; en CPU fp16 se desactiva
    clip_grad: float = None             # norma maxima del gradiente; None -> sin recorte
    factor_lr: float = 0.5              # ReduceLROnPlateau sobre la perdida de validacion
    paciencia_lr: int = 3
    lr_min: float = 1e-5
    parada_temprana: int = None         # epocas sin mejorar en val antes de parar; None -> nunca
    workers: int = 2                    # procesos del DataLoader (simulan on-the-fly)
    dispositivo: str = "auto"           # "auto", "cpu" o "cuda"
    semilla: int = 0                    # pesos iniciales y orden de los batches
    log_cada: int = 50                  # iteraciones entre lineas de log_iter.csv
    datos_train: str = None             # carpeta de pregenerar.py; None -> on-the-fly
    datos_val: str = None
    carpeta_corridas: str = None        # None -> localizacion/corridas/; si no, relativa al directorio de trabajo

    def __post_init__(self):
        self.validar()

    def validar(self):
        _exige(self.epocas >= 1 and self.batch >= 1 and self.acumulacion >= 1,
               "epocas, batch y acumulacion deben ser >= 1")
        _exige(self.lr > 0 and self.weight_decay >= 0, "se requiere lr > 0 y weight_decay >= 0")
        _exige(self.amp in MODOS_AMP, f"amp desconocido: {self.amp}")
        _exige(self.clip_grad is None or self.clip_grad > 0, "clip_grad debe ser positivo (o null)")
        _exige(0 < self.factor_lr < 1 and self.paciencia_lr >= 0, "se requiere 0 < factor_lr < 1 y paciencia_lr >= 0")
        _exige(self.parada_temprana is None or self.parada_temprana >= 1, "parada_temprana debe ser >= 1 (o null)")
        _exige(self.workers >= 0 and self.log_cada >= 1, "se requiere workers >= 0 y log_cada >= 1")
        _exige(self.dispositivo in ("auto", "cpu", "cuda"), f"dispositivo desconocido: {self.dispositivo}")


CONFIANZAS = ("pico", "masa")


@dataclass
class ConfigPostproc:
    """Salida de la red -> detecciones (postproc.py). No afecta el entrenamiento:
    infer.py y evaluate.py la cambian con --set postproc.campo=valor."""
    # por defecto: la mejor combinacion del barrido de checks/check_postproc.py en validacion (Jaccard 0.825)
    umbral: float = 0.4                 # confianza minima de una deteccion
    radio_xy_vox: int = 3               # semiancho de la ventana del maximo local, en voxeles (x, y)
    radio_z_planos: int = 3             # ... y en planos (z)
    radio_cm_xy_vox: int = 2            # semiancho de la ventana del centro de masa
    radio_cm_z_planos: int = 2
    confianza: str = "masa"             # "pico": maximo / escala; "masa": suma en la ventana del centro de masa / la de
                                        # una gaussiana del target (puede pasar de 1); "masa" depende menos del umbral

    def __post_init__(self):
        self.validar()

    def validar(self):
        _exige(self.umbral >= 0, "postproc.umbral debe ser >= 0")
        _exige(all(int(r) == r and r >= 0 for r in (self.radio_xy_vox, self.radio_z_planos,
                                                     self.radio_cm_xy_vox, self.radio_cm_z_planos)),
               "los radios de postproc deben ser enteros >= 0")
        _exige(self.confianza in CONFIANZAS, f"postproc.confianza desconocida: {self.confianza}")


@dataclass
class ConfigCorrida:
    datos: ConfigDatos = field(default_factory=ConfigDatos)
    target: ConfigTarget = field(default_factory=ConfigTarget)
    modelo: ConfigModelo = field(default_factory=ConfigModelo)
    perdida: ConfigPerdida = field(default_factory=ConfigPerdida)
    entrenamiento: ConfigEntrenamiento = field(default_factory=ConfigEntrenamiento)
    postproc: ConfigPostproc = field(default_factory=ConfigPostproc)


SECCIONES = {f.name: f.type for f in dataclasses.fields(ConfigCorrida)}


# ----------------------------------------------------------------------------
# YAML y linea de comandos
# ----------------------------------------------------------------------------

def aplicar_overrides(d, overrides):
    """Modifica el dict anidado d con entradas "a.b.c=valor" (valor en sintaxis YAML)."""
    for o in overrides:
        clave, sep, valor = o.partition("=")
        _exige(sep and clave.strip(), f"override invalido (se espera clave=valor): {o!r}")
        *ruta, hoja = clave.strip().split(".")
        nodo = d
        for k in ruta:
            nodo = nodo.setdefault(k, {})
            _exige(isinstance(nodo, dict), f"override {o!r}: {k} no es una seccion")
        nodo[hoja] = yaml.safe_load(valor)
    return d


def desde_dict(d):
    """ConfigCorrida desde un dict {seccion: {campo: valor}}; rechaza claves desconocidas."""
    desconocidas = set(d) - set(SECCIONES)
    _exige(not desconocidas, f"secciones desconocidas: {sorted(desconocidas)}")
    kw = {}
    for nombre, clase in SECCIONES.items():
        sec = d.get(nombre) or {}
        extra = set(sec) - {f.name for f in dataclasses.fields(clase)}
        _exige(not extra, f"{nombre}: campos desconocidos {sorted(extra)}")
        kw[nombre] = clase(**sec)
    return ConfigCorrida(**kw)


def cargar_config(ruta=None, overrides=()):
    """Lee el YAML (o los valores por defecto si ruta es None) y aplica los overrides."""
    d = yaml.safe_load(Path(ruta).read_text()) if ruta else None
    return desde_dict(aplicar_overrides(d or {}, overrides))


def _a_listas(x):
    """Tuplas -> listas, recursivo (yaml.safe_dump no acepta tuplas)."""
    if isinstance(x, dict):
        return {k: _a_listas(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_a_listas(v) for v in x]
    return x


def a_dict(cfg):
    """Dataclass (anidada) -> dict serializable a YAML."""
    return _a_listas(asdict(cfg))


def guardar_yaml(d, ruta):
    Path(ruta).write_text(yaml.safe_dump(_a_listas(d), sort_keys=False))


def agregar_argumentos(parser: argparse.ArgumentParser):
    """--config y --set, comunes a todos los scripts."""
    parser.add_argument("--config", default=None, help="YAML de la corrida (sin el: valores por defecto)")
    parser.add_argument("--set", nargs="*", default=[], metavar="CLAVE=VALOR",
                        help="modifica campos del YAML, p. ej. --set datos.simulador.H=32 target.dz_nm=100")
    return parser


def config_desde_args(args):
    return cargar_config(args.config, args.set)
