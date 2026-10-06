# Documentacion de `sim_videos.py`

Este documento explica en detalle el simulador de **videos** de microscopia
de fluorescencia (`sim_videos.py`), construido como extension temporal de
`sim_imagenes.py`. Esta pensado para alguien que conoce `sim_imagenes.py`
(o lo esta leyendo en paralelo) y quiere entender exactamente que cambia,
que se reutiliza, y por que cada decision de diseno se tomo asi.

---

## 1. Idea central: de "imagenes independientes" a "video"

`sim_imagenes.py` genera `N` imagenes **independientes**: en cada una se
sortean emisores nuevos, en posiciones nuevas, sin relacion con la imagen
anterior. Eso sirve para entrenar/evaluar un localizador que trabaja
imagen por imagen, pero **no** sirve para simular una pelicula real de
microscopia, donde:

- las mismas particulas **persisten** de un cuadro al siguiente,
- se **mueven** (difusion Browniana, deriva de la platina),
- **parpadean** (fluorescencia intermitente) y eventualmente se
  **fotoblanquean** (dejan de emitir para siempre),
- pueden **entrar y salir** del campo de vision con el tiempo.

`sim_videos.py` agrega exactamente esos cuatro ingredientes (movimiento,
fotofisica, poblacion, tiempo) sobre la misma base optica. Todo lo que
tiene que ver con "como se ve un emisor puntual en la camara" (la PSF, el
modelo de camara, el fondo) se **importa sin modificar** desde
`sim_imagenes.py`:

```python
from sim_imagenes import Config as ConfigImagen, BancoPSF, generar_fondo, aplicar_camara
```

Esto es deliberado: evita tener dos copias del modelo de PSF que se puedan
desincronizar, y dice explicitamente "el video y las imagenes fijas vienen
de la misma optica".

---

## 2. Que se reutiliza tal cual de `sim_imagenes.py`

| Pieza | Que hace | Por que no cambia para video |
|---|---|---|
| `BancoPSF` | Precalcula la PSF (con mascara de fase *corkscrew*) en una rejilla de `z`, y la coloca en la imagen con desplazamiento subpixel exacto via fase lineal en Fourier. | La forma de la PSF depende solo de la optica (NA, longitud de onda, mascara de fase) y de `z`, no del tiempo. Un emisor en `z` se ve igual en el frame 1 que en el frame 300. |
| `generar_fondo` | Genera un fondo con posible domo gaussiano de iluminacion no uniforme. | En este simulador se llama **una sola vez por video** (no por frame): representa una iluminacion de campo ancho que no cambia durante la adquisicion. Ver seccion 6 para las implicaciones. |
| `aplicar_camara` | Aplica la cadena Poisson -> ganancia -> ruido de lectura -> offset -> `uint16`. | El sensor de la camara no sabe si esta grabando un video o una foto suelta; el modelo de ruido por cuadro es identico. |

---

## 3. `ConfigVideo`: que agrega sobre `Config`

`ConfigVideo` hereda de `Config` (la clase de `sim_imagenes.py`), asi que
automaticamente tiene todos los parametros opticos, de camara y de PSF
(`NA`, `n_medio`, `lambda_nm`, `pixel_camara_um`, `magnificacion`, `H`,
`W`, `gain`, `offset`, `read_noise`, `mascara`, `zonas_corkscrew`,
`M_pupila`, `K_psf`, `z_paso_nm`, `blur_extra_px`, `z_rango_nm`,
`fondo_rango`, `fondo_no_uniforme`, `seed`).

Dos campos heredados cambian de significado en el contexto de video:

- **`n_emisores_rango`**: en imagenes fijas era "cuantos emisores hay en
  *esta* imagen". En video pasa a ser "cuantas particulas hay **al
  arrancar el video** (frame 0)"; la poblacion evoluciona despues por
  fotoblanqueo, salida del campo de vision e inmigracion (ver seccion 5).
- **`fotones_rango`**: en imagenes fijas era el numero de fotones exacto
  de esa unica observacion. En video es el rango del **brillo promedio**
  `mu_fotones` de cada particula mientras esta en estado ON; el numero de
  fotones real de cada cuadro fluctua alrededor de ese promedio (seccion
  4.3).

El campo heredado `n_imagenes` no se usa (se reemplaza por `n_frames`).

### 3.1 Campos nuevos

**Temporales**
- `n_frames` (int): numero de cuadros del video.
- `fps` (float): cuadros por segundo. De aqui sale `dt = 1/fps`, el paso de
  tiempo real (en segundos) que avanza la simulacion en cada iteracion.

**Movimiento (difusion Browniana)**
- `D_um2_s`: coeficiente de difusion lateral (x, y), en um²/s.
- `D_z_um2_s`: coeficiente de difusion axial (z), en um²/s (`0` = sin
  movimiento en z; util para difusion casi-2D pegada al cubreobjetos).
- `fraccion_inmovil`: fraccion de particulas con `D = 0`, es decir,
  ancladas (por ejemplo, moleculas fijadas a una superficie).

**Deriva de platina**
- `drift_x_nm_s`, `drift_y_nm_s`: velocidad constante y comun a todas las
  particulas (nm/s), para simular el corrimiento mecanico de la platina
  del microscopio durante una adquisicion larga.

**Fotofisica (parpadeo + fotoblanqueo)**
- `k_on_s`, `k_off_s`, `k_bleach_s`: tasas (1/s) de la cadena de Markov de
  tres estados (seccion 4.2).
- `frac_on_inicial`: fraccion de la poblacion inicial que arranca en
  estado ON en el frame 0.
- `fotones_cv`: coeficiente de variacion del brillo frame a frame de una
  particula ON (ruido de emision, no ruido de camara).

**Dinamica de poblacion**
- `tasa_llegada_s`: tasa (particulas/s) de aparicion de nuevas particulas
  dentro del campo de vision extendido (inmigracion). `0` = poblacion
  cerrada (nadie entra despues del frame 0).
- `margen_px`: borde alrededor del campo de vision donde pueden aparecer o
  seguir existiendo particulas cuyo centro esta fuera del sensor pero cuya
  PSF todavia alcanza a tocarlo.
- `factor_margen_salida`: multiplica a `margen_px` para definir un margen
  mas amplio, mas alla del cual una particula se considera perdida para
  siempre (seccion 5.2).

---

## 4. Dinamica de cada particula

Cada particula viva se representa con la clase `Emisor`:

```python
@dataclass
class Emisor:
    id: int
    x: float          # px, columna
    y: float           # px, fila
    z: float           # nm, profundidad
    movil: bool        # False si D = 0 (particula inmovil)
    mu_fotones: float  # brillo promedio (fotones/frame) mientras esta ON
    estado: str        # "on" | "off" | "bleached"
```

En cada frame, para cada particula viva, se ejecutan en orden tres pasos:
**difusion -> fotofisica -> emision de fotones**.

### 4.1 Movimiento: difusion Browniana (`paso_difusion`)

Para una particula difundiendo libremente en 1D con coeficiente `D`, la
posicion en el instante `t + dt` sigue:

```
pos(t + dt) ~ Normal( pos(t),  sigma^2 ),      sigma^2 = 2 * D * dt
```

Esta es la relacion de Einstein para el movimiento Browniano: la varianza
del desplazamiento crece linealmente con el tiempo. El codigo aplica esto
de forma independiente en `x` y en `y` (difusion isotropica en el plano),
convirtiendo unidades de nm a pixeles de la muestra con `cfg.pixel_nm`
(la misma propiedad que usa `sim_imagenes.py`), y opcionalmente en `z`
(en nm directamente, ya que asi las espera `BancoPSF`).

La deriva de platina se suma como un termino determinista
`drift * dt` encima del paso aleatorio: representa un corrimiento mecanico
constante, no aleatorio.

Las particulas con `movil = False` (fraccion `fraccion_inmovil`) no se
mueven en absoluto: sirven para simular moleculas fijas o inmovilizadas.

Una vez que una particula se fotoblanquea, se congela en su ultima
posicion. Es una simplificacion deliberada: como ya no emite fotones, su
posicion fisica real (si sigue moviendose o no) no tiene ningun efecto
observable en el video, asi que seguir actualizandola seria trabajo
computacional sin ningun efecto visible.

### 4.2 Fotofisica: cadena de Markov de tres estados (`paso_fotofisica`)

Cada particula esta en uno de tres estados:

```
        k_on
  OFF ---------> ON ---------> BLEACHED
        <---------   k_bleach
           k_off
```

- **OFF**: apagada pero recuperable (parpadeo tipico de fluoroforos:
  transiciones a estados oscuros/triplete, protonacion, etc.).
- **ON**: emitiendo fotones.
- **BLEACHED**: fotoblanqueada, estado absorbente — una vez ahi, nunca
  vuelve a emitir.

Esto es un **proceso de Markov de tiempo continuo**: cada transicion tiene
una tasa en 1/s, y el tiempo de espera hasta que ocurre es exponencial. La
simulacion, sin embargo, avanza en pasos discretos de tamano `dt = 1/fps`,
asi que hay que convertir tasas a probabilidades por paso:

```
P(la transicion con tasa k ya ocurrio en una ventana dt) = 1 - exp(-k * dt)
```

Esta es la funcion de distribucion acumulada de una exponencial: es exacta
(no una aproximacion de primer orden), asi que el resultado no depende de
si `fps` es alto o bajo.

Desde el estado ON hay dos transiciones en competencia: a OFF (tasa
`k_off_s`) y a BLEACHED (tasa `k_bleach_s`). Para un proceso de Poisson con
dos "relojes" en competencia, la probabilidad de que *alguno* de los dos
ya haya sonado en la ventana `dt` es:

```
p_deja_on = 1 - exp(-(k_off_s + k_bleach_s) * dt)
```

y, condicionado a que la particula efectivamente deja ON, la probabilidad
de que haya sido el reloj de fotoblanqueo (y no el de apagado reversible)
es simplemente la proporcion de sus tasas:

```
P(fue bleach | dejo ON) = k_bleach_s / (k_off_s + k_bleach_s)
```

Esto es la propiedad de "adelgazamiento" (*thinning*) de procesos de
Poisson, y es exacto, no una heuristica.

**Cantidades derivadas utiles** (calculadas en `resumen_video`):

- *Duty cycle* en estado estacionario (ignorando fotoblanqueo):
  `k_on / (k_on + k_off)` — la fraccion del tiempo que, en promedio, una
  particula esta ON.
- *Tiempo ON acumulado antes de fotoblanquearse*: exactamente
  `1 / k_bleach_s`. Esto es un resultado no trivial pero conocido en
  fotofisica: como el reloj de fotoblanqueo **solo corre mientras la
  particula esta ON**, el tiempo total (acumulado, sumando todas las
  rachas ON) que una particula pasa emitiendo antes de fotoblanquearse
  tiene distribucion exponencial de tasa `k_bleach_s`, sin importar cuantas
  veces haya parpadeado OFF/ON en el camino (por la propiedad de perdida
  de memoria de la exponencial aplicada al "reloj interno" de tiempo ON).

### 4.3 Fotones emitidos por frame (`fotones_frame`)

Si la particula no esta ON, aporta 0 fotones. Si esta ON, en vez de emitir
siempre exactamente `mu_fotones`, el brillo fluctua frame a frame
alrededor de ese promedio (variaciones de orientacion del dipolo emisor,
de la intensidad de excitacion local, etc.):

```
N ~ Gamma(forma = 1/cv^2, escala = mu_fotones * cv^2)
```

Se eligio una distribucion Gamma (en vez de, por ejemplo, una Normal)
porque:

1. Tiene **soporte no negativo**: nunca puede dar un numero de fotones
   negativo, algo que una Normal si podria hacer para `cv` grande.
2. Se puede parametrizar exactamente por su media (`mu_fotones`) y su
   coeficiente de variacion (`fotones_cv`), lo cual hace que `fotones_cv`
   tenga una interpretacion directa e independiente de la escala de
   brillo.

Este numero (`N`) es el numero **esperado** de fotones de esa particula en
ese frame; el ruido de disparo real (Poisson) se aplica una sola vez sobre
la imagen ya sumada (particulas + fondo) dentro de `aplicar_camara`, igual
que en `sim_imagenes.py`.

---

## 5. Dinamica de poblacion

### 5.1 Inmigracion: aparicion de nuevas particulas

Si `tasa_llegada_s > 0`, en cada frame se sortea cuantas particulas nuevas
"entran" al campo de vision extendido:

```
n_nuevas ~ Poisson(tasa_llegada_s * dt)
```

(el numero de eventos de un proceso de Poisson en una ventana de tiempo
`dt` es, por definicion, Poisson de parametro `tasa * dt`). Cada particula
nueva se crea con `crear_emisor`, con posicion aleatoria uniforme en el
campo de vision (mas el margen `margen_px`), profundidad `z` aleatoria en
`z_rango_nm`, y estado inicial ON/OFF segun `frac_on_inicial`.

Esto permite mantener una **densidad de particulas visibles
aproximadamente constante** a lo largo de un video largo, compensando las
que se van fotoblanqueando — el regimen tipico de una adquisicion real de
localizacion de una sola molecula (SMLM), donde el fotoblanqueo continuo
se compensa con activacion continua de nuevas moleculas.

Con `tasa_llegada_s = 0` (el valor por defecto) la poblacion es **cerrada**:
solo existen las particulas sorteadas en el frame 0, y su numero solo
puede disminuir (por fotoblanqueo o por salir del campo de vision).

### 5.2 Cuando se elimina una particula de la simulacion

Al final de cada frame, una particula se descarta de la lista de
particulas vivas si:

- **se fotoblanqueo** (`estado == "bleached"`): no tiene sentido seguir
  simulandola, nunca mas va a aportar luz; o
- **salio del campo de vision extendido**
  (`dentro_del_fov_extendido` es `False`, es decir, se alejo mas de
  `margen_px * factor_margen_salida` pixeles del sensor).

La segunda condicion es una **simplificacion deliberada**: una particula
que difunde muy lejos del campo de vision podria, en teoria, volver a
acercarse mas tarde por puro azar, pero en un video de duracion tipica esa
probabilidad es despreciable, y no vale la pena pagar el costo
computacional de seguir simulando su camino aleatorio en un area que no se
va a observar. Ver la seccion 7 (Limitaciones) para cuando esta suposicion
podria dejar de ser razonable.

Notar que **antes** de descartarla, la particula siempre se registra una
ultima vez en el ground truth (con `estado="bleached"` y `photons=0`, o con
su ultimo estado si salio del campo de vision), de modo que quien analice
el CSV puede distinguir "la trayectoria termino porque se fotoblanqueo" de
"la trayectoria termino porque salio de cuadro".

---

## 6. Generacion del video completo (`generar_video`)

Resumen del bucle principal:

```
fondo = generar_fondo(cfg, rng)          # una sola vez para todo el video

para t en 0..n_frames-1:
    (inmigracion: agregar particulas nuevas segun tasa_llegada_s)

    img = ceros(H, W)
    para cada particula viva:
        paso_difusion(...)                # mueve x, y, z
        paso_fotofisica(...)              # actualiza estado on/off/bleached
        N = fotones_frame(...)            # 0 si no esta ON
        si N > 0: banco.colocar(img, x, y, z, N)   # suma N * PSF(z) en (x,y)
        registrar fila de ground truth

    video[t] = aplicar_camara(img + fondo, cfg, rng)   # Poisson -> camara -> uint16
```

**Por que el fondo se calcula una sola vez, fuera del bucle**: representa
la iluminacion de campo ancho del microscopio, que en una adquisicion real
no cambia de forma apreciable durante los segundos que dura un video (a
diferencia del ruido de disparo, que si es distinto en cada frame porque
es una realizacion aleatoria nueva de Poisson cada vez que se lee el
sensor). Si se quisiera modelar, por ejemplo, un fondo que decae con el
tiempo por fotoblanqueo del fondo mismo, habria que mover ese calculo
dentro del bucle — no esta implementado aqui porque no era parte del pedido
original, pero el punto de extension queda claro.

---

## 7. Salidas

### 7.1 `video.tif`

Stack de forma `(n_frames, H, W)`, `dtype=uint16`, escrito con
`tifffile.imwrite(..., imagej=True, metadata={"axes": "TYX"})` para que se
abra correctamente como una pila temporal en ImageJ/Fiji (a diferencia de
`sim_imagenes.py`, que usa `axes="ZYX"` porque alli el eje no es tiempo
sino "imagen independiente numero i").

### 7.2 `ground_truth_video.csv`

Una fila por cada combinacion (frame, particula viva en ese frame):

| columna | significado |
|---|---|
| `frame` | indice de cuadro, `0..n_frames-1` |
| `track_id` | identificador **persistente** de la particula (a diferencia de `sim_imagenes.py`, donde `img_id` solo identificaba la imagen, aqui `track_id` permite reconstruir toda la trayectoria de una particula uniendo filas) |
| `x_px`, `y_px` | posicion en pixeles de la muestra en ese frame |
| `z_nm` | profundidad en nm en ese frame |
| `photons` | fotones esperados aportados por esa particula en ese frame (0 si esta OFF o BLEACHED) |
| `estado` | `"on"`, `"off"` o `"bleached"` en ese frame |

Este formato es directamente util para evaluar un algoritmo de *tracking*
(agrupando por `track_id` se recupera la trayectoria real) y para evaluar
un localizador de una sola molecula (filtrando `estado == "on"` se
recuperan las posiciones realmente visibles en cada frame).

---

## 8. Limitaciones y simplificaciones (explicitas)

Estas son decisiones de diseno conscientes, no descuidos, pero conviene
tenerlas presentes al interpretar resultados o al decidir si hay que
extender el simulador:

1. **Particulas perdidas no vuelven a aparecer.** Una vez que una
   particula sale del margen extendido, se elimina para siempre (seccion
   5.2). Para videos muy largos con difusion muy rapida relativa al campo
   de vision, esto subestima ligeramente cuantas particulas "regresan" por
   azar.
2. **El fondo es estatico durante todo el video** (salvo el ruido de
   disparo, que si varia). No hay fotoblanqueo del fondo ni fluctuaciones
   lentas de iluminacion.
3. **Sin interacciones entre particulas.** Cada particula difunde y
   parpadea de forma completamente independiente; no hay exclusion de
   volumen, ni FRET, ni efectos de aglomeracion.
4. **La cadena de fotofisica es la misma para todas las particulas** salvo
   por la aleatoriedad de sus tiradas: no hay heterogeneidad de tasas
   `k_on`/`k_off`/`k_bleach` entre subpoblaciones (por ejemplo, distintas
   especies de fluoroforo). Si se necesitara, se podria sortear estas tasas
   por particula en `crear_emisor` en vez de tomarlas fijas de `cfg`.
5. **La deriva de platina es una velocidad constante conocida.** Una
   deriva real suele tener componentes de baja frecuencia mas complejos
   (no lineales); aqui se modelo de la forma mas simple que sigue siendo
   util para probar algoritmos de correccion de deriva.

---

## 9. Como correrlo

```bash
python3 sim_videos.py
```

Esto imprime en consola un resumen de la optica (heredado de
`sim_imagenes.py`, via `cfg.resumen()`) y un resumen especifico del video
(`cfg.resumen_video()`: duracion, tamano de paso de difusion, duty cycle,
vida media antes de fotoblanquear, tasa de inmigracion), y escribe
`video.tif` y `ground_truth_video.csv` en el directorio actual.

Para generar un video con otros parametros, se instancia `ConfigVideo` con
los campos que se quieran cambiar, por ejemplo:

```python
from sim_videos import ConfigVideo, BancoPSF, generar_video
import tifffile

cfg = ConfigVideo(
    n_frames=500,
    fps=100.0,
    D_um2_s=0.1,          # difusion mas rapida
    k_off_s=5.0,          # parpadeo mas frecuente
    k_bleach_s=0.02,      # fotoblanqueo mas lento
    tasa_llegada_s=0.3,   # activacion continua de nuevas moleculas
    seed=42,
)
banco = BancoPSF(cfg)               # se puede reutilizar entre corridas
video, gt = generar_video(cfg, banco)
tifffile.imwrite("mi_video.tif", video, imagej=True, metadata={"axes": "TYX"})
gt.to_csv("mi_ground_truth.csv", index=False)
```
