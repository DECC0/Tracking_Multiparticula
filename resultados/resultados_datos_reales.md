# Datos reales del montaje anterior (Yeast6_1): validación del modelo del espejo

Datos de Alex: `/home/decar/Datos Alex/Yeast6_1_MMStack_Pos0.ome.tif` (Micro-Manager 2.0.1, Hamamatsu DCAM, 100 cuadros de 1696 × 900 px, uint16; no están en el repo). Se analizan con

```bash
$PY figuras/analisis_datos_reales.py [--tif RUTA] [--beta-deg 56] [--radio-um 1.5] [--salida figuras/datos_reales]   # ~30 s
```

que imprime todas las cifras de este documento y escribe `figuras/datos_reales/fig_datos_reales_canales.png/.pdf` y `figuras/datos_reales/fig_datos_reales_medidas.png/.pdf`. No usa el simulador ni `localizacion`. Convención del repo: +z hacia el objetivo, β desde el eje óptico, píxel (fila i, columna j) centrado en (x = j, y = i), φ = azimut de m (dirección lateral línea del espejo → muestra, de +x hacia +y). Píxel de 98.48 nm (el de los metadatos, 0.4 µm, está mal).

**Metadatos útiles.** Exposición de 100.0 ms y `ElapsedTime` que avanza 100 ms por cuadro, así que se confirman los 10 Hz. ROI `204-536-1696-900`, binning 1. El comentario de la adquisición dice *"198mW 100ms, around 45 degree angle. 3407Yeasts, sphericalHalfIIntensity"*. Ese "45 degree" coincide con lo que sugiere el acople en m (sección 4), no con 56°.

## Resumen

| | canal izquierdo | canal derecho |
|---|---|---|
| separación real–reflejo | 4416 ± 245 nm | 4633 ± 232 nm |
| φ (reflejo → real, coords. locales) | +84.9° | −169.9° |
| d al espejo con β = 56° / 45° | 3.95 / 3.12 µm | 4.14 / 3.28 µm |
| z_reflejo − z_real con 56° / 45° | −6.55 / −4.42 µm | −6.87 / −4.63 µm |
| ratio reflejo/real, r = 1.5 µm, cociente por cuadro (fondo anillo) | 0.53 ± 0.11 | 0.43 ± 0.03 |
| ídem por regresión a lo largo del fotoblanqueo (fondo anillo) | 0.37 ± 0.01 | 0.42 ± 0.01 |
| acople en t: correlación / pendiente OLS / pendiente VI | 0.39 / 0.26 ± 0.06 / 1.22 ± 0.33 | 0.85 / 0.95 ± 0.06 / 1.20 ± 0.14 |
| pendiente VI en m (56° → +0.375; 45° → 0) | −0.00 ± 0.11 | −0.23 ± 0.10 |

- **Hay espejado entre canales**, sin ambigüedad. La similitud impropia deja un residuo de 143 nm RMS y la propia de 650 nm; en el bootstrap de la afín general, 0 de 2000 remuestreos tienen det > 0. La parte lineal es casi la antidiagonal (x_der ≈ −y_izq, y_der ≈ −x_izq), como se había visto. Con los puntos queda un espejado puro respecto de la recta a −42.6° (la antidiagonal está a −45°). La escala no se puede fijar con una sola partícula (sección 2).
- **Brillo izquierdo / derecho**: 0.361 ± 0.020 sumando real + reflejo (CV 5.6 % entre cuadros). Confirma el ~36 %.
- **Acople a lo largo de la línea del espejo**: se confirma en el canal derecho (correlación 0.85, pendiente 0.95 ± 0.06). En el izquierdo la pendiente OLS (0.26) está atenuada por el ruido de localización del real. Con variable instrumental, usando el mismo real medido en el otro canal, ambos canales dan una pendiente compatible con 1 (1.22 ± 0.33 y 1.20 ± 0.14).
- **β = 56° no encaja con el acople en m.** Con 56°, el reflejo debería moverse +0.375 por cada paso del real perpendicular a la línea; lo medido es −0.00 ± 0.11 (izq) y −0.23 ± 0.10 (der). El β implícito es 45° (IC95 39–51) y 38° (32–44), compatible con el "around 45 degree" del comentario. Hay una salvedad (sección 4), pero las distancias al espejo calculadas con 56° deben tomarse con cautela.
- **El ratio reflejo/real depende mucho del fondo.** El canal izquierdo, que es 3 veces más débil, tiene un sesgo aditivo bajo el reflejo (la franja del espejo). Que el ratio sea distinto por canal **no es robusto**: el cociente por cuadro da izq > der y la regresión da izq < der.
- **Una sola partícula** en todo el video: 2 pistas por canal, presentes en el 100 % de los cuadros. La dispersión entre partículas no se puede estimar.

## 1. Corte y detección

Las dos aperturas de campo (casi circulares, una por canal) **se tocan** cerca del centro, en las filas 335–562, así que no hay columnas vacías en todas las filas. El corte se toma en el valle del perfil de columnas en esas filas: **x = 840**, no en W/2 = 848. El valle mide 99.1 ADU contra 97.6 fuera de las aperturas, lo que indica un leve solapamiento de luz. Es vertical: por bloques de 20 filas cae en x = 837–846. Canal izquierdo = columnas 0–839 (840 px), canal derecho = 840–1695 (856 px). Todas las coordenadas del canal derecho son locales (columna − 840). El cociente de áreas de las aperturas da √(A_der/A_izq) = 1.017.

Detección por cuadro: diferencia de gaussianas (σ 3 y 18 px), máximos locales sobre 10 MAD, fusión de picos a < 16 px (la PSF esférica con anillos da varios máximos) y centroide iterativo (r = 10 px, fondo = mediana del anillo). Después vienen el enlace húngaro en pistas y los pares de pistas de separación estable; el real es la de más señal. Resultado: canal izquierdo, 2 detecciones en los 100 cuadros; canal derecho, 2 en 97 y 3 en 3 cuadros (una espuria que no forma pista). Un par por canal.

## 2. Transformación izquierdo → derecho

Ajuste por mínimos cuadrados con 100 cuadros × 2 puntos (real y reflejo). Las coordenadas son las locales de cada mitad; la transformación es x_der = M x_izq + t.

| modelo | RMS real | RMS reflejo | det | escala(s) | ángulo |
|---|---|---|---|---|---|
| similitud propia | 672 nm | 627 nm | +1.02 | 1.010 | +105.4° |
| **similitud impropia** | **187 nm** | **77 nm** | −1.10 | 1.049 ± 0.003 | −85.2 ± 0.2° |
| rígida impropia (s = 1) | 216 nm | 136 nm | −1 | 1 | −85.2° |
| afín general | 181 nm | 82 nm | −1.03 ± 0.03 | 1.058 / 0.975 | −86.5° |
| similitud impropia solo con el real | 181 nm | 554 nm | −1.00 | 1.005 ± 0.040 | −91.6 ± 1.8° |

Similitud impropia: M = [[+0.0878, −1.0454], [−1.0454, −0.0878]], t = (837.55, 944.41) px. El ángulo θ es el de Q = Rot(θ)·diag(1, −1), un espejado respecto de la recta a θ/2.

- **Espejado: sí.** La propia es 4.5 veces peor y ningún remuestreo bootstrap de la afín tiene det > 0. Con un solo vector (el segmento reflejo → real) la rotación de 90° y el espejado serían indistinguibles. Lo que decide es el recorrido del real perpendicular al segmento (14 px a lo largo del video).
- **La escala y el ángulo no quedan determinados con una partícula**, y hay dos lecturas compatibles:
  1. Con todos los puntos, s = 1.049 y θ = −85.2°. Esa escala sale casi entera del cociente de separaciones (4633 / 4416 = 1.049). El ángulo coincide con el que dan las franjas del espejo de los dos canales, que es independiente de las partículas (−83.5°).
  2. Solo con el real, s = 1.005 ± 0.040 y θ = −91.6 ± 1.8°, cerca de la antidiagonal exacta y del 1.017 de las aperturas. Pero entonces el reflejo del canal izquierdo cae 0.55 µm lejos del derecho, es decir, el reflejo estaría en otro sitio según la polarización (por ejemplo, por un recorte de pupila o una reflectancia que dependen de la polarización).

  La afín general (escalas 1.058 a lo largo del segmento y 0.975 perpendicular) también apunta a que el 5 % de diferencia en la separación no es magnificación. Para resolverlo hace falta calibrar el registro con varias beads o una rejilla repartidas por el campo. El archivo `PSFTimelapse_2_MMStack_Pos0_1.ome.tif` (4.3 GB, en la misma carpeta) podría servir, pero no lo revisé.
- Los residuos del real (~180 nm) son mayores que los del reflejo porque el real del canal izquierdo, débil y con anillos, se localiza peor. La diferencia de pasos del real entre canales da un ruido combinado de ~120 nm a lo largo de t y ~70 nm a lo largo de m.

## 3. Medidas por canal

**Geometría.** La separación es 2d cos β y φ es el azimut de m, con el valor que va directo a `orientacion_espejo_deg` (en ImageJ, −φ). Valores en la tabla del resumen. La separación fluctúa ±0.24 µm entre cuadros (rango 3.95–5.35 µm), porque d cambia con el movimiento.

**Franja del espejo.** Es una franja de luz paralela a t, **asimétrica**: tiene un borde nítido del lado de la muestra y una cola ancha del otro lado (panel superior derecho de `fig_datos_reales_medidas`). Amplitud: ~4 ADU sobre un fondo de ~101. El reflejo cae en el flanco nítido, a +1.7 µm (izq) y +0.8 µm (der) del borde; el real, a +6.1 y +5.5 µm. Si el borde es la intersección espejo/plano focal, z_real sería +0.6 / +1.3 µm con β = 56° y −1.7 / −0.8 µm con 45°. Es indicativo: los dos canales difieren en ~1 µm y no está probado que el borde sea esa línea. Además, el borde forma ~5° con t en ambos canales (−5.0° y +6.5°, con los signos opuestos que impone el espejado), mientras que el modelo exige que el segmento real–reflejo sea perpendicular a la línea del espejo. Queda abierto: o la franja no es exactamente la línea o los centroides tienen un sesgo a lo largo de t.

**Ratio reflejo/real.** Es la intensidad integrada en un disco de radio r menos el fondo, con tres fondos:
- anillo: mediana del anillo [r+3, r+10] px, sin el otro punto;
- franja: se resta antes el perfil medido de la franja y luego el anillo;
- global: una constante, la mediana de la apertura lejos de todo.

También se calcula la **regresión I_reflejo = ρ·I_real + c a lo largo del fotoblanqueo**. Real y reflejo son el mismo emisor, así que c absorbe un error aditivo constante del fondo, cosa que el cociente por cuadro no hace.

| r [µm] | izq, cociente (anillo) | izq, regresión (anillo), c/⟨I_refl⟩ | der, cociente (anillo) | der, regresión (anillo), c/⟨I_refl⟩ |
|---|---|---|---|---|
| 0.50 | 1.00 ± 0.24 | 0.89 (+0.09) | 0.70 ± 0.12 | 0.81 (−0.14) |
| 1.00 | 0.79 ± 0.08 | 0.78 (+0.01) | 0.63 ± 0.05 | 0.69 (−0.08) |
| 1.25 | 0.62 ± 0.09 | 0.50 (+0.16) | 0.52 ± 0.03 | 0.53 (−0.02) |
| **1.50** | **0.53 ± 0.11** | **0.37 (+0.27)** | **0.43 ± 0.03** | **0.42 (+0.03)** |
| 1.75 | 0.51 ± 0.11 | 0.29 (+0.38) | 0.41 ± 0.05 | 0.35 (+0.13) |
| 2.00 | 0.51 ± 0.10 | 0.31 (+0.36) | 0.43 ± 0.05 | 0.37 (+0.13) |
| 2.20 | 0.52 ± 0.08 | 0.38 (+0.25) | 0.45 ± 0.04 | 0.39 (+0.11) |

Con el fondo global, a r = 1.5 µm el cociente sube a 0.71 (izq) y 0.55 (der), y la regresión da 0.44 y 0.46. El fondo con el perfil de la franja da casi lo mismo que el anillo. Lectura:

- El ratio baja con el radio hasta ~1.5–1.75 µm y luego sube un poco por la contaminación del halo del otro punto (los discos casi se tocan a r = 2.2 µm). El real tiene un halo de anillos ancho y el reflejo es compacto. Por eso el ratio de apertura **no es** el `ratio_reflejo` del simulador, que es un cociente de fotones: para calibrarlo hay que aplicar esta misma fotometría a imágenes simuladas con la misma geometría.
- En el canal derecho todos los estimadores coinciden en 0.42–0.46 a r = 1.5 µm, sin tendencia temporal (0.426 / 0.441 entre las dos mitades del video).
- En el canal izquierdo, el cociente por cuadro sube de 0.48 a 0.59 a lo largo del video, y la ordenada de la regresión es el 27 % del reflejo. El real parece blanquearse 3.8× y el reflejo solo 2.3×, algo imposible para un mismo emisor. Es un fondo aditivo mal restado bajo el reflejo (la franja), que pesa 3 veces más en este canal porque es más débil. El valor robusto del canal izquierdo está entre 0.37 (regresión) y 0.44 (regresión con fondo global).
- **Ratio de brillo entre canales** (izq / der, r = 1.5 µm): real 0.339 ± 0.028 (CV 8 %), reflejo 0.413 ± 0.063 (CV 15 %), suma 0.361 ± 0.020 (CV 6 %). Por regresión: real 0.377, reflejo 0.333, suma 0.365. El total es estable (~0.36). En cambio, cuál de los dos ratios reflejo/real es mayor depende del estimador: ρ_izq/ρ_der = 1.22 con el cociente por cuadro y 0.88 con la regresión. Con estos datos, **una diferencia de ratio entre polarizaciones no está demostrada**. Si existe, es de ±20 % como mucho.
- Entre partículas: no estimable (hay una).

**Fotoblanqueo** (no se modela; trabajo futuro): el real del canal derecho pasa de ~14 100 a ~4 750 e⁻ en r = 1.5 µm entre los cuadros 0–4 y 95–99 (×3.0, con 0.46 e⁻/ADU de la Orca Flash 4.0). El reflejo del derecho también cae ×3.0.

## 4. Acople real–reflejo

Pasos cuadro a cuadro proyectados en t (a lo largo de la línea del espejo, t = m rotado 90°) y en m, con m tomado de la dirección media reflejo → real. El modelo (`geometria_espejo.py`) predice:

- a lo largo de t: Δt' = Δt, pendiente 1;
- a lo largo de m: Δm' = −cos(2β)·Δm − sin(2β)·Δz, o sea pendiente +0.375 con β = 56° y 0 con 45° si Δz no se correlaciona con Δm. La deriva rígida de la platina daría 1 en las dos direcciones.

La pendiente OLS (reflejo ~ real) se atenúa por el ruido de localización del real. La **pendiente con variable instrumental (VI)** usa como instrumento los pasos del mismo real medidos en el otro canal, cuyo ruido es independiente: cov(Δreflejo, Δinstrumento) / cov(Δreal, Δinstrumento). Así se elimina esa atenuación (es invariante a la escala y al signo del instrumento).

| | n | corr | OLS ± bootstrap | 1/OLS inversa | VI ± bootstrap (IC95) | corr. real–instrumento |
|---|---|---|---|---|---|---|
| izq, t | 99 | 0.39 | 0.26 ± 0.06 | 1.72 | 1.22 ± 0.33 (0.82–2.05) | 0.44 |
| der, t | 99 | 0.85 | 0.95 ± 0.06 | 1.31 | 1.20 ± 0.14 (0.95–1.51) | 0.44 |
| izq, m | 99 | −0.10 | −0.08 ± 0.07 | — | −0.00 ± 0.11 (−0.21–0.22) | 0.68 |
| der, m | 99 | 0.07 | 0.05 ± 0.08 | — | −0.23 ± 0.10 (−0.45–−0.03) | 0.68 |
| combinados*, t | 99 | 0.69 | 0.65 ± 0.07 | 1.37 | — | — |

\* Posiciones del izquierdo transformado y del derecho, promediadas con igual peso. Sale peor que el derecho solo porque el izquierdo aporta mucho ruido.

- **A lo largo de t se cumple el modelo.** En el derecho la OLS (cota inferior) da 0.95 y la VI 1.20 ± 0.14; en el izquierdo la VI da 1.22 ± 0.33. Las trayectorias del real y del reflejo a lo largo de t se superponen (panel inferior izquierdo de `fig_datos_reales_medidas`).
- **A lo largo de m**, ambos canales excluyen +0.375 (a 3.4σ y 5.8σ). El β implícito, ½·arccos(−pendiente), es 44.9° (IC95 38.8–51.4) en el izquierdo y 38.4° (31.5–44.1) en el derecho. La salvedad: esto supone que Δz no se correlaciona con Δm. Una partícula confinada en una levadura que toca el espejo podría correlacionarlos. Además, el centroide del reflejo, recortado por la pupila, podría depender de z'. Aun así, junto con el comentario de la adquisición, lo más probable es que **β estuviera cerca de 45°, no de 56°**. Habría que confirmarlo con beads (separación lateral contra separación de foco, como en `resultados_calibracion_espejo.md`).

## 5. Figuras

- `figuras/datos_reales/fig_datos_reales_canales.png`: arriba, el campo completo (media de los cuadros 0–9) con el corte en x = 840. Abajo, cada canal ampliado con el real (círculo), el reflejo (cuadrado), la flecha reflejo → real y el borde de la franja del espejo; a la derecha, el canal izquierdo transformado (magenta) sobre el derecho (verde). Los anillos del real coinciden en los dos canales, así que ambos tienen el mismo desenfoque.
- `figuras/datos_reales/fig_datos_reales_medidas.png`: ratio contra radio con los tres fondos, acople en t y en m, perfil de la franja, trayectorias a lo largo de t, ratio de brillo entre canales, intensidad integrada contra cuadro y residuos de la transformación.

## 6. Comparación con los resultados preliminares (una partícula)

| | preliminar | este análisis |
|---|---|---|
| acople der (corr / pendiente) | 0.86 / 0.91 | 0.85 / 0.95 ± 0.06 (VI 1.20 ± 0.14) |
| acople izq | 0.42 | corr 0.39, OLS 0.26 ± 0.06, VI 1.22 ± 0.33 |
| ratio der / izq, r ≥ 1.5 µm | 0.52 / 0.63 | 0.43 / 0.53 (anillo); 0.55 / 0.71 (global); 0.42 / 0.37 (regresión) |
| transformación | det < 0, x_der ≈ −y_izq, y_der ≈ −x_izq | confirmado, con un espejado respecto de la recta a −42.6° |
| brillo izq / der | ~36 % | 36.1 ± 2.0 % |

Los ratios preliminares quedan entre los de fondo local y fondo global, así que la diferencia es de fondo. "Con todas las partículas" no aplica: el video tiene una sola.

## 7. Cómo llevar los dos canales al simulador y a la red

En los dos casos, el registro izq → der tiene que salir de una calibración con beads o una rejilla, no de esta partícula: hoy la escala está entre 1.00 y 1.05 y el ángulo entre −85° y −92°.

**(a) Alinear y sumar en una sola imagen.**
- En los datos: remuestrear el canal izquierdo en la rejilla del derecho con la transformación impropia y sumar.
- En el simulador: un solo canal con fotones N_izq + N_der (el izquierdo aporta ~27 % del total), un ratio efectivo ρ = (f_izq ρ_izq + f_der ρ_der) / (f_izq + f_der), fondo sumado y varianza de lectura doble.
- Pros: la red y el simulador no cambian (`canales = 1`). Se gana ~36 % de señal. La suma de Poisson sigue siendo Poisson.
- Contras:
  - El remuestreo interpola, correlaciona el ruido y rompe la estadística Poisson + Gaussiana que simula `aplicar_camara`. Con s ≈ 1 y −90° exactos bastaría una antitransposición más un desplazamiento subpíxel de Fourier, pero los ~5° residuales obligan a interpolar.
  - Un error de registro de ~100–200 nm ensancha la PSF sumada, del orden de la tolerancia de 150 nm de `metricas.py`.
  - Si el reflejo cae en sitios distintos según la polarización (lectura 2 de la sección 2), la suma lo emborrona.
  - El canal izquierdo, 3 veces más débil, suma su fondo y su ruido de lectura completos.
  - Se pierde la información de polarización.

**(b) Dos canales alineados como entrada (`canales = 2`).**
- En los datos: el canal izquierdo remuestreado en la rejilla del derecho, apilado como (2, H, W).
- En el simulador: renderizar cada canal con su propio número de fotones, ratio, fondo y ruido. Hay dos opciones: en la rejilla común, con el mismo remuestreo que se aplica a los datos, o en la rejilla nativa del canal izquierdo con las posiciones transformadas y después remuestreado igual que los datos reales, que es lo más fiel.
- Pros:
  - La red aprende a pesar el canal ruidoso: el acople del izquierdo solo tiene correlación 0.39 frente a 0.85 del derecho, y promediar con igual peso empeoró el resultado.
  - Admite PSF, ratio y fondo distintos por canal, y es robusta a un registro imperfecto y a diferencias reales entre polarizaciones.
  - La normalización `robusta` ya es por canal.
  - El mecanismo `canales_entrada = canales + 1` de `canal_distancia_espejo` muestra que la entrada ya se generaliza.
- Contras:
  - Hay que implementar `canales > 1` en `localizacion` (hoy lanza `NotImplementedError` a la espera de un simulador biplano) y el render multicanal en `simulador_nuevo`.
  - Datos el doble de grandes y un poco más de cómputo en la primera capa.
  - Sigue haciendo falta el registro.

**Recomendación:** usar (a) como línea base rápida, porque no toca la red, y (b) como objetivo para los datos nuevos a 45°. El canal izquierdo es mucho más débil y la diferencia entre polarizaciones (ratio, posición del reflejo) no está resuelta: (b) no obliga a fijarla de antemano.

**Ratio de reflejo distinto por canal.** Los datos no muestran que exista: es ±20 % como mucho y depende del estimador. Conviene entrenar para ser robusto a esa diferencia en vez de fijarla.
- Simulador: un campo por canal en `Config`, por ejemplo `ratio_reflejo_canales = (ρ_izq, ρ_der)` y `fraccion_canales = (0.27, 0.73)` para el reparto de fotones. Con un solo canal se usa `ratio_reflejo` como hoy, para que todo siga bit-exacto. `config.json` lo guarda como cualquier otro campo del espejo.
- Datos de entrenamiento: sortear por muestra un ρ global (por ejemplo en [0.3, 0.7], que cubre todas las estimaciones de arriba) y un factor por canal ρ_c = ρ·(1 + δ_c), con δ_c en ±0.2. Debe ir en un stream propio, `spawn_key = (idx, 1003, ...)`, igual que la geometría (1001) y la célula (1002), para no mover las demás muestras.
- Modelo físico alternativo: reflectancias de Fresnel R_s/R_p del espejo según el ángulo de incidencia y la orientación del dipolo. Antes de adoptarlo habría que medirlo con beads en las dos polarizaciones.
- En el caso (a), todo esto se reduce al ratio efectivo ponderado por fotones.
- En los dos casos, calibrar `ratio_reflejo` (cociente de fotones) contra el ratio de apertura medido aquí aplicando la misma fotometría a imágenes simuladas con la misma geometría.

## Pendiente

- Registro entre canales con beads o una rejilla (escala, ángulo y si el reflejo coincide entre polarizaciones).
- Confirmar β (¿45° o 56°?) con un z-stack de beads, y si el borde de la franja es la línea espejo/plano focal (y por qué forma ~5° con t).
- Calibrar el ratio de apertura contra el `ratio_reflejo` del simulador.
- Fotoblanqueo (~3× en 100 cuadros): no modelado.
